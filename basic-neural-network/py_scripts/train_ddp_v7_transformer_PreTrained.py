import os
import pathlib
import pickle
import time
import random
import warnings

warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import numpy as np
import pretty_midi as pm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.hub import download_url_to_file

# Transformers library integration
from transformers import GPT2Config, GPT2LMHeadModel

# ==========================================
# 1. DATASET DOWNLOADING & PREPROCESSING
# ==========================================

def download_maestro_dataset(dest_dir: str = 'data') -> pathlib.Path:
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    base_path = pathlib.Path(dest_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    zip_target = base_path / 'maestro-v3.0.0-midi.zip'
    extracted_folder = base_path / 'maestro-v3.0.0'

    if not (zip_target.exists() and extracted_folder.exists()):
        print("Downloading dataset...")
        download_url_to_file(url, str(zip_target), progress=True)
        import zipfile
        with zipfile.ZipFile(zip_target, 'r') as zip_ref:
            zip_ref.extractall(base_path)

    return extracted_folder


def convert_midi_to_notes_remi(midi_file_path: str) -> np.ndarray:
    """Parses a MIDI file into a 1D sequence of REMI tokens (TimeShift, Pitch, Duration)."""
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    if not midi_data.instruments:
        return np.array([], dtype=np.int64)

    instrument = midi_data.instruments[0]
    sorted_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    if not sorted_notes:
        return np.array([], dtype=np.int64)

    tokens = []
    prev_start = sorted_notes[0].start
    TICKS_PER_SECOND = 32 

    for note in sorted_notes:
        step_sec = note.start - prev_start
        duration_sec = note.end - note.start
        
        step_ticks = min(int(step_sec * TICKS_PER_SECOND), 127) 
        duration_ticks = min(max(int(duration_sec * TICKS_PER_SECOND), 1), 127)
        
        time_shift_token = 128 + step_ticks
        pitch_token = note.pitch
        duration_token = 256 + duration_ticks
        
        tokens.extend([time_shift_token, pitch_token, duration_token])
        prev_start = note.start

    return np.array(tokens, dtype=np.int64)

def convert_all_songs_to_notes(dataset_root: pathlib.Path, years_to_use=None) -> list:
    if years_to_use is None:
        all_midi_files = list(dataset_root.glob('**/*.midi'))
    else:
        all_midi_files = []
        for year in years_to_use:
            all_midi_files.extend((dataset_root / str(year)).glob('*.midi'))

    if not all_midi_files:
        raise FileNotFoundError(f"No MIDI files found in {dataset_root} for years {years_to_use}")

    all_songs = []
    for midi_file in all_midi_files:
        tokens_array = convert_midi_to_notes_remi(midi_file)
        if tokens_array.size > 0:
            all_songs.append(tokens_array)

    return all_songs


def load_or_create_note_cache(dataset_root: pathlib.Path, is_main_process: bool, years_to_use=None) -> list:
    cache_version = "v6_remi_tokens" 
    year_tag = "all" if years_to_use is None else "_".join(map(str, years_to_use))
    cache_file = dataset_root / f'converted_notes_{cache_version}_{year_tag}.pkl'

    if is_main_process:
        if cache_file.exists():
            try:
                with cache_file.open('rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                cache_file.unlink()

        download_maestro_dataset()
        converted_notes = convert_all_songs_to_notes(dataset_root, years_to_use)
        temp_cache = cache_file.with_name('converted_notes.tmp')
        with temp_cache.open('wb') as f:
            pickle.dump(converted_notes, f)
        os.replace(temp_cache, cache_file)
        return converted_notes
    else:
        while not cache_file.exists():
            time.sleep(1)
        while True:
            try:
                with cache_file.open('rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                time.sleep(1)

# ==========================================
# 2. PYTORCH DATASET & SPLITTING
# ==========================================

class BasicRNNForMusic(data.Dataset):
    """Causal sequence-to-sequence dataset for REMI tokens with pitch shift augmentation."""
    def __init__(self, song_note_arrays, seq_len, hop_length, augment):
        self.seq_len = seq_len
        self.augment = augment
        self.hop_length = hop_length
        self.song_pitches = []
        self.index_map = []
        
        for song_notes in song_note_arrays:
            notes_array = np.asarray(song_notes, dtype=np.int64)
            if len(notes_array) <= self.seq_len:
                continue

            song_idx = len(self.song_pitches)
            self.song_pitches.append(torch.tensor(notes_array, dtype=torch.long))
            
            self.index_map.extend(
                (song_idx, start_idx) 
                for start_idx in range(0, len(notes_array) - self.seq_len, self.hop_length)
            )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_idx, start_idx = self.index_map[idx]
        end_idx = start_idx + self.seq_len + 1
        
        full_seq = self.song_pitches[song_idx][start_idx:end_idx].clone()
        
        if self.augment:
            shift = random.randint(-5, 5)
            is_pitch = full_seq < 128
            if is_pitch.any():
                pitches = full_seq[is_pitch]
                if (pitches.min() + shift >= 0) and (pitches.max() + shift <= 127):
                    full_seq[is_pitch] += shift

        input_seq = full_seq[:-1]   # Shape: (seq_len,)
        target_seq = full_seq[1:]   # Shape: (seq_len,)

        return input_seq, target_seq

def split_song_arrays(song_note_arrays, seed, train_ratio=0.8, val_ratio=0.1):
    song_indices = np.random.default_rng(seed).permutation(len(song_note_arrays))
    train_cutoff = int(len(song_indices) * train_ratio)
    val_cutoff = int(len(song_indices) * (train_ratio + val_ratio))

    return (
        [song_note_arrays[i] for i in song_indices[:train_cutoff]],
        [song_note_arrays[i] for i in song_indices[train_cutoff:val_cutoff]],
        [song_note_arrays[i] for i in song_indices[val_cutoff:]]
    )

# ==========================================
# 3. PRETRAINED TRANSFORMER WRAPPER
# ==========================================

class PretrainedMusicTransformer(nn.Module):
    """
    Wraps a Pretrained Hugging Face GPT-2 backbone and adapts token embeddings 
    and output head to match custom REMI vocabulary size (384 tokens).
    """
    def __init__(self, pretrained_model_name="gpt2", num_tokens=384, seq_len=768, dropout_rate=0.15):
        super().__init__()
        
        # Load Pretrained Backbone
        self.transformer = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        
        # Adjust Token Embeddings for REMI Token Set Size (384)
        self.transformer.resize_token_embeddings(num_tokens)
        
        # Update Configuration Dropout & Sequence Settings
        self.transformer.config.resid_pdrop = dropout_rate
        self.transformer.config.embd_pdrop = dropout_rate
        self.transformer.config.attn_pdrop = dropout_rate
        self.transformer.config.n_positions = seq_len

    def forward(self, token_seq):
        # Hugging Face GPT2LMHeadModel produces Causal Attention internally
        outputs = self.transformer(input_ids=token_seq)
        logits = outputs.logits  # Shape: (batch_size, seq_len, num_tokens)
        return {'pitch': logits}

# ==========================================
# 4. MAIN WORKER & TRAINING LOOP
# ==========================================

def main_worker(gpu, world_size, hparams):
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    is_main_process = (rank == 0)

    if is_main_process:
        wandb_api_key = "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"
        wandb.login(key=wandb_api_key)

        wandb.init(
            project="music-rnn-ddp",
            entity="riya-rajbhut-student",
            config=hparams
        )

    dataset_root = pathlib.Path('data/maestro-v3.0.0')
    if is_main_process:
        download_maestro_dataset()
    dist.barrier()

    converted_notes = load_or_create_note_cache(dataset_root, is_main_process, hparams['years_to_use'])
    train_notes, val_notes, test_notes = split_song_arrays(converted_notes, seed=hparams['seed'])

    train_dataset = BasicRNNForMusic(train_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['train_augment'])
    val_dataset = BasicRNNForMusic(val_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['val_augment'])
    test_dataset = BasicRNNForMusic(test_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['test_augment'])

    if is_main_process:
        print(f"Dataset split — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    num_workers = min(8, max(4, (os.cpu_count() or 4) // world_size))
    train_loader = data.DataLoader(train_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=True, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=num_workers, persistent_workers=True)

    # Initialize Pretrained Model Backbone
    model = PretrainedMusicTransformer(
        pretrained_model_name=hparams['pretrained_model_name'],
        num_tokens=384,
        seq_len=hparams['seq_len'],
        dropout_rate=hparams['dropout_rate']
    ).cuda(gpu)
    model = DDP(model, device_ids=[gpu])

    criterion_pitch = nn.CrossEntropyLoss(label_smoothing=hparams['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'], fused=True)

    warmup_epochs = hparams['warmup_epochs']
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(hparams['epochs'] - warmup_epochs, 1), eta_min=1e-5)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = GradScaler("cuda")

    artifacts_root = pathlib.Path("artifacts")
    best_checkpoint_path = artifacts_root / "best_model.pt"
    if is_main_process:
        artifacts_root.mkdir(parents=True, exist_ok=True)

    best_val_pitch_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(hparams["epochs"]):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss_t = torch.zeros((), device=gpu, dtype=torch.float64)
        running_pitch_loss_t = torch.zeros((), device=gpu, dtype=torch.float64)
        train_correct_t = torch.zeros((), device=gpu, dtype=torch.float64)
        train_total = 0

        for batch_idx, (x_pitch, y_pitch) in enumerate(train_loader):
            x_pitch = x_pitch.cuda(gpu, non_blocking=True)
            y_pitch = y_pitch.cuda(gpu, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                preds = model(x_pitch)
                flat_y = y_pitch.reshape(-1)
                flat_pitch_logits = preds['pitch'].reshape(-1, 384)

                predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                train_correct_t += (predicted_pitch == flat_y).sum()
                train_total += flat_y.size(0)
                loss_pitch = criterion_pitch(flat_pitch_logits, flat_y)

            scaler.scale(loss_pitch).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss_t += loss_pitch.detach()
            running_pitch_loss_t += loss_pitch.detach()

        running_loss = running_loss_t.item()
        running_pitch_loss = running_pitch_loss_t.item()
        train_correct = train_correct_t.item()

        # Validation Loop
        model.eval()
        val_loss_tot, val_loss_p = 0.0, 0.0
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for x_pitch, y_pitch in val_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch)
                    flat_y = y_pitch.reshape(-1)
                    flat_pitch_logits = preds['pitch'].reshape(-1, 384)

                    predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                    val_correct += (predicted_pitch == flat_y).sum().item()
                    val_total += flat_y.size(0)
                    
                    loss_pitch = criterion_pitch(flat_pitch_logits, flat_y)
                    val_loss_tot += loss_pitch.item()
                    val_loss_p += loss_pitch.item()

        metrics = torch.tensor(
            [
                running_loss / len(train_loader),
                running_pitch_loss / len(train_loader),
                val_loss_tot / len(val_loader),
                val_loss_p / len(val_loader),
                train_correct,
                train_total,
                val_correct,
                val_total
            ],
            device=gpu,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        (
            train_l, train_p_l, 
            val_l, val_p_l, 
            train_correct_all, train_total_all,
            val_correct_all, val_total_all
        ) = metrics.tolist()

        train_l /= world_size
        train_p_l /= world_size
        val_l /= world_size
        val_p_l /= world_size

        train_acc = train_correct_all / train_total_all if train_total_all else 0.0
        val_acc = val_correct_all / val_total_all if val_total_all else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if is_main_process:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train/loss": train_l,
                "train/accuracy": train_acc,
                "val/loss": val_l,
                "val/accuracy": val_acc,
            }, step=epoch + 1)

            print(
                f"Epoch [{epoch+1}/{hparams['epochs']}] | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_l:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_l:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"Time: {time.time() - epoch_start:.1f}s"
            )

            if val_p_l < best_val_pitch_loss:
                best_val_pitch_loss = val_p_l
                epochs_without_improvement = 0
                torch.save({"model_state_dict": model.module.state_dict()}, best_checkpoint_path)
            else:
                epochs_without_improvement += 1

        stop_signal = torch.tensor([1 if epochs_without_improvement >= hparams["patience"] else 0], device=gpu)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        if stop_signal.item() > 0:
            break

    # ==========================================
    # 5. TEST EVALUATION (POST-TRAINING)
    # ==========================================
    dist.barrier()
    
    if is_main_process:
        print("\n=== Commencing Test Evaluation using Best Checkpoint ===")
        best_ckpt = torch.load(best_checkpoint_path, map_location=f'cuda:{gpu}', weights_only=True)
        model.module.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()
        
        test_loader = data.DataLoader(test_dataset, batch_size=hparams['batch_size_per_gpu'], shuffle=False, num_workers=num_workers)
        test_correct, test_total = 0, 0
        
        with torch.no_grad():
            for x_pitch, y_pitch in test_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch)
                    flat_y = y_pitch.reshape(-1)
                    flat_pitch_logits = preds['pitch'].reshape(-1, 384)
                    predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                    test_correct += (predicted_pitch == flat_y).sum().item()
                    test_total += flat_y.size(0)

        test_accuracy = (test_correct / test_total) if test_total > 0 else 0.0
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
        wandb.log({"test/accuracy": test_accuracy})
        wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    hyperparameters = {
        'pretrained_model_name': 'gpt2', # Pretrained Hugging Face GPT-2 base architecture
        'seq_len': 768,             
        'batch_size_per_gpu': 32,   
        'epochs': 40,              
        'patience': 10,              
        'lr': 5e-5,                  # Fine-tuning requires lower learning rate (e.g., 5e-5 vs 3e-4)          
        'warmup_epochs': 2,         
        'weight_decay': 0.01,        
        'label_smoothing': 0.05,     
        'dropout_rate': 0.1,        
        'seed': 53,
        'years_to_use': None,
        'hop_length': 256,
        'train_augment': True,       
        'val_augment': False,
        'test_augment': False
    }
    gpus_available = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if gpus_available > 1:
        print(f"Running DDP across {gpus_available} GPUs.")
        torch.multiprocessing.spawn(main_worker, args=(gpus_available, hyperparameters), nprocs=gpus_available, join=True)
    else:
        print("Running single-process fallback.")
        main_worker(0, 1, hyperparameters)