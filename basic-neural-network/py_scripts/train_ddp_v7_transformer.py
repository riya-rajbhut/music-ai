import os
import pathlib
import pickle
import collections
import time
import random
import warnings

warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import numpy as np
import pandas as pd
import pretty_midi as pm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.hub import download_url_to_file

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


def convert_midi_to_notes(midi_file_path: str) -> pd.DataFrame:
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    if not midi_data.instruments:
        return pd.DataFrame()

    instrument = midi_data.instruments[0]
    sorted_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    if not sorted_notes:
        return pd.DataFrame()

    pitches = [note.pitch for note in sorted_notes]
    return pd.DataFrame({'pitch': np.array(pitches, dtype=np.float32)})


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
        notes_df = convert_midi_to_notes(midi_file)
        if not notes_df.empty:
            all_songs.append(notes_df[['pitch']].to_numpy(dtype=np.float32))

    return all_songs


def load_or_create_note_cache(dataset_root: pathlib.Path, is_main_process: bool, years_to_use=None) -> list:
    cache_version = "v5_causal_seq2seq" 
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
    """Causal sequence-to-sequence dataset."""
    def __init__(self, song_note_arrays, seq_len=128, augment=False, hop_length=None):
        self.seq_len = seq_len
        self.augment = augment
        # Hop length defaults to seq_len for non-overlapping contiguous chunks
        self.hop_length = hop_length if hop_length is not None else seq_len
        self.song_pitches = []
        self.index_map = []
        
        for song_notes in song_note_arrays:
            notes_array = np.asarray(song_notes, dtype=np.float32)
            # Need seq_len + 1 notes to form (input, target) shifted by 1 position
            if len(notes_array) <= self.seq_len:
                continue

            pitches = notes_array[:, 0].astype(np.int64)
            song_idx = len(self.song_pitches)
            self.song_pitches.append(torch.tensor(pitches, dtype=torch.long))
            
            self.index_map.extend(
                (song_idx, start_idx) 
                for start_idx in range(0, len(pitches) - self.seq_len, self.hop_length)
            )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_idx, start_idx = self.index_map[idx]
        end_idx = start_idx + self.seq_len + 1
        
        full_seq = self.song_pitches[song_idx][start_idx:end_idx]
        
        if self.augment:
            shift = random.randint(-5, 5)
            full_seq = (full_seq + shift).clamp_(0, 127)

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
# 3. MODEL ARCHITECTURE
# ==========================================

class OptimizedMusicTransformer(nn.Module):
    """Predicts all next pitches in sequence using causal self-attention."""
    def __init__(self, num_pitches=128, pitch_embed_dim=128, hidden_size=512, num_layers=3, num_heads=8, seq_len=128, dropout_rate=0.1):
        super().__init__()
        self.num_pitches = num_pitches
        
        pc_dim = (pitch_embed_dim * 3) // 4
        oct_dim = pitch_embed_dim - pc_dim
        
        self.pitch_class_embed = nn.Embedding(num_embeddings=12, embedding_dim=pc_dim)
        self.octave_embed = nn.Embedding(num_embeddings=11, embedding_dim=oct_dim)
        self.pos_embed = nn.Embedding(num_embeddings=seq_len, embedding_dim=pitch_embed_dim)
        self.input_norm = nn.LayerNorm(pitch_embed_dim)
        self.embed_proj = nn.Linear(pitch_embed_dim, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pc_feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.oct_feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.pc_head = nn.Linear(hidden_size, 12)
        self.oct_head = nn.Linear(hidden_size, 11)
        self.fusion_layer = nn.Linear(hidden_size * 2, num_pitches)

    def forward(self, pitch_seq):
        batch_size, seq_length = pitch_seq.size()
        
        pitch_class = pitch_seq % 12
        octave = torch.clamp(pitch_seq // 12, 0, 10)
        
        pc_embeds = self.pitch_class_embed(pitch_class)
        oct_embeds = self.octave_embed(octave)
        pitch_embeds = torch.cat([pc_embeds, oct_embeds], dim=2)
        
        positions = torch.arange(seq_length, device=pitch_seq.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_embeds = self.pos_embed(positions)
        
        x = pitch_embeds + pos_embeds
        x = self.input_norm(x)
        x = self.embed_proj(x)
        
        # Upper-triangular causal mask prevents position t from attending to t+1...T
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=pitch_seq.device)
        transformer_out = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # Full Sequence Output: (Batch, Seq_Len, Hidden)
        pc_features = self.pc_feature_extractor(transformer_out)
        oct_features = self.oct_feature_extractor(transformer_out)
        
        pc_logits = self.pc_head(pc_features)
        oct_logits = self.oct_head(oct_features)
        
        combined_features = torch.cat([pc_features, oct_features], dim=-1)
        joint_logits = self.fusion_layer(combined_features)
        
        return {
            'pitch': joint_logits,      # (B, T, 128)
            'pc_logits': pc_logits,    # (B, T, 12)
            'oct_logits': oct_logits    # (B, T, 11)
        }


# ==========================================
# 4. MAIN WORKER & TRAINING LOOP
# ==========================================

def compute_pitch_frequency_bucket_ids(song_note_arrays):
    pitch_counts = np.zeros(128, dtype=np.int64)
    for song_notes in song_note_arrays:
        notes_array = np.asarray(song_notes, dtype=np.float32)
        if notes_array.size == 0:
            continue
        pitches = notes_array[:, 0].astype(np.int64)
        pitch_counts += np.bincount(pitches, minlength=128)

    nonzero_counts = pitch_counts[pitch_counts > 0]
    bucket_ids = np.full(128, -1, dtype=np.int64)
    bucket_names = ["Rare", "Medium", "Common"]

    if nonzero_counts.size == 0:
        return bucket_ids, bucket_names, pitch_counts

    lower_cutoff, upper_cutoff = np.quantile(nonzero_counts, [1 / 3, 2 / 3])
    observed_mask = pitch_counts > 0
    bucket_ids[np.logical_and(observed_mask, pitch_counts < upper_cutoff)] = 1
    bucket_ids[np.logical_and(observed_mask, pitch_counts < lower_cutoff)] = 0
    bucket_ids[np.logical_and(observed_mask, pitch_counts >= upper_cutoff)] = 2
    return bucket_ids, bucket_names, pitch_counts


def summarize_pitch_confusions(confusion_matrix, top_n=5):
    confusion = torch.as_tensor(confusion_matrix, dtype=torch.int64).clone()
    if confusion.numel() == 0:
        return "None"

    confusion.fill_diagonal_(0)
    flat = confusion.reshape(-1)
    nonzero = int((flat > 0).sum().item())
    if nonzero == 0:
        return "None"

    top_vals, top_idx = torch.topk(flat, k=min(top_n, nonzero))
    entries = []
    for flat_idx, count in zip(top_idx.tolist(), top_vals.tolist()):
        if count <= 0:
            continue
        target_pitch = flat_idx // 128
        predicted_pitch = flat_idx % 128
        entries.append(f"{target_pitch}->{predicted_pitch}: {int(count)}")
    return ", ".join(entries) if entries else "None"


def main_worker(gpu, world_size, hparams):
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    is_main_process = (rank == 0)

    if is_main_process:
        wandb_api_key = "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"
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            wandb.login()
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

    pitch_bucket_ids_np, _, _ = compute_pitch_frequency_bucket_ids(train_notes)
    pitch_bucket_ids_t = torch.tensor(pitch_bucket_ids_np, device=gpu, dtype=torch.long)

    train_dataset = BasicRNNForMusic(train_notes, seq_len=hparams['seq_len'], augment=True, hop_length=hparams['seq_len'])
    val_dataset = BasicRNNForMusic(val_notes, seq_len=hparams['seq_len'], augment=False, hop_length=hparams['seq_len'])
    test_dataset = BasicRNNForMusic(test_notes, seq_len=hparams['seq_len'], augment=False, hop_length=hparams['seq_len'])

    if is_main_process:
        print(f"Dataset split — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    num_workers = min(8, max(4, (os.cpu_count() or 4) // world_size))
    train_loader = data.DataLoader(train_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=True, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=num_workers, persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=hparams['batch_size_per_gpu'], pin_memory=True, num_workers=num_workers, shuffle=False)

    model = OptimizedMusicTransformer(
        hidden_size=hparams['hidden_size'], 
        num_layers=hparams['num_layers'],
        seq_len=hparams['seq_len']
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

        running_loss, running_pitch_loss, running_pc_loss, running_oct_loss = 0.0, 0.0, 0.0, 0.0
        train_correct, train_total = 0, 0

        for batch_idx, (x_pitch, y_pitch) in enumerate(train_loader):
            x_pitch = x_pitch.cuda(gpu, non_blocking=True)
            y_pitch = y_pitch.cuda(gpu, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                preds = model(x_pitch)
                
                # Flatten sequence tokens: (B, T, C) -> (B*T, C)
                flat_y = y_pitch.reshape(-1)
                flat_pitch_logits = preds['pitch'].reshape(-1, 128)
                flat_pc_logits = preds['pc_logits'].reshape(-1, 12)
                flat_oct_logits = preds['oct_logits'].reshape(-1, 11)

                predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                train_correct += (predicted_pitch == flat_y).sum().item()
                train_total += flat_y.size(0)

                loss_pitch = criterion_pitch(flat_pitch_logits, flat_y)
                loss_pc = criterion_pitch(flat_pc_logits, flat_y % 12) 
                loss_oct = criterion_pitch(flat_oct_logits, torch.clamp(flat_y // 12, 0, 10))
                
                train_loss = loss_pitch + (hparams['lambda_pc'] * loss_pc) + (hparams['lambda_oct'] * loss_oct)

            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += train_loss.item()
            running_pitch_loss += loss_pitch.item()
            running_pc_loss += loss_pc.item()
            running_oct_loss += loss_oct.item()

        # Validation Loop
        model.eval()
        val_loss_tot, val_loss_p, val_loss_pc, val_loss_oct = 0.0, 0.0, 0.0, 0.0
        val_correct, val_total = 0, 0
        val_pc_correct, val_oct_correct, val_pure_octave_errors = 0, 0, 0
        val_topk_correct = torch.zeros(3, device=gpu, dtype=torch.float64)
        val_confusion = torch.zeros((128, 128), device=gpu, dtype=torch.int64)

        with torch.no_grad():
            for x_pitch, y_pitch in val_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch)
                    
                    flat_y = y_pitch.reshape(-1)
                    flat_pitch_logits = preds['pitch'].reshape(-1, 128)
                    flat_pc_logits = preds['pc_logits'].reshape(-1, 12)
                    flat_oct_logits = preds['oct_logits'].reshape(-1, 11)

                    predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                    top5_indices = torch.topk(flat_pitch_logits, k=5, dim=1).indices

                    val_correct += (predicted_pitch == flat_y).sum().item()
                    val_total += flat_y.size(0)
                    
                    y_pc = flat_y % 12
                    pred_pc = predicted_pitch % 12
                    y_oct = flat_y // 12
                    pred_oct = predicted_pitch // 12
                    
                    val_pc_correct += (pred_pc == y_pc).sum().item()
                    val_oct_correct += (pred_oct == y_oct).sum().item()
                    val_pure_octave_errors += ((pred_pc == y_pc) & (predicted_pitch != flat_y)).sum().item()

                    val_topk_correct[0] += (top5_indices[:, :1] == flat_y.unsqueeze(1)).any(dim=1).sum().item()
                    val_topk_correct[1] += (top5_indices[:, :3] == flat_y.unsqueeze(1)).any(dim=1).sum().item()
                    val_topk_correct[2] += (top5_indices == flat_y.unsqueeze(1)).any(dim=1).sum().item()

                    confusion_indices = flat_y * 128 + predicted_pitch
                    val_confusion += torch.bincount(confusion_indices, minlength=128 * 128).reshape(128, 128)

                    loss_pitch = criterion_pitch(flat_pitch_logits, flat_y)
                    loss_pc = criterion_pitch(flat_pc_logits, flat_y % 12)
                    loss_oct = criterion_pitch(flat_oct_logits, torch.clamp(flat_y // 12, 0, 10))
                    
                    val_loss = loss_pitch + (hparams['lambda_pc'] * loss_pc) + (hparams['lambda_oct'] * loss_oct)

                    val_loss_tot += val_loss.item()
                    val_loss_p += loss_pitch.item()
                    val_loss_pc += loss_pc.item()
                    val_loss_oct += loss_oct.item()

        metrics = torch.tensor(
            [
                running_loss / len(train_loader),
                running_pitch_loss / len(train_loader),
                running_pc_loss / len(train_loader),
                running_oct_loss / len(train_loader),
                val_loss_tot / len(val_loader),
                val_loss_p / len(val_loader),
                val_loss_pc / len(val_loader),
                val_loss_oct / len(val_loader),
                train_correct,
                train_total,
                val_correct,
                val_total,
                val_pc_correct,
                val_oct_correct,
                val_pure_octave_errors
            ],
            device=gpu,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_topk_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_confusion, op=dist.ReduceOp.SUM)

        (
            train_l, train_p_l, train_pc_l, train_oct_l,
            val_l, val_p_l, val_pc_l, val_oct_l,
            train_correct_all, train_total_all,
            val_correct_all, val_total_all,
            val_pc_correct_all, val_oct_correct_all, val_pure_octave_errors_all
        ) = metrics.tolist()

        train_l /= world_size
        train_p_l /= world_size
        train_pc_l /= world_size
        train_oct_l /= world_size
        val_l /= world_size
        val_p_l /= world_size
        val_pc_l /= world_size
        val_oct_l /= world_size

        train_acc = train_correct_all / train_total_all if train_total_all else 0.0
        val_acc = val_correct_all / val_total_all if val_total_all else 0.0
        val_acc_top3 = val_topk_correct[1].item() / val_total_all if val_total_all else 0.0
        val_acc_top5 = val_topk_correct[2].item() / val_total_all if val_total_all else 0.0
        
        val_pc_acc = val_pc_correct_all / val_total_all if val_total_all else 0.0
        val_oct_acc = val_oct_correct_all / val_total_all if val_total_all else 0.0
        val_pure_oct_err_rate = val_pure_octave_errors_all / val_total_all if val_total_all else 0.0

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
                "val/accuracy_top3": val_acc_top3,
                "val/accuracy_top5": val_acc_top5,
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

    if is_main_process:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    hyperparameters = {
        'seq_len': 128,
        'hidden_size': 512,
        'num_layers': 3,
        'batch_size_per_gpu': 128,  # Adjusted to accommodate 128 target predictions per batch item
        'epochs': 40,
        'patience': 8,
        'lr': 5e-4,               
        'warmup_epochs': 5,         
        'weight_decay': 1e-4,
        'lambda_pc': 0.1,       
        'lambda_oct': 0.3,      
        'label_smoothing': 0.0,
        'seed': 53,
        'years_to_use': None
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