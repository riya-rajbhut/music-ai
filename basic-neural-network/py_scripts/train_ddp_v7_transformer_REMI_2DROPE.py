import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pathlib
import pickle
import time
import random
import warnings
from contextlib import nullcontext

warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import numpy as np
import pretty_midi as pm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        print("Downloading dataset...", flush=True)
        download_url_to_file(url, str(zip_target), progress=True)
        import zipfile
        with zipfile.ZipFile(zip_target, 'r') as zip_ref:
            zip_ref.extractall(base_path)

    return extracted_folder


def convert_midi_to_notes_remi(midi_file_path: str) -> np.ndarray:
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
        raise FileNotFoundError(f"No MIDI files found in {dataset_root}")

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

class RemiTokenDataset(data.Dataset):
    def __init__(self, song_note_arrays, seq_len, hop_length, augment):
        self.seq_len = seq_len
        self.augment = augment
        self.hop_length = hop_length
        self.song_pitches = []
        self.song_active_pitches = []
        self.index_map = []
        
        for song_notes in song_note_arrays:
            notes_array = np.asarray(song_notes, dtype=np.int64)
            if len(notes_array) <= self.seq_len:
                continue

            # Vectorized Active Pitch tracking (Forward Fill)
            is_pitch = (notes_array < 128)
            pitch_ids = np.where(is_pitch, notes_array, -1)
            idx = np.maximum.accumulate(np.where(is_pitch, np.arange(len(notes_array)), 0))
            active_pitches = pitch_ids[idx]
            active_pitches[active_pitches == -1] = 60 # Default to Middle C if no prior pitch exists
            
            song_idx = len(self.song_pitches)
            self.song_pitches.append(torch.tensor(notes_array, dtype=torch.long))
            self.song_active_pitches.append(torch.tensor(active_pitches, dtype=torch.long))
            
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
        full_active = self.song_active_pitches[song_idx][start_idx:end_idx].clone()
        
        if self.augment:
            shift = random.randint(-5, 5)
            is_pitch = full_seq < 128
            if is_pitch.any():
                if (full_seq[is_pitch].min() + shift >= 0) and (full_seq[is_pitch].max() + shift <= 127):
                    full_seq[is_pitch] += shift
                    full_active = torch.clamp(full_active + shift, 0, 127)

        input_seq = full_seq[:-1]     
        target_seq = full_seq[1:]     
        input_active = full_active[:-1] 

        return input_seq, target_seq, input_active

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
# 3. NATIVE 2D ROPE TRANSFORMER ARCHITECTURE
# ==========================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class Harmonic2DRotaryEmbedding(nn.Module):
    def __init__(self, dim_per_head, max_seq_len=4096, max_pitch=128):
        super().__init__()
        self.half_dim = dim_per_head // 2
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        
        # Precompute Temporal Frequencies
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs_t = torch.einsum("i,j->ij", t, inv_freq)
        emb_t = torch.cat((freqs_t, freqs_t), dim=-1)
        self.register_buffer("cos_time", emb_t.cos(), persistent=False)
        self.register_buffer("sin_time", emb_t.sin(), persistent=False)
        
        # Precompute Pitch Frequencies
        p = torch.arange(max_pitch, dtype=torch.float32)
        freqs_p = torch.einsum("i,j->ij", p, inv_freq)
        emb_p = torch.cat((freqs_p, freqs_p), dim=-1)
        self.register_buffer("cos_pitch", emb_p.cos(), persistent=False)
        self.register_buffer("sin_pitch", emb_p.sin(), persistent=False)

    def forward(self, q, k, pitch_seq):
        B, num_heads, S, _ = q.shape
        
        q_time, q_pitch = q.chunk(2, dim=-1)
        k_time, k_pitch = k.chunk(2, dim=-1)
        
        time_seq = torch.arange(S, device=q.device)
        cos_t = self.cos_time[time_seq].view(1, 1, S, self.half_dim)
        sin_t = self.sin_time[time_seq].view(1, 1, S, self.half_dim)
        
        cos_p = self.cos_pitch[pitch_seq].unsqueeze(1) # (B, 1, S, half_dim)
        sin_p = self.sin_pitch[pitch_seq].unsqueeze(1)
        
        q_t_rot = apply_rotary_pos_emb(q_time, cos_t, sin_t)
        k_t_rot = apply_rotary_pos_emb(k_time, cos_t, sin_t)
        
        q_p_rot = apply_rotary_pos_emb(q_pitch, cos_p, sin_p)
        k_p_rot = apply_rotary_pos_emb(k_pitch, cos_p, sin_p)
        
        q_rot = torch.cat([q_t_rot, q_p_rot], dim=-1)
        k_rot = torch.cat([k_t_rot, k_p_rot], dim=-1)
        return q_rot, k_rot

class CausalSelfAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.attn_dropout = dropout_rate
        
        self.rope2d = Harmonic2DRotaryEmbedding(self.head_dim)

    def forward(self, x, pitch_seq):
        B, S, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope2d(q, k, pitch_seq)
        
        # PyTorch Fused SDPA kernel
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        out = self.resid_dropout(self.c_proj(out))
        return out

class TransformerBlock2D(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention2D(embed_dim, num_heads, dropout_rate)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, pitch_seq):
        x = x + self.attn(self.ln_1(x), pitch_seq)
        x = x + self.mlp(self.ln_2(x))
        return x

class Harmonic2DMusicTransformer(nn.Module):
    def __init__(self, num_tokens=384, embed_dim=768, num_heads=12, num_layers=12, dropout_rate=0.1):
        super().__init__()
        # Position embedding is entirely removed (handled organically by 2D RoPE in attention)
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.drop = nn.Dropout(dropout_rate)
        
        self.blocks = nn.ModuleList([
            TransformerBlock2D(embed_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_tokens, bias=False)
        self.head.weight = self.token_embedding.weight # Weight Tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_seq, pitch_seq):
        x = self.token_embedding(token_seq)
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x, pitch_seq)
            
        x = self.ln_f(x)
        logits = self.head(x)
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

    train_dataset = RemiTokenDataset(train_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['train_augment'])
    val_dataset = RemiTokenDataset(val_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['val_augment'])
    test_dataset = RemiTokenDataset(test_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['test_augment'])

    if is_main_process:
        print(f"Dataset split — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}", flush=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    num_workers = min(8, max(4, (os.cpu_count() or 4) // world_size))
    train_loader = data.DataLoader(train_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=True, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=num_workers, persistent_workers=True)

    # Initialize Custom 2D RoPE Transformer
    model = Harmonic2DMusicTransformer(
        num_tokens=384,
        embed_dim=hparams['embed_dim'],
        num_heads=hparams['num_heads'],
        num_layers=hparams['num_layers'],
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
        train_correct_t = torch.zeros((), device=gpu, dtype=torch.float64)
        train_total = 0

        grad_accum_steps = hparams.get('grad_accum_steps', 1)

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_pitch, y_pitch, x_active) in enumerate(train_loader):
            x_pitch = x_pitch.cuda(gpu, non_blocking=True)
            y_pitch = y_pitch.cuda(gpu, non_blocking=True)
            x_active = x_active.cuda(gpu, non_blocking=True)

            is_last_micro_batch = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader)
            sync_context = nullcontext() if is_last_micro_batch else model.no_sync()

            with sync_context:
                with autocast("cuda"):
                    preds = model(x_pitch, x_active)
                    flat_y = y_pitch.reshape(-1)
                    flat_pitch_logits = preds['pitch'].reshape(-1, 384)

                    predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                    train_correct_t += (predicted_pitch == flat_y).sum()
                    train_total += flat_y.size(0)
                    
                    loss_pitch = criterion_pitch(flat_pitch_logits, flat_y) / grad_accum_steps

                scaler.scale(loss_pitch).backward()

            if is_last_micro_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss_t += (loss_pitch.detach() * grad_accum_steps)

            if is_main_process and (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{hparams['epochs']}] | "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {loss_pitch.item() * grad_accum_steps:.4f}",
                    flush=True
                )
        running_loss = running_loss_t.item()
        train_correct = train_correct_t.item()

        # Validation Loop
        model.eval()
        val_loss_tot = 0.0
        val_correct, val_total = 0, 0

        # NEW: Cache for high-loss sequences
        hard_mistakes = []
        MISTAKE_THRESHOLD = 4.5 # Tweak this: only log sequences with loss higher than this
        
        with torch.no_grad():
            for x_pitch, y_pitch, x_active in val_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)
                x_active = x_active.cuda(gpu, non_blocking=True)
                B, S = x_pitch.shape

                with autocast("cuda"):
                    preds = model(x_pitch, x_active)
                    flat_y = y_pitch.reshape(-1)
                    flat_pitch_logits = preds['pitch'].reshape(-1, 384)

                    predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                    val_correct += (predicted_pitch == flat_y).sum().item()
                    val_total += flat_y.size(0)
                    
                    loss_pitch = criterion_pitch(flat_pitch_logits, flat_y)
                    val_loss_tot += loss_pitch.item()

                    # NEW: Efficient Mistake Profiling
                    if is_main_process: # Only profile on GPU 0 to avoid DDP sync overhead
                        # Calculate per-token loss without reducing to a scalar
                        unreduced_loss = F.cross_entropy(flat_pitch_logits, flat_y, reduction='none')
                        # Reshape to (Batch, Sequence) and average across the sequence length
                        seq_losses = unreduced_loss.view(B, S).mean(dim=1)
                        
                        # Find indices of sequences that exceed the threshold
                        hard_idx = (seq_losses > MISTAKE_THRESHOLD).nonzero(as_tuple=True)[0]
                        
                        for idx in hard_idx:
                            # Move immediately to CPU to keep VRAM free
                            hard_mistakes.append({
                                'epoch': epoch + 1,
                                'loss': seq_losses[idx].item(),
                                'input_seq': x_pitch[idx].cpu().clone().numpy(),
                            })

        metrics = torch.tensor(
            [
                running_loss / len(train_loader),
                val_loss_tot / len(val_loader),
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
            train_l,
            val_l,
            train_correct_all, train_total_all,
            val_correct_all, val_total_all
        ) = metrics.tolist()

        train_l /= world_size
        val_l /= world_size

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
                "hard_mistakes_logged": len(hard_mistakes)
            }, step=epoch + 1)

            print(
                f"Epoch [{epoch+1}/{hparams['epochs']}] | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_l:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_l:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"Mistakes Logged: {len(hard_mistakes)} | "
                f"Time: {time.time() - epoch_start:.1f}s",
                flush=True
            )

# NEW: Save the mistakes to disk sequentially
            if hard_mistakes:
                mistake_file = artifacts_root / "hard_mistakes_dataset.pkl"
                # Append if file exists, write if new
                mode = "ab" if mistake_file.exists() else "wb"
                with open(mistake_file, mode) as f:
                    pickle.dump(hard_mistakes, f)

            if val_l < best_val_pitch_loss:
                best_val_pitch_loss = val_l
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
        print("\n=== Commencing Test Evaluation using Best Checkpoint ===", flush=True)
        best_ckpt = torch.load(best_checkpoint_path, map_location=f'cuda:{gpu}', weights_only=True)
        model.module.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()
        
        test_loader = data.DataLoader(test_dataset, batch_size=hparams['batch_size_per_gpu'], shuffle=False, num_workers=num_workers)
        test_correct, test_total = 0, 0
        
        with torch.no_grad():
            for x_pitch, y_pitch, x_active in test_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)
                x_active = x_active.cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch, x_active)
                    flat_y = y_pitch.reshape(-1)
                    flat_pitch_logits = preds['pitch'].reshape(-1, 384)
                    predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                    test_correct += (predicted_pitch == flat_y).sum().item()
                    test_total += flat_y.size(0)

        test_accuracy = (test_correct / test_total) if test_total > 0 else 0.0
        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%", flush=True)
        wandb.log({"test/accuracy": test_accuracy})
        wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    hyperparameters = {
        'seq_len': 768,             
        'batch_size_per_gpu': 32,             # Increased from 16. Utilizes GPU fully and halves steps.
        'grad_accum_steps': 1,                # Lowered to 1 since batch size is increased.
        'epochs': 40,              
        'patience': 10,              
        'lr': 1e-4,                           # Boosted slightly for training a clean initialization from scratch
        'warmup_epochs': 2,         
        'weight_decay': 0.01,        
        'label_smoothing': 0.05,     
        'dropout_rate': 0.1,        
        'seed': 53,
        'years_to_use': None,
        'hop_length': 512,                    # Pushed from 256. Slashes dataset overlap, massively speeds up epochs.
        'train_augment': True,       
        'val_augment': False,
        'test_augment': False,
        'embed_dim': 768,                     # Base GPT-2 Architecture Sizing
        'num_heads': 12,
        'num_layers': 12,
    }
    gpus_available = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if gpus_available > 1:
        print(f"Running DDP across {gpus_available} GPUs.", flush=True)
        torch.multiprocessing.spawn(main_worker, args=(gpus_available, hyperparameters), nprocs=gpus_available, join=True)
    else:
        print("Running single-process fallback.", flush=True)
        main_worker(0, 1, hyperparameters)