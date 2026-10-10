import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pathlib
import pickle
import time
import random
import warnings
import math
import csv
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


def load_or_create_note_cache(dataset_root: pathlib.Path, is_main_process: bool) -> dict:
    cache_version = "v11_official_split_cache"
    cache_file = dataset_root / f'converted_notes_{cache_version}.pkl'

    if is_main_process:
        if cache_file.exists():
            try:
                with cache_file.open('rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                cache_file.unlink()

        download_maestro_dataset()
        all_midi_files = list(dataset_root.glob('**/*.midi'))
        song_dict = {}
        for midi_file in all_midi_files:
            rel_path = midi_file.relative_to(dataset_root)
            tokens = convert_midi_to_notes_remi(str(midi_file))
            if tokens.size > 0:
                song_dict[str(rel_path).replace('\\', '/')] = tokens

        temp_cache = cache_file.with_name('converted_notes.tmp')
        with temp_cache.open('wb') as f:
            pickle.dump(song_dict, f)
        os.replace(temp_cache, cache_file)
        return song_dict
    else:
        while not cache_file.exists():
            time.sleep(1)
        while True:
            try:
                with cache_file.open('rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                time.sleep(1)


def get_official_split_songs(dataset_root: pathlib.Path, song_dict: dict):
    csv_path = dataset_root / 'maestro-v3.0.0.csv'
    train_songs, val_songs, test_songs = [], [], []

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_key = row['midi_filename'].replace('\\', '/')
            if midi_key in song_dict:
                tokens = song_dict[midi_key]
                split = row['split']
                if split == 'train':
                    train_songs.append(tokens)
                elif split == 'validation':
                    val_songs.append(tokens)
                elif split == 'test':
                    test_songs.append(tokens)

    return train_songs, val_songs, test_songs

# ==========================================
# 2. PYTORCH DATASET & MODULO-3 TRACKING
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

            is_pitch = (notes_array < 128)
            pitch_ids = np.where(is_pitch, notes_array, -1)
            idx = np.maximum.accumulate(np.where(is_pitch, np.arange(len(notes_array)), 0))
            active_pitches = pitch_ids[idx]
            active_pitches[active_pitches == -1] = 60
            
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

        return input_seq, target_seq, input_active, start_idx

# ==========================================
# 3. CYCLIC 3D ROPE & ARCFACE TRANSFORMER
# ==========================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class Harmonic3DRotaryEmbedding(nn.Module):
    def __init__(self, dim_per_head, max_seq_len=4096, max_pitch=128):
        super().__init__()
        self.dim_time = dim_per_head // 2
        self.dim_chroma = dim_per_head // 4
        self.dim_octave = dim_per_head - self.dim_time - self.dim_chroma

        inv_freq_time = 1.0 / (10000 ** (torch.arange(0, self.dim_time, 2).float() / self.dim_time))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs_t = torch.einsum("i,j->ij", t, inv_freq_time)
        emb_t = torch.cat((freqs_t, freqs_t), dim=-1)
        self.register_buffer("cos_time", emb_t.cos(), persistent=False)
        self.register_buffer("sin_time", emb_t.sin(), persistent=False)

        num_chroma_pairs = self.dim_chroma // 2
        chroma_k = torch.arange(1, num_chroma_pairs + 1, dtype=torch.float32)
        inv_freq_chroma = (2.0 * torch.pi * chroma_k) / 12.0
        c = torch.arange(12, dtype=torch.float32)
        freqs_c = torch.einsum("i,j->ij", c, inv_freq_chroma)
        emb_c = torch.cat((freqs_c, freqs_c), dim=-1)
        self.register_buffer("cos_chroma", emb_c.cos(), persistent=False)
        self.register_buffer("sin_chroma", emb_c.sin(), persistent=False)

        inv_freq_octave = 1.0 / (10000 ** (torch.arange(0, self.dim_octave, 2).float() / self.dim_octave))
        o = torch.arange(16, dtype=torch.float32)
        freqs_o = torch.einsum("i,j->ij", o, inv_freq_octave)
        emb_o = torch.cat((freqs_o, freqs_o), dim=-1)
        self.register_buffer("cos_octave", emb_o.cos(), persistent=False)
        self.register_buffer("sin_octave", emb_o.sin(), persistent=False)

    def forward(self, q, k, pitch_seq):
        B, num_heads, S, _ = q.shape
        q_time, q_chroma, q_octave = torch.split(q, [self.dim_time, self.dim_chroma, self.dim_octave], dim=-1)
        k_time, k_chroma, k_octave = torch.split(k, [self.dim_time, self.dim_chroma, self.dim_octave], dim=-1)

        chroma_seq = torch.clamp(pitch_seq % 12, 0, 11)
        octave_seq = torch.clamp(pitch_seq // 12, 0, 15)

        time_seq = torch.arange(S, device=q.device)
        cos_t = self.cos_time[time_seq].view(1, 1, S, self.dim_time)
        sin_t = self.sin_time[time_seq].view(1, 1, S, self.dim_time)

        cos_c = self.cos_chroma[chroma_seq].unsqueeze(1)
        sin_c = self.sin_chroma[chroma_seq].unsqueeze(1)

        cos_o = self.cos_octave[octave_seq].unsqueeze(1)
        sin_o = self.sin_octave[octave_seq].unsqueeze(1)

        q_t_rot = apply_rotary_pos_emb(q_time, cos_t, sin_t)
        k_t_rot = apply_rotary_pos_emb(k_time, cos_t, sin_t)
        q_c_rot = apply_rotary_pos_emb(q_chroma, cos_c, sin_c)
        k_c_rot = apply_rotary_pos_emb(k_chroma, cos_c, sin_c)
        q_o_rot = apply_rotary_pos_emb(q_octave, cos_o, sin_o)
        k_o_rot = apply_rotary_pos_emb(k_octave, cos_o, sin_o)

        return torch.cat([q_t_rot, q_c_rot, q_o_rot], dim=-1), torch.cat([k_t_rot, k_c_rot, k_o_rot], dim=-1)


class CausalSelfAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.attn_dropout = dropout_rate
        self.rope3d = Harmonic3DRotaryEmbedding(self.head_dim)

    def forward(self, x, pitch_seq):
        B, S, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope3d(q, k, pitch_seq)
        
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.resid_dropout(self.c_proj(out))


class TransformerBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention3D(embed_dim, num_heads, dropout_rate)
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


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes=128, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features, labels=None):
        cosine = F.linear(F.normalize(input_features, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        
        if labels is None or not self.training:
            return self.s * cosine

        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, -1 + 1e-7, 1 - 1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class FactorizedHarmonic3DMusicTransformer(nn.Module):
    def __init__(self, num_tokens=384, embed_dim=768, num_heads=12, num_layers=12, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.drop = nn.Dropout(dropout_rate)
        
        self.blocks = nn.ModuleList([
            TransformerBlock3D(embed_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.time_head = nn.Linear(embed_dim, 128, bias=False)
        self.pitch_head = ArcFaceHead(embed_dim, num_classes=128, s=30.0, m=0.50)
        self.duration_head = nn.Linear(embed_dim, 128, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_seq, pitch_seq, start_indices=None, targets=None):
        x = self.token_embedding(token_seq)
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x, pitch_seq)
            
        x = self.ln_f(x)
        B, S, C = x.shape
        
        if start_indices is not None:
            positions = start_indices.view(B, 1) + torch.arange(S, device=x.device).view(1, S) + 1
            token_types = positions % 3
        else:
            token_types = (torch.arange(S, device=x.device).view(1, S) + 1) % 3

        time_mask = (token_types == 0)
        pitch_mask = (token_types == 1)
        dur_mask = (token_types == 2)

        out_logits = torch.full((B, S, 384), float('-inf'), device=x.device, dtype=x.dtype)

        if time_mask.any():
            out_logits[time_mask, 128:256] = self.time_head(x[time_mask]).to(out_logits.dtype)

        if pitch_mask.any():
            pitch_labels = targets[pitch_mask] if targets is not None else None
            out_logits[pitch_mask, 0:128] = self.pitch_head(x[pitch_mask], pitch_labels).to(out_logits.dtype)

        if dur_mask.any():
            out_logits[dur_mask, 256:384] = self.duration_head(x[dur_mask]).to(out_logits.dtype)

        return {'logits': out_logits, 'token_types': token_types}

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

    song_dict = load_or_create_note_cache(dataset_root, is_main_process)
    train_songs, val_songs, test_songs = get_official_split_songs(dataset_root, song_dict)

    train_dataset = RemiTokenDataset(train_songs, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['train_augment'])
    val_dataset = RemiTokenDataset(val_songs, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['val_augment'])
    test_dataset = RemiTokenDataset(test_songs, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['test_augment'])

    if is_main_process:
        print(f"Dataset split (Official CSV) — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}", flush=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    num_workers = min(8, max(4, (os.cpu_count() or 4) // world_size))
    train_loader = data.DataLoader(train_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=True, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=num_workers, persistent_workers=True)

    model = FactorizedHarmonic3DMusicTransformer(
        num_tokens=384,
        embed_dim=hparams['embed_dim'],
        num_heads=hparams['num_heads'],
        num_layers=hparams['num_layers'],
        dropout_rate=hparams['dropout_rate']
    ).cuda(gpu)
    
    model = DDP(model, device_ids=[gpu])

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

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(hparams["epochs"]):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        model.train()

        tr_loss_time_t = torch.zeros((), device=gpu, dtype=torch.float64)
        tr_loss_pitch_t = torch.zeros((), device=gpu, dtype=torch.float64)
        tr_loss_dur_t = torch.zeros((), device=gpu, dtype=torch.float64)
        
        tr_pitch_corr_t = torch.zeros((), device=gpu, dtype=torch.float64)
        tr_pitch_tot_t = torch.zeros((), device=gpu, dtype=torch.float64)
        tr_time_corr_t  = torch.zeros((), device=gpu, dtype=torch.float64)
        tr_time_tot_t   = torch.zeros((), device=gpu, dtype=torch.float64)
        tr_dur_corr_t   = torch.zeros((), device=gpu, dtype=torch.float64)
        tr_dur_tot_t    = torch.zeros((), device=gpu, dtype=torch.float64)

        grad_accum_steps = hparams.get('grad_accum_steps', 1)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_pitch, y_pitch, x_active, start_indices) in enumerate(train_loader):
            x_pitch = x_pitch.cuda(gpu, non_blocking=True)
            y_pitch = y_pitch.cuda(gpu, non_blocking=True)
            x_active = x_active.cuda(gpu, non_blocking=True)
            start_indices = start_indices.cuda(gpu, non_blocking=True)

            is_last_micro_batch = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader)
            sync_context = nullcontext() if is_last_micro_batch else model.no_sync()

            with sync_context:
                with autocast(device_type="cuda"):
                    preds = model(x_pitch, x_active, start_indices, targets=y_pitch)
                    flat_logits = preds['logits'].reshape(-1, 384)
                    flat_y = y_pitch.reshape(-1)
                    token_types = preds['token_types'].reshape(-1)
                    
                    time_idx = (token_types == 0)
                    pitch_idx = (token_types == 1)
                    dur_idx = (token_types == 2)

                    loss = 0.0
                    if time_idx.any():
                        t_logits = flat_logits[time_idx, 128:256]
                        t_targets = flat_y[time_idx] - 128
                        l_time = F.cross_entropy(t_logits, t_targets)
                        loss += l_time
                        tr_loss_time_t += l_time.detach() * t_targets.numel()
                        
                        t_preds = torch.argmax(t_logits, dim=1)
                        tr_time_corr_t += (torch.abs(t_preds - t_targets) <= 2).sum()
                        tr_time_tot_t += t_targets.numel()

                    if pitch_idx.any():
                        p_logits = flat_logits[pitch_idx, 0:128]
                        p_targets = flat_y[pitch_idx]
                        l_pitch = F.cross_entropy(p_logits, p_targets)
                        loss += l_pitch
                        tr_loss_pitch_t += l_pitch.detach() * p_targets.numel()
                        
                        p_preds = torch.argmax(p_logits, dim=1)
                        tr_pitch_corr_t += (p_preds == p_targets).sum()
                        tr_pitch_tot_t += p_targets.numel()

                    if dur_idx.any():
                        d_logits = flat_logits[dur_idx, 256:384]
                        d_targets = flat_y[dur_idx] - 256
                        l_dur = F.cross_entropy(d_logits, d_targets)
                        loss += l_dur
                        tr_loss_dur_t += l_dur.detach() * d_targets.numel()
                        
                        d_preds = torch.argmax(d_logits, dim=1)
                        tr_dur_corr_t += (torch.abs(d_preds - d_targets) <= 2).sum()
                        tr_dur_tot_t += d_targets.numel()
                    
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

            if is_last_micro_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # Validation Loop
        model.eval()
        val_loss_time_tot, val_loss_pitch_tot, val_loss_dur_tot = 0.0, 0.0, 0.0
        val_pitch_corr, val_pitch_tot = 0, 0
        val_time_corr, val_time_tot = 0, 0
        val_dur_corr, val_dur_tot = 0, 0

        with torch.no_grad():
            for x_pitch, y_pitch, x_active, start_indices in val_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)
                x_active = x_active.cuda(gpu, non_blocking=True)
                start_indices = start_indices.cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch, x_active, start_indices, targets=None)
                    flat_logits = preds['logits'].reshape(-1, 384)
                    flat_y = y_pitch.reshape(-1)
                    token_types = preds['token_types'].reshape(-1)

                    time_idx = (token_types == 0)
                    pitch_idx = (token_types == 1)
                    dur_idx = (token_types == 2)

                    if time_idx.any():
                        t_logits = flat_logits[time_idx, 128:256]
                        t_targets = flat_y[time_idx] - 128
                        val_loss_time_tot += F.cross_entropy(t_logits, t_targets, reduction='sum').item()
                        t_preds = torch.argmax(t_logits, dim=1)
                        val_time_corr += (torch.abs(t_preds - t_targets) <= 2).sum().item()
                        val_time_tot += t_targets.numel()

                    if pitch_idx.any():
                        p_logits = flat_logits[pitch_idx, 0:128]
                        p_targets = flat_y[pitch_idx]
                        val_loss_pitch_tot += F.cross_entropy(p_logits, p_targets, reduction='sum').item()
                        p_preds = torch.argmax(p_logits, dim=1)
                        val_pitch_corr += (p_preds == p_targets).sum().item()
                        val_pitch_tot += p_targets.numel()

                    if dur_idx.any():
                        d_logits = flat_logits[dur_idx, 256:384]
                        d_targets = flat_y[dur_idx] - 256
                        val_loss_dur_tot += F.cross_entropy(d_logits, d_targets, reduction='sum').item()
                        d_preds = torch.argmax(d_logits, dim=1)
                        val_dur_corr += (torch.abs(d_preds - d_targets) <= 2).sum().item()
                        val_dur_tot += d_targets.numel()

        metrics = torch.tensor(
            [
                tr_loss_time_t.item(), tr_loss_pitch_t.item(), tr_loss_dur_t.item(),
                tr_pitch_corr_t.item(), tr_pitch_tot_t.item(),
                tr_time_corr_t.item(),  tr_time_tot_t.item(),
                tr_dur_corr_t.item(),   tr_dur_tot_t.item(),
                val_loss_time_tot,      val_loss_pitch_tot,     val_loss_dur_tot,
                float(val_pitch_corr),  float(val_pitch_tot),
                float(val_time_corr),   float(val_time_tot),
                float(val_dur_corr),    float(val_dur_tot)
            ],
            device=gpu, dtype=torch.float64
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        m = metrics.tolist()
        tr_tot_tokens = m[4] + m[6] + m[8]
        val_tot_tokens = m[13] + m[15] + m[17]

        train_l = (m[0] + m[1] + m[2]) / tr_tot_tokens if tr_tot_tokens else 0.0
        val_l = (m[9] + m[10] + m[11]) / val_tot_tokens if val_tot_tokens else 0.0

        train_ppl = math.exp(min(train_l, 20.0))
        val_ppl = math.exp(min(val_l, 20.0))

        train_pitch_acc = m[3] / m[4] if m[4] else 0.0
        train_time_acc  = m[5] / m[6] if m[6] else 0.0
        train_dur_acc   = m[7] / m[8] if m[8] else 0.0
        
        val_pitch_acc   = m[12] / m[13] if m[13] else 0.0
        val_time_acc    = m[14] / m[15] if m[15] else 0.0
        val_dur_acc     = m[16] / m[17] if m[17] else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if is_main_process:
            wandb.log({
                "epoch": epoch + 1, "learning_rate": current_lr,
                "train/loss": train_l, "train/perplexity": train_ppl,
                "train/pitch_accuracy": train_pitch_acc,
                "train/time_shift_acc_tol2": train_time_acc,
                "train/duration_acc_tol2": train_dur_acc,
                "val/loss": val_l, "val/perplexity": val_ppl,
                "val/pitch_accuracy": val_pitch_acc,
                "val/time_shift_acc_tol2": val_time_acc,
                "val/duration_acc_tol2": val_dur_acc
            }, step=epoch + 1)

            print(
                f"Epoch [{epoch+1}/{hparams['epochs']}] | LR: {current_lr:.6f} | "
                f"Train Loss: {train_l:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {val_l:.4f} (PPL: {val_ppl:.2f}) \n"
                f"   -> Pitch Acc: {val_pitch_acc*100:.2f}% | "
                f"Time Acc (±2 tick): {val_time_acc*100:.2f}% | "
                f"Dur Acc (±2 tick): {val_dur_acc*100:.2f}% | "
                f"Time: {time.time() - epoch_start:.1f}s",
                flush=True
            )

            if val_l < best_val_loss:
                best_val_loss = val_l
                epochs_without_improvement = 0
                torch.save({"model_state_dict": model.module.state_dict()}, best_checkpoint_path)

                artifact = wandb.Artifact(
                    name="factorized_harmonic_transformer_best", 
                    type="model",
                    description=f"Best model saved at epoch {epoch + 1} with val_loss {val_l:.4f}"
                )
                artifact.add_file(str(best_checkpoint_path))
                wandb.log_artifact(artifact)
            else:
                epochs_without_improvement += 1

        stop_signal = torch.tensor([1 if epochs_without_improvement >= hparams["patience"] else 0], device=gpu)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        if stop_signal.item() > 0:
            break

    # ==========================================
    # 5. TEST EVALUATION
    # ==========================================
    dist.barrier()
    if is_main_process:
        print("\n=== Commencing Test Evaluation using Best Checkpoint ===", flush=True)
        best_ckpt = torch.load(best_checkpoint_path, map_location=f'cuda:{gpu}', weights_only=True)
        model.module.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()
        
        test_loader = data.DataLoader(test_dataset, batch_size=hparams['batch_size_per_gpu'], shuffle=False, num_workers=num_workers)
        test_pitch_corr, test_pitch_tot = 0, 0
        test_time_corr, test_time_tot = 0, 0
        test_dur_corr, test_dur_tot = 0, 0
        test_loss_sum = 0.0
        
        with torch.no_grad():
            for x_pitch, y_pitch, x_active, start_indices in test_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)
                x_active = x_active.cuda(gpu, non_blocking=True)
                start_indices = start_indices.cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch, x_active, start_indices, targets=None)
                    flat_logits = preds['logits'].reshape(-1, 384)
                    flat_y = y_pitch.reshape(-1)
                    token_types = preds['token_types'].reshape(-1)

                    time_idx = (token_types == 0)
                    pitch_idx = (token_types == 1)
                    dur_idx = (token_types == 2)

                    if time_idx.any():
                        t_logits = flat_logits[time_idx, 128:256]
                        t_targets = flat_y[time_idx] - 128
                        l_val = F.cross_entropy(t_logits, t_targets, reduction='sum').item()
                        test_loss_sum += l_val
                        t_preds = torch.argmax(t_logits, dim=1)
                        test_time_corr += (torch.abs(t_preds - t_targets) <= 2).sum().item()
                        test_time_tot += t_targets.numel()

                    if pitch_idx.any():
                        p_logits = flat_logits[pitch_idx, 0:128]
                        p_targets = flat_y[pitch_idx]
                        l_val = F.cross_entropy(p_logits, p_targets, reduction='sum').item()
                        test_loss_sum += l_val
                        p_preds = torch.argmax(p_logits, dim=1)
                        test_pitch_corr += (p_preds == p_targets).sum().item()
                        test_pitch_tot += p_targets.numel()

                    if dur_idx.any():
                        d_logits = flat_logits[dur_idx, 256:384]
                        d_targets = flat_y[dur_idx] - 256
                        l_val = F.cross_entropy(d_logits, d_targets, reduction='sum').item()
                        test_loss_sum += l_val
                        d_preds = torch.argmax(d_logits, dim=1)
                        test_dur_corr += (torch.abs(d_preds - d_targets) <= 2).sum().item()
                        test_dur_tot += d_targets.numel()

        test_total_tokens = test_pitch_tot + test_time_tot + test_dur_tot
        test_loss = test_loss_sum / test_total_tokens if test_total_tokens else 0.0
        test_ppl = math.exp(min(test_loss, 20.0))
        
        test_pitch_acc = test_pitch_corr / test_pitch_tot if test_pitch_tot else 0.0
        test_time_acc = test_time_corr / test_time_tot if test_time_tot else 0.0
        test_dur_acc = test_dur_corr / test_dur_tot if test_dur_tot else 0.0

        print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}", flush=True)
        print(f"Test Pitch Accuracy (Exact): {test_pitch_acc * 100:.2f}%", flush=True)
        print(f"Test Time Shift Accuracy (±2 tick tolerance): {test_time_acc * 100:.2f}%", flush=True)
        print(f"Test Duration Accuracy (±2 tick tolerance): {test_dur_acc * 100:.2f}%", flush=True)

        wandb.log({
            "test/loss": test_loss,
            "test/perplexity": test_ppl,
            "test/pitch_accuracy": test_pitch_acc,
            "test/time_shift_acc_tol2": test_time_acc,
            "test/duration_acc_tol2": test_dur_acc
        })
        wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    hyperparameters = {
        'seq_len': 768,             
        'embed_dim': 768,           
        'num_layers': 12,            
        'num_heads': 12,
        'batch_size_per_gpu': 32,   
        'epochs': 40,             
        'patience': 8,
        'lr': 1e-4,                            
        'warmup_epochs': 3,         
        'weight_decay': 0.01,
        'seed': 53,
        'hop_length': 384,
        'train_augment': True,
        'val_augment': False,
        'test_augment': False,
        'grad_accum_steps': 1,      
        'dropout_rate': 0.1,        
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