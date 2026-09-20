#REMI - Revamped MIDI its a music tokenizaton method that converts MIDI files into a 
# sequence of tokens that represent musical events like pitch, time, and duration. 
# It is designed to capture the structure and nuances of music, making it suitable for 
# training machine learning models, especially in the context of music generation and analysis.

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
import torch.nn.functional as F

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


def convert_midi_to_notes_remi(midi_file_path: str) -> np.ndarray:
    """Parses a MIDI file into a 1D sequence of REMI tokens (TimeShift, Pitch, Duration)."""
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    if not midi_data.instruments:
        return np.array([], dtype=np.int64)

    instrument = midi_data.instruments[0]
    sorted_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    if not sorted_notes:
        return np.array([], dtype=np.int64)
    #print(f"sorted_notes: {sorted_notes[:32]}")  # Print the first 10 notes for debugging
    tokens = []
    prev_start = sorted_notes[0].start
    
    # Define our temporal resolution (e.g., 32 bins per second = ~31.25ms per tick)
    TICKS_PER_SECOND = 32 

    for note in sorted_notes:
        # 1. Calculate time in seconds (your original logic)
        step_sec = note.start - prev_start
        duration_sec = note.end - note.start
        
        # 2. Quantize seconds into discrete integer ticks
        # Cap at 127 so we don't exceed our vocabulary limits
        step_ticks = min(int(step_sec * TICKS_PER_SECOND), 127) 
        duration_ticks = min(max(int(duration_sec * TICKS_PER_SECOND), 1), 127) # Ensure duration is at least 1
        
        # 3. Shift integers into their designated vocabulary blocks
        time_shift_token = 128 + step_ticks
        pitch_token = note.pitch
        duration_token = 256 + duration_ticks
        
        # 4. Append as a flat sequence: "Wait -> Play Note -> Hold Note"
        tokens.extend([time_shift_token, pitch_token, duration_token])
        
        prev_start = note.start

    #print(tokens[:32])
    # Return a 1D array of integers, not a DataFrame
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
        # Check if the numpy array is not empty
        if tokens_array.size > 0:
            all_songs.append(tokens_array)

    print(f"midi files found: {len(all_midi_files)}")
    print(f"Converted {len(all_songs)} songs to REMI token sequences.")
    print(f"Sample token sequence (first 32 tokens) from the first song: {all_songs[0][:32]}")
    print(f"Sample token sequence (last 32 tokens) from the last song: {all_songs[-1][-32:]}")

    return all_songs


def load_or_create_note_cache(dataset_root: pathlib.Path, is_main_process: bool, years_to_use=None) -> list:
    # Bumped version tag to "v6_remi_tokens" to invalidate any old/corrupted pkl cache
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
    """Causal sequence-to-sequence dataset for REMI tokens."""
    def __init__(self, song_note_arrays, seq_len, hop_length,augment):
        self.seq_len = seq_len
        self.augment = augment
        self.hop_length = hop_length
        self.song_pitches = []
        self.index_map = []
        
        
        for song_notes in song_note_arrays:
            # song_notes is a 1D array of REMI tokens now
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
        
        # Clone so we don't accidentally modify the original data in memory
        full_seq = self.song_pitches[song_idx][start_idx:end_idx].clone()
        
       ## if self.augment:
       ##     shift = random.randint(-5, 5)
       ##     # Identify which tokens are pitch tokens (0-127)
       ##     is_pitch = full_seq < 128
       ##     # Shift only the pitches, and clamp them to stay within 0-127 bounds
       ##     full_seq[is_pitch] = (full_seq[is_pitch] + shift).clamp_(0, 127)

        if self.augment:
            is_pitch = full_seq < 128
            pitches = full_seq[is_pitch]

            if pitches.numel() > 0:
                # A conservative starting range. Preserve intervals exactly.
                min_shift = max(-3, -int(pitches.min()))
                max_shift = min(3, 127 - int(pitches.max()))

                shift = random.randint(min_shift, max_shift)
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
# 3. MODEL ARCHITECTURE
# ==========================================

class OptimizedMusicTransformer(nn.Module):
    """Predicts next REMI tokens in sequence using causal self-attention."""
    def __init__(self, num_tokens, hidden_size, num_layers, num_heads, seq_len, dropout_rate):
        super().__init__()

        self.num_tokens = num_tokens

        self.token_embed = nn.Embedding(num_tokens, hidden_size)
        self.pos_embed = nn.Embedding(seq_len, hidden_size)

        # 0: pitch, 1: time shift, 2: duration
        self.type_embed = nn.Embedding(3, hidden_size)

        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False,
        )

        self.output_head = nn.Linear(hidden_size, num_tokens)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

        # Independently initialize each cloned encoder layer.
        self.apply(self._init_weights)

        # MultiheadAttention's combined QKV weight is not an nn.Linear.
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_seq):
        _, length = token_seq.shape

        if length > self.pos_embed.num_embeddings:
            raise ValueError("Input exceeds configured seq_len.")

        positions = torch.arange(
            length, device=token_seq.device
        ).unsqueeze(0)

        token_types = token_seq // 128

        x = (
            self.token_embed(token_seq)
            + self.pos_embed(positions)
            + self.type_embed(token_types)
        )
        x = self.input_dropout(self.input_norm(x))

        x = self.transformer(
            x,
            mask=self.causal_mask[:length, :length],
            is_causal=True,
        )

        return {"pitch": self.output_head(x)}
#        super().__init__()
#        self.num_tokens = num_tokens
#        
#        # Single embedding layer for all 384 tokens
#        self.token_embed = nn.Embedding(num_embeddings=num_tokens, embedding_dim=hidden_size)
#        self.pos_embed = nn.Embedding(num_embeddings=seq_len, embedding_dim=hidden_size)
#        self.input_norm = nn.LayerNorm(hidden_size)
#        
#        encoder_layer = nn.TransformerEncoderLayer(
#            d_model=hidden_size,
#            nhead=num_heads,
#            dim_feedforward=hidden_size * 4,
#            dropout=dropout_rate,
#            activation='gelu',
#            batch_first=True,
#            norm_first=True
#        )
#        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#        
#        # Single output head mapping back to the 384-token vocabulary
#        self.output_head = nn.Linear(hidden_size, num_tokens)
#
#    def forward(self, token_seq):
#        batch_size, seq_length = token_seq.size()
#        
#        # Embed tokens and add positional encodings
#        tok_embeds = self.token_embed(token_seq)
#        
#        positions = torch.arange(seq_length, device=token_seq.device).unsqueeze(0).expand(batch_size, seq_length)
#        pos_embeds = self.pos_embed(positions)
#        
##        x = tok_embeds + pos_embeds
#        x = self.input_norm(x)
#        
#        # Upper-triangular causal mask prevents position t from attending to t+1...T
#        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=token_seq.device)
#        transformer_out = self.transformer(x, mask=causal_mask, is_causal=True)
#        
#        # Final logits for the 384 vocabulary: (B, T, 384)
#        logits = self.output_head(transformer_out)
#        
#        return {'pitch': logits} # Kept key as 'pitch' to minimize changes in your training loop

class ExactDistributedEvalSampler(data.Sampler):
    """Partitions examples without padding or duplication."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(
            range(self.rank, len(self.dataset), self.world_size)
        )

    def __len__(self):
        return len(
            range(self.rank, len(self.dataset), self.world_size)
        )

@torch.inference_mode()
def evaluate(model, loader, gpu):
    # Exact sharding can produce different batch counts across ranks.
    # Bypass DDP forward-time collectives during evaluation.
    net = model.module if isinstance(model, DDP) else model
    net.eval()

    device = torch.device("cuda", gpu)

    # NLL sum, token count, correct count,
    # pitch correct/count, time correct/count, duration correct/count
    stats = torch.zeros(9, device=device, dtype=torch.float64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast("cuda"):
            logits = net(x)["pitch"]

        flat_logits = logits.reshape(-1, logits.size(-1))
        targets = y.reshape(-1)
        predictions = flat_logits.argmax(dim=-1)
        correct = predictions.eq(targets)

        stats[0] += F.cross_entropy(
            flat_logits.float(),
            targets,
            reduction="sum",
        ).double()

        stats[1] += targets.numel()
        stats[2] += correct.sum()

        token_types = targets // 128

        for token_type in range(3):
            mask = token_types.eq(token_type)
            offset = 3 + 2 * token_type
            stats[offset] += (correct & mask).sum()
            stats[offset + 1] += mask.sum()

    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    values = stats.cpu().tolist()

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    return {
        "loss": ratio(values[0], values[1]),
        "accuracy": ratio(values[2], values[1]),
        "pitch_accuracy": ratio(values[3], values[4]),
        "time_accuracy": ratio(values[5], values[6]),
        "duration_accuracy": ratio(values[7], values[8]),
    }


# ==========================================
# 4. MAIN WORKER & TRAINING LOOP
# ==========================================

def main_worker(gpu, world_size, hparams):
    rank = gpu
#    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)

    seed = hparams["seed"] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device("cuda", gpu),
    )
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
        print("Hyperparameters:")
        for key, value in hparams.items():
            print(f"  {key}: {value}")

        wandb.config.update(hparams)

    dataset_root = pathlib.Path('data/maestro-v3.0.0')
    if is_main_process:
        download_maestro_dataset()
    dist.barrier()

    converted_notes = load_or_create_note_cache(dataset_root, is_main_process, hparams['years_to_use'])
    train_notes, val_notes, test_notes = split_song_arrays(converted_notes, seed=hparams['seed'])

    train_dataset = BasicRNNForMusic(train_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'],augment=hparams['train_augment'])
    val_dataset = BasicRNNForMusic(val_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['val_augment'])
    test_dataset = BasicRNNForMusic(test_notes, seq_len=hparams['seq_len'], hop_length=hparams['hop_length'], augment=hparams['test_augment'])

    if is_main_process:
        print(f"Dataset split — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,seed=hparams["seed"])
    #val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_sampler = ExactDistributedEvalSampler(
        val_dataset, rank, world_size
    )
    test_sampler = ExactDistributedEvalSampler(
        test_dataset, rank, world_size
    )
    num_workers = min(8, max(4, (os.cpu_count() or 4) // world_size))
    train_loader = data.DataLoader(train_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=True, persistent_workers=True)


    val_loader = data.DataLoader(
        val_dataset,
        batch_size=hparams["batch_size_per_gpu"],
        sampler=val_sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=hparams["batch_size_per_gpu"],
        sampler=test_sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "No training batches. Reduce batch_size_per_gpu "
            "or check preprocessing and dataset size."
        )

    if len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Validation and test datasets must be nonempty.")
    #val_loader = data.DataLoader(val_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=num_workers, persistent_workers=True)
    #test_loader = data.DataLoader(test_dataset, batch_size=hparams['batch_size_per_gpu'], pin_memory=True, num_workers=num_workers, shuffle=False)

    model = OptimizedMusicTransformer(
        num_tokens=hparams['num_tokens'],
        hidden_size=hparams['hidden_size'], 
        num_layers=hparams['num_layers'],
        num_heads=hparams['num_heads'],
        seq_len=hparams['seq_len'],
        dropout_rate=hparams['dropout_rate']
    ).cuda(gpu)
    model = DDP(model, device_ids=[gpu])

    criterion_pitch = nn.CrossEntropyLoss(label_smoothing=hparams['label_smoothing'])
#    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'], fused=True)
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    optimizer = optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": hparams["weight_decay"],
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=hparams["lr"],
        betas=(0.9, 0.999),
        fused=True,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-3,
        threshold_mode="rel",
        min_lr=1e-6,
    )
   # warmup_epochs = hparams['warmup_epochs']
   # warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1))
   # cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(hparams['epochs'] - warmup_epochs, 1), eta_min=1e-5)
   # scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

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
#        running_pitch_loss_t = torch.zeros((), device=gpu, dtype=torch.float64)
        train_correct_t = torch.zeros((), device=gpu, dtype=torch.float64)
        train_total = 0

        for batch_idx, (x_pitch, y_pitch) in enumerate(train_loader):
            x_pitch = x_pitch.cuda(gpu, non_blocking=True)
            y_pitch = y_pitch.cuda(gpu, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                preds = model(x_pitch)
                
                # Flatten sequence tokens: (B, T, C) -> (B*T, C)
                flat_y = y_pitch.reshape(-1)
                logits = preds["pitch"]
                flat_pitch_logits = logits.reshape(-1, logits.size(-1))
                predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
                train_correct_t += (predicted_pitch == flat_y).sum()
                train_total += flat_y.size(0)
                loss_pitch = criterion_pitch(flat_pitch_logits, flat_y)
                
                train_loss = loss_pitch

            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            # .detach() only — stays on GPU, no CPU/GPU sync happens here

            running_loss_t += train_loss.detach()
#            running_pitch_loss_t += loss_pitch.detach()

        # Single sync point for the whole epoch's training stats, instead of one per batch
#        running_loss = running_loss_t.item()
#        running_pitch_loss = running_pitch_loss_t.item()
#        train_correct = train_correct_t.item()


        # Validation Loop
#        model.eval()
#        val_loss_tot, val_loss_p = 0.0, 0.0
#        val_correct, val_total = 0, 0
#        
#        with torch.no_grad():
#            for x_pitch, y_pitch in val_loader:
#                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
#                y_pitch = y_pitch.cuda(gpu, non_blocking=True)
#
#                with autocast("cuda"):
#                    preds = model(x_pitch)
#                    
#                    flat_y = y_pitch.reshape(-1)
#                    flat_pitch_logits = preds['pitch'].reshape(-1, 384)
#
#                    predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
#
#                    val_correct += (predicted_pitch == flat_y).sum().item()
#                    val_total += flat_y.size(0)
#                    
#                    loss_pitch = criterion_pitch(flat_pitch_logits, flat_y)
#                    
#                    val_loss = loss_pitch
#                    val_loss_tot += val_loss.item()
#                    val_loss_p += loss_pitch.item()
#
#
#        metrics = torch.tensor(
#            [
#                running_loss / len(train_loader),
##                running_pitch_loss / len(train_loader),
#                val_loss_tot / len(val_loader),
#                val_loss_p / len(val_loader),
#                train_correct,
#                train_total,
#                val_correct,
#                val_total
#            ],
#            device=gpu,
#            dtype=torch.float64,
#        )
#        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
#        
#        (
#            train_l, train_p_l, 
#            val_l, val_p_l, 
#            train_correct_all, train_total_all,
#            val_correct_all, val_total_all
#        ) = metrics.tolist()
#
#        train_l /= world_size
#        train_p_l /= world_size
#        val_l /= world_size
#        val_p_l /= world_size
#
#        train_acc = train_correct_all / train_total_all if train_total_all else 0.0
#        val_acc = val_correct_all / val_total_all if val_total_all else 0.0
#        
#
#        #current_lr = optimizer.param_groups[0]["lr"]
#        #scheduler.step()
#        current_lr = optimizer.param_groups[0]["lr"]
#        scheduler.step(val_l)

        validation = evaluate(model, val_loader, gpu)

        # Training batches have a fixed size because drop_last=True.
        train_stats = torch.stack([
            running_loss_t,
            train_correct_t,
            torch.tensor(float(train_total), device=gpu, dtype=torch.float64),
            torch.tensor(float(len(train_loader)), device=gpu, dtype=torch.float64),
        ])

        dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)

        loss_sum, correct_sum, token_count, batch_count = train_stats.tolist()

        train_l = loss_sum / batch_count
        train_acc = correct_sum / token_count

        val_l = validation["loss"]
        val_p_l = val_l  # Keeps your existing checkpoint logic working.
        val_acc = validation["accuracy"]
        if not np.isfinite(val_l):
            raise RuntimeError(f"Non-finite validation loss: {val_l}")

        # LR used during the epoch that just completed.
        current_lr = optimizer.param_groups[0]["lr"]

        # Execute on EVERY rank, using the globally aggregated validation loss.
        scheduler.step(val_l)

        # LR that will be used for the next epoch.
        next_lr = optimizer.param_groups[0]["lr"]

        if is_main_process:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "next_learning_rate": next_lr,
                "train/loss": train_l,
                "train/accuracy": train_acc,
                "val/loss": val_l,
                "val/accuracy": val_acc,
                "val/pitch_accuracy": validation["pitch_accuracy"],
                "val/time_accuracy": validation["time_accuracy"],
                "val/duration_accuracy": validation["duration_accuracy"],                
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
    
    # Ensure all GPUs wait for training to completely finish
    dist.barrier()
    
    # Load the best weights saved during the validation loop
    # map_location ensures the weights are loaded safely across the correct GPUs
    best_ckpt = torch.load(best_checkpoint_path, map_location=f'cuda:{gpu}', weights_only=True)
    model.module.load_state_dict(best_ckpt["model_state_dict"])
    
    if is_main_process:
        print("\n=== Commencing Test Evaluation using Best Checkpoint ===")
        
#    model.eval()
#    test_correct = torch.zeros((), device=gpu, dtype=torch.float64)
#    test_total = 0
#
##    with torch.no_grad():
 #       for x_pitch, y_pitch in test_loader:
#            x_pitch = x_pitch.cuda(gpu, non_blocking=True)
#            y_pitch = y_pitch.cuda(gpu, non_blocking=True)
#
#            with autocast("cuda"):
#                preds = model(x_pitch)
#                
#                flat_y = y_pitch.reshape(-1)
#                flat_pitch_logits = preds['pitch'].reshape(-1, 384)
#                
#                predicted_pitch = torch.argmax(flat_pitch_logits, dim=1)
#                test_correct += (predicted_pitch == flat_y).sum()
#                test_total += flat_y.size(0)
#
    # Aggregate test results across all GPUs
#    dist.all_reduce(test_correct, op=dist.ReduceOp.SUM)
    
#    test_accuracy = (test_correct.item() / (test_total * world_size)) if test_total > 0 else 0.0

#    if is_main_process:
#        print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
#        wandb.log({"test/accuracy": test_accuracy})

    test_metrics = evaluate(model, test_loader, gpu)

    if is_main_process:
        print(
            f"Test NLL: {test_metrics['loss']:.4f} | "
            f"Overall: {100 * test_metrics['accuracy']:.2f}% | "
            f"Pitch: {100 * test_metrics['pitch_accuracy']:.2f}% | "
            f"Time: {100 * test_metrics['time_accuracy']:.2f}% | "
            f"Duration: {100 * test_metrics['duration_accuracy']:.2f}%"
        )

        wandb.log({
            f"test/{name}": value
            for name, value in test_metrics.items()
        })

    if is_main_process:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    hyperparameters = {
        "seq_len": 768,
        "hidden_size": 768,
        "num_layers": 6,
        "num_heads": 8,
        "dropout_rate": 0.2,

        "batch_size_per_gpu": 32,
        "epochs": 40,
        "patience": 10,

        "lr": 1e-4,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,

        "seed": 53,
        "years_to_use": None,
        "hop_length": 256,

        "train_augment": True,
        "val_augment": False,
        "test_augment": False,
        'num_tokens': 384
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