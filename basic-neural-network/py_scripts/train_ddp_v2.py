import os
import pathlib
import pickle
import collections
import time
import warnings


warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import numpy as np
import pandas as pd
import pretty_midi as pm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.hub import download_url_to_file

"""
Trains an LSTM to predict the next note (pitch, timing-until-it-starts, and how long
it rings) in a piano performance, using the MAESTRO dataset. Runs across both GPUs on
a Kaggle T4x2 machine using PyTorch's DistributedDataParallel (DDP): this script's
main_worker() function runs once per GPU, each on its own slice of the data, syncing
gradients after every batch so both GPUs end up training the same model together.

Pipeline: download & parse MIDI -> clip/normalize timing features -> split into
train/val/test by song -> (optionally) augment the train set -> train an LSTM with
three prediction heads (pitch / step / duration) -> evaluate on held-out test songs.
"""

# ==========================================
# 1. DATASET DOWNLOADING & PREPROCESSING
# ==========================================

def download_maestro_dataset(dest_dir: str = 'data') -> pathlib.Path:
    """Downloads and unzips the MAESTRO MIDI dataset if not present."""
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
    """Parses a MIDI file into note pitches, steps, and durations."""
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    if not midi_data.instruments:
        return pd.DataFrame()

    instrument = midi_data.instruments[0]
    sorted_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    if not sorted_notes:
        return pd.DataFrame()

    notes = collections.defaultdict(list)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        notes['pitch'].append(note.pitch)
        notes['step'].append(note.start - prev_start)
        notes['duration'].append(note.end - note.start)
        prev_start = note.start

    return pd.DataFrame({name: np.array(values, dtype=np.float32) for name, values in notes.items()})


def convert_all_songs_to_notes(dataset_root: pathlib.Path, years_to_use=None) -> list:
    """Parses .midi files from selected year folders into numpy arrays."""
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
            all_songs.append(notes_df[['pitch', 'step', 'duration']].to_numpy(dtype=np.float32))

    return all_songs


def load_or_create_note_cache(dataset_root: pathlib.Path, is_main_process: bool, years_to_use=None) -> list:
    """Handles cached dataset reading and writing for multi-GPU safety."""
    cache_version = "v2_sorted_chords"
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


def compute_time_clip_bounds(song_note_arrays, upper_percentile=99.5):
    """Finds a cutoff for unusually long step/duration values (e.g. 99.5th percentile).

    A tiny fraction of notes have huge gaps before them (the performer paused) or ring
    for a very long time — outliers a 64-note window has no way to predict. Left alone,
    these outliers also distort the mean/std used for normalization above. We find the
    cutoff from TRAIN data only, then apply it everywhere (see clip_extreme_time_values).
    """
    raw_time = np.vstack([np.asarray(s, dtype=np.float32)[:, 1:3] for s in song_note_arrays if len(s) > 0])
    step_max = float(np.percentile(raw_time[:, 0], upper_percentile))
    duration_max = float(np.percentile(raw_time[:, 1], upper_percentile))
    return step_max, duration_max


def clip_extreme_time_values(song_note_arrays, step_max, duration_max):
    """Clips time steps and durations to the calculated percentiles."""
    clipped = []
    for song_notes in song_note_arrays:
        notes_array = np.asarray(song_notes, dtype=np.float32).copy()
        if notes_array.size > 0:
            notes_array[:, 1] = np.minimum(notes_array[:, 1], step_max)
            notes_array[:, 2] = np.minimum(notes_array[:, 2], duration_max)
        clipped.append(notes_array)
    return clipped


def compute_time_feature_stats(song_note_arrays):
    """Computes mean/std of log1p transformed step and duration features."""
    logged_time_features = [np.log1p(np.asarray(s)[:, 1:3]) for s in song_note_arrays if len(s) > 0]
    if not logged_time_features:
        return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)

    stacked = np.vstack(logged_time_features)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


# ==========================================
# 2. PYTORCH DATASET & SPLITTING
# ==========================================

class BasicRNNForMusic(data.Dataset):
    """Sliding window dataset for sequence prediction."""

    def __init__(self, song_note_arrays, seq_len=64, time_feature_mean=None, time_feature_std=None):
        self.seq_len = seq_len
        self.song_pitches = []
        self.song_time_features = []
        self.index_map = []
        
        time_mean = np.zeros(2, dtype=np.float32) if time_feature_mean is None else time_feature_mean
        time_std = np.ones(2, dtype=np.float32) if time_feature_std is None else time_feature_std

        for song_notes in song_note_arrays:
            notes_array = np.asarray(song_notes, dtype=np.float32)
            if len(notes_array) <= self.seq_len:
                continue

            pitches = notes_array[:, 0].astype(np.int64)
            time_features = (np.log1p(notes_array[:, 1:3]) - time_mean) / time_std

            song_idx = len(self.song_pitches)
            self.song_pitches.append(torch.tensor(pitches, dtype=torch.long))
            self.song_time_features.append(torch.tensor(time_features, dtype=torch.float32))
            self.index_map.extend((song_idx, start_idx) for start_idx in range(len(pitches) - self.seq_len))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_idx, start_idx = self.index_map[idx]
        end_idx = start_idx + self.seq_len
        return (
            (self.song_pitches[song_idx][start_idx:end_idx], self.song_time_features[song_idx][start_idx:end_idx]),
            (self.song_pitches[song_idx][end_idx], self.song_time_features[song_idx][end_idx])
        )


def split_song_arrays(song_note_arrays, seed, train_ratio=0.8, val_ratio=0.1):
    """Splits full songs into train, validation, and test splits."""
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

class OptimizedMusicRNN(nn.Module):
    """Predicts (pitch, step, duration) for the next note, given the past `seq_len` notes.

    pitch_embedding: turns each pitch (0-127, a category) into a learned vector, so the
        model can learn how pitches relate to each other instead of treating pitch as
        a plain number.
    lstm: reads the sequence one note at a time, carrying a "memory" (hidden state)
        forward so later predictions can be informed by earlier notes.
    pitch/step/duration heads: small networks that turn the LSTM's final memory state
        into the three predictions. pitch is classification (128 possible notes);
        step/duration are regression (real-valued numbers in normalized space).
    """
    def __init__(self, num_pitches=128, pitch_embed_dim=128, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super().__init__()

        self.pitch_embedding = nn.Embedding(num_embeddings=num_pitches, embedding_dim=pitch_embed_dim)
        input_size = pitch_embed_dim + 2
        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.pitch_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_pitches),
        )
        self.step_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pitch_seq, time_seq):
        pitch_embeds = self.pitch_embedding(pitch_seq)
        x = torch.cat([pitch_embeds, time_seq], dim=2)
        x = self.input_norm(x)

        _, (h_n, _) = self.lstm(x)
        last_out = self.dropout(h_n[-1])

        return {
            'pitch': self.pitch_head(last_out),
            'step': self.step_head(last_out),
            'duration': self.duration_head(last_out),
        }


def pitch_to_label(pitch_value):
    """Formats a MIDI pitch number as '<num>(<note>)', e.g. '60(C4)'."""
    pitch_int = int(pitch_value)
    return f"{pitch_int}({pm.note_number_to_name(pitch_int)})"


def format_pitch_sequence(pitch_sequence):
    """Formats a full pitch sequence with MIDI numbers and note names."""
    return "[" + ", ".join(pitch_to_label(p) for p in pitch_sequence) + "]"


def format_topk_predictions(topk_indices, topk_probs):
    """Formats top-k pitch predictions with probabilities."""
    return "[" + ", ".join(
        f"{pitch_to_label(pitch)} ({prob:.3f})"
        for pitch, prob in zip(topk_indices, topk_probs)
    ) + "]"


def format_pitch_prediction_row(split_name, epoch_num, input_sequence, pred_pitch, target_pitch, topk_indices, topk_probs):
    """Creates a readable console log row for one sample prediction."""
    return (
        f"{split_name} Sample | Epoch {epoch_num} | "
        f"Input: {format_pitch_sequence(input_sequence)} -> "
        f"Pred Pitch: {pitch_to_label(pred_pitch)} | "
        f"Target Pitch: {pitch_to_label(target_pitch)} | "
        f"Top-k: {format_topk_predictions(topk_indices, topk_probs)}"
    )


def compute_pitch_frequency_bucket_ids(song_note_arrays):
    """Buckets pitches into rare / medium / common using train-set frequency tertiles."""
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


def summarize_top_pitch_counts(counts, top_n=5):
    """Formats the most frequent pitches in a count vector."""
    counts_tensor = torch.as_tensor(counts, dtype=torch.float64)
    total = counts_tensor.sum().item()
    if total <= 0:
        return "None"

    top_n = min(top_n, counts_tensor.numel())
    top_vals, top_idx = torch.topk(counts_tensor, k=top_n)
    entries = []
    for pitch, count in zip(top_idx.tolist(), top_vals.tolist()):
        if count <= 0:
            continue
        entries.append(f"{pitch_to_label(pitch)}: {100.0 * count / total:.1f}%")
    return ", ".join(entries) if entries else "None"


def summarize_pitch_confusions(confusion_matrix, top_n=5):
    """Formats the most common wrong target->prediction pitch confusions."""
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
        entries.append(f"{pitch_to_label(target_pitch)} -> {pitch_to_label(predicted_pitch)}: {int(count)}")
    return ", ".join(entries) if entries else "None"


def format_named_accuracy_line(label, names, correct_counts, total_counts):
    """Formats per-bucket accuracies for logging."""
    entries = []
    for name, correct, total in zip(names, correct_counts, total_counts):
        correct_value = float(correct)
        total_value = float(total)
        accuracy = 100.0 * correct_value / total_value if total_value > 0 else 0.0
        entries.append(f"{name}: {accuracy:.2f}% ({int(correct_value)}/{int(total_value)})")
    return f"{label} | " + " | ".join(entries)


# ==========================================
# 4. MAIN WORKER & TRAINING LOOP
# ==========================================

def main_worker(gpu, world_size, hparams):

    """Everything one GPU process does: load data, build the model, train, evaluate.

    Because we're training on 2 GPUs, this exact function runs TWICE at the same time
    — once per GPU — each with a different `gpu` number (0 or 1). PyTorch's
    DistributedDataParallel (DDP) keeps both copies of the model in sync: after every
    batch, gradients computed on each GPU are averaged together (all-reduce) so both
    GPUs are always training the same model, just splitting the work of going through
    the data in half.
    """
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    is_main_process = (rank == 0)

    # --- Data Loading ---
    dataset_root = pathlib.Path('data/maestro-v3.0.0')
    if is_main_process:
        download_maestro_dataset()
    dist.barrier()

    converted_notes = load_or_create_note_cache(dataset_root, is_main_process, hparams['years_to_use'])
    train_notes, val_notes, test_notes = split_song_arrays(converted_notes, seed=hparams['seed'])

    step_max, duration_max = compute_time_clip_bounds(train_notes, hparams['time_clip_upper_percentile'])
    train_notes = clip_extreme_time_values(train_notes, step_max, duration_max)
    val_notes = clip_extreme_time_values(val_notes, step_max, duration_max)
    test_notes = clip_extreme_time_values(test_notes, step_max, duration_max)

    mean, std = compute_time_feature_stats(train_notes)
    pitch_bucket_ids_np, pitch_bucket_names, _ = compute_pitch_frequency_bucket_ids(train_notes)
    pitch_bucket_ids_t = torch.tensor(pitch_bucket_ids_np, device=gpu, dtype=torch.long)
    interval_bucket_names = ["Repeat", "Step<=2", "Leap3-5", "Leap6-12", "Leap>12"]

    train_dataset = BasicRNNForMusic(train_notes, seq_len=hparams['seq_len'], time_feature_mean=mean, time_feature_std=std)
    val_dataset = BasicRNNForMusic(val_notes, seq_len=hparams['seq_len'], time_feature_mean=mean, time_feature_std=std)
    test_dataset = BasicRNNForMusic(test_notes, seq_len=hparams['seq_len'], time_feature_mean=mean, time_feature_std=std)

    if is_main_process:
        print(f"Dataset split — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = data.DataLoader(train_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=2, drop_last=True, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=2, persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=hparams['batch_size_per_gpu'], pin_memory=True, num_workers=2, shuffle=False)

    # --- Model Setup ---
    model = OptimizedMusicRNN(hidden_size=hparams['hidden_size'], num_layers=hparams['num_layers']).cuda(gpu)
    model = DDP(model, device_ids=[gpu])

    ## Cross-entropy loss is for classification like pitch, looks at the probability the model assigned to the correct answer and asks: how confident and correct was this?
    # Loss = −log(probability assigned to the correct class)
    #If the model gives the correct answer a high probability (close to 1, i.e., very confident and correct), −log(p) is close to 0. Almost no penalty — great job.
    #If the model gives the correct answer a low probability (it barely considered it, or was confident in a wrong answer), −log(p) shoots up toward a huge number. Big penalty.

    #For SmoothL1Loss, the loss is a combination of L1 and L2 loss. For small errors (less than 1), it uses L2 loss (squared error), which is smooth and differentiable. F
    # For larger errors (greater than 1), it switches to L1 loss (absolute error), which is less sensitive to outliers. This makes SmoothL1Loss more robust to outliers compared to pure L2 loss.
    # Score = distance away (this is basically "L1 loss"). Miss by 2 inches → lose 2 points. Miss by 10 inches → lose 10 points
    #Score = distance squared (this is "L2 loss"). Miss by 2 inches → lose 4 points. Miss by 10 inches → lose 100 points.

    criterion_pitch = nn.CrossEntropyLoss(label_smoothing=hparams['label_smoothing'])
    criterion_time = nn.SmoothL1Loss()

    #Adam looks at momentum (the running average of past gradients) and the variance of the gradients to adaptively adjust the learning rate for each parameter. 
    # This helps the optimizer converge faster and more reliably, especially in scenarios where the loss landscape is complex or has varying curvature.
    #AdamW is a variant of the Adam optimizer that decouples weight decay from the gradient update. In standard Adam, weight decay is applied as part of the gradient update, 
    # which can lead to suboptimal regularization. AdamW applies weight decay directly to the weights after the gradient update, leading to better generalization and performance in many cases.
    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

    # Learning-rate schedule: a short linear warmup (so a high LR doesn't destabilize the
    # very first few steps), followed by one smooth cosine decay down to eta_min over the
    # rest of training.
    warmup_epochs = hparams['warmup_epochs']
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(hparams['epochs'] - warmup_epochs, 1), eta_min=1e-5)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = GradScaler("cuda")

    artifacts_root = pathlib.Path("artifacts")
    best_checkpoint_path = artifacts_root / "best_model.pt"
    history_csv_path = artifacts_root / "history.csv"
    if is_main_process:
        artifacts_root.mkdir(parents=True, exist_ok=True)

    best_val_pitch_loss = float("inf")
    epochs_without_improvement = 0
    history_rows = []

    # --- Training Loop ---
    for epoch in range(hparams["epochs"]):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss, running_pitch_loss = 0.0, 0.0
        train_correct, train_total = 0, 0
        train_debug_rows = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            x_pitch = inputs[0].cuda(gpu, non_blocking=True)
            x_time = inputs[1].cuda(gpu, non_blocking=True)
            y_pitch = targets[0].cuda(gpu, non_blocking=True)
            y_time = targets[1].cuda(gpu, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                preds = model(x_pitch, x_time)
                predicted_pitch = torch.argmax(preds["pitch"], dim=1)

                train_correct += (predicted_pitch == y_pitch).sum().item()
                train_total += y_pitch.size(0)

                loss_pitch = criterion_pitch(preds["pitch"], y_pitch)
                loss_step = criterion_time(preds["step"], y_time[:, 0:1])
                loss_duration = criterion_time(preds["duration"], y_time[:, 1:2])
                train_loss = loss_pitch + hparams["time_loss_weight"] * (loss_step + loss_duration)

                if is_main_process and len(train_debug_rows) < 3:
                    sample_idx = torch.randint(0, y_pitch.size(0), (1,), device=gpu).item()
                    probs = torch.softmax(preds["pitch"][sample_idx], dim=0)
                    topk_probs, topk_indices = torch.topk(probs, k=3)

                    train_debug_rows.append({
                        "epoch": epoch + 1,
                        "target_pitch": int(y_pitch[sample_idx].item()),
                        "pred_pitch": int(predicted_pitch[sample_idx].item()),
                        "input_sequence": x_pitch[sample_idx].detach().cpu().tolist(),
                        "topk_indices": topk_indices.detach().cpu().tolist(),
                        "topk_probs": topk_probs.detach().cpu().tolist(),
                    })

            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += train_loss.item()
            running_pitch_loss += loss_pitch.item()

        # Evaluation
        model.eval()
        val_loss_tot, val_loss_p = 0.0, 0.0
        val_correct, val_total = 0, 0
        debug_rows = []
        val_topk_correct = torch.zeros(3, device=gpu, dtype=torch.float64)
        val_confusion = torch.zeros((128, 128), device=gpu, dtype=torch.int64)
        val_pred_counts = torch.zeros(128, device=gpu, dtype=torch.int64)
        val_target_counts = torch.zeros(128, device=gpu, dtype=torch.int64)
        val_bucket_correct = torch.zeros(3, device=gpu, dtype=torch.float64)
        val_bucket_total = torch.zeros(3, device=gpu, dtype=torch.float64)
        val_interval_correct = torch.zeros(5, device=gpu, dtype=torch.float64)
        val_interval_total = torch.zeros(5, device=gpu, dtype=torch.float64)

        with torch.no_grad():
            for inputs, targets in val_loader:
                x_pitch = inputs[0].cuda(gpu, non_blocking=True)
                x_time = inputs[1].cuda(gpu, non_blocking=True)
                y_pitch = targets[0].cuda(gpu, non_blocking=True)
                y_time = targets[1].cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch, x_time)
                    predicted_pitch = torch.argmax(preds["pitch"], dim=1)
                    top5_indices = torch.topk(preds["pitch"], k=5, dim=1).indices

                    val_correct += (predicted_pitch == y_pitch).sum().item()
                    val_total += y_pitch.size(0)

                    val_topk_correct[0] += (top5_indices[:, :1] == y_pitch.unsqueeze(1)).any(dim=1).sum().item()
                    val_topk_correct[1] += (top5_indices[:, :3] == y_pitch.unsqueeze(1)).any(dim=1).sum().item()
                    val_topk_correct[2] += (top5_indices == y_pitch.unsqueeze(1)).any(dim=1).sum().item()

                    val_pred_counts += torch.bincount(predicted_pitch, minlength=128)
                    val_target_counts += torch.bincount(y_pitch, minlength=128)
                    confusion_indices = y_pitch * 128 + predicted_pitch
                    val_confusion += torch.bincount(confusion_indices, minlength=128 * 128).reshape(128, 128)

                    batch_bucket_ids = pitch_bucket_ids_t[y_pitch]
                    valid_bucket_mask = batch_bucket_ids >= 0
                    if valid_bucket_mask.any():
                        valid_bucket_ids = batch_bucket_ids[valid_bucket_mask]
                        val_bucket_total += torch.bincount(valid_bucket_ids, minlength=3).to(torch.float64)
                        correct_bucket_ids = valid_bucket_ids[(predicted_pitch[valid_bucket_mask] == y_pitch[valid_bucket_mask])]
                        if correct_bucket_ids.numel() > 0:
                            val_bucket_correct += torch.bincount(correct_bucket_ids, minlength=3).to(torch.float64)

                    interval_sizes = (y_pitch - x_pitch[:, -1]).abs()
                    interval_bucket_ids = torch.full_like(interval_sizes, 4)
                    interval_bucket_ids[interval_sizes == 0] = 0
                    interval_bucket_ids[(interval_sizes > 0) & (interval_sizes <= 2)] = 1
                    interval_bucket_ids[(interval_sizes >= 3) & (interval_sizes <= 5)] = 2
                    interval_bucket_ids[(interval_sizes >= 6) & (interval_sizes <= 12)] = 3
                    val_interval_total += torch.bincount(interval_bucket_ids, minlength=5).to(torch.float64)
                    correct_interval_ids = interval_bucket_ids[predicted_pitch == y_pitch]
                    if correct_interval_ids.numel() > 0:
                        val_interval_correct += torch.bincount(correct_interval_ids, minlength=5).to(torch.float64)

                    loss_pitch = criterion_pitch(preds["pitch"], y_pitch)
                    loss_step = criterion_time(preds["step"], y_time[:, 0:1])
                    loss_duration = criterion_time(preds["duration"], y_time[:, 1:2])
                    val_loss = loss_pitch + hparams["time_loss_weight"] * (loss_step + loss_duration)

                    val_loss_tot += val_loss.item()
                    val_loss_p += loss_pitch.item()

                if is_main_process and len(debug_rows) < 10:
                    sample_idx = torch.randint(0, y_pitch.size(0), (1,), device=gpu).item()
                    probs = torch.softmax(preds["pitch"][sample_idx], dim=0)
                    topk_probs, topk_indices = torch.topk(probs, k=3)

                    debug_rows.append({
                        "epoch": epoch + 1,
                        "target_pitch": int(y_pitch[sample_idx].item()),
                        "pred_pitch": int(predicted_pitch[sample_idx].item()),
                        "input_sequence": x_pitch[sample_idx].detach().cpu().tolist(),
                        "topk_indices": topk_indices.detach().cpu().tolist(),
                        "topk_probs": topk_probs.detach().cpu().tolist(),
                    })

        metrics = torch.tensor(
            [
                running_loss / len(train_loader),
                running_pitch_loss / len(train_loader),
                val_loss_tot / len(val_loader),
                val_loss_p / len(val_loader),
                train_correct,
                train_total,
                val_correct,
                val_total,
            ],
            device=gpu,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_topk_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_confusion, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_pred_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_target_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_bucket_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_bucket_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_interval_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_interval_total, op=dist.ReduceOp.SUM)

        (
            train_l,
            train_p_l,
            val_l,
            val_p_l,
            train_correct_all,
            train_total_all,
            val_correct_all,
            val_total_all,
        ) = metrics.tolist()

        train_l /= world_size
        train_p_l /= world_size
        val_l /= world_size
        val_p_l /= world_size

        train_acc = train_correct_all / train_total_all if train_total_all else 0.0
        val_acc = val_correct_all / val_total_all if val_total_all else 0.0
        val_acc_top3 = val_topk_correct[1].item() / val_total_all if val_total_all else 0.0
        val_acc_top5 = val_topk_correct[2].item() / val_total_all if val_total_all else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if is_main_process:
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "train_loss": train_l,
                    "train_pitch_loss": train_p_l,
                    "train_acc": train_acc,
                    "val_loss": val_l,
                    "val_pitch_loss": val_p_l,
                    "val_acc": val_acc,
                    "val_acc_top3": val_acc_top3,
                    "val_acc_top5": val_acc_top5,
                }
            )

            if (epoch + 1) % 5 == 0 and debug_rows:
                pd.DataFrame(debug_rows).to_csv(
                    artifacts_root / f"predictions_epoch_{epoch+1}.csv",
                    index=False,
                )

            print(
                f"Epoch [{epoch+1}/{hparams['epochs']}] | "
                f"Train Loss: {train_l:.4f} (Pitch: {train_p_l:.4f}) | "
                f"Val Loss: {val_l:.4f} (Pitch: {val_p_l:.4f}) | "
                f"Time: {time.time() - epoch_start:.1f}s"
            )

            for row in train_debug_rows:
                print(
                    format_pitch_prediction_row(
                        split_name="Train",
                        epoch_num=row["epoch"],
                        input_sequence=row["input_sequence"],
                        pred_pitch=row["pred_pitch"],
                        target_pitch=row["target_pitch"],
                        topk_indices=row["topk_indices"],
                        topk_probs=row["topk_probs"],
                    )
                )

            for row in debug_rows[:3]:
                print(
                    format_pitch_prediction_row(
                        split_name="Val",
                        epoch_num=row["epoch"],
                        input_sequence=row["input_sequence"],
                        pred_pitch=row["pred_pitch"],
                        target_pitch=row["target_pitch"],
                        topk_indices=row["topk_indices"],
                        topk_probs=row["topk_probs"],
                    )
                )

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{hparams['epochs']}] | "
                    f"LR: {current_lr:.6f} | "
                    f"Train Loss: {train_l:.4f} | Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_l:.4f} | Val Acc: {val_acc*100:.2f}% | "
                    f"Time: {time.time() - epoch_start:.1f}s"
                )
                print(
                    f"Val Pitch Top-k Accuracy | "
                    f"Acc@1: {val_acc*100:.2f}% | "
                    f"Acc@3: {val_acc_top3*100:.2f}% | "
                    f"Acc@5: {val_acc_top5*100:.2f}%"
                )
                print(
                    format_named_accuracy_line(
                        "Val Pitch Frequency Bucket Accuracy",
                        pitch_bucket_names,
                        val_bucket_correct.detach().cpu().tolist(),
                        val_bucket_total.detach().cpu().tolist(),
                    )
                )
                print(
                    format_named_accuracy_line(
                        "Val Interval Bucket Accuracy",
                        interval_bucket_names,
                        val_interval_correct.detach().cpu().tolist(),
                        val_interval_total.detach().cpu().tolist(),
                    )
                )
                print(
                    "Val Most Common Pitch Confusions | "
                    + summarize_pitch_confusions(val_confusion.detach().cpu(), top_n=5)
                )
                print(
                    "Val Pitch Distribution | "
                    f"Pred Top: {summarize_top_pitch_counts(val_pred_counts.detach().cpu(), top_n=5)} | "
                    f"Target Top: {summarize_top_pitch_counts(val_target_counts.detach().cpu(), top_n=5)}"
                )

            if val_p_l < best_val_pitch_loss:
                best_val_pitch_loss = val_p_l
                epochs_without_improvement = 0
                torch.save({"model_state_dict": model.module.state_dict()}, best_checkpoint_path)
            else:
                epochs_without_improvement += 1

            pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)

        stop_signal = torch.tensor(
            [1 if epochs_without_improvement >= hparams["patience"] else 0],
            device=gpu,
        )
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        if stop_signal.item() > 0:
            break

    # --- Test Evaluation ---
    if is_main_process:
        print("\n--- Running Evaluation On Test Set ---")
        checkpoint = torch.load(best_checkpoint_path, map_location='cuda:0')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        correct_pitch, total_samples = 0, 0
        sum_err_step, sum_err_duration = 0.0, 0.0
        test_debug_rows = []
        time_mean_t, time_std_t = torch.tensor(mean, device=0), torch.tensor(std, device=0)

        with torch.no_grad():
            for inputs, targets in test_loader:
                x_pitch, x_time = inputs[0].cuda(0, non_blocking=True), inputs[1].cuda(0, non_blocking=True)
                y_pitch, y_time = targets[0].cuda(0, non_blocking=True), targets[1].cuda(0, non_blocking=True)
                with autocast('cuda'):
                    preds = model.module(x_pitch, x_time)
                    predicted_classes = torch.argmax(preds['pitch'], dim=1)
                    correct_pitch += (predicted_classes == y_pitch).sum().item()
                    total_samples += y_pitch.size(0)

                    pred_step_sec = torch.expm1(preds['step'].squeeze(-1) * time_std_t[0] + time_mean_t[0])
                    pred_dur_sec = torch.expm1(preds['duration'].squeeze(-1) * time_std_t[1] + time_mean_t[1])
                    true_step_sec = torch.expm1(y_time[:, 0] * time_std_t[0] + time_mean_t[0])
                    true_dur_sec = torch.expm1(y_time[:, 1] * time_std_t[1] + time_mean_t[1])

                    sum_err_step += (pred_step_sec - true_step_sec).abs().sum().item()
                    sum_err_duration += (pred_dur_sec - true_dur_sec).abs().sum().item()

                if len(test_debug_rows) < 5:
                    samples_to_add = min(5 - len(test_debug_rows), y_pitch.size(0))
                    for sample_idx in range(samples_to_add):
                        probs = torch.softmax(preds['pitch'][sample_idx], dim=0)
                        topk_probs, topk_indices = torch.topk(probs, k=3)
                        test_debug_rows.append({
                            "input_sequence": x_pitch[sample_idx].detach().cpu().tolist(),
                            "pred_pitch": int(predicted_classes[sample_idx].item()),
                            "target_pitch": int(y_pitch[sample_idx].item()),
                            "topk_indices": topk_indices.detach().cpu().tolist(),
                            "topk_probs": topk_probs.detach().cpu().tolist(),
                        })

        if total_samples > 0:
            print(f"Final Pitch Accuracy: {(correct_pitch / total_samples) * 100:.2f}%")
            print(f"Final Step MAE: {sum_err_step / total_samples:.4f}s | Duration MAE: {sum_err_duration / total_samples:.4f}s")
            for row in test_debug_rows:
                print(
                    format_pitch_prediction_row(
                        split_name="Test",
                        epoch_num="final",
                        input_sequence=row["input_sequence"],
                        pred_pitch=row["pred_pitch"],
                        target_pitch=row["target_pitch"],
                        topk_indices=row["topk_indices"],
                        topk_probs=row["topk_probs"],
                    )
                )

    dist.destroy_process_group()


if __name__ == '__main__':
    hyperparameters = {
        'seq_len': 64,
        'hidden_size': 384,
        'num_layers': 3,
        'batch_size_per_gpu': 1536,
        'epochs': 35,
        'patience': 8,
        'lr': 2e-3,
        'warmup_epochs': 2,
        'weight_decay': 1e-4,
        'time_loss_weight': 0.5,
        'label_smoothing': 0.03,
        'seed': 53,
        'time_clip_upper_percentile': 99.5,
        'years_to_use': [2004, 2006, 2008, 2009, 2011]
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