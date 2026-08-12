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

import os
import pathlib
import pickle
import collections
import time
import warnings

# pretty_midi's own internals still import the deprecated pkg_resources module, which
# newer setuptools versions warn loudly about. That warning comes from pretty_midi's
# code, not ours, and doesn't affect training, so we silence just that one message.
# This must run BEFORE `import pretty_midi` below, since the warning fires at import time.
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import numpy as np
import pandas as pd
import pretty_midi as pm

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

def download_maestro_dataset(dest_dir: str = 'data'):
    """Downloads and unzips the MAESTRO MIDI dataset, unless it's already on disk."""
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    base_path = pathlib.Path(dest_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    zip_target_path = pathlib.Path(f'{dest_dir}/maestro-v3.0.0-midi.zip')
    extracted_folder_path = pathlib.Path(f'{dest_dir}/maestro-v3.0.0')

    if not (zip_target_path.exists() and extracted_folder_path.exists()):
        print("Downloading the MAESTRO dataset...")
        download_url_to_file(url, str(zip_target_path), progress=True)
        if zip_target_path.exists():
            print("Extracting the dataset...")
            import zipfile
            with zipfile.ZipFile(zip_target_path, 'r') as zip_ref:
                zip_ref.extractall(base_path)
    return extracted_folder_path


def convert_midi_to_notes(midi_file_path: str) -> pd.DataFrame:
    """Reads one MIDI file and returns a table of (pitch, step, duration) per note.

    pitch: MIDI note number, 0-127.
    step: seconds since the previous note started (how long we waited to hear this note).
    duration: seconds this note rings for (note.end - note.start).
    """
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    if not midi_data.instruments:
        return pd.DataFrame()

    instrument = midi_data.instruments[0]
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start if len(sorted_notes) > 0 else 0

    for note in sorted_notes:
        notes['pitch'].append(note.pitch)
        notes['step'].append(note.start - prev_start)
        notes['duration'].append(note.end - note.start)
        prev_start = note.start

    return pd.DataFrame({name: np.array(values) for name, values in notes.items()})


def convert_all_songs_to_notes(data_roots) -> list:
    """Runs convert_midi_to_notes() over every .midi file under the given folder(s).

    Returns a list where each entry is one song's notes as a (num_notes, 3) numpy array
    of [pitch, step, duration].
    """
    all_songs_notes_array = []
    root_paths = [pathlib.Path(p) for p in (data_roots if isinstance(data_roots, list) else [data_roots])]

    all_midi_files = []
    for root_path in root_paths:
        all_midi_files.extend(root_path.glob('**/*.midi'))

    if not all_midi_files:
        raise FileNotFoundError(f"No MIDI files found in paths: {root_paths}")

    for midi_file in all_midi_files:
        notes_df = convert_midi_to_notes(midi_file)
        if notes_df.empty:
            continue
        single_song_notes_array = notes_df[['pitch', 'step', 'duration']].to_numpy(dtype=np.float32)
        all_songs_notes_array.append(single_song_notes_array)

    return all_songs_notes_array


def list_available_years(dataset_root: pathlib.Path):
    """Every year-folder MAESTRO ships (e.g. '2004', '2006', ...), found on disk."""
    year_paths = sorted(
        p for p in dataset_root.iterdir() if p.is_dir() and p.name.isdigit()
    )
    return [p.name for p in year_paths]


def resolve_training_years(dataset_root: pathlib.Path, max_years=None, seed=0):
    """Picks which years to train on: all of them by default, or a capped, random subset.

    max_years=None (the default) uses every year MAESTRO has. Passing a number instead
    caps it to that many years, chosen deterministically (seeded, so it's reproducible
    and identical across the DDP processes) from whatever's available on disk — no need
    to hardcode or guess exact year names to control how much data you train on.
    """
    available_year_names = list_available_years(dataset_root)
    selected_years = available_year_names

    if max_years is not None and max_years < len(selected_years):
        rng = np.random.default_rng(seed)
        selected_years = sorted(rng.choice(selected_years, size=max_years, replace=False).tolist())

    resolved_paths = [dataset_root / year for year in selected_years]
    return resolved_paths, selected_years


def build_cache_tag(selected_years) -> str:
    """Short label for cache/checkpoint filenames, e.g. '2004-2009-2011-2013-2015-2018'."""
    joined_years = "-".join(selected_years)
    return joined_years if len(joined_years) <= 80 else f"{selected_years[0]}-{selected_years[-1]}-{len(selected_years)}years"


def compute_time_feature_stats(song_note_arrays):
    """Mean/std of log1p(step) and log1p(duration), used to z-score-normalize timing.

    Raw timing values are skewed (most gaps are tiny, a few are seconds long), so we
    compress them with log1p first, then normalize to mean 0 / std 1 — this keeps the
    values in a range that's easy for the network to learn from. Always computed from
    the TRAIN split only, then reused for val/test too, so no information about
    val/test leaks into how the data is scaled.
    """
    logged_time_features = []
    for song_notes in song_note_arrays:
        notes_array = np.asarray(song_notes, dtype=np.float32)
        if notes_array.size == 0:
            continue
        logged_time_features.append(np.log1p(notes_array[:, 1:3]))

    if not logged_time_features:
        return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)

    stacked_time_features = np.vstack(logged_time_features)
    feature_mean = stacked_time_features.mean(axis=0).astype(np.float32)
    feature_std = stacked_time_features.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    return feature_mean, feature_std


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
    """Caps step/duration at the bounds above. Returns new arrays; doesn't mutate input."""
    clipped = []
    for song_notes in song_note_arrays:
        notes_array = np.asarray(song_notes, dtype=np.float32).copy()
        if notes_array.size > 0:
            notes_array[:, 1] = np.minimum(notes_array[:, 1], step_max)
            notes_array[:, 2] = np.minimum(notes_array[:, 2], duration_max)
        clipped.append(notes_array)
    return clipped


def load_or_create_note_cache(data_roots, dataset_root: pathlib.Path, cache_tag: str, is_main_process: bool):
    """Parses every MIDI file into notes once, then reuses the cached result on future runs.

    Parsing MIDI is slow; re-doing it every run would waste a lot of time. Only the
    main process (rank 0) manages the cache file: it checks whether a good one already
    exists, rebuilds it if it's missing OR unreadable (e.g. left corrupted by an earlier
    run that crashed mid-write), and only ever gives the file its real name once writing
    is completely finished. Every other GPU process just waits for a valid file to show
    up and reads it — they never do the parsing themselves, so the work only happens once.

    Why the temp-file-then-rename step matters: opening a file for writing creates it on
    disk immediately, before any data is in it. If rank 0 wrote directly to the real
    filename, another process polling for "does the file exist yet" could see it too
    early and try to read an empty or half-written file. Renaming a file is atomic on
    Linux, so the real filename only ever appears once, fully written — there's no
    in-between state for another process to catch it in.
    """
    song_cache_file = dataset_root / f'converted_notes_array_{cache_tag}.pkl'

    if is_main_process:
        if song_cache_file.exists():
            try:
                with song_cache_file.open('rb') as cache_handle:
                    return pickle.load(cache_handle)
            except (EOFError, pickle.UnpicklingError):
                # Left behind by a run that got killed mid-write — not usable, so
                # rebuild instead of crashing on it.
                print(f"Cache file {song_cache_file} looks incomplete/corrupted (likely an interrupted earlier run) — rebuilding it.")
                song_cache_file.unlink()

        download_maestro_dataset()
        converted_notes = convert_all_songs_to_notes(data_roots)
        temp_cache_file = song_cache_file.with_name(song_cache_file.name + '.tmp')
        with temp_cache_file.open('wb') as cache_handle:
            pickle.dump(converted_notes, cache_handle)
        os.replace(temp_cache_file, song_cache_file)  # atomic — no reader ever sees a partial file
        return converted_notes
    else:
        # Retry rather than crash on a failed load: rank 0 might still be in the
        # middle of detecting and deleting a leftover corrupted file above.
        while True:
            if song_cache_file.exists():
                try:
                    with song_cache_file.open('rb') as cache_handle:
                        return pickle.load(cache_handle)
                except (EOFError, pickle.UnpicklingError):
                    pass
            time.sleep(1)


# ==========================================
# 2. PYTORCH DATASET & DATA-PREP HELPERS
# ==========================================

class BasicRNNForMusic(data.Dataset):
    """Turns songs into (past 64 notes -> next note) training examples.

    For each song, slides a window of `seq_len` notes across it: notes 0..63 predict
    note 64, notes 1..64 predict note 65, and so on. `index_map` stores every
    (song_index, start_index) pair so __getitem__ can look up any window in O(1).
    """

    def __init__(self, song_note_arrays, seq_len=64, time_feature_mean=None, time_feature_std=None):
        self.seq_len = seq_len
        self.song_pitches = []
        self.song_time_features = []
        self.index_map = []
        self.time_feature_mean = np.zeros(2, dtype=np.float32) if time_feature_mean is None else np.asarray(time_feature_mean, dtype=np.float32)
        self.time_feature_std = np.ones(2, dtype=np.float32) if time_feature_std is None else np.asarray(time_feature_std, dtype=np.float32)

        for song_notes in song_note_arrays:
            notes_array = np.asarray(song_notes, dtype=np.float32)
            if len(notes_array) <= self.seq_len:
                continue  # song too short to form even one full window

            pitches = notes_array[:, 0].astype(np.int64)
            time_features = np.log1p(notes_array[:, 1:3])
            time_features = (time_features - self.time_feature_mean) / self.time_feature_std

            song_index = len(self.song_pitches)
            self.song_pitches.append(torch.tensor(pitches, dtype=torch.long))
            self.song_time_features.append(torch.tensor(time_features, dtype=torch.float32))
            self.index_map.extend((song_index, start_idx) for start_idx in range(len(pitches) - self.seq_len))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_index, start_idx = self.index_map[idx]
        return (
            (self.song_pitches[song_index][start_idx: start_idx + self.seq_len],
             self.song_time_features[song_index][start_idx: start_idx + self.seq_len]),
            (self.song_pitches[song_index][start_idx + self.seq_len],
             self.song_time_features[song_index][start_idx + self.seq_len])
        )


def split_song_arrays(song_note_arrays, seed, train_ratio=0.8, val_ratio=0.1):
    """Splits songs (not individual notes!) into train/val/test.

    Splitting by whole song, rather than by note-window, keeps windows from the same
    song out of both train and test — otherwise the model could partly memorize a
    song it's supposedly being evaluated on.
    """
    song_indices = np.random.default_rng(seed).permutation(len(song_note_arrays))
    train_cutoff = int(len(song_indices) * train_ratio)
    val_cutoff = int(len(song_indices) * (train_ratio + val_ratio))

    return (
        [song_note_arrays[i] for i in song_indices[:train_cutoff]],
        [song_note_arrays[i] for i in song_indices[train_cutoff:val_cutoff]],
        [song_note_arrays[i] for i in song_indices[val_cutoff:]]
    )


def augment_with_pitch_transposition(song_note_arrays, semitone_shifts=(-3, 3)):
    """Adds transposed copies of each training song (shifted up/down by a few semitones).

    Relative intervals and patterns stay identical under transposition, so this gives
    the model more distinct pitch sequences to learn from without downloading any new
    data. Only ever applied to the TRAIN split — val/test must stay untouched so
    evaluation reflects the real, un-augmented data distribution.
    """
    augmented = list(song_note_arrays)  # keep all the originals
    for song_notes in song_note_arrays:
        notes_array = np.asarray(song_notes, dtype=np.float32)
        if notes_array.size == 0:
            continue
        pitches = notes_array[:, 0]
        min_pitch, max_pitch = pitches.min(), pitches.max()
        for shift in semitone_shifts:
            if min_pitch + shift < 0 or max_pitch + shift > 127:
                continue  # skip shifts that would push a note outside the valid MIDI range
            shifted = notes_array.copy()
            shifted[:, 0] = shifted[:, 0] + shift
            augmented.append(shifted)
    return augmented


def build_pitch_class_weights(song_note_arrays, num_pitches=128, power=-0.5):
    """Optionally up-weights rare pitches in the loss, so the model doesn't ignore them.

    Some pitches appear far more often than others. Without weighting, the model could
    get a decent-looking loss just by nailing the common pitches. `power` controls the
    strength: -0.5 up-weights rare pitches moderately; 0.0 disables weighting entirely
    (returns None, so the loss falls back to plain unweighted cross-entropy — matches
    a plain unweighted accuracy metric exactly).
    """
    if power == 0.0:
        return None

    pitch_counts = np.zeros(num_pitches, dtype=np.float32)
    for song_notes in song_note_arrays:
        pitches = np.asarray(song_notes)[:, 0].astype(np.int64)
        pitch_counts += np.bincount(pitches, minlength=num_pitches)

    observed = pitch_counts > 0
    pitch_weights = np.zeros(num_pitches, dtype=np.float32)
    pitch_weights[observed] = np.power(pitch_counts[observed], power)
    pitch_weights[observed] /= pitch_weights[observed].mean()
    return torch.tensor(pitch_weights, dtype=torch.float32)


# ==========================================
# 3. MODEL: EMBEDDING -> LSTM -> 3 HEADS
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
        super(OptimizedMusicRNN, self).__init__()

        self.pitch_embedding = nn.Embedding(num_embeddings=num_pitches, embedding_dim=pitch_embed_dim)
        input_size = pitch_embed_dim + 2  # + step, duration
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

        # h_n[-1] is the last LSTM layer's final hidden state after reading the whole
        # sequence — a compact summary of everything the model has seen so far.
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = self.dropout(h_n[-1])

        # step/duration targets are log1p'd then z-score normalized, so roughly half of
        # them are negative — a plain linear output (no squashing function) is used here
        # so the model can actually reach the full target range. Non-negativity in real
        # seconds gets enforced later, when converting a prediction back out of
        # normalized space (see the evaluation code below), not inside the model itself.
        return {
            'pitch': self.pitch_head(last_out),
            'step': self.step_head(last_out),
            'duration': self.duration_head(last_out),
        }


# ===========================================
# 4. TRAINING (RUNS ONCE PER GPU, VIA DDP)
# ===========================================

def main_worker(gpu, world_size, hyperparameters):
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
    torch.backends.cudnn.benchmark = True  # input shapes are fixed every step, so cuDNN can cache its fastest kernels
    is_main_process = (rank == 0)

    # --- Load & prepare data (identically on every GPU process) ---
    dataset_root = pathlib.Path('data/maestro-v3.0.0')
    if is_main_process:
        download_maestro_dataset()
    dist.barrier()  # other rank(s) wait here until rank 0 finishes downloading

    selected_year_paths, resolved_years = resolve_training_years(
        dataset_root, max_years=hyperparameters.get('max_years'), seed=hyperparameters['seed']
    )
    cache_tag = build_cache_tag(resolved_years)
    converted_notes_array = load_or_create_note_cache(selected_year_paths, dataset_root, cache_tag, is_main_process)

    train_notes, val_notes, test_notes = split_song_arrays(converted_notes_array, seed=hyperparameters['seed'])

    if hyperparameters.get('use_pitch_augmentation', False):
        train_notes = augment_with_pitch_transposition(
            train_notes, semitone_shifts=hyperparameters.get('augmentation_shifts', (-3, 3))
        )

    # Clip outlier step/duration values, then compute normalization stats from the
    # (now-clipped) train data, then apply both to all three splits.
    step_clip_max, duration_clip_max = compute_time_clip_bounds(
        train_notes, upper_percentile=hyperparameters.get('time_clip_upper_percentile', 99.5)
    )
    train_notes = clip_extreme_time_values(train_notes, step_clip_max, duration_clip_max)
    val_notes = clip_extreme_time_values(val_notes, step_clip_max, duration_clip_max)
    test_notes = clip_extreme_time_values(test_notes, step_clip_max, duration_clip_max)
    time_feature_mean, time_feature_std = compute_time_feature_stats(train_notes)

    train_dataset = BasicRNNForMusic(train_notes, seq_len=hyperparameters['seq_len'], time_feature_mean=time_feature_mean, time_feature_std=time_feature_std)
    val_dataset = BasicRNNForMusic(val_notes, seq_len=hyperparameters['seq_len'], time_feature_mean=time_feature_mean, time_feature_std=time_feature_std)
    test_dataset = BasicRNNForMusic(test_notes, seq_len=hyperparameters['seq_len'], time_feature_mean=time_feature_mean, time_feature_std=time_feature_std)

    if is_main_process:
        print(f"Years used: {resolved_years}")
        print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}, Test examples: {len(test_dataset)}")

    # --- DataLoaders: split data across the 2 GPUs, load it in the background ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # persistent_workers=True keeps the background data-loading processes alive across
    # epochs instead of recreating them every epoch (the default) — recreating them
    # repeatedly re-imports every library from scratch, which is real wasted time.
    train_loader = data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=2, drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_loader = data.DataLoader(val_dataset, batch_size=hyperparameters['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=2, drop_last=False, persistent_workers=True, prefetch_factor=4)
    test_loader = data.DataLoader(test_dataset, batch_size=hyperparameters['batch_size_per_gpu'], pin_memory=True, num_workers=2, shuffle=False)

    # --- Model, wrapped in DDP so gradients sync across both GPUs automatically ---
    model = OptimizedMusicRNN(hidden_size=hyperparameters['hidden_size'], num_layers=hyperparameters['num_layers']).cuda(gpu)
    model = DDP(model, device_ids=[gpu])

    pitch_class_weights = build_pitch_class_weights(train_notes, power=hyperparameters.get('pitch_weight_power', -0.5))
    weight_tensor = pitch_class_weights.cuda(gpu) if pitch_class_weights is not None else None
    criterion_pitch = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=hyperparameters['label_smoothing'])
    criterion_time = nn.SmoothL1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

    # Learning-rate schedule: a short linear warmup (so a high LR doesn't destabilize the
    # very first few steps), followed by one smooth cosine decay down to eta_min over the
    # rest of training.
    warmup_epochs = hyperparameters.get('warmup_epochs', 0)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(hyperparameters['epochs'] - warmup_epochs, 1), eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    scaler = GradScaler('cuda')  # for mixed-precision (float16) training

    best_val_pitch_loss = float('inf')
    epochs_without_improvement = 0
    # Filename includes the architecture + augmentation setting, so different experiments
    # don't silently overwrite each other's checkpoints.
    aug_tag = "augT" if hyperparameters.get('use_pitch_augmentation', False) else "augF"
    arch_tag = f"h{hyperparameters['hidden_size']}_l{hyperparameters['num_layers']}_{aug_tag}"
    best_checkpoint_path = pathlib.Path('artifacts') / f'ddp_lstm_music_best_{cache_tag}_{arch_tag}.pt'
    if is_main_process:
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    for epoch in range(hyperparameters['epochs']):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_loss_pitch = 0.0

        for inputs, targets in train_loader:
            x_pitch, x_time = inputs[0].cuda(gpu, non_blocking=True), inputs[1].cuda(gpu, non_blocking=True)
            y_pitch, y_time = targets[0].cuda(gpu, non_blocking=True), targets[1].cuda(gpu, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                predictions = model(x_pitch, x_time)
                loss_pitch = criterion_pitch(predictions['pitch'], y_pitch)
                loss_step = criterion_time(predictions['step'], y_time[:, 0:1])
                loss_duration = criterion_time(predictions['duration'], y_time[:, 1:2])
                total_loss = loss_pitch + hyperparameters['time_loss_weight'] * (loss_step + loss_duration)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # caps runaway gradient spikes
            scaler.step(optimizer)
            scaler.update()
            running_loss += total_loss.item()
            running_loss_pitch += loss_pitch.item()

        model.eval()
        running_val_loss = 0.0
        running_val_loss_pitch = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                x_pitch, x_time = inputs[0].cuda(gpu, non_blocking=True), inputs[1].cuda(gpu, non_blocking=True)
                y_pitch, y_time = targets[0].cuda(gpu, non_blocking=True), targets[1].cuda(gpu, non_blocking=True)
                with autocast('cuda'):
                    preds = model(x_pitch, x_time)
                    loss_pitch = criterion_pitch(preds['pitch'], y_pitch)
                    loss_step = criterion_time(preds['step'], y_time[:, 0:1])
                    loss_duration = criterion_time(preds['duration'], y_time[:, 1:2])
                    val_loss = loss_pitch + hyperparameters['time_loss_weight'] * (loss_step + loss_duration)
                    running_val_loss += val_loss.item()
                    running_val_loss_pitch += loss_pitch.item()

        # Average each metric across both GPUs, so the printed numbers reflect the
        # whole dataset, not just whatever half this one GPU happened to process.
        metrics_tensor = torch.tensor([
            running_loss / len(train_loader), running_loss_pitch / len(train_loader),
            running_val_loss / len(val_loader), running_val_loss_pitch / len(val_loader),
        ], device=gpu)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        global_train_loss = metrics_tensor[0].item() / world_size
        global_train_loss_pitch = metrics_tensor[1].item() / world_size
        global_val_loss = metrics_tensor[2].item() / world_size
        global_val_loss_pitch = metrics_tensor[3].item() / world_size

        scheduler.step()
        epoch_seconds = time.time() - epoch_start_time

        if is_main_process:
            gpu_mem_gb = torch.cuda.max_memory_allocated(gpu) / (1024 ** 3)
            print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}] -> "
                  f"Train Loss: {global_train_loss:.4f} (Pitch: {global_train_loss_pitch:.4f}), "
                  f"Val Loss: {global_val_loss:.4f} (Pitch: {global_val_loss_pitch:.4f}), "
                  f"Epoch time: {epoch_seconds:.1f}s, GPU peak mem: {gpu_mem_gb:.2f} GB")

            # Checkpoint on pitch-only val loss, not the combined loss — the final
            # metric we report is pitch accuracy alone, and a model can improve a lot
            # on timing while pitch barely moves, which the combined number would hide.
            if global_val_loss_pitch < best_val_pitch_loss:
                best_val_pitch_loss = global_val_loss_pitch
                epochs_without_improvement = 0
                torch.save({'model_state_dict': model.module.state_dict(), 'val_pitch_loss': best_val_pitch_loss}, best_checkpoint_path)
            else:
                epochs_without_improvement += 1

        # All GPUs need to agree on whether to stop, so rank 0's decision is broadcast
        # to the rest via all_reduce before anyone breaks out of the loop.
        stop_signal = torch.tensor([1 if epochs_without_improvement >= hyperparameters['patience'] else 0], device=gpu)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        if stop_signal.item() > 0:
            break

    # --- Final evaluation on held-out test songs (main process only) ---
    if is_main_process:
        print("\n--- Running Evaluation Pipeline On Testing Dataset ---")
        checkpoint = torch.load(best_checkpoint_path, map_location='cuda:0')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        correct_pitch = 0
        total_samples = 0
        sum_abs_err_step = 0.0
        sum_abs_err_duration = 0.0
        time_mean_t = torch.tensor(time_feature_mean, device=0)
        time_std_t = torch.tensor(time_feature_std, device=0)

        with torch.no_grad():
            for inputs, targets in test_loader:
                x_pitch, x_time = inputs[0].cuda(0, non_blocking=True), inputs[1].cuda(0, non_blocking=True)
                y_pitch = targets[0].cuda(0, non_blocking=True)
                y_time = targets[1].cuda(0, non_blocking=True)
                with autocast('cuda'):
                    preds = model.module(x_pitch, x_time)
                    predicted_classes = torch.argmax(preds['pitch'], dim=1)
                    correct_pitch += (predicted_classes == y_pitch).sum().item()
                    total_samples += y_pitch.size(0)

                    # Undo normalization (z-score, then log1p) to get errors in real
                    # seconds, which is easier to reason about than normalized units.
                    pred_step_sec = torch.expm1(preds['step'].squeeze(-1) * time_std_t[0] + time_mean_t[0])
                    pred_duration_sec = torch.expm1(preds['duration'].squeeze(-1) * time_std_t[1] + time_mean_t[1])
                    true_step_sec = torch.expm1(y_time[:, 0] * time_std_t[0] + time_mean_t[0])
                    true_duration_sec = torch.expm1(y_time[:, 1] * time_std_t[1] + time_mean_t[1])

                    sum_abs_err_step += (pred_step_sec - true_step_sec).abs().sum().item()
                    sum_abs_err_duration += (pred_duration_sec - true_duration_sec).abs().sum().item()

        if total_samples > 0:
            print(f"Final Pitch Accuracy ({cache_tag}, {arch_tag}): {(correct_pitch / total_samples) * 100:.2f}%")
            print(f"Final Step MAE: {sum_abs_err_step / total_samples:.4f}s, Duration MAE: {sum_abs_err_duration / total_samples:.4f}s")

    dist.destroy_process_group()


if __name__ == '__main__':
    hparams = {
        # --- Core model/training hyperparameters ---
        'seq_len': 64,             # how many past notes the model sees before predicting the next one
        'hidden_size': 384,        # size of the LSTM's memory (hidden state) at each layer
        'num_layers': 3,           # how many LSTM layers are stacked
        'batch_size_per_gpu': 1536,  # training examples processed together per GPU per step
        'epochs': 35,               # maximum number of passes over the training data
        'patience': 8,               # stop early if val pitch-loss hasn't improved for this many epochs
        'lr': 2e-3,                  # optimizer step size (paired with batch_size_per_gpu above)
        'warmup_epochs': 2,          # epochs of linearly ramping LR up before the cosine decay begins
        'weight_decay': 1e-4,        # L2 regularization strength, discourages overly large weights
        'time_loss_weight': 0.5,     # how much step/duration loss counts vs. pitch loss in total_loss
        'label_smoothing': 0.03,     # softens pitch targets slightly, discourages overconfidence
        'seed': 53,                  # makes the train/val/test song split reproducible
        'max_years': None,           # None = use every year MAESTRO has; set to a number (e.g. 6) to cap it and speed up epochs

        # --- Data-volume / loss-shaping options ---
        'use_pitch_augmentation': False,   # add transposed copies of each training song (see augment_with_pitch_transposition)
        'augmentation_shifts': (-3, 3),    # semitone shifts to use, if augmentation is on
        'pitch_weight_power': 0.0,         # 0.0 = no class weighting; -0.5 = moderately up-weight rare pitches
        'time_clip_upper_percentile': 99.5,  # cap step/duration outliers above this percentile of train data
    }

    gpus_available = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if gpus_available > 1:
        print(f"Multi-GPU Target Evaluated: Orchestrating native DDP over {gpus_available} visible cards.")
        torch.multiprocessing.spawn(main_worker, args=(gpus_available, hparams), nprocs=gpus_available, join=True)
    else:
        print("Falling back to single process setup.")
        dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
        main_worker(0, 1, hparams)
