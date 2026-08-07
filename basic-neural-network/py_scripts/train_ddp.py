import copy
import os
import pathlib
import pickle
import collections
import numpy as np
import pandas as pd
import pretty_midi as pm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.hub import download_url_to_file

# ==========================================
# 1. DATASET DOWNLOADING & PREPROCESSING
# ==========================================

def download_maestro_dataset(dest_dir: str='data'):
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
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    # Fixed Bug: Safely select the first available instrument object track
    if not midi_data.instruments:
        return pd.DataFrame()
        
    instrument = midi_data.instruments[0]  
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

    # Fixed Bug: Corrected attribute access on list reference
    prev_start = sorted_notes[0].start if len(sorted_notes) > 0 else 0

    for note in sorted_notes:
        notes['pitch'].append(note.pitch)
        notes['step'].append(note.start - prev_start)
        notes['duration'].append(note.end - note.start)
        prev_start = note.start

    return pd.DataFrame({name: np.array(values) for name, values in notes.items()})

def convert_all_songs_to_notes(data_roots) -> list:
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

def build_cache_tag(selected_years) -> str:
    joined_years = "-".join(selected_years)
    return joined_years if len(joined_years) <= 80 else f"{selected_years[0]}-{selected_years[-1]}-{len(selected_years)}years"

def resolve_training_years(dataset_root: pathlib.Path, selected_years):
    available_year_paths = sorted(
        year_path for year_path in dataset_root.iterdir() if year_path.is_dir() and year_path.name.isdigit()
    )
    available_year_names = [year_path.name for year_path in available_year_paths]

    unknown_years = [year for year in selected_years if year not in available_year_names]
    if unknown_years:
        raise ValueError(f"Requested years not found in dataset: {unknown_years}.")

    resolved_paths = [dataset_root / year for year in selected_years]
    return resolved_paths, available_year_names

def compute_time_feature_stats(song_note_arrays):
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

def load_or_create_note_cache(data_roots, dataset_root: pathlib.Path, cache_tag: str, is_main_process: bool):
    song_cache_file = dataset_root / f'converted_notes_array_{cache_tag}.pkl'

    if song_cache_file.exists():
        with song_cache_file.open('rb') as cache_handle:
            return pickle.load(cache_handle)

    if is_main_process:
        download_maestro_dataset()
        converted_notes = convert_all_songs_to_notes(data_roots)
        with song_cache_file.open('wb') as cache_handle:
            pickle.dump(converted_notes, cache_handle)
        return converted_notes
    else:
        import time
        while not song_cache_file.exists():
            time.sleep(1)
        with song_cache_file.open('rb') as cache_handle:
            return pickle.load(cache_handle)

# ==========================================
# 2. PYTORCH CUSTOM STRUCTURE DEFINITIONS
# ==========================================

class BasicRNNForMusic(data.Dataset):
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
                continue

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
            (self.song_pitches[song_index][start_idx : start_idx + self.seq_len],
             self.song_time_features[song_index][start_idx : start_idx + self.seq_len]),
            (self.song_pitches[song_index][start_idx + self.seq_len],
             self.song_time_features[song_index][start_idx + self.seq_len])
        )

def split_song_arrays(song_note_arrays, seed, train_ratio=0.8, val_ratio=0.1):
    song_indices = np.random.default_rng(seed).permutation(len(song_note_arrays))
    train_cutoff = int(len(song_indices) * train_ratio)
    val_cutoff = int(len(song_indices) * (train_ratio + val_ratio))

    return (
        [song_note_arrays[i] for i in song_indices[:train_cutoff]],
        [song_note_arrays[i] for i in song_indices[train_cutoff:val_cutoff]],
        [song_note_arrays[i] for i in song_indices[val_cutoff:]]
    )

def build_pitch_class_weights(song_note_arrays, num_pitches=128):
    pitch_counts = np.zeros(num_pitches, dtype=np.float32)
    for song_notes in song_note_arrays:
        pitches = np.asarray(song_notes)[:, 0].astype(np.int64)
        pitch_counts += np.bincount(pitches, minlength=num_pitches)

    observed = pitch_counts > 0
    pitch_weights = np.zeros(num_pitches, dtype=np.float32)
    pitch_weights[observed] = np.power(pitch_counts[observed], -0.5)
    pitch_weights[observed] /= pitch_weights[observed].mean()
    return torch.tensor(pitch_weights, dtype=torch.float32)

# ==========================================
# 3. OPTIMIZED MUSIC RNN MODEL ARCHITECTURE
# ==========================================

class OptimizedMusicRNN(nn.Module):
    def __init__(self, num_pitches=128, pitch_embed_dim=128, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super(OptimizedMusicRNN, self).__init__()
        
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
        
        # Widen prediction heads to 256 dimensions to prevent task gradient conflicts
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
        
        # properly build input and run LSTM
        x = torch.cat([pitch_embeds, time_seq], dim=2)
        x = self.input_norm(x)

        # Pull terminal sequence state tokens from true LSTM cell memory (h_n)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = self.dropout(h_n[-1])

        return {
            'pitch': self.pitch_head(last_out),
            'step': F.softplus(self.step_head(last_out)),
            'duration': F.softplus(self.duration_head(last_out)),
        }


# ===========================================
# 4. TRAINING AND DISTRIBUTED MAIN LOOP
# ===========================================


def main_worker(gpu, world_size, selected_years, hyperparameters):
    rank = gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(gpu)
    is_main_process = (rank == 0)

    dataset_root = pathlib.Path('data/maestro-v3.0.0')
    if is_main_process:
        download_maestro_dataset()
    dist.barrier()

    selected_year_paths, _ = resolve_training_years(dataset_root, selected_years)
    cache_tag = build_cache_tag(selected_years)
    converted_notes_array = load_or_create_note_cache(selected_year_paths, dataset_root, cache_tag, is_main_process)

    train_notes, val_notes, test_notes = split_song_arrays(converted_notes_array, seed=hyperparameters['seed'])
    time_feature_mean, time_feature_std = compute_time_feature_stats(train_notes)

    train_dataset = BasicRNNForMusic(train_notes, seq_len=hyperparameters['seq_len'], time_feature_mean=time_feature_mean, time_feature_std=time_feature_std)
    val_dataset = BasicRNNForMusic(val_notes, seq_len=hyperparameters['seq_len'], time_feature_mean=time_feature_mean, time_feature_std=time_feature_std)
    test_dataset = BasicRNNForMusic(test_notes, seq_len=hyperparameters['seq_len'], time_feature_mean=time_feature_mean, time_feature_std=time_feature_std)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=hyperparameters['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=2, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=hyperparameters['batch_size_per_gpu'], pin_memory=True, num_workers=2, shuffle=False)

    model = OptimizedMusicRNN(hidden_size=hyperparameters['hidden_size'], num_layers=hyperparameters['num_layers']).cuda(gpu)
    model = DDP(model, device_ids=[gpu])

    pitch_class_weights = build_pitch_class_weights(train_notes).cuda(gpu)
    criterion_pitch = nn.CrossEntropyLoss(weight=pitch_class_weights, label_smoothing=hyperparameters['label_smoothing'])
    criterion_time = nn.SmoothL1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_checkpoint_path = pathlib.Path('artifacts') / f'ddp_lstm_music_best_{cache_tag}.pt'
    if is_main_process:
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(hyperparameters['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            x_pitch, x_time = inputs[0].cuda(gpu, non_blocking=True), inputs[1].cuda(gpu, non_blocking=True)
            y_pitch, y_time = targets[0].cuda(gpu, non_blocking=True), targets[1].cuda(gpu, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                predictions = model(x_pitch, x_time)
                loss_pitch = criterion_pitch(predictions['pitch'], y_pitch)
                loss_step = criterion_time(predictions['step'], y_time[:, 0:1])
                loss_duration = criterion_time(predictions['duration'], y_time[:, 1:2])
                total_loss = loss_pitch + hyperparameters['time_loss_weight'] * (loss_step + loss_duration)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += total_loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                x_pitch, x_time = inputs[0].cuda(gpu, non_blocking=True), inputs[1].cuda(gpu, non_blocking=True)
                y_pitch, y_time = targets[0].cuda(gpu, non_blocking=True), targets[1].cuda(gpu, non_blocking=True)
                with torch.cuda.amp.autocast():
                    preds = model(x_pitch, x_time)
                    loss_pitch = criterion_pitch(preds['pitch'], y_pitch)
                    loss_step = criterion_time(preds['step'], y_time[:, 0:1])
                    loss_duration = criterion_time(preds['duration'], y_time[:, 1:2])
                    val_loss = loss_pitch + hyperparameters['time_loss_weight'] * (loss_step + loss_duration)
                    running_val_loss += val_loss.item()

        metrics_tensor = torch.tensor([running_loss / len(train_loader), running_val_loss / len(val_loader)], device=gpu)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        global_train_loss = metrics_tensor[0].item() / world_size
        global_val_loss = metrics_tensor[1].item() / world_size

        scheduler.step()

        if is_main_process:
            print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}] -> Train Loss: {global_train_loss:.4f}, Val Loss: {global_val_loss:.4f}")

            if global_val_loss < best_val_loss:
                best_val_loss = global_val_loss
                epochs_without_improvement = 0
                torch.save({'model_state_dict': model.module.state_dict(), 'val_loss': best_val_loss}, best_checkpoint_path)
            else:
                epochs_without_improvement += 1

        stop_signal = torch.tensor([1 if epochs_without_improvement >= hyperparameters['patience'] else 0], device=gpu)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        if stop_signal.item() > 0:
            break

    if is_main_process:
        print("\n--- Running Evaluation Pipeline On Testing Dataset ---")
        checkpoint = torch.load(best_checkpoint_path, map_location=f'cuda:{0}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        correct_pitch = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                x_pitch, x_time = inputs[0].cuda(0, non_blocking=True), inputs[1].cuda(0, non_blocking=True)
                y_pitch = targets[0].cuda(0, non_blocking=True)
                with torch.cuda.amp.autocast():
                    preds = model.module(x_pitch, x_time)
                    predicted_classes = torch.argmax(preds['pitch'], dim=1)
                    correct_pitch += (predicted_classes == y_pitch).sum().item()
                    total_samples += y_pitch.size(0)

        if total_samples > 0:
            print(f"Final Pitch Accuracy over 5-Year Configuration: {(correct_pitch / total_samples) * 100:.2f}%")

    dist.destroy_process_group()


if __name__ == '__main__':
    selected_years_config = ['2004', '2008', '2011', '2015', '2018']
    hparams = {'seq_len': 64,
               'hidden_size': 256,
               'num_layers': 2,
               'batch_size_per_gpu': 256,
               'epochs': 50,
               'patience': 10,
               'lr': 1e-3,
               'weight_decay': 1e-4,
               'time_loss_weight': 0.5,
               'label_smoothing': 0.03,
               'seed': 53}

    gpus_available = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if gpus_available > 1:
        print(f"Multi-GPU Target Evaluated: Orchestrating native DDP over {gpus_available} visible cards.")
        torch.multiprocessing.spawn(main_worker, args=(gpus_available, selected_years_config, hparams), nprocs=gpus_available, join=True)
    else:
        print("Falling back to single process setup.")
        dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
        main_worker(0, 1, selected_years_config, hparams)