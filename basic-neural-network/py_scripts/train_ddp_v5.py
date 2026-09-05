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
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

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
    cache_version = "v5_12_pitch_only" 
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
    """Dataset returning pitch classes (0-11)."""
    def __init__(self, song_note_arrays, seq_len=64, augment=False):
        self.seq_len = seq_len
        self.augment = augment
        self.song_pitches = []
        self.index_map = []
        
        for song_notes in song_note_arrays:
            notes_array = np.asarray(song_notes, dtype=np.float32)
            if len(notes_array) <= self.seq_len:
                continue

            pitches = notes_array[:, 0].astype(np.int64)
            song_idx = len(self.song_pitches)
            self.song_pitches.append(torch.tensor(pitches, dtype=torch.long))
            self.index_map.extend((song_idx, start_idx) for start_idx in range(len(pitches) - self.seq_len))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_idx, start_idx = self.index_map[idx]
        end_idx = start_idx + self.seq_len
        
        # Squeeze down to 0-11 pitch classes
        raw_pitch_seq = self.song_pitches[song_idx][start_idx:end_idx].clone()
        raw_target = self.song_pitches[song_idx][end_idx].clone()
        
        pitch_seq = raw_pitch_seq % 12
        target_pitch = raw_target % 12
        
        if self.augment:
            shift = torch.randint(-5, 6, (1,)).item()
            pitch_seq = (pitch_seq + shift) % 12
            target_pitch = (target_pitch + shift) % 12

        return pitch_seq, target_pitch

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
# 3. MODEL ARCHITECTURE (SIMPLIFIED)
# ==========================================

class OptimizedMusicRNN(nn.Module):
    """Predicts next pitch class (0-11)."""
    def __init__(self, num_pitches=12, pitch_embed_dim=64, hidden_size=384, num_layers=3, dropout_rate=0.3):
        super().__init__()
        self.num_pitches = num_pitches

        self.pitch_embed = nn.Embedding(num_embeddings=num_pitches, embedding_dim=pitch_embed_dim)
        self.input_norm = nn.LayerNorm(pitch_embed_dim)

        self.lstm = nn.LSTM(
            input_size=pitch_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.skip_proj = nn.Linear(pitch_embed_dim, hidden_size)
        self.gate_proj = nn.Linear(hidden_size, hidden_size)

        self.fusion_layer = nn.Linear(hidden_size, num_pitches)

    def forward(self, pitch_seq):
        x = self.pitch_embed(pitch_seq)
        x = self.input_norm(x)

        _, (h_n, _) = self.lstm(x)
        last_out = self.dropout(h_n[-1])

        last_pitch_embed = x[:, -1, :]
        gate = torch.sigmoid(self.gate_proj(last_out))
        pitch_input = last_out + gate * self.skip_proj(last_pitch_embed)

        logits = self.fusion_layer(pitch_input)

        return {
            'pitch': logits
        }

# --- Formatting Utilities (Adapted for 12 classes) ---
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def pitch_to_label(pitch_value):
    pitch_int = int(pitch_value) % 12
    return f"{pitch_int}({PITCH_CLASS_NAMES[pitch_int]})"

def format_pitch_sequence(pitch_sequence):
    return "[" + ", ".join(pitch_to_label(p) for p in pitch_sequence) + "]"

def format_topk_predictions(topk_indices, topk_probs):
    return "[" + ", ".join(
        f"{pitch_to_label(pitch)} ({prob:.3f})"
        for pitch, prob in zip(topk_indices, topk_probs)
    ) + "]"

def format_pitch_prediction_row(split_name, epoch_num, input_sequence, pred_pitch, target_pitch, topk_indices, topk_probs):
    return (
        f"{split_name} Sample | Epoch {epoch_num} | "
        f"Input: {format_pitch_sequence(input_sequence[-5:])} (last 5) -> "
        f"Pred: {pitch_to_label(pred_pitch)} | "
        f"Target: {pitch_to_label(target_pitch)} | "
        f"Top-k: {format_topk_predictions(topk_indices, topk_probs)}"
    )

def compute_pitch_frequency_bucket_ids(song_note_arrays):
    """Buckets pitch classes into rare / medium / common."""
    pitch_counts = np.zeros(12, dtype=np.int64)
    for song_notes in song_note_arrays:
        notes_array = np.asarray(song_notes, dtype=np.float32)
        if notes_array.size == 0:
            continue
        pitches = (notes_array[:, 0].astype(np.int64)) % 12
        pitch_counts += np.bincount(pitches, minlength=12)

    nonzero_counts = pitch_counts[pitch_counts > 0]
    bucket_ids = np.full(12, -1, dtype=np.int64)
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
        target_pitch = flat_idx // 12
        predicted_pitch = flat_idx % 12
        entries.append(f"{pitch_to_label(target_pitch)} -> {pitch_to_label(predicted_pitch)}: {int(count)}")
    return ", ".join(entries) if entries else "None"

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
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            wandb.login() 
        wandb.init(project="music-rnn-ddp-12classes", entity="riya-rajbhut-student", config=hparams)
        weights_history = []
        sampled_epochs = []

    dataset_root = pathlib.Path('data/maestro-v3.0.0')
    if is_main_process:
        download_maestro_dataset()
    dist.barrier()

    converted_notes = load_or_create_note_cache(dataset_root, is_main_process, hparams['years_to_use'])
    train_notes, val_notes, test_notes = split_song_arrays(converted_notes, seed=hparams['seed'])

    pitch_bucket_ids_np, pitch_bucket_names, _ = compute_pitch_frequency_bucket_ids(train_notes)
    pitch_bucket_ids_t = torch.tensor(pitch_bucket_ids_np, device=gpu, dtype=torch.long)

    train_dataset = BasicRNNForMusic(train_notes, seq_len=hparams['seq_len'], augment=True)
    val_dataset = BasicRNNForMusic(val_notes, seq_len=hparams['seq_len'], augment=False)
    test_dataset = BasicRNNForMusic(test_notes, seq_len=hparams['seq_len'], augment=False)

    if is_main_process:
        print(f"Dataset split — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = data.DataLoader(train_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=train_sampler, pin_memory=True, num_workers=2, drop_last=True, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=hparams['batch_size_per_gpu'], sampler=val_sampler, pin_memory=True, num_workers=2, persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=hparams['batch_size_per_gpu'], pin_memory=True, num_workers=2, shuffle=False)

    model = OptimizedMusicRNN(num_pitches=12, hidden_size=hparams['hidden_size'], num_layers=hparams['num_layers']).cuda(gpu)
    model = DDP(model, device_ids=[gpu])

    criterion_pitch = nn.CrossEntropyLoss(label_smoothing=hparams['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])

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

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(hparams["epochs"]):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        train_correct, train_total = 0, 0

        for batch_idx, (x_pitch, y_pitch) in enumerate(train_loader):
            x_pitch = x_pitch.cuda(gpu, non_blocking=True)
            y_pitch = y_pitch.cuda(gpu, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                preds = model(x_pitch)
                predicted_pitch = torch.argmax(preds["pitch"], dim=1)

                train_correct += (predicted_pitch == y_pitch).sum().item()
                train_total += y_pitch.size(0)

                train_loss = criterion_pitch(preds["pitch"], y_pitch)

            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += train_loss.item()

        model.eval()
        val_loss_tot = 0.0
        val_correct, val_total = 0, 0
        debug_rows = []
        
        val_topk_correct = torch.zeros(3, device=gpu, dtype=torch.float64)
        val_confusion = torch.zeros((12, 12), device=gpu, dtype=torch.int64)
        val_pred_counts = torch.zeros(12, device=gpu, dtype=torch.int64)
        val_target_counts = torch.zeros(12, device=gpu, dtype=torch.int64)

        with torch.no_grad():
            for x_pitch, y_pitch in val_loader:
                x_pitch = x_pitch.cuda(gpu, non_blocking=True)
                y_pitch = y_pitch.cuda(gpu, non_blocking=True)

                with autocast("cuda"):
                    preds = model(x_pitch)
                    predicted_pitch = torch.argmax(preds["pitch"], dim=1)
                    top5_indices = torch.topk(preds["pitch"], k=5, dim=1).indices

                    val_correct += (predicted_pitch == y_pitch).sum().item()
                    val_total += y_pitch.size(0)

                    val_topk_correct[0] += (top5_indices[:, :1] == y_pitch.unsqueeze(1)).any(dim=1).sum().item()
                    val_topk_correct[1] += (top5_indices[:, :3] == y_pitch.unsqueeze(1)).any(dim=1).sum().item()
                    val_topk_correct[2] += (top5_indices == y_pitch.unsqueeze(1)).any(dim=1).sum().item()

                    val_pred_counts += torch.bincount(predicted_pitch, minlength=12)
                    val_target_counts += torch.bincount(y_pitch, minlength=12)
                    confusion_indices = y_pitch * 12 + predicted_pitch
                    val_confusion += torch.bincount(confusion_indices, minlength=12 * 12).reshape(12, 12)

                    val_loss = criterion_pitch(preds["pitch"], y_pitch)
                    val_loss_tot += val_loss.item()

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
                val_loss_tot / len(val_loader),
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

        train_l, val_l, train_c, train_t, val_c, val_t = metrics.tolist()
        train_l /= world_size
        val_l /= world_size

        train_acc = train_c / train_t if train_t else 0.0
        val_acc = val_c / val_t if val_t else 0.0
        val_acc_top3 = val_topk_correct[1].item() / val_t if val_t else 0.0
        val_acc_top5 = val_topk_correct[2].item() / val_t if val_t else 0.0
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if is_main_process:
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "train_loss": train_l,
                    "train_acc": train_acc,
                    "val_loss": val_l,
                    "val_acc": val_acc,
                    "val_acc_top3": val_acc_top3,
                    "val_acc_top5": val_acc_top5
                }
            )

            log_payload = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train/loss": train_l,
                "train/accuracy": train_acc,
                "val/loss": val_l,
                "val/accuracy": val_acc,
                "val/accuracy_top3": val_acc_top3,
                "val/accuracy_top5": val_acc_top5,
            }

            if (epoch + 1) % 10 == 0:
                val_confusion_cpu = val_confusion.detach().cpu().numpy()
                fig_conf, ax_conf = plt.subplots(figsize=(8, 7))
                sns.heatmap(
                    np.log1p(val_confusion_cpu),
                    cmap="viridis",
                    ax=ax_conf,
                    cbar_kws={"label": "log(1 + count)"},
                    xticklabels=PITCH_CLASS_NAMES,
                    yticklabels=PITCH_CLASS_NAMES
                )
                ax_conf.set_title(f"Val Pitch Confusion Matrix (Epoch {epoch + 1})")
                ax_conf.set_xlabel("Predicted Pitch Class")
                ax_conf.set_ylabel("Target Pitch Class")
                plt.tight_layout()

                log_payload["val/confusion_matrix"] = wandb.Image(fig_conf)
                plt.close(fig_conf)

            wandb.log(log_payload, step=epoch + 1)

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
                "Val Most Common Pitch Confusions | "
                + summarize_pitch_confusions(val_confusion.detach().cpu(), top_n=5)
            )

            if val_l < best_val_loss:
                best_val_loss = val_l
                epochs_without_improvement = 0
                torch.save({"model_state_dict": model.module.state_dict()}, best_checkpoint_path)
            else:
                epochs_without_improvement += 1

            pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)

        stop_signal = torch.tensor([1 if epochs_without_improvement >= hparams["patience"] else 0], device=gpu)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
        if stop_signal.item() > 0:
            break

    if is_main_process:
        print("\n--- Running Evaluation On Test Set ---")
        checkpoint = torch.load(best_checkpoint_path, map_location='cuda:0')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        correct_pitch, total_samples = 0, 0
        test_debug_rows = []

        with torch.no_grad():
            for x_pitch, y_pitch in test_loader:
                x_pitch = x_pitch.cuda(0, non_blocking=True)
                y_pitch = y_pitch.cuda(0, non_blocking=True)
                
                with autocast('cuda'):
                    preds = model.module(x_pitch)
                    predicted_classes = torch.argmax(preds['pitch'], dim=1)
                    correct_pitch += (predicted_classes == y_pitch).sum().item()
                    total_samples += y_pitch.size(0)

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
            wandb.log({"test/pitch_accuracy": correct_pitch / total_samples})

        print("\n--- Generating Embedding Visualizations ---")
        
        embedding_weights = model.module.pitch_embed.weight.detach().cpu().numpy()
        
        # 1. Cosine Similarity Heatmap
        similarity_matrix = cosine_similarity(embedding_weights)
        fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix, 
            xticklabels=PITCH_CLASS_NAMES, 
            yticklabels=PITCH_CLASS_NAMES, 
            cmap="coolwarm", 
            annot=True, 
            fmt=".2f", 
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax_sim
        )
        ax_sim.set_title("Pitch Class Embedding Similarity")
        ax_sim.set_xlabel("Pitch Class")
        ax_sim.set_ylabel("Pitch Class")
        plt.tight_layout()
        wandb.log({"embeddings/cosine_similarity": wandb.Image(fig_sim)})
        plt.close(fig_sim)

        # 2. PCA Projection
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embedding_weights)

        fig_pca, ax_pca = plt.subplots(figsize=(8, 8))
        ax_pca.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='blue', s=100)

        for i, note in enumerate(PITCH_CLASS_NAMES):
            ax_pca.annotate(
                note, 
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=12,
                fontweight='bold'
            )

        ax_pca.set_title("2D PCA Projection of Pitch Class Embeddings")
        ax_pca.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax_pca.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        ax_pca.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        wandb.log({"embeddings/pca_2d": wandb.Image(fig_pca)})
        plt.close(fig_pca)

        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    hyperparameters = {
        'seq_len': 128,
        'hidden_size': 512,
        'num_layers': 3,
        'batch_size_per_gpu': 256,
        'epochs': 40,
        'patience': 8,
        'lr': 1e-3,
        'warmup_epochs': 2,
        'weight_decay': 1e-4,
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