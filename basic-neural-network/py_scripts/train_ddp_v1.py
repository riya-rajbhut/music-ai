"""
A minimal LSTM that learns to predict the next note in a piece of music.

This is deliberately simplified for learning, not for accuracy. Every line
here has a clear job, and there's nothing extra to explain away. Given the
last SEQ_LEN notes played, the model predicts what note comes next - the
exact same idea as "given the last few words, predict the next word."

To keep things fast and easy to follow, this only uses a handful of songs
and trains for a few epochs. Don't expect a great model - expect one you
can read from top to bottom and understand.
"""

import pathlib
import zipfile

import pretty_midi as pm
import torch
import torch.nn as nn
from torch.hub import download_url_to_file
from torch.utils.data import Dataset, DataLoader

# --- Settings you can play with ---
MAX_SONGS = 50        # fewer songs = faster to run, but a worse model
SEQ_LEN = 32           # how many past notes the model gets to look at
HIDDEN_SIZE = 128      # size of the LSTM's internal memory
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================================================================
# Step 1: Get some real piano music to learn from
# ======================================================================

def download_maestro():
    """Downloads and unzips a public dataset of piano performances (MIDI files)."""
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    data_dir = pathlib.Path('data')
    data_dir.mkdir(exist_ok=True)
    zip_path = data_dir / 'maestro.zip'
    extracted_path = data_dir / 'maestro-v3.0.0'

    if not extracted_path.exists():
        print("Downloading MAESTRO dataset (piano performances)...")
        download_url_to_file(url, str(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(data_dir)

    return extracted_path


def midi_file_to_pitches(midi_path):
    """Turns one MIDI file into a plain list of note pitches (0-127), in the order played."""
    midi_data = pm.PrettyMIDI(str(midi_path))
    if not midi_data.instruments:
        return []
    notes_in_order = sorted(midi_data.instruments[0].notes, key=lambda note: note.start)
    return [note.pitch for note in notes_in_order]


# ======================================================================
# Step 2: Turn songs into (past notes -> next note) training examples
# ======================================================================

class NoteDataset(Dataset):
    """
    Each training example is SEQ_LEN notes in a row, plus the note that comes
    right after them. The model's whole job is to predict that next note from
    the notes before it.
    """

    def __init__(self, songs, seq_len):
        self.examples = []
        for pitches in songs:
            for start in range(len(pitches) - seq_len):
                window = pitches[start:start + seq_len]
                next_note = pitches[start + seq_len]
                self.examples.append((window, next_note))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        window, next_note = self.examples[idx]
        return torch.tensor(window, dtype=torch.long), torch.tensor(next_note, dtype=torch.long)


# ======================================================================
# Step 3: The LSTM model itself
# ======================================================================

class NoteLSTM(nn.Module):
    """
    Reads a sequence of note pitches and predicts the pitch of the next note.

    The pipeline is: pitch numbers -> embeddings -> LSTM -> linear layer ->
    a score for each of the 128 possible next pitches (higher = more likely).
    """

    def __init__(self, num_pitches=128, embed_dim=32, hidden_size=HIDDEN_SIZE):
        super().__init__()
        # Turns each pitch number (0-127) into a small vector of learned features,
        # instead of treating pitch numbers as if their raw size were meaningful.
        self.embedding = nn.Embedding(num_pitches, embed_dim)
        # Reads the sequence of embeddings one note at a time, updating an internal
        # "memory" (the hidden state) as it goes.
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        # Turns the LSTM's final memory into a score for each possible next pitch.
        self.output_layer = nn.Linear(hidden_size, num_pitches)

    def forward(self, pitch_sequence):
        embedded = self.embedding(pitch_sequence)             # (batch, seq_len, embed_dim)
        _, (final_hidden, _) = self.lstm(embedded)             # run the LSTM over the whole sequence
        return self.output_layer(final_hidden[-1])             # (batch, num_pitches) scores


# ======================================================================
# Step 4: Load the data, train the model, see how it did
# ======================================================================

def main():
    data_root = download_maestro()
    midi_files = list(data_root.glob('**/*.midi'))[:MAX_SONGS]
    print(f"Using {len(midi_files)} songs")

    songs = [midi_file_to_pitches(path) for path in midi_files]
    songs = [notes for notes in songs if len(notes) > SEQ_LEN]  # drop songs too short to learn from

    split_point = int(len(songs) * 0.9)
    train_songs, val_songs = songs[:split_point], songs[split_point:]

    train_data = NoteDataset(train_songs, SEQ_LEN)
    val_data = NoteDataset(val_songs, SEQ_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    print(f"Train examples: {len(train_data)}, Val examples: {len(val_data)}")

    model = NoteLSTM().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # --- Training: the model learns from the training examples ---
        model.train()
        train_loss = 0.0
        for windows, targets in train_loader:
            windows, targets = windows.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(windows)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- Validation: just checking how it does on notes it didn't train on ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for windows, targets in val_loader:
                windows, targets = windows.to(device), targets.to(device)
                predictions = model(windows)
                val_loss += loss_function(predictions, targets).item()
                correct += (predictions.argmax(dim=1) == targets).sum().item()
                total += targets.size(0)

        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.1f}%")


if __name__ == '__main__':
    main()