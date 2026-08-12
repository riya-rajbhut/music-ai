import pathlib
import zipfile
import pretty_midi as pm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.hub import download_url_to_file

# ==========================================
# 1. DATASET DOWNLOADING & MIDI PARSING
# ==========================================
def download_sample_data():
    """Downloads the MAESTRO dataset zip if you don't have it."""
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    zip_path = pathlib.Path("maestro.zip")
    extract_path = pathlib.Path("maestro_data")
    
    if not extract_path.exists():
        print("Downloading MAESTRO dataset sample...")
        download_url_to_file(url, str(zip_path), progress=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
    # Find the first .midi file in the extracted folder
    midi_files = list(extract_path.glob("**/*.midi"))
    return midi_files[0]


def load_midi_notes(midi_file_path):
    """Reads a MIDI file and returns a tensor of [pitch, step, duration] notes."""
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    instrument = midi_data.instruments[0]  # Grab the piano track
    
    # Sort notes by when they start playing
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    
    notes_list = []
    prev_start_time = 0.0
    
    for note in sorted_notes:
        pitch = float(note.pitch)                  # MIDI note (0 to 127)
        step = float(note.start - prev_start_time) # Wait time before this note starts
        duration = float(note.end - note.start)    # How long the note rings
        
        notes_list.append([pitch, step, duration])
        prev_start_time = note.start
        
    return torch.tensor(notes_list, dtype=torch.float32)


# ==========================================
# 2. SLIDING WINDOW PYTORCH DATASET
# ==========================================
class SimpleMusicDataset(Dataset):
    """Turns a list of notes into 10-note sequences that predict the 11th note."""
    def __init__(self, notes_tensor, seq_len=10):
        self.inputs = []
        self.targets = []
        
        # Slide a window across the song notes
        for i in range(len(notes_tensor) - seq_len):
            # Input: Past `seq_len` notes (e.g., notes 0 to 9)
            self.inputs.append(notes_tensor[i : i + seq_len])
            # Target: Next note to predict (e.g., note 10)
            self.targets.append(notes_tensor[i + seq_len])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ==========================================
# 3. THE SIMPLE LSTM MODEL
# ==========================================
class SimpleMusicLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()
        # The LSTM reads a sequence of past notes
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        
        # A single linear layer converts the LSTM memory into the final note prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the LSTM's output at the VERY LAST note of the sequence
        last_note_memory = lstm_out[:, -1, :]
        
        # Predict the next note (pitch, step, duration)
        prediction = self.fc(last_note_memory)
        return prediction


# ==========================================
# 4. MAIN TRAINING LOOP (REAL DATA)
# ==========================================
if __name__ == '__main__':
    # Hyperparameters
    seq_len = 10        # Number of past notes the model looks at
    features = 3       # [Pitch, Step, Duration]
    batch_size = 16
    epochs = 5
    hidden_size = 64
    learning_rate = 0.001

    # Step 1: Download & get path to a MIDI file
    sample_midi_path = download_sample_data()
    print(f"Reading MIDI file: {sample_midi_path.name}")

    # Step 2: Read the real notes out of the MIDI file
    notes = load_midi_notes(sample_midi_path)
    print(f"Total notes in song: {len(notes)}")

    # Step 3: Package into a PyTorch Dataset & DataLoader
    dataset = SimpleMusicDataset(notes_tensor=notes, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 4: Initialize Model, Loss Function, and Optimizer
    model = SimpleMusicLSTM(input_size=features, hidden_size=hidden_size, output_size=features)
    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 5: Run Training Loop over real DataLoader batches
    print("\nStarting Training on Real Data...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for x_batch, y_batch in dataloader:
            # 1. Reset gradients from the previous step
            optimizer.zero_grad()

            # 2. Forward Pass: Make predictions
            predictions = model(x_batch)

            # 3. Calculate Loss: How wrong was the prediction?
            loss = criterion(predictions, y_batch)

            # 4. Backward Pass: Calculate gradients
            loss.backward()

            # 5. Update Model Weights
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}")

    print("\nTraining Complete!")