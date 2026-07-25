import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import pathlib
import numpy as np

from torch.hub import download_url_to_file
def download_maestro_dataset(dest_dir: str='data'):
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    base_path = pathlib.Path(dest_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    zip_target_path = pathlib.Path(f'{dest_dir}/maestro-v3.0.0-midi.zip')
    extracted_folder_path = pathlib.Path(f'{dest_dir}/maestro-v3.0.0')

    if zip_target_path.exists() and extracted_folder_path.exists():
        print(f"Dataset already downloaded and extracted at {extracted_folder_path}.")
    else:
        print("Downloading the MAESTRO dataset...")
        download_url_to_file(url, zip_target_path, progress=True)
        print("Download complete...")

        if(zip_target_path.exists()):
            print("Extracting the dataset...")
            import zipfile
            with zipfile.ZipFile(zip_target_path, 'r') as zip_ref:
                zip_ref.extractall(base_path)
            print(f"Extraction complete at {extracted_folder_path}.")

    
    sample_files = list(extracted_folder_path.glob('**/*.midi'))
    print(f"Total MIDI Files Discovered: {len(sample_files)}")
    #print(f"Sample files: {sample_files[:3]}")
    return extracted_folder_path

import pandas as pd
import collections
def convert_midi_to_notes(midi_file_path: str) -> pd.DataFrame:
    import pretty_midi as pm
    midi_data = pm.PrettyMIDI(midi_file_path)

    #print(f'Total Instruments in MIDI: {len(midi_data.instruments)}')
    #print all available instruments
    
    print(f'Instrument[0]: {midi_data.instruments[0].name}, Program: {midi_data.instruments[0].program}, Is Drum: {midi_data.instruments[0].is_drum}')

    instrument = midi_data.instruments[0]  # Assuming single instrument for simplicity
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

    prev_start=sorted_notes[0].start if len(sorted_notes) > 0 else 0

    #loop through each sorted note and construct a DataFrame with start time, end time, pitch, and velocity
    for note in sorted_notes:
        notes['pitch'].append(note.pitch)
        notes['step'].append(note.start- prev_start)
        notes['duration'].append(note.end - note.start)

        prev_start = note.start

    
    converted_notes = pd.DataFrame({name: np.array(values) for name, values in notes.items()})
    print(f'Sample of converted notes:\n{converted_notes.head()}')
    return converted_notes

def convert_all_songs_to_notes(data_path: str) -> np.ndarray:
    all_songs_notes = []
    all_midi_files = list(pathlib.Path(data_path).glob('**/*.midi'))

    for midi_file in all_midi_files:
        notes_df = convert_midi_to_notes(midi_file)
        all_songs_notes.append(notes_df)
    
    # Concatenate all notes into a single DataFrame
    all_notes_df = pd.concat(all_songs_notes, ignore_index=True)
    print(f'Total notes collected from all songs: {len(all_notes_df)}')
    print(f'Sample of converted notes:\n{all_notes_df.head()}')

    # Convert to numpy array for further processing
    my_notes_array = all_notes_df[['pitch', 'step', 'duration']].to_numpy(dtype=np.float32)
    print(f'Shape of the final notes array: {my_notes_array.shape}')
    return my_notes_array


path_to_data = pathlib.Path('data/maestro-v3.0.0/2004')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

download_maestro_dataset()
converted_notes_array = convert_all_songs_to_notes(path_to_data)
print(f'Converted Notes Array:\n{converted_notes_array[:5]}')  # Print first 5 rows of the array

class BasicRNNForMusic(data.Dataset):
    def __init__(self, notes_array, seq_len=30):
        self.seq_len = seq_len
        self.data = notes_array.copy()
        self.data[:, 0] = self.data[:, 0] / 128.0  # Normalize pitch

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

seq_len = 30
batch_size=50
deterministic_data_seed = 53

full_dataset = BasicRNNForMusic(converted_notes_array, seq_len=seq_len)
full_len = len(full_dataset)

training_data_len = int(0.8 * full_len)
validation_data_len = int(0.1 * full_len)
testing_data_len = full_len - training_data_len - validation_data_len

training_dataset, validation_dataset, testing_dataset = data.random_split(full_dataset, [training_data_len, validation_data_len, testing_data_len], generator=torch.Generator().manual_seed(deterministic_data_seed))

training_loader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#just check once 
for x, y in training_loader:
    print(f"Input batch shape: {x.shape}, Target batch shape: {y.shape}")
    break  # Just check the first batch 


class MusicRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, dropout_rate=0.2):
        super(MusicRNN, self).__init__()
        
        # LSTM layer expects input shape: (batch, sequence, features)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-Output Heads branching out from the hidden state
        self.pitch_head = nn.Linear(hidden_size, 1)
        self.step_head = nn.Linear(hidden_size, 1)
        self.duration_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, 3)
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Isolate the hidden state of the LAST sequence step only
        last_step_out = lstm_out[:, -1, :]
        last_step_out = self.dropout(last_step_out)
        
        # Forward pass through individual prediction heads
        pitch_pred = self.pitch_head(last_step_out)
        step_pred = self.step_head(last_step_out)
        duration_pred = self.duration_head(last_step_out)
        
        # Return a dictionary of predictions for clear downstream routing
        return {
            'pitch': pitch_pred,
            'step': step_pred,
            'duration': duration_pred
        }

# Instantiate the network and push to active hardware
model = MusicRNN().to(device)
print(model)

# Objective Function and Optimizer bound to model parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 5  # Set to 5 or 10 epochs for local testing
model.train()  # Flag training state (activates dropout)

print("\n--- Starting Training Optimization Loop ---")
for epoch in range(epochs):
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(training_loader):
        # Push batch variables onto target hardware device (CPU or GPU)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Deconstruct target slices matching prediction dimensions: shape (batch_size, 1)
        true_pitch = targets[:, 0].unsqueeze(1)
        true_step = targets[:, 1].unsqueeze(1)
        true_duration = targets[:, 2].unsqueeze(1)
        
        # 1. Clear previous gradient memory
        optimizer.zero_grad()
        
        # 2. Compute network predictions
        predictions = model(inputs)
        
        # 3. Calculate individual element losses and aggregate them
        loss_pitch = criterion(predictions['pitch'], true_pitch)
        loss_step = criterion(predictions['step'], true_step)
        loss_duration = criterion(predictions['duration'], true_duration)
        
        total_loss = loss_pitch + loss_step + loss_duration
        
        # 4. Backward error propagation
        total_loss.backward()
        
        # 5. Fine-tune model parameters
        optimizer.step()
        
        running_loss += total_loss.item()
        
    epoch_loss = running_loss / len(training_loader)
    print(f"Epoch [{epoch+1}/{epochs}] -> Mean Composite Loss: {epoch_loss:.4f}")

model.eval()  # Freeze dropout changes for static evaluations
generated_notes = []

# 1. Grab a starting seed block from your dataset (e.g., first 25 notes)
# Shape will be: (1, 25, 3) -> adding a manual dummy batch axis
start_sequence, _ = dataset[0]
current_input = start_sequence.unsqueeze(0).to(device)

print("\n--- Generating 10 New Notes Iteratively ---")
with torch.no_grad():  # Turn off gradient engine to maximize inference performance
    for i in range(10):
        # Predict the parameters for the next note
        preds = model(current_input)
        
        # Extract scalar float properties from the model output tensor
        pred_pitch_norm = preds['pitch'].item()
        pred_step = max(0.0, preds['step'].item())         # Force timing values positive
        pred_duration = max(0.1, preds['duration'].item()) # Force a minimum audible duration
        
        # Denormalize the pitch back up to raw standard MIDI values (0-127)
        pred_pitch = int(np.clip(pred_pitch_norm * 128.0, 0, 127))
        
        # Store predicted notes for inspection
        generated_notes.append([pred_pitch, pred_step, pred_duration])
        print(f"Note {i+1:02d}: Pitch={pred_pitch}, Step={pred_step:.3f}s, Duration={pred_duration:.3f}s")
        
        # 2. Construct the next step input vector: shape (1, 1, 3)
        next_note_tensor = torch.tensor([[[pred_pitch_norm, pred_step, pred_duration]]], dtype=torch.float32).to(device)
        
        # 3. Slide your window: drop the oldest note and append the newly predicted note
        current_input = torch.cat((current_input[:, 1:, :], next_note_tensor), dim=1)

print("\nSample Generation Loop Completed Successfully!")
