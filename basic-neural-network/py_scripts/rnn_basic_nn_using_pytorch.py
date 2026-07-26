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
        download_url_to_file(url, str(zip_target_path), progress=True)
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
import pretty_midi as pm
def convert_midi_to_notes(midi_file_path: str) -> pd.DataFrame:
    midi_data = pm.PrettyMIDI(str(midi_file_path))

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
    all_songs_notes_array = []
    all_midi_files = list(pathlib.Path(data_path).glob('**/*.midi'))

    if not all_midi_files:
        raise FileNotFoundError(f"No MIDI files found in path: {data_path}")
    
    for midi_file in all_midi_files:
        notes_df = convert_midi_to_notes(midi_file)
        if notes_df.empty:
            continue

        single_song_notes_array= notes_df[['pitch', 'step', 'duration']].to_numpy(dtype=np.float32)
        all_songs_notes_array.append(single_song_notes_array)
    
    if not all_songs_notes_array:
        raise ValueError("No valid note data extracted from any of the MIDI files.")

    print(f"Successfully processed {len(all_songs_notes_array)} individual songs.")
    return all_songs_notes_array # Returns a list of arrays: [ [song1_notes], [song2_notes], ... ]

path_to_data = pathlib.Path('data/maestro-v3.0.0/2004')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Training loopwas taking much more time,hence caching converted notes.
cache_file = pathlib.Path('data/maestro-v3.0.0/converted_notes_cache.npy')
if cache_file.exists():
    print(f"Loading converted notes from cache: {cache_file}")
    converted_notes_array = np.load(cache_file, allow_pickle=True)
else:
    download_maestro_dataset()
    converted_notes_array = convert_all_songs_to_notes(path_to_data)
    
print(f'Converted Notes Array:\n{converted_notes_array[:5]}')  # Print first 5 rows of the array
all_notes_flat = np.vstack(converted_notes_array) # np.concatenate(converted_notes_array, axis=0)
print(f"Flattened Array Shape: {all_notes_flat.shape}")

if not cache_file.exists():
    np.save(cache_file, all_notes_flat)

class BasicRNNForMusic(data.Dataset):
    def __init__(self, notes_array, seq_len=30):
        self.seq_len = seq_len

        notes_array = np.array(notes_array)
        if notes_array.ndim == 1:
            raise ValueError(f"Expected 2D array with shape (N, 3), but got 1D array with shape {notes_array.shape}")
        
        pitches = notes_array[:, 0].astype(int)       # Keep as integers for CrossEntropy
        time_features = notes_array[:, 1:3].copy()    # step and duration
        
        self.pitches = torch.tensor(pitches, dtype=torch.long)
        self.time_features = torch.tensor(time_features, dtype=torch.float32)

    def __len__(self):
        return len(self.pitches) - self.seq_len

    def __getitem__(self, idx):
        # We need pitch as float for the LSTM input layers, combined with times
        x_pitch = self.pitches[idx : idx + self.seq_len]
        x_time = self.time_features[idx : idx + self.seq_len]
            
        # 2. Extract separate targets (Y)
        y_pitch = self.pitches[idx + self.seq_len]                 # Scalar tensor (Long) for Classification
        y_time = self.time_features[idx + self.seq_len]             # Tensor of 2 floats for Regression
        
        return (x_pitch, x_time), (y_pitch, y_time)

seq_len = 30
batch_size=1024
deterministic_data_seed = 53

torch.manual_seed(deterministic_data_seed)


total_notes = len(all_notes_flat)
train_cutoff = int(0.8 * total_notes)
val_cutoff = int(0.9 * total_notes)
 
train_notes = all_notes_flat[:train_cutoff]
val_notes = all_notes_flat[train_cutoff:val_cutoff]
test_notes = all_notes_flat[val_cutoff:]

training_dataset = BasicRNNForMusic(train_notes, seq_len=seq_len)
validation_dataset = BasicRNNForMusic(val_notes, seq_len=seq_len)
testing_dataset = BasicRNNForMusic(test_notes, seq_len=seq_len)
print(f"Training Dataset Length: {len(training_dataset)}, Validation Dataset Length: {len(validation_dataset)}, Testing Dataset Length: {len(testing_dataset)}")

training_loader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
testing_loader = data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
import torch
import torch.nn as nn

class MusicRNNV2(nn.Module):
    def __init__(self, num_pitches=128, pitch_embed_dim=32, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super(MusicRNNV2, self).__init__()
        
        # 1. Discrete Pitch Embedding
        self.pitch_embedding = nn.Embedding(num_embeddings=num_pitches, embedding_dim=pitch_embed_dim)
        
        # Total feature size = pitch_embed_dim + 2 (step & duration)
        input_size = pitch_embed_dim + 2
        
        # 2. Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. Prediction Heads
        self.pitch_head = nn.Linear(hidden_size, num_pitches)
        self.step_head = nn.Linear(hidden_size, 1)
        self.duration_head = nn.Linear(hidden_size, 1)

    def forward(self, pitch_seq, time_seq):
        # pitch_seq shape: (batch_size, seq_len) [LongTensor]
        # time_seq shape:  (batch_size, seq_len, 2) [FloatTensor]
        
        # Embed pitches -> shape: (batch_size, seq_len, pitch_embed_dim)
        pitch_embeds = self.pitch_embedding(pitch_seq)
        
        # Concatenate embedded pitch with step & duration
        x = torch.cat([pitch_embeds, time_seq], dim=2)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Extract last step output
        last_out = self.dropout(lstm_out[:, -1, :])
        
        return {
            'pitch': self.pitch_head(last_out),
            'step': torch.relu(self.step_head(last_out)),
            'duration': torch.relu(self.duration_head(last_out))
        }

# Instantiate the network and push to active hardware
model = MusicRNNV2().to(device)
print(model)

#=======================================================
# Let's Start with Training Loop
#=======================================================
from tqdm import tqdm  # pip install tqdm

# Separate loss criteria for multi-task outputs
criterion_pitch = nn.CrossEntropyLoss()
criterion_time = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5  # Set to 5 or 10 epochs for local testing
print("\n--- Starting Training Optimization Loop ---")
for epoch in range(epochs):
    model.train()  # Flag training state (activates dropout)
    running_loss = 0.0

    print("Starting with Epoch: {}/{}".format(epoch+1, epochs))
    for batch_idx, (inputs, targets) in enumerate(training_loader):
        print("Starting Training Batch: {}/{}".format(batch_idx+1, len(training_loader)))

        # Push batch variables onto target hardware device (CPU or GPU)
        x_pitch, x_time = inputs
        x_pitch, x_time = x_pitch.to(device, non_blocking=True), x_time.to(device, non_blocking=True)

        # Deconstruct target slices matching prediction dimensions: shape (batch_size, 1)
        y_pitch, y_time = targets
        actual_pitch = y_pitch.to(device)
        actual_step = y_time[:, 0:1].to(device)
        actual_duration = y_time[:, 1:2].to(device)
        
        # 1. Clear previous gradient memory in each loop
        optimizer.zero_grad()
        
        # 2. Get the predictions from the model for the current batch
        predictions = model(x_pitch, x_time)
        
        # 3. Calculate individual element losses and aggregate them
        loss_pitch = criterion_pitch(predictions['pitch'], actual_pitch)
        loss_step = criterion_time(predictions['step'], actual_step)
        loss_duration = criterion_time(predictions['duration'], actual_duration)
        
        total_loss = loss_pitch + 10.0 *(loss_step + loss_duration)
        
        # 4. Backward error propagation
        total_loss.backward()
        
        # 5. Fine-tune model parameters
        optimizer.step()
        
        running_loss += total_loss.item()
        
    epoch_loss = running_loss / len(training_loader)
    print(f"Epoch [{epoch+1}/{epochs}] -> Mean Composite Loss: {epoch_loss:.4f}")

#=======================================================
# Let's Validate our model now
#=======================================================
model.eval()  # We don't need to drop out neurons during validations.. hence set this func
running_val_loss = 0.0
generated_notes = []

print("\n--- Generating 10 New Notes Iteratively during VALIDATION---")
with torch.no_grad():  # Turn off gradient engine to maximize inference performance- this is Validation phase hence we dont need
    for inputs, targets in validation_loader:
        x_pitch, x_time = inputs
        x_pitch, x_time = x_pitch.to(device, non_blocking=True), x_time.to(device, non_blocking=True)

        y_pitch, y_time = targets
    
        # Deconstruct target slices matching prediction dimensions: shape (batch_size, 1)
        actual_pitch = y_pitch.to(device)
        actual_step = y_time[:, 0:1].to(device)
        actual_duration = y_time[:, 1:2].to(device)
    
        # Predict the parameters for the next note
        preds = model(x_pitch, x_time)
        
        # Compute combined loss
        loss_pitch = criterion_pitch(preds['pitch'], actual_pitch)
        loss_step = criterion_time(preds['step'], actual_step)
        loss_duration = criterion_time(preds['duration'], actual_duration)

        val_loss = loss_pitch + loss_step + loss_duration
        running_val_loss += val_loss.item()

avg_val_loss = running_val_loss / len(validation_loader)
print(f"Validation Loss: {avg_val_loss:.4f}")

#=======================================================
# TESTING - Let's test our model now
#=======================================================
print("\n--- Running Final Testing Loop ---")
running_test_loss = 0.0
correct_pitch_predictions = 0
total_samples = 0

model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation for testing
    for inputs, targets in validation_loader:
        x_pitch, x_time = inputs
        x_pitch, x_time = x_pitch.to(device, non_blocking=True), x_time.to(device, non_blocking=True)

        y_pitch, y_time = targets
    
        # Deconstruct target slices matching prediction dimensions: shape (batch_size, 1)
        actual_pitch = y_pitch.to(device)
        actual_step = y_time[:, 0:1].to(device)
        actual_duration = y_time[:, 1:2].to(device)

        preds = model(x_pitch, x_time)

        loss_pitch = criterion_pitch(preds['pitch'], actual_pitch)
        loss_step = criterion_time(preds['step'], actual_step)
        loss_duration = criterion_time(preds['duration'], actual_duration)
        test_loss = loss_pitch + loss_step + loss_duration
        running_test_loss += test_loss.item()

        predicted_pitch_classes = torch.argmax(preds['pitch'], dim=1)
        correct_pitch_predictions += (predicted_pitch_classes == actual_pitch).sum().item()
        total_samples += actual_pitch.size(0)

avg_test_loss = running_test_loss / len(validation_loader)
accuracy = correct_pitch_predictions / total_samples * 100
print(f"Test Loss: {avg_test_loss:.4f}, Pitch Prediction Accuracy: {accuracy:.2f}%")
print("Pitch prediction accuracy on TESTING dataset: {accuracy:.2f}%".format(accuracy=accuracy))

print("\nSample Generation Loop Completed Successfully!")
