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

def convert_all_songs_to_midi(data_path: str) -> np.ndarray:
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
converted_notes_array = convert_all_songs_to_midi(path_to_data)
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

dataset= BasicRNNForMusic(converted_notes_array, seq_len=seq_len)

training_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

