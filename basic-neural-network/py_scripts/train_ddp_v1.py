import pathlib
import pretty_midi as pm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.hub import download_url_to_file
import zipfile

# ==========================================
# 1. DOWNLOAD A SAMPLE FROM MAESTRO
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


# ==========================================
# 2. CONVERT A MIDI FILE TO NUMBERS
# ==========================================
def load_midi_notes(midi_file_path):
    """Reads a MIDI file and returns a list of [pitch, step, duration] notes."""
    midi_data = pm.PrettyMIDI(str(midi_file_path))
    instrument = midi_data.instruments[0]  # Grab the piano track
    
    # Sort notes by when they start playing
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    
    notes_list = []
    prev_start_time = 0.0
    
    for note in sorted_notes:
        pitch = float(note.pitch)                       # MIDI note (0 to 127)
        step = float(note.start - prev_start_time)      # Wait time before this note starts
        duration = float(note.end - note.start)         # How long the note rings
        
        notes_list.append([pitch, step, duration])
        prev_start_time = note.start
        
    return torch.tensor(notes_list, dtype=torch.float32)


# ==========================================
# 3. SLIDING WINDOW PYTORCH DATASET
# ==========================================
class SimpleMusicDataset(Dataset):
    """Turns a list of notes into 10-note sequences that predict the 11th note."""
    def __init__(self, notes_tensor, seq_len=10):
        self.inputs = []
        self.targets = []
        
        # Slide a window across the song notes
        for i in range(len(notes_tensor) - seq_len):
            # Input: Past `seq_len` notes (e.g. notes 0 to 9)
            self.inputs.append(notes_tensor[i : i + seq_len])
            # Target: Next note to predict (e.g. note 10)
            self.targets.append(notes_tensor[i + seq_len])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ==========================================
# 4. HOW TO USE IT WITH THE LSTM
# ==========================================
if __name__ == '__main__':
    # Step 1: Download & get path to a MIDI file
    sample_midi_path = download_sample_data()
    print(f"Reading MIDI file: {sample_midi_path.name}")

    # Step 2: Read the notes out of the MIDI file
    notes = load_midi_notes(sample_midi_path)
    print(f"Total notes in song: {len(notes)}")

    # Step 3: Package into a PyTorch Dataset & DataLoader
    dataset = SimpleMusicDataset(notes_tensor=notes, seq_len=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Step 4: Grab a single batch to inspect
    x_batch, y_batch = next(iter(dataloader))
    
    print("\n--- Batch Shapes ready for the LSTM ---")
    print(f"Input batch shape (X):  {x_batch.shape} -> (Batch Size, Sequence Length, Features)")
    print(f"Target batch shape (Y): {y_batch.shape} -> (Batch Size, Target Note Features)")