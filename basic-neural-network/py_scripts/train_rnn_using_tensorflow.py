import tensorflow as tf
import numpy as np

import pretty_midi as pm
import pandas as pd
    

#to make sure,the randomness is deterministic, I am setting the seed to 53.
SEED = 53
tf.random.set_seed(SEED)
np.random.seed(SEED)

SAMPLING_RATE = 16000   #I am feeding music as 16k per second

print(f"\n TensorFlow Version: {tf.__version__}")
print(f"\n GPUs Available: {tf.config.list_physical_devices('GPU')}")

# Method to download MASESTRO dataset and extract it to a folder
def extract_midifiles_from_maestro_dataset(cache_dir="./data"):
    
    zip_path = tf.keras.utils.get_file(
        fname="maestro-v3.0.0.zip",    
        origin='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
        extract=True,
        cache_dir=cache_dir,
        cache_subdir="maestro")
    
    import pathlib
    base_path = pathlib.Path(zip_path).parent / "maestro-v3_extracted"
    print(f"\n Zip path Extracted: {zip_path}")
    print(f"\n Base path: {base_path}")

    filenames = list(base_path.glob("**/*.midi"))                                   
    print(f"\n Total number of midi files: {len(filenames)}")

    return filenames

def parse_midi_file(midi_file_path: str) -> pd.DataFrame:
    import collections
    try:
        pm_data = pm.PrettyMIDI(midi_file_path)
        instrument = pm_data.instruments[0]
        notes = collections.defaultdict(list)

        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        previous_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['step'].append(start - previous_start)
            notes['duration'].append(end - start)

            previous_start = start
        
        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
    except Exception as e:
        print(f"Error parsing MIDI file {midi_file_path}: {e}, sending empty DataFrame")
        return pd.DataFrame()
    
def creat_train_dataset(filenames, num_files=50, sequence_length=25, batch_size=64):
    
    all_notes = []
    for idx, f in enumerate(filenames[:num_files]):
        print(f"Parsing {idx+1}/{num_files}")
        notes_df = parse_midi_file(f)
        if notes_df is not None:
            all_notes.append(notes_df) 
    
    pf_df_dataset_notes = pd.concat(all_notes, ignore_index=True)

    #We have to make sure that the Columns are in correct order hence below
    key_order = ['pitch', 'step', 'duration']

    # As Tensor flow can't work with padnas dataframe, lets convertto numPy aarays so that 
    # we flatten the data in 2D-numpy array for further processing.
    traing_df_notes = np.stack([pf_df_dataset_notes[key].values for key in key_order], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(traing_df_notes)

    seq_ds = notes_ds.window(sequence_length + 1, shift=1, drop_remainder=True)
    seq_ds = seq_ds.flat_map(lambda window: window.batch(sequence_length + 1))
    
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[1:]
        
        # Structure labels as dictionary mapping outputs back to loss functions easily
        # Labels map directly onto predicted metrics
        return inputs, {
            'pitch': labels_dense[:, 0],
            'step': labels_dense[:, 1],
            'duration': labels_dense[:, 2]
        }

    # Apply structural mapping and cache sequence data locally to prevent re-reads
    dataset = seq_ds.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
    
    return dataset
    
#-------------------------------------------------------------------------------
# Actual program starts here
# I will train the RNN model such that it can predict the next note in a sequence of notes.
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    midi_files = extract_midifiles_from_maestro_dataset()

    if not midi_files:
        print("\n No midi files found. Please check the dataset path.")
        exit(1)
    train_dataset = creat_train_dataset(
        filenames=midi_files,
        num_files=10,
        sequence_length=25,
        batch_size=32,
    )
    
    # to check the data we are getting, print
    if train_dataset is not None:
        print(f"\n Train Dataset Shape: {train_dataset.element_spec}")
    else:
        print("\n Warning: train_dataset is None. Check if MIDI files were found and parsed correctly.")
    
    for batch in train_dataset.take(1):
        print(batch)
