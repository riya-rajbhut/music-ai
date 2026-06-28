import tensorflow as tf
import numpy as np

#to make sure,the randomness is deterministic, I am setting the seed to 53.
SEED = 53
tf.random.set_seed(SEED)
np.random.seed(SEED)

SAMPLING_RATE = 16000   #I am feeding music as 16k per second

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPUs Available: {tf.config.list_physical_devices('GPU')}")

# Method to download MASESTRO dataset and extract it to a folder
def extract_midifiles_from_maestro_dataset(cache_dir="./data"):

    zip_path = tf.keras.utils.get_file(
        fname="maestro-v3.0.0.zip",    
        origin='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
        extract=True,
        cache_dir=cache_dir,
        cache_subdir="maestro")
    
    base_path = pathlib.Path(zip_path).parent / "maestro-v3.0.0"
    print(f"\n Zip path Extracted: {zip_path}")
    print(f"\n Base path: {base_path}")

    filenames = list(base_path.glob("**/*.midi"))
    print(f"\n Total number of midi files: {len(filenames)}")

    return filenames

    
