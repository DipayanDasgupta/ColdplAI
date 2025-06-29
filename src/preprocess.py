import warnings
from pathlib import Path
from miditok import REMI
import json

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
MIDI_DIR = Path("data/midi")
TOKENIZER_PATH = Path("models/tokenizer.json")
TOKEN_OUTPUT_FILE = Path("data/all_tokens.json") 

def preprocess_midi_files():
    """
    Final, robust version built for miditok v3.x.
    It lets the library use its new 'symusic' backend, which is correct.
    """
    print("Starting MIDI preprocessing...")
    midi_files = list(MIDI_DIR.glob('**/*.mid'))
    print(f"Found {len(midi_files)} MIDI files.")

    # 1. Create and train the tokenizer on the file paths
    # This automatically uses the new symusic backend.
    tokenizer = REMI()
    tokenizer.train(vocab_size=1000, files_paths=midi_files)
    
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer vocabulary learned and saved to {TOKENIZER_PATH}")

    # 2. Tokenize all files using the modern API
    all_tokens = []
    success_count = 0
    for path in midi_files:
        try:
            # tokenizer(path) now correctly loads and tokenizes with symusic
            # It returns a list of TokSequence objects.
            tokens_list = tokenizer(path)
            
            # We extract the integer IDs from the first track's TokSequence
            if tokens_list and tokens_list[0].ids:
                all_tokens.append(tokens_list[0].ids)
                success_count += 1
        except Exception as e:
            print(f"--> Could not process {path.name}, skipping. Error: {e}")
            continue
    
    if success_count == 0:
        raise RuntimeError("CRITICAL ERROR: No MIDI files could be processed.")

    # 3. Save the consolidated list of tokens
    TOKEN_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_OUTPUT_FILE, 'w') as f:
        # We now have a simple list of lists, save it directly.
        json.dump({"tokens": all_tokens}, f)

    print(f"Successfully processed {success_count} / {len(midi_files)} files.")
    print(f"All tokens saved to a single file: {TOKEN_OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_midi_files()