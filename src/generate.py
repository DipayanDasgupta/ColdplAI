import torch
from miditok import REMI
from transformers import GPT2LMHeadModel
from pathlib import Path
import os

# --- Configuration ---
MODEL_DIR = Path("models/coldplay_gpt2")
TOKENIZER_PATH = Path("models/tokenizer.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_MIDI_NAME = "my_coldplay_song.mid"
OUTPUT_AUDIO_NAME = "my_coldplay_song.wav"
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.9
TOP_K = 50

# ==============================================================================
#  IMPORTANT: YOU MUST CHANGE THIS PATH TO GET A .wav AUDIO FILE!
# ==============================================================================
# Download a soundfont like "GeneralUser GS v1.471.sf2" and put its path here.
# Example for Linux/WSL: SOUNDFONT_PATH = "/usr/share/sounds/sf2/GeneralUser_GS.sf2"
# Example for Windows: SOUNDFONT_PATH = "C:/Users/YourUser/Documents/SoundFonts/GeneralUser_GS.sf2"
SOUNDFONT_PATH = "/path/to/your/soundfont.sf2"
# ==============================================================================

def generate_music():
    print("Starting music generation...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    tokenizer = REMI(params=TOKENIZER_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt_tokens = [tokenizer["BOS_None"]]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            do_sample=True,
            eos_token_id=tokenizer["EOS_None"]
        )
    
    generated_ids = output[0].cpu().tolist()
    
    # --- THE FINAL FIX IS HERE ---
    # miditok.decode expects a list of tracks (a list of lists of token ids).
    generated_score = tokenizer.decode([generated_ids])
    # -----------------------------

    output_midi_path = OUTPUT_DIR / OUTPUT_MIDI_NAME
    generated_score.dump_midi(output_midi_path)
    print(f"Generated MIDI file saved to: {output_midi_path}")

    # This part converts the MIDI to a .wav file if you have a soundfont.
    if os.path.exists(SOUNDFONT_PATH):
        try:
            # You may need to install fluidsynth: sudo apt-get install fluidsynth
            os.system(f"fluidsynth -ni {SOUNDFONT_PATH} {output_midi_path} -F {output_audio_path} -r 44100")
            print(f"Synthesized audio saved to: {output_audio_path}")
        except Exception as e:
            print(f"Could not synthesize audio. Is 'fluidsynth' installed? Error: {e}")
    else:
        print(f"\n--- WARNING: Soundfont not found at '{SOUNDFONT_PATH}'. Audio file (.wav) was not created. ---")
        print("--- You can play the .mid file in a music player like MuseScore or a DAW. ---")

if __name__ == "__main__":
    generate_music()