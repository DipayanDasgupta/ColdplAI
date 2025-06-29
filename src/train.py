import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, Trainer,
    TrainingArguments
)
from miditok import REMI
from pathlib import Path
import json

# --- Configuration ---
TOKENIZER_PATH = Path("models/tokenizer.json")
TOKEN_FILE = Path("data/all_tokens.json") 
MODEL_OUTPUT_DIR = Path("models/coldplay_gpt2")

class MusicDataset(Dataset):
    """
    This is the definitive, robust Dataset class.
    It handles ALL padding and formatting internally, so the Trainer
    doesn't have to. This bypasses all API incompatibilities.
    """
    def __init__(self, file_path: Path, tokenizer: REMI, max_length=512):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer["PAD_None"]

        with open(file_path, 'r') as f:
            data = json.load(f)
        self.examples = data['tokens']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Get the sequence
        seq = self.examples[i]
        
        # Truncate if necessary
        seq = seq[:self.max_length]
        
        # Manually pad the sequence to max_length
        padding_to_add = self.max_length - len(seq)
        input_ids = seq + ([self.pad_token_id] * padding_to_add)
        
        # For language modeling, labels are the input_ids.
        # We set padded token labels to -100 so they are ignored in the loss calculation.
        labels = list(input_ids)
        for j, token in enumerate(labels):
            if token == self.pad_token_id:
                labels[j] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def train_model():
    print("Starting model training...")
    tokenizer = REMI(params=TOKENIZER_PATH)
    
    # Pass the tokenizer to the dataset so it knows the pad token ID
    dataset = MusicDataset(TOKEN_FILE, tokenizer=tokenizer, max_length=512)

    if len(dataset) < 1:
        raise ValueError("Dataset is empty. Check preprocessing.")
    print(f"Dataset loaded with {len(dataset)} sequences.")

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        pad_token_id=tokenizer["PAD_None"],
        eos_token_id=tokenizer["EOS_None"]
    )
    model = GPT2LMHeadModel(config)
    
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # We provide NO data_collator, so the Trainer uses its default,
        # which is exactly what we want now that our data is pre-padded.
        data_collator=None,
    )
    print("Training is starting... This will run to completion.")
    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    print(f"Model training complete. Model saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()