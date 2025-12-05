import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

# --- 1. CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "Finetuned_model/Qwen2.5-Coder-7B-rStar-Full"  # Standard flat path

# --- 2. LOAD DATASET ---
# We use the rStar dataset for reasoning
print("Loading microsoft/rStar-Coder dataset...")
dataset = load_dataset("microsoft/rStar-Coder", split="train")

def format_rstar_to_chat(row):
    # Convert rStar 'problem'/'solution' to standard Chat format
    messages = [
        {"role": "system", "content": "You are an expert coding assistant that thinks step-by-step."},
        {"role": "user", "content": row['problem']},
        {"role": "assistant", "content": row['solution']}
    ]
    # SFTTrainer expects a column named 'messages' by default
    return {"messages": messages}

print("Formatting dataset...")
dataset = dataset.map(format_rstar_to_chat)

# --- 3. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,             # 1 epoch is usually enough for 7B on large datasets
    per_device_train_batch_size=4,  # DeepSpeed will shard this. 4 is safe for A40.
    gradient_accumulation_steps=4,  # Effective Batch Size = 4 * 4 * 2 GPUs = 32
    learning_rate=2e-5,             # Low LR for full fine-tuning (standard is 2e-5)
    weight_decay=0.01,
    bf16=True,                      # Use Ampere BF16 precision
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,                 # Save checkpoints every 100 steps
    save_total_limit=2,             # Only keep the last 2 checkpoints to save disk space
    gradient_checkpointing=True,    # CRITICAL: Saves VRAM by trading compute
    report_to="none",               # Turn off WandB/Tensorboard for now
    dataloader_num_workers=4,       # Speed up data loading
)

# --- 4. LOAD MODEL & TOKENIZER ---
# Note: With DeepSpeed ZeRO-3, we do NOT use device_map="auto"
print(f"Loading Model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,               # improved stability for training
    attn_implementation="flash_attention_2", # Use A40 specific speedup
    torch_dtype=torch.bfloat16
)

# --- 5. INITIALIZE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="messages", # SFTTrainer automatically handles chat templating for this column
    max_seq_length=4096,           # rStar has long reasoning chains
    packing=False,                 # Set to True if you want to pack sequences (faster, but complex)
)

# --- 6. START TRAINING ---
print("Starting DeepSpeed training...")
trainer.train()

# --- 7. SAVE FINAL MODEL ---
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")
