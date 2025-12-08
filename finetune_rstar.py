import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig

# --- 1. CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "Qwen2.5-Coder-7B-rStar-Full"

# --- 2. LOAD DATASET ---
print("Loading microsoft/rStar-Coder (synthetic_sft)...")
dataset = load_dataset("microsoft/rStar-Coder", "synthetic_sft", split="train")

def format_rstar_to_chat(row):
    # --- A. Prepare User Prompt ---
    user_content = row['question']
    
    if row.get('starter_code') and len(str(row['starter_code'])) > 0:
        user_content += f"\n\nHere is the starter code:\n```python\n{row['starter_code']}\n```"

    # --- B. Prepare Assistant Response ---
    assistant_content = f"{row['response']}\n\nHere is the solution:\n```python\n{row['code']}\n```"
    
    messages = [
        {"role": "system", "content": "You are an expert coding assistant. Think through the problem step-by-step before writing the code."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    return {"messages": messages}

print("Formatting dataset...")
dataset = dataset.map(format_rstar_to_chat)

# --- 3. TRAINING ARGUMENTS (The "Bulletproof" Config) ---
# We initialize with only the safe arguments first
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="messages",
    
    # Standard training params
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none",
    dataloader_num_workers=0,
    dataset_num_proc=1,
)

# --- THE FIX: Manually inject the problematic arguments ---
# This bypasses the version check but SFTTrainer will still respect them
training_args.max_seq_length = 4096
training_args.packing = False

# --- 4. LOAD MODEL ---
print(f"Loading Model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# Safety net: explicitly tell tokenizer its limit
tokenizer.model_max_length = 4096 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,                # Must be False for gradient checkpointing
    attn_implementation="sdpa",     # Using SDPA
    torch_dtype=torch.bfloat16
)

# --- 5. INITIALIZE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    # Note: We do NOT pass max_seq_length here anymore
)

# --- 6. START TRAINING ---
print("Starting DeepSpeed training...")
trainer.train()

# --- 7. SAVE FINAL MODEL ---
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")
