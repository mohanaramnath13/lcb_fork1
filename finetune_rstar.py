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
OUTPUT_DIR = "Qwen2.5-Coder-7B-rStar-Full"

# --- 2. LOAD DATASET ---
print("Loading microsoft/rStar-Coder dataset...")
dataset = load_dataset("microsoft/rStar-Coder", "synthetic_sft", split="train")
#dataset = load_dataset("microsoft/rStar-Coder", split="train")

def format_rstar_to_chat(row):
    # --- A. Prepare User Prompt ---
    # Start with the main question
    user_content = row['question']
    
    # If starter code exists, append it so the model knows where to start
    if row.get('starter_code') and len(str(row['starter_code'])) > 0:
        user_content += f"\n\nHere is the starter code:\n```python\n{row['starter_code']}\n```"

    # --- B. Prepare Assistant Response ---
    # CRITICAL STEP: Combine "Thinking" (response) with "Doing" (code)
    # This teaches the model to reason first, then output the solution.
    assistant_content = f"{row['response']}\n\nHere is the solution:\n```python\n{row['code']}\n```"
    
    messages = [
        {"role": "system", "content": "You are an expert coding assistant. Think through the problem step-by-step before writing the code."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    # SFTTrainer expects a 'messages' column
    return {"messages": messages}

print("Formatting dataset with correct columns...")
dataset = dataset.map(format_rstar_to_chat)

# --- 3. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,  # DeepSpeed will shard this
    gradient_accumulation_steps=4,  # Total batch size = 32
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,                      # Use A40's native bfloat16
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    gradient_checkpointing=True,    # Saves VRAM
    report_to="none",
    dataloader_num_workers=4,
)

# --- 4. LOAD MODEL ---
# Note: No device_map="auto" because DeepSpeed manages placement
print(f"Loading Model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16
)

# --- 5. INITIALIZE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="messages", 
    max_seq_length=4096, 
    packing=False,
)

# --- 6. START TRAINING ---
print("Starting DeepSpeed training...")
trainer.train()

# --- 7. SAVE FINAL MODEL ---
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")
