import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

import pdb

# --- 1. CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "finetuned-models/adapter-weights/Qwen/Qwen2.5-Coder-7B-Instruct-taco-I2O-2000x2-lora-adapter-weights"

# --- 2. LOAD DATASET ---
print("Loading TACO dataset...")
#dataset = load_dataset("microsoft/rStar-Coder", "synthetic_sft", split="train[:20000]")
dataset = load_dataset("BAAI/TACO", split="train[:2000]")

def format_taco_to_chat(batch):
    new_messages = []
    try:
        for i, question, solutions, ip_op in zip(range(len(batch)), batch["question"], batch["solutions"], batch["input_output"]):
            try:
                _=question.encode("utf-8")
                _=solutions.encode("utf-8")
                _=ip_op.encode("utf-8")
            except UnicodeEncodeError:
                continue
            if "<image>" in question or "figure" in question:
                continue
            # inner loop for multiple solutions
            try:
                solutions = eval(solutions)
                ip_op = eval(ip_op)
            except:
                continue
            try:
                if len(ip_op["inputs"]) > 0:
                    sample_1 = "\nSample Input 1:\n" + ip_op["inputs"][0] + "\nSample Output 1:\n" + ip_op["outputs"][0]
                else:
                    sample_1 = ""
                if len(ip_op["inputs"]) > 1:
                    sample_2 = "\n\nSample Input 2:\n" + ip_op["inputs"][1] + "\nSample Output 2:\n" + ip_op["outputs"][1]
                else:
                    sample_2 = ""
            except:
                #pdb.set_trace()
                sample_1, sample_2 = "", ""
            for sol in solutions:
                assistant_content = f"```python\n{sol}\n```"
                messages = [
                    {"role": "system", "content": "You are a competitive code generator. You must output **ONLY** the required Python code in a single markdown block, with NO extra text, explanation, or comments."},
                    {"role": "user", "content": question + sample_1 + sample_2},
                    {"role": "assistant", "content": assistant_content}
                    ]
                new_messages.append(messages)
    except:
        pass
    
    return {"messages": new_messages}

print("Formatting dataset...")
dataset = dataset.map(format_taco_to_chat, batched=True, remove_columns=dataset.features, batch_size=1)
print("len_dataset:", len(dataset))

# --- 3. TRAINING ARGUMENTS (The "Bulletproof" Config) ---
# We initialize with only the safe arguments first
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="messages",
    
    # Standard training params
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
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

# --- 5. [NEW] CONFIGURE LoRA ---
peft_config = LoraConfig(
    r=16,                        # Rank
    lora_alpha=32,               # Alpha (usually 2x rank)
    lora_dropout=0.05,           # Dropout
    bias="none",                 # Bias setting
    task_type="CAUSAL_LM",       # Task type
    target_modules=[             # Target all linear layers for best performance
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ],
)

# --- 6. INITIALIZE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    # Note: We do NOT pass max_seq_length here anymore
    peft_config=peft_config,     # [NEW] Pass the LoRA config here
)

# --- 7. START TRAINING ---
print("Starting DeepSpeed training...")
trainer.train()

# --- 8. SAVE FINAL MODEL ---
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")
