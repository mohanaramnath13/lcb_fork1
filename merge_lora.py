import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
BASE_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
# The folder where your LoRA adapters are currently saved (from the error log)
ADAPTER_PATH = "/home/kbs/LCB_setup2/lcb_fork1/finetuned-models/adapter-weights/deepseek-coder-1.3b-instruct-taco-I2O-2000x2-lora-adapter-weights"
# Where to save the final merged model
OUTPUT_PATH = "finetuned-models/taco/deepseek-coder-1.3b-instruct-taco-I2O-2000x2"

print(f"Loading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading LoRA adapters from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapters into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Done! You can now point LCB to the MERGED folder.")
