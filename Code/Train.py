import os
# MUST BE AT THE VERY TOP BEFORE ANY IMPORTS
os.environ["HF_HOME"] = "D:/Study/BTU University Subjects/Computing at Scale/Pytorch_Tutorial/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/Study/BTU University Subjects/Computing at Scale/Pytorch_Tutorial/huggingface/datasets"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. Load the dataset
print("Loading OpenHermes dataset...")
dataset = load_dataset("teknium/OpenHermes-2.5", split="train")

# 2. Configure 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Changed from torch.float16
)

# 3. Load Mistral 7B (Ungated, no token required)
model_name = "mistralai/Mistral-7B-v0.1"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Define LoRA Adapters (Let SFTTrainer apply them)
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# DELETED: model = get_peft_model(...)

# 5. Define Training Arguments
training_args = SFTConfig(
    output_dir="./mistral-chatbot-results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    max_steps=1000, 
    bf16=True,       # Changed from fp16=True
    max_grad_norm=0.3,
    warmup_steps=30, 
    lr_scheduler_type="cosine",
    max_length=512, 
)

# 6. Format the dataset manually and drop all problematic columns
print("Formatting dataset...")
def map_format(example):
    text = ""
    for msg in example['conversations']:
        role = "user" if msg['from'] == 'human' else "assistant"
        content = msg['value']
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return {"text": text}

# This applies the format and permanently deletes all original metadata columns 
# so they cannot crash the Arrow data loader.
dataset = dataset.map(map_format, remove_columns=dataset.column_names)

# 7. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer, 
    # Notice we completely removed formatting_func because the data is already formatted
)

# 8. Start Training
print("Starting training...")
trainer.train()

# 9. Save the LoRA weights
trainer.model.save_pretrained("my-mistral-chatbot")
tokenizer.save_pretrained("my-mistral-chatbot")
print("Training complete! Adapter saved to 'my-mistral-chatbot'.")