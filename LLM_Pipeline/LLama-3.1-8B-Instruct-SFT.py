import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

load_dotenv()
hf_token = os.getenv("hf_token")


# Load Base Model
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length = 8192,       # Job descriptions are long (~7k tokens)
    load_in_4bit = True,
    dtype = torch.bfloat16,
    device_map = {"": 0},
    token = hf_token
)


# Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None
)


# Train Val Split & Preprocess
dataset = load_dataset("json", data_files="llama3_finetune_1000.jsonl")["train"]
dataset = dataset.train_test_split(test_size = 0.1, seed = 42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

def format_and_tokenize(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}

train_dataset = train_dataset.map(format_and_tokenize, num_proc=8)
val_dataset = val_dataset.map(format_and_tokenize, num_proc=8)


# Model Training 
training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,           
    learning_rate = 5e-5,           
    warmup_steps = 10,
    weight_decay = 0.02,
    lr_scheduler_type = "linear",
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    seed = 3407,
    output_dir = "outputs_refined",
    save_strategy = "steps",
    save_steps = 25,
    save_total_limit = 2,
    report_to = "none"
)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    dataset_num_proc = 2,
    packing = False,
    args = training_args
)

trainer.train()


# Save LoRA adapters 
save_dir = "llama3_8b_jobextractor_sft"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ Fine-tuning completed! LoRA adapters saved to '{save_dir}'")