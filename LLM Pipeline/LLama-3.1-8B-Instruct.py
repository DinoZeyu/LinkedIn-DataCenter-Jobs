import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
from dotenv import load_dotenv
import os
import tqdm

# Load huggingface token
load_dotenv()  
hf_token = os.getenv("HF_TOKEN")

# Load data
df = pd.read_csv("linkedin_data_center_jobs.csv")


# Load Model 
Model_ID = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(Model_ID, 
                                          use_fast=True,
                                          trust_remote_code=True)
                                          
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    Model_ID,
    dtype=torch.float16,   
    device_map="auto"
)
model.eval()


# Generation Function
def extract_skills_batched(df, batch_size=8, max_new_tokens=100):
    all_skills = []

    for i in tqdm.tqdm(range(0, len(df), batch_size), desc="Extracting Skills"):
        batch = df.iloc[i:i+batch_size]["job_description"].tolist()

        # Build prompts using chat template
        prompts = []
        for job_desc in batch:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that extracts **precise, non-redundant technical skills** "
                                              "from job descriptions. Keep distinct fields separate, "
                                              "but do NOT include duplicates or minor rewordings — keep only the most specific terms. "
                                              "Avoid repeating equivalent terms. "
                                              "Return a concise, comma-separated list of unique technical skills. Do not explain."},
                {"role": "user", "content": job_desc},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Tokenize and run generation
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and clean outputs
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for out in decoded:
            last_line = out.strip().split("\n")[-1]
            skills = [s.strip() for s in last_line.split(",") if s.strip()]
            all_skills.append(", ".join(sorted(set(skills))))

    return all_skills

df["Technical Skills Extracted from LLaMA"] = extract_skills_batched(df, batch_size=5)
df.to_csv("llama_extracted_skills.csv", index=False)