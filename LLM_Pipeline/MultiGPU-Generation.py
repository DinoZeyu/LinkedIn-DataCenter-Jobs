import torch
import json
import os
import re
from tqdm import tqdm
import pandas as pd
from unsloth import FastLanguageModel

tqdm.pandas(desc="🔍 Extracting job info")

# Load fine-tuned model
model_path = "llama3_8b_jobextractor_sft"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length = 8192,
    dtype = torch.bfloat16,
    device_map = "auto",
)
model.eval()


# Define generation function
def generate_structured_output(description):
    prompt = f"""
You are an expert job-analysis assistant.

Extract the following information **as JSON only**:
- role
- domain
- core_skills
- soft_skills
- summary

Return **nothing else** except a valid JSON object.
Wrap the JSON inside <json>...</json> tags.

### Job Description:
{description}

### Output:
<json>{{"role": "", "domain": "", "core_skills": [], "soft_skills": [], "summary": ""}}</json>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.0,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def process_all_csvs(input_dir="Datasets", output_dir="Generated_Outputs", total_parts=20):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(14, total_parts + 1):  
        part_str = f"{i:02d}"
        input_file = os.path.join(input_dir, f"unannotated_jobs_part-{part_str}of{total_parts}_input.csv")
        output_file = os.path.join(output_dir, f"annotated_jobs_part-{part_str}of{total_parts}_output.csv")

        if not os.path.exists(input_file):
            print(f"❌ Skipping missing file: {input_file}")
            continue

        print(f"\n🚀 Processing Part {i}/{total_parts}: {input_file}")
        df = pd.read_csv(input_file)
        df["raw_output"] = df["job_description"].fillna("").progress_apply(generate_structured_output)
        df.to_csv(output_file, index=False)
        print(f"✅ Saved to: {output_file}")

if __name__ == "__main__":
    process_all_csvs()