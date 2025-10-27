import os
import re
import json
import torch
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

# Configuration
BASE_DIR = os.path.dirname(__file__)                          
CLEANED_DIR = BASE_DIR                                       
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../..", "llama3_8b_jobextractor_sft"))  
RAW_COLUMN = "raw_output"

tqdm.pandas(desc="Regenerating missing rows")

# Load model 
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=4096,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()


# Regeneration and Extraction
def fix_json_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return (
        s.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
         .replace("\u200b", "")
         .strip()
    )


def extract_last_json(text: str):
    if not isinstance(text, str):
        return None
    matches = re.findall(r"<json>(.*?)</json>", text, re.DOTALL)
    if not matches:
        return None
    last = fix_json_text(matches[-1])
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def generate_structured_output(description: str):
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
            temperature=0.2,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def regenerate_missing_rows():
    cleaned_files = sorted([f for f in os.listdir(CLEANED_DIR) if f.endswith("_clean.csv")])
    for fname in cleaned_files:
        fpath = os.path.join(CLEANED_DIR, fname)
        df = pd.read_csv(fpath)

        if "parsed_json" not in df.columns or "job_description" not in df.columns:
            continue

        # find missing or None parsed_json
        mask = df["parsed_json"].isnull() | df["parsed_json"].eq("None")
        if not mask.any():
            continue

        print(f"Processing {fname} ({mask.sum()} rows to regenerate)")

        # regenerate only missing ones
        df.loc[mask, RAW_COLUMN] = df.loc[mask, "job_description"].progress_apply(generate_structured_output)

        # re-extract JSONs
        df["parsed_json"] = df[RAW_COLUMN].apply(extract_last_json)
        df["parsed_json_str"] = df["parsed_json"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else None
        )

        # overwrite same CSV (no new file)
        df.to_csv(fpath, index=False)

    print("Regeneration complete — all cleaned CSVs updated in place.")


# Runing
if __name__ == "__main__":
    regenerate_missing_rows()
