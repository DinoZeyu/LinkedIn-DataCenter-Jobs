import os
import time
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv() 
client = OpenAI(api_key=os.getenv("open_api_key"))


# We used GPT-4o model to get annotations from job description as labels for fine-tuning LLama3
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_annotation(text, model_name="gpt-4o-mini"):
    """
    Use GPT to extract structured info from a long job description.
    Returns a dict with parsed annotation (no seniority_level).
    """
    system_prompt = (
        "You are an expert annotator for LinkedIn job posts. "
        "Read the job description carefully and output a concise JSON object "
        "with the following keys only: role, domain, core_skills, soft_skills, summary. "
        "Avoid explanations; just output valid JSON."
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text[:12000]},
        ],
    )

    return json.loads(response.choices[0].message.content)

# Get the files for fine-tuning
def annotate_dataframe(input_csv, output_csv, finetune_jsonl, model="gpt-4o-mini", n_rows=1000):
    """
    Annotate first n_rows of a CSV with GPT labels and export both:
    1) annotated CSV
    2) instruction-tuning JSONL file
    """
    df = pd.read_csv(input_csv)
    annotations = []

    for idx in tqdm(range(min(n_rows, len(df)))):
        desc = df.loc[idx, "job_description"]
        if not isinstance(desc, str) or len(desc.strip()) == 0:
            continue
        try:
            data = get_annotation(desc, model)
            data["row_id"] = idx
            annotations.append(data)
        except Exception as e:
            print(f"Row {idx}: {e}")
            continue

    ann_df = pd.DataFrame(annotations)
    merged = df.join(ann_df.set_index("row_id"), how="left")

    # --- Save annotated CSV ---
    merged.to_csv(output_csv, index=False)
    print(f"Saved annotated CSV → {output_csv}")

    # --- Build fine-tuning JSONL ---
    jsonl_records = []
    for _, row in merged.head(len(ann_df)).iterrows():
        if pd.isna(row.get("role")):
            continue
        record = {
            "instruction": "Extract structured job information from this job description.",
            "input": row["job_description"],
            "output": json.dumps({
                "role": row.get("role"),
                "domain": row.get("domain"),
                "core_skills": row.get("core_skills"),
                "soft_skills": row.get("soft_skills"),
                "summary": row.get("summary")
            }, ensure_ascii=False)
        }
        jsonl_records.append(record)

    with open(finetune_jsonl, "w", encoding="utf-8") as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved fine-tuning JSONL → {finetune_jsonl}")
    return merged


# Acquire required datasets
if __name__ == "__main__":
    annotate_dataframe(
        input_csv="linkedin_data_center_jobs.csv",
        output_csv="annotated_jobs_1000.csv",
        finetune_jsonl="llama3_finetune_1000.jsonl",
        model="gpt-4o-mini",
        n_rows=1000
    )