import os
import json
import torch
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm


def load_model(model_path: str, device: str = "cuda"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=8192,
        load_in_4bit=True,
        dtype=torch.float16,  # faster on A100
        device_map={"": device},
    )
    model = torch.compile(model)
    model.eval()
    return model, tokenizer



def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        return {k: None for k in ["role", "domain", "core_skills", "soft_skills", "summary"]}
    try:
        return json.loads(text[start:end].strip())
    except Exception:
        return {k: None for k in ["role", "domain", "core_skills", "soft_skills", "summary"]}



def generate_batch(model, tokenizer, descriptions, device="cuda", batch_size=2):
    prompts = [
        f"""You are an expert job-analysis assistant.
Read the following job description and extract key structured information.

Return ONLY a valid JSON object with the following keys:
role, domain, core_skills, soft_skills, and summary.

### Job Description:
{desc}
""" for desc in descriptions
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            top_p=0.85,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)



def run_inference(model_path, input_csv, output_csv, batch_size=2):
    model, tokenizer = load_model(model_path)
    df = pd.read_csv(input_csv)

    df_to_annotate = df.iloc[1000:].copy()

    for start in tqdm(range(0, len(df_to_annotate), batch_size), desc="Annotating"):
        batch = df_to_annotate.iloc[start:start + batch_size]
        descs = [str(d)[:8000] for d in batch["job_description"]]

        outputs = generate_batch(model, tokenizer, descs, batch_size=batch_size)
        for j, text in enumerate(outputs):
            result = extract_json(text)
            idx = batch.index[j]
            for key in ["role", "domain", "core_skills", "soft_skills", "summary"]:
                df_to_annotate.at[idx, key] = result.get(key)

        if (start + batch_size) % 1000 == 0:
            partial = pd.concat([df.iloc[:1000], df_to_annotate], ignore_index=True)
            partial.to_csv(output_csv, index=False)

    merged = pd.concat([df.iloc[:1000], df_to_annotate], ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"Finished annotation. Final CSV saved as {output_csv}")



if __name__ == "__main__":
    torch.cuda.empty_cache()
    run_inference(
        model_path="llama3_8b_jobextractor_sft",
        input_csv="annotated_jobs_1000.csv",
        output_csv="annotated_jobs_full_fast.csv",
        batch_size=16,
    )
