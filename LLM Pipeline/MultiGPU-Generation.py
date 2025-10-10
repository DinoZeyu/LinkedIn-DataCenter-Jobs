import os
import json
import torch
import re
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from torch.multiprocessing import Process, set_start_method


def load_model(model_path: str, device_id: int):
    device = f"cuda:{device_id}"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=8192,
        load_in_4bit=True,
        dtype=torch.float16,
        device_map={"": device},
    )
    model = torch.compile(model)
    model.eval()
    return model, tokenizer, device


def extract_json_safe(text: str) -> dict:
    match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    candidate = match.group(1) if match else text
    
    candidate = candidate.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    candidate = candidate.replace("'", '"')  
    candidate = re.sub(r",\s*}", "}", candidate)  
    candidate = re.sub(r",\s*\]", "]", candidate)

    blocks = re.findall(r"\{.*?\}", candidate, re.DOTALL)

    for b in blocks[::-1]:  
        try:
            data = json.loads(b)
            if all(k in data for k in ["role", "domain", "core_skills", "soft_skills", "summary"]):
                return data
        except json.JSONDecodeError:
            continue
    return {k: None for k in ["role", "domain", "core_skills", "soft_skills", "summary"]}



def generate_batch(model, tokenizer, descriptions, device, batch_size=2):
    prompts = [
    f"""You are an expert job-analysis assistant.

Extract the following information **as JSON only**:
- role
- domain
- core_skills
- soft_skills
- summary

Return **nothing else** except a valid JSON object.
Wrap the JSON inside <json>...</json> tags.

### Job Description:
{desc}

### Output:
<json>{{"role": "", "domain": "", "core_skills": [], "soft_skills": [], "summary": ""}}</json>
"""
for desc in descriptions
]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=8192).to(device)

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


def process_chunk(rank, model_path, df_chunk, save_prefix, batch_size=2):
    model, tokenizer, device = load_model(model_path, rank)
    out_file = f"{save_prefix}_gpu{rank}.csv"

    for start in tqdm(range(0, len(df_chunk), batch_size), desc=f"GPU{rank}"):
        batch = df_chunk.iloc[start:start + batch_size]
        descs = [str(d)[:8000] for d in batch["job_description"]]
        outputs = generate_batch(model, tokenizer, descs, device, batch_size=batch_size)

        for j, text in enumerate(outputs):
            result = extract_json_safe(text)
            idx = batch.index[j]
            for key in ["role", "domain", "core_skills", "soft_skills", "summary"]:
                df_chunk.at[idx, key] = result.get(key)

        # checkpoint every 1000 rows
        if (start + batch_size) % 1000 == 0:
            df_chunk.to_csv(out_file, index=False)

    df_chunk.to_csv(out_file, index=False)
    print(f"[GPU {rank}] Finished chunk → {out_file}")


def split_dataframe(df, num_splits):
    idx = torch.arange(len(df))
    parts = torch.chunk(idx, num_splits)
    return [df.iloc[p.cpu().numpy()] for p in parts]


def merge_partials(prefix, num_gpus, base_df, output_csv):
    partials = [f"{prefix}_gpu{i}.csv" for i in range(num_gpus)]
    dfs = [pd.read_csv(f) for f in partials if os.path.exists(f)]
    merged = pd.concat([base_df.iloc[:1000]] + dfs, ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"✅ All chunks merged → {output_csv}")

    # optional: clean up
    for f in partials:
        if os.path.exists(f):
            os.remove(f)


def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    model_path = "llama3_8b_jobextractor_sft"
    input_csv = "annotated_jobs_1000.csv"
    output_csv = "annotated_jobs_full_multigpu.csv"
    save_prefix = "annotated_jobs_partial"

    df = pd.read_csv(input_csv)
    df_to_annotate = df.iloc[1000:].copy()

    num_gpus = torch.cuda.device_count()
    chunks = split_dataframe(df_to_annotate, num_gpus)

    processes = []
    for rank in range(num_gpus):
        p = Process(target=process_chunk,
                    args=(rank, model_path, chunks[rank], save_prefix, 8))  # batch=8
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_partials(save_prefix, num_gpus, df, output_csv)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
