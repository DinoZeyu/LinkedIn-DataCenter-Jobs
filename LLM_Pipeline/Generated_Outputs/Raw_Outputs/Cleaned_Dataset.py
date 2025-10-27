import os
import re
import json
import pandas as pd

# Configuration
TOTAL_PARTS = 20
RAW_COLUMN = "raw_output"               # column containing ### Output:
NTH_JSON = 2                            # pick the 2nd <json>...</json> block
RAW_DIR = os.path.dirname(__file__)
CLEANED_DIR = os.path.join(RAW_DIR, "..", "Cleaned_Outputs")
CLEANED_DIR = os.path.abspath(CLEANED_DIR) 

# Extraction Functions
def fix_json_text(s: str) -> str:
    """Clean common Unicode and quote issues before json.loads."""
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


def extract_nth_json_after_output(text: str, n: int = 2):
    """Extract the nth <json>...</json> block that appears after ### Output:"""
    if not isinstance(text, str) or not text.strip():
        return None

    match = re.search(r"###\s*Output:(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    section = match.group(1)

    json_blocks = re.findall(r"<json>(.*?)</json>", section, flags=re.DOTALL | re.IGNORECASE)
    if len(json_blocks) < n:
        return None

    candidate = fix_json_text(json_blocks[n - 1])
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}\s*$", candidate, flags=re.DOTALL)
        if brace_match:
            try:
                return json.loads(fix_json_text(brace_match.group(0)))
            except json.JSONDecodeError:
                return None
        return None


# Processing all files
def process_one_csv(part_idx: int, total_parts: int = TOTAL_PARTS):
    part_str = f"{part_idx:02d}"
    filename = f"annotated_jobs_part-{part_str}of{total_parts}_output.csv"
    input_path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(input_path):
        return  # silently skip missing files

    print(f"Processing {filename} ...") 

    df = pd.read_csv(input_path)
    if RAW_COLUMN not in df.columns:
        return

    # Extract JSONs
    # Invalid or missing <json> blocks will simply be recorded as None
    df["parsed_json"] = df[RAW_COLUMN].apply(lambda x: extract_nth_json_after_output(x, NTH_JSON))
    df["parsed_json_str"] = df["parsed_json"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else None
    )

    # Save to Cleaned_Outputs
    # Keep all rows (even those with None)
    output_csv = os.path.join(CLEANED_DIR, filename.replace("_output.csv", "_clean.csv"))
    df.to_csv(output_csv, index=False)


# Running
def process_all_csvs(total_parts: int = TOTAL_PARTS):
    for i in range(1, total_parts + 1):
        process_one_csv(i, total_parts)


if __name__ == "__main__":
    process_all_csvs()