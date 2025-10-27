import os
import pandas as pd
import json

# ======================================================
# CONFIGURATION
# ======================================================
BASE_DIR = os.path.dirname(__file__)       # current folder = Cleaned_Outputs
CLEANED_DIR = BASE_DIR                     # cleaned CSVs are here
TARGET_COLUMN = "parsed_json"              # column to check
DESC_COLUMN = "job_description"            # column to extract
OUTPUT_JSON = os.path.join(BASE_DIR, "all_missing_descriptions.json")


# ======================================================
# MAIN LOGIC
# ======================================================
def export_missing_descriptions():
    cleaned_files = sorted([f for f in os.listdir(CLEANED_DIR) if f.endswith("_clean.csv")])
    all_missing = []

    for fname in cleaned_files:
        fpath = os.path.join(CLEANED_DIR, fname)
        df = pd.read_csv(fpath)

        if TARGET_COLUMN not in df.columns or DESC_COLUMN not in df.columns:
            continue

        # find missing or None parsed_json
        mask = (
            df[TARGET_COLUMN].isnull()
            | df[TARGET_COLUMN].eq("None")
            | (df[TARGET_COLUMN].astype(str).str.strip() == "")
        )
        if not mask.any():
            continue

        missing_rows = df.loc[mask, [DESC_COLUMN]].copy()
        for i, row in missing_rows.iterrows():
            all_missing.append({
                "file": fname,
                "row_index": int(i),
                "job_description": str(row[DESC_COLUMN]).strip()
            })

    # save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_missing, f, ensure_ascii=False, indent=2)

    print(f"✅ Exported {len(all_missing)} missing descriptions to:")
    print(f"   {OUTPUT_JSON}")


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    export_missing_descriptions()
