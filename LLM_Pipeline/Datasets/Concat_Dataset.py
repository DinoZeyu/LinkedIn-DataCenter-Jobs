import os
import json
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
CLEAN_DIR = os.path.join(PARENT_DIR, "Generated_Outputs", "Cleaned_Outputs")

MAIN_CSV = os.path.join(BASE_DIR, "annotated_jobs_1000.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "final_dataset.csv")

print("📂 MAIN_CSV:", MAIN_CSV)
print("📂 CLEAN_DIR:", CLEAN_DIR)

# Load and expand cleaned CSVs
cleaned_files = sorted([f for f in os.listdir(CLEAN_DIR) if f.endswith("_clean.csv")])
if not cleaned_files:
    raise FileNotFoundError(f"❌ No _clean.csv files found in {CLEAN_DIR}")

cleaned_dfs = []
for fname in cleaned_files:
    fpath = os.path.join(CLEAN_DIR, fname)
    df = pd.read_csv(fpath)
    if "orig_index" not in df.columns:
        raise ValueError(f"{fname} missing 'orig_index' column")

    # expand parsed_json
    parsed_cols = {"role": [], "domain": [], "core_skills": [], "soft_skills": [], "summary": []}
    if "parsed_json" in df.columns:
        for raw in df["parsed_json"]:
            try:
                data = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
            except json.JSONDecodeError:
                data = {}

            for key in parsed_cols:
                val = data.get(key)
                if isinstance(val, list):
                    val = ", ".join(map(str, val))
                parsed_cols[key].append(val)
    else:
        # fallback: if no parsed_json
        for key in parsed_cols:
            parsed_cols[key] = [None] * len(df)

    # attach expanded cols
    for key, vals in parsed_cols.items():
        df[key] = vals

    # keep only orig_index + expanded cols
    df = df[["orig_index", "role", "domain", "core_skills", "soft_skills", "summary"]]
    cleaned_dfs.append(df)


# Merge datasets
cleaned_all = pd.concat(cleaned_dfs, ignore_index=True)

main_df = pd.read_csv(MAIN_CSV)
main_df = main_df.reset_index().rename(columns={"index": "orig_index"})
print(f"📄 Loaded main file: {len(main_df)} rows")
merged = pd.merge(main_df, cleaned_all, on="orig_index", how="left", suffixes=("", "_cleaned"))



# Clean datasets
cols_before = len(merged.columns)

for col in list(merged.columns):
    if col.endswith("_x") and col[:-2] + "_y" in merged.columns:
        merged.drop(columns=[col], inplace=True)
        merged.rename(columns={col[:-2] + "_y": col[:-2]}, inplace=True)

merged = merged[[c for c in merged.columns if not c.endswith("_cleaned")]]
merged.drop(columns=["orig_index"], errors="ignore", inplace=True)

cols_after = len(merged.columns)
print(f"🧹 Removed {cols_before - cols_after} redundant columns.")


# Reorder columns
new_cols = ["role", "domain", "core_skills", "soft_skills", "summary"]
existing_cols = [c for c in merged.columns if c not in new_cols]
merged = merged[existing_cols + [c for c in new_cols if c in merged.columns]]

# Save final dataset
merged.to_csv(OUTPUT_CSV, index=False)
print(f"🎉 Final dataset saved to: {OUTPUT_CSV}")
print(f"📊 Total rows: {len(merged)}")
print(f"🧱 Columns: {merged.columns.tolist()}")