import pandas as pd
import numpy as np

INPUT_CSV = "annotated_jobs_1000.csv"
SKIP_FIRST = 1000
NUM_PARTS = 20
PREFIX = "unannotated_jobs_part"

df = pd.read_csv(INPUT_CSV)

# Keep the ORIGINAL index (don't reset)
df_rem = df.iloc[SKIP_FIRST:]
n = len(df_rem)

if n == 0:
    print("Nothing to split.")
else:
    # Record original index into a column so it survives CSV round-trips
    df_rem = df_rem.copy()
    df_rem["orig_index"] = df_rem.index

    parts = min(NUM_PARTS, n)  # if fewer rows than parts, reduce parts
    # Split by position, not by index values
    pos = np.arange(n)
    pos_chunks = np.array_split(pos, parts)

    for i, pos_idx in enumerate(pos_chunks, 1):
        chunk = df_rem.iloc[pos_idx]
        # Helpful: show the true original index range in the filename printout
        idx_min, idx_max = int(chunk["orig_index"].min()), int(chunk["orig_index"].max())
        out = f"{PREFIX}-{i:02d}of{parts:02d}_input.csv"
        # Save WITHOUT writing the pandas index; we keep orig_index as a normal column
        chunk.to_csv(out, index=False)
        print(f"Saved {out} ({len(chunk)} rows)  [orig_index {idx_min}..{idx_max}]")

    print(f"Done. Total remaining rows: {n}, files: {parts}.")
