import pandas as pd

input_csv = "annotated_jobs_1000.csv"
df = pd.read_csv(input_csv)

# Previous 1000 lines are already 
df_remaining = df.iloc[1000:].copy()
total = len(df_remaining)
print(f"🔹 Total remaining rows: {total}")

# Split the large dataset into 2 small datasets
half = total // 2
df_part1 = df_remaining.iloc[:half].copy()
df_part2 = df_remaining.iloc[half:].copy()

# Save new csvs for later usage
df_part1.to_csv("unannotated_jobs_part1_input.csv", index=False)
df_part2.to_csv("unannotated_jobs_part2_input.csv", index=False)

print(f" - Part1: unannotated_jobs_part1_input.csv ({len(df_part1)} rows)")
print(f" - Part2: unannotated_jobs_part2_input.csv ({len(df_part2)} rows)")
