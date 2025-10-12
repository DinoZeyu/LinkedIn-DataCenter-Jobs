import pandas as pd

# 1️⃣ Load the base file (first 1000 annotated rows)
base = pd.read_csv("annotated_jobs_1000.csv").iloc[:1000]

# 2️⃣ Load your two processed halves
part1 = pd.read_csv("annotated_jobs_part1_output.csv")
part2 = pd.read_csv("annotated_jobs_part2_output.csv")

# 3️⃣ Combine everything in the correct order
merged = pd.concat([base, part1, part2], ignore_index=True)

# 4️⃣ Optional sanity check
print("✅ Merge summary:")
print(f" - Base rows (first 1000): {len(base)}")
print(f" - Part1 rows: {len(part1)}")
print(f" - Part2 rows: {len(part2)}")
print(f" - Total combined: {len(merged)}")

# 5️⃣ Save to final file
merged.to_csv("annotated_jobs_full_multigpu.csv", index=False)
print("\n🎯 Final merged file saved → annotated_jobs_full_multigpu.csv")
