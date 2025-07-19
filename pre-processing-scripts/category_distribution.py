import json
import pandas as pd
from collections import defaultdict, Counter

json_file = "semantic_category (4).json"

# === LOAD DATA ===
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# === INITIALIZE ===
total_counts = Counter()
difficulty_counts = defaultdict(Counter)

for function in data["functions"]:
    difficulty = function.get("difficulty", "unspecified")

    for category, value in function.items():
        if category == "difficulty":
            continue

        if category == "method_declaration":
            if isinstance(value, str) and value.strip():
                total_counts["method_declaration"] += 1
                difficulty_counts[difficulty]["method_declaration"] += 1

        elif isinstance(value, list):
            count = len(value)
            total_counts[category] += count
            difficulty_counts[difficulty][category] += count

# Total counts
total_df = pd.DataFrame(total_counts.items(), columns=["Category", "Total Token Count"])
total_df = total_df.sort_values(by="Total Token Count", ascending=False)

# Per-difficulty breakdown
rows = []
for diff, cat_counts in difficulty_counts.items():
    for cat, count in cat_counts.items():
        rows.append({"Difficulty": diff, "Category": cat, "Token Count": count})
difficulty_df = pd.DataFrame(rows)
difficulty_df = difficulty_df.sort_values(by=["Difficulty", "Token Count"], ascending=False)

print("=== Total Token Counts ===")
print(total_df)

print("\n=== Token Counts by Difficulty ===")
print(difficulty_df)

total_df.to_csv("semantic_totals.csv", index=False)
difficulty_df.to_csv("semantic_by_difficulty.csv", index=False)
