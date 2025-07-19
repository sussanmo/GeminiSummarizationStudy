import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# Load dataset
df = pd.read_csv("participant_summary_with_total.csv")

# Clean and map experience
experience_dict = {
    '100': 'High', '110': 'Low', '120': 'High', '130': 'High', '150': 'Low', '160': 'High',
    '170': 'Low', '180': 'High', '210': 'High', '220': 'High', '230': 'High', '240': 'High',
    '250': 'High', '260': 'Low', '270': 'Low', '280': 'Low', '290': 'High', '300': 'High',
    '310': 'High', '320': 'High', '330': 'High', '340': 'High', '350': 'High', '360': 'Low'
}
df["ParticipantID"] = df["ParticipantID"].str.strip()
df["ParticipantID_clean"] = df["ParticipantID"].str.extract(r"(\d+)")
df["Experience"] = df["ParticipantID_clean"].map(experience_dict)

# Filter missing values
df = df.dropna(subset=["Experience", "FixationDuration_mean", "FixationCount_mean", "SemanticCategory", "Phase"])

# Function to run ANOVA for one DV
def run_mixed_anova(df, dv_col):
    results = []
    for category in df["SemanticCategory"].unique():
        sub = df[df["SemanticCategory"] == category]
        try:
            aov = pg.mixed_anova(
                dv=dv_col,
                within='Phase',
                between='Experience',
                subject='ParticipantID',
                data=sub
            )
            aov["Category"] = category
            aov["DV"] = dv_col
            results.append(aov)
        except Exception as e:
            print(f"⚠️ Skipped category {category} for {dv_col} due to error: {e}")
    return pd.concat(results, ignore_index=True)

# Run ANOVA for both FixationDuration and FixationCount
aov_duration = run_mixed_anova(df, "FixationDuration_mean")
aov_count = run_mixed_anova(df, "FixationCount_mean")

# Combine and export
combined = pd.concat([aov_duration, aov_count], ignore_index=True)
combined["p-corrected"] = multipletests(combined["p-unc"], method="fdr_bh")[1]
combined.to_csv("anova_by_category_duration_and_count_c.csv", index=False)

print("✅ Done. Results saved to 'anova_by_category_duration_and_count_c.csv'")