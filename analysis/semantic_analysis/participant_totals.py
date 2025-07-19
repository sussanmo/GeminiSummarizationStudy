import os
import pandas as pd

# === CONFIG ===
PRE_DIR = 'analysis/combined_by_category'      # Replace with your actual path
POST_DIR = 'analysis/combined_by_category_post'    # Replace with your actual path
OUTPUT_FILE = 'participant_summary_with_total.csv'
# ==============

def extract_phase_and_participant(file_path, phase_label):
    parts = file_path.split(os.sep)
    participant = next((p for p in parts if 'participant' in p.lower()), 'Unknown' + file_path)
    return phase_label, participant

def process_file(file, phase_label):
    try:
        df = pd.read_excel(file, index_col=0)
        df = df.map(lambda x: pd.to_numeric(str(x).replace(',', ''), errors='coerce'))

        phase, participant = extract_phase_and_participant(file, phase_label)

        # Get fixation count and fixation duration
        fixation_count_row = df[df.index.str.lower().str.contains('fixation count')]
        fixation_duration_row = df[df.index.str.lower().str.contains('fixation duration')]

        rows = []

        for col in df.columns:
            if col.strip() == '' or col.lower().startswith('unnamed'):
                continue

            count = fixation_count_row.get(col, pd.Series([None])).values[0]
            duration = fixation_duration_row.get(col, pd.Series([None])).values[0]

            rows.append({
                'ParticipantID': participant,
                'Phase': phase,
                'SemanticCategory': col.strip(),
                'FixationCount': count,
                'FixationDuration': duration
            })
        return rows
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")
        return []

def gather_all_data(root_dir, phase_label):
    data = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.xlsx'):
                full_path = os.path.join(root, file)
                data.extend(process_file(full_path, phase_label))
    return data

# Gather pre and post data
pre_data = gather_all_data(PRE_DIR, 'Pre')
post_data = gather_all_data(POST_DIR, 'Post')

combined_data = pd.DataFrame(pre_data + post_data)

# Drop rows with missing data
combined_data.dropna(subset=['FixationCount', 'FixationDuration'], inplace=True)

# Convert columns to numeric
combined_data['FixationCount'] = pd.to_numeric(combined_data['FixationCount'], errors='coerce')
combined_data['FixationDuration'] = pd.to_numeric(combined_data['FixationDuration'], errors='coerce')

# === Per-Participant Summary ===
participant_summary = (
    combined_data.groupby(['ParticipantID', 'SemanticCategory', 'Phase'])
    .agg(
        FixationCount_mean=('FixationCount', 'mean'),
        FixationCount_std=('FixationCount', 'std'),
        FixationDuration_mean=('FixationDuration', 'mean'),
        FixationDuration_std=('FixationDuration', 'std')
    )
    .reset_index()
)

# === Total Averages (across all participants) ===
overall_summary = (
    combined_data.groupby(['SemanticCategory', 'Phase'])
    .agg(
        FixationCount_mean=('FixationCount', 'mean'),
        FixationCount_std=('FixationCount', 'std'),
        FixationDuration_mean=('FixationDuration', 'mean'),
        FixationDuration_std=('FixationDuration', 'std')
    )
    .reset_index()
)
overall_summary['ParticipantID'] = 'ALL'

# === Combine Summaries ===
final_summary = pd.concat([participant_summary, overall_summary], ignore_index=True)

# === Export to CSV ===
final_summary.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved full summary with participant and overall stats to {OUTPUT_FILE}")