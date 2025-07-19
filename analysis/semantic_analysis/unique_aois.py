import os
import pandas as pd

# ==== CONFIG ====
DATA_DIR = 'analysis/pre_gemini_data'  # <-- Replace with your actual folder
OUTPUT_AOI_FILE = 'unique_aois_from_columns_3.csv'
OUTPUT_UNNAMED_REPORT = 'unnamed_aoi_report.csv'
# =================

unique_aois = set()
unnamed_aois = []

# Recursively search all Excel files in subfolders
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.xlsx'):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_excel(filepath, engine='openpyxl')
                for col in df.columns:
                    col_str = str(col).strip()

                    # Skip 'Unnamed: 0'
                    if col_str.lower() == 'unnamed: 0':
                        continue

                    unique_aois.add(col_str)

                    # Log truly suspicious unnamed columns
                    if 'unnamed' in col_str.lower():
                        unnamed_aois.append({
                            'File': filepath,
                            'Column': col_str
                        })
            except Exception as e:
                print(f"❌ Error reading {filepath}: {e}")

# Save unique AOIs (excluding 'Unnamed: 0')
aoi_df = pd.DataFrame({'AOI_Label': sorted(unique_aois)})
aoi_df.to_csv(OUTPUT_AOI_FILE, index=False)

# Save unnamed column report (excluding 'Unnamed: 0')
if unnamed_aois:
    unnamed_df = pd.DataFrame(unnamed_aois)
    unnamed_df.to_csv(OUTPUT_UNNAMED_REPORT, index=False)
    print(f"⚠️ Found unnamed AOI columns in {len(unnamed_df['File'].unique())} files. Saved to {OUTPUT_UNNAMED_REPORT}")
else:
    print("✅ No suspicious unnamed AOI columns found.")

print(f"✅ Extracted {len(unique_aois)} unique AOI names to {OUTPUT_AOI_FILE}")