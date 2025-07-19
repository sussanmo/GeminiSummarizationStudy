import os
import pandas as pd

# ==== CONFIG ====
DATA_DIR = 'analysis/pre_gemini_data'  # Your folder with original Excel files
AOI_MAP_FILE = 'aoi_to_category_mapping_3.csv'  # Mapping of AOI → SemanticCategory
OUTPUT_FOLDER = 'analysis/combined_by_category'  # Where to save processed files
# =================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load AOI → Category mapping
aoi_map_df = pd.read_csv(AOI_MAP_FILE)
aoi_to_category = dict(zip(aoi_map_df['AOI_Label'].astype(str), aoi_map_df['SemanticCategory']))

# Process all Excel files
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.xlsx'):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_excel(filepath, engine='openpyxl')

                # Store the first column name (used for row names)
                first_col_name = df.columns[0]

                # Prepare new DataFrame for semantic category sums
                category_df = pd.DataFrame(index=df.index)

                for col in df.columns:
                    col_str = str(col).strip()
                    category = aoi_to_category.get(col_str)

                    if category:
                        # Convert to numeric, coerce strings to NaN, fill with 0
                        numeric_data = pd.to_numeric(df[col_str], errors='coerce').fillna(0)
                        if category not in category_df.columns:
                            category_df[category] = numeric_data
                        else:
                            category_df[category] += numeric_data

                # Combine the first original column + aggregated semantic categories
                final_df = pd.concat([df[[first_col_name]], category_df], axis=1)

                # Determine output path
                relative_path = os.path.relpath(filepath, DATA_DIR)
                output_path = os.path.join(OUTPUT_FOLDER, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save file
                final_df.to_excel(output_path, index=False)
                print(f"✅ Combined and saved: {output_path}")

            except Exception as e:
                print(f"❌ Failed to process {filepath}: {e}")