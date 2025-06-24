import pandas as pd
import os
from collections import defaultdict
import numpy as np

import pandas as pd
import os
import pandas as pd

import os
import pandas as pd

def extract_fixation_metrics_from_directory_tester(base_dir):
    for participant_id in os.listdir(base_dir):
        participant_path = os.path.join(base_dir, participant_id)
        if not os.path.isdir(participant_path):
            continue

        for filename in os.listdir(participant_path):
            if not filename.endswith('.xlsx'):
                continue

            file_path = os.path.join(participant_path, filename)
            print(f"\n📂 Reading: {file_path}")
            df = pd.read_excel(file_path, engine='openpyxl', header=None)

            # Look for the AOI row (likely row 1 or 2)
            aoi_row_idx = None
            for i in range(5):  # look at first 5 rows only
                row = df.iloc[i].astype(str).str.lower().tolist()
                if "code" in row and "gemini" in row:
                    aoi_row_idx = i
                    break

            if aoi_row_idx is None:
                print("⚠️ AOI row not found.")
                continue

            # Look for the fixation count and duration rows
            fixation_count_row = None
            fixation_dur_row = None

            for i in range(len(df)):
                cell = str(df.iloc[i, 0]).lower()
                if "fixation count" in cell:
                    fixation_count_row = i
                elif "fixation duration" in cell:
                    fixation_dur_row = i

            if fixation_count_row is None or fixation_dur_row is None:
                print("⚠️ Fixation rows not found.")
                continue

            column_names = df.iloc[aoi_row_idx]
            fixation_count = df.iloc[fixation_count_row][1:]
            fixation_duration = df.iloc[fixation_dur_row][1:]

            fixation_count.index = column_names[1:]
            fixation_duration.index = column_names[1:]

            print("\n🔢 Fixation Count:")
            print(fixation_count)

            print("\n⏱ Fixation Duration:")
            print(fixation_duration)

            return  # ⛔ Just test one file for now


import os
import pandas as pd
from scipy.stats import zscore

def categorize_columns(columns):
    code_keywords = ['code']
    gemini_keywords = ['gemini']
    summary_keywords = ['summary']  # Only columns that exactly match or contain "summary"
    
    code, gemini, summary, other = [], [], [], []

    for col in columns:
        col_str = str(col).strip().lower()
        
        if any(k == col_str or k in col_str for k in code_keywords):
            code.append(col)
        elif any(k == col_str or k in col_str for k in gemini_keywords):
            gemini.append(col)
        elif any(k == col_str or k in col_str for k in summary_keywords):
            summary.append(col)
        else:
            other.append(col)

    return {"code": code, "gemini": gemini, "summary": summary, "other": other}


def extract_fixation_metrics_from_directory(base_dir, condition_name):
    all_records = []

    for participant_id in os.listdir(base_dir):
        participant_path = os.path.join(base_dir, participant_id)
        
        if not os.path.isdir(participant_path):
            continue

        for filename in os.listdir(participant_path):
            if not filename.endswith('.xlsx'):
                continue

            file_path = os.path.join(participant_path, filename)
            # print(f"\n📂 Reading: {file_path}")
            df = pd.read_excel(file_path, engine='openpyxl', header=None)

            # Identify AOI row
            aoi_row_idx = None
            if not df.empty:
                for i in range(5):  # look at first few rows
                    row = df.iloc[i].astype(str).str.lower().tolist()
                    if "code" in row and "gemini" in row:
                        aoi_row_idx = i
                        break

                if aoi_row_idx is None:
                    continue

                # Identify fixation rows
                fixation_count_row = None
                fixation_dur_row = None
                for i in range(len(df)):
                    val = str(df.iloc[i, 0]).lower()
                    if "fixation count" in val:
                        fixation_count_row = i
                    elif "fixation duration" in val:
                        fixation_dur_row = i

                if fixation_count_row is None or fixation_dur_row is None:
                    continue

                # Pull column names from AOI row
                column_names = df.iloc[aoi_row_idx]
                fixation_count = df.iloc[fixation_count_row][1:]
                fixation_duration = df.iloc[fixation_dur_row][1:]

                fixation_count.index = column_names[1:]
                fixation_duration.index = column_names[1:]

                # Group columns
                grouped = categorize_columns(fixation_count.index)
                print(f"File: {filename} | AOI groups:")
                for group_name, cols in grouped.items():
                    print(f"  {group_name}: {cols}")

                # Get mean fixation stats per group
                def safe_mean(series, cols): 
                    values = pd.to_numeric(series[cols], errors='coerce').dropna()
                    return values.mean() if not values.empty else None

                record = {
                    "participant": participant_id,
                    "task": filename,
                    "condition": condition_name,
                    "fix_count_code": safe_mean(fixation_count, grouped["code"]),
                    "fix_count_gemini": safe_mean(fixation_count, grouped["gemini"]),
                    "fix_count_summary": safe_mean(fixation_count, grouped["summary"]),
                    "fix_count_other": safe_mean(fixation_count, grouped["other"]),
                    "fix_dur_code": safe_mean(fixation_duration, grouped["code"]),
                    "fix_dur_gemini": safe_mean(fixation_duration, grouped["gemini"]),
                    "fix_dur_summary": safe_mean(fixation_duration, grouped["summary"]),
                    "fix_dur_other": safe_mean(fixation_duration, grouped["other"]),
                }

                all_records.append(record)

    return pd.DataFrame(all_records)

if __name__ == '__main__':
    # directory = 'GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    # directory = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    # create_directories(directory, 'pre_gemini_data', 'post_gemini_data')

    # testing one file 
    # pre_gem_dir = "/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/pre_gemini_data"

    # extract_fixation_metrics_from_directory(pre_gem_dir)

    pre_gem_dir = "/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/pre_gemini_data"
    post_gem_dir = "/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/post_gemini_data"

    df_pre = extract_fixation_metrics_from_directory(pre_gem_dir, "pre")
    df_pre.to_csv('pre_gemini_basic_fixations.csv', index=False)  # save without row index
    print("Pre Gemini Data:")
    print(df_pre.head())   # prints first 5 rows, change or remove .head() to see more

    df_post = extract_fixation_metrics_from_directory(post_gem_dir, "post")
    df_post.to_csv('post_gemini_basic_fixations.csv', index=False)  # save without row index
    # print("Post Gemini Data:")
    print(df_post.head())

    # Combine
    df_all = pd.concat([df_pre, df_post], ignore_index=True)
    # print("Combined Data:")
    print(df_all.head())

    # # Normalize fixation metrics (z-score)
    # metrics_to_normalize = [col for col in df_all.columns if col.startswith("fix_")]
    # df_z = df_all.copy()
    # df_z[metrics_to_normalize] = df_all[metrics_to_normalize].apply(zscore)

    # # Save
    # output_path = "zscore_fixation_analysis.csv"
    # df_z.to_csv(output_path, index=False)
    # print(f"✅ Saved z-score normalized fixation analysis to {output_path}")

