import os
import pandas as pd
import openpyxl
import os
import pandas as pd
from difflib import get_close_matches

import difflib

def fuzzy_match_columns(ref_cols, target_cols, cutoff=0.6):
    mapping = {}
    for ref_col in ref_cols:
        ref_clean = clean_post_column(ref_col)
        matches = difflib.get_close_matches(ref_clean, [clean_post_column(tc) for tc in target_cols], n=1, cutoff=cutoff)
        if matches:
            matched_clean = matches[0]
            # Map to original name in target_cols
            for tc in target_cols:
                if clean_post_column(tc) == matched_clean:
                    mapping[ref_col] = tc
                    break
    return mapping
    
def clean_post_column(col):
    
    original_col = col
    col = str(col).lower()
    if 'post_' in col:
        cleaned = col.replace('post_', '').strip()
    elif '_post' in col:
        cleaned = col.replace('_post', '').strip()
    elif col.endswith('post'):
        cleaned = col.replace('post', '').strip()
    else:
        cleaned = col.strip()
    
    # print(f"Original: '{original_col}' → Cleaned: '{cleaned}'")
    return cleaned

def handle_outlier_participant(file_path, pre_dir, post_dir):
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)

    xls = pd.ExcelFile(file_path)
    participant_id = os.path.splitext(os.path.basename(file_path))[0]

    for sheet_name in xls.sheet_names:
        print(f"\n=== Processing sheet: {sheet_name} ===")
        df = xls.parse(sheet_name)
        df_total = df.iloc[:8].copy()
        df_post = df.iloc[11:].copy()

        # Set first row as column headers
        raw_columns = df_post.iloc[0]
        print(raw_columns)
        cleaned_columns = [clean_post_column(col) for col in raw_columns]
        # print("Original header row:", list(raw_columns))
        # print("Cleaned post columns:", cleaned_columns)

        df_post.columns = cleaned_columns
        df_post = df_post.iloc[1:].copy()  # Drop the header row

        # Save cleaned post df
        post_out_path = os.path.join(post_dir, f"{participant_id}_{sheet_name}_post.xlsx")
        df_post.to_excel(post_out_path, index=False)

        # Get common numeric columns
        # print(df_total.columns)
        # print(df_post.columns)
        match_map = fuzzy_match_columns(df_total.columns, df_post.columns)
        print("Fuzzy match mapping:")
        print(match_map)

        # Filter numeric columns from df_total that matched
        numeric_cols = [col for col in match_map if pd.api.types.is_numeric_dtype(df_total[col])]
        print("Numeric columns used for subtraction:", numeric_cols)

        df_total_numeric = df_total[numeric_cols]
        df_post_numeric = df_post[[match_map[col] for col in numeric_cols]].reindex(columns=[match_map[col] for col in numeric_cols], fill_value=0)

        df_pre = df_total_numeric.subtract(df_post_numeric.values, fill_value=0).clip(lower=0)

        pre_out_path = os.path.join(pre_dir, f"{participant_id}_{sheet_name}_pre.xlsx")
        df_pre.to_excel(pre_out_path, index=False)

        print(f"Sheet '{sheet_name}': saved pre → {pre_out_path}, post → {post_out_path}")
        
        # common_cols = df_total.columns.intersection(df_post.columns)
        # # print(common_cols.tolist())
        # numeric_cols = df_total[common_cols].select_dtypes(include='number').columns
        # print("Numeric columns used for subtraction:", numeric_cols.tolist())

        # df_total_numeric = df_total[numeric_cols]
        # df_post_numeric = df_post[numeric_cols].reindex(columns=numeric_cols, fill_value=0)

        # # Confirm dtypes
        # # print("dtypes of df_total_numeric:")
        # # print(df_total_numeric.dtypes)
        # # print("dtypes of df_post_numeric:")
        # # print(df_post_numeric.dtypes)

        # # Compute pre = total - post
        # df_pre = df_total_numeric.subtract(df_post_numeric, fill_value=0).clip(lower=0)

        # # Save result
        # pre_out_path = os.path.join(pre_dir, f"{participant_id}_{sheet_name}_pre.xlsx")
        # df_pre.to_excel(pre_out_path, index=False)

        # print(f"Sheet '{sheet_name}': saved pre → {pre_out_path}, post → {post_out_path}")

# create directories for pre v. post gemini data per participant/task
def create_directories(dir_input, dir_pre, dir_post):
    # create directories
    os.makedirs(dir_pre, exist_ok=True)
    os.makedirs(dir_post, exist_ok=True)
    taskCounter=0
    #iterate through extracted data
    for pid_file in os.listdir(dir_input):
       
       if pid_file.endswith('.xlsx'):
        # Extract participant ID from filename
        participant_id = pid_file.replace('.xlsx', '')

        os.makedirs(os.path.join(dir_pre, participant_id), exist_ok=True)
        os.makedirs(os.path.join(dir_post, participant_id), exist_ok=True)
        # df_pre.to_excel(os.path.join(dir_pre, "Participant01", "task1.xlsx"))   

        #for each file, copy contents above line break (pre/post gem) into respective outputdirectory
        taskCounter+=1 
        file_path = os.path.join(dir_input, pid_file)
        
        try:
            # excel_data = pd.read_excel(file_path, sheet_name=None)
            excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

            # excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

        except Exception as e:
            print(f" Error reading file {pid_file}: {e}")
            continue  # Skip to next file if error
    # in each sheet
        # print(excel_data)
        
        for sheet_name, data in excel_data.items(): 
            # print(f"Processing file #{taskCounter}: {pid_file} | Sheet: {sheet_name}")
            # locate the row containing the phrase 'post gemini' and split the data into pre and post sections. 
            split_index = None
            for idx, row in data.iterrows(): 
                row_strs = row.astype(str).str.lower()
                if row_strs.str.startswith('pre').any() or row_strs.str.startswith('post').any():
                    split_index = idx
                    break

            if split_index is None:
                print(f"No 'post gemini or pre gemini' marker found in {pid_file} | {sheet_name}")
                # Create empty DataFrames
                df_pre = pd.DataFrame()
                df_post = pd.DataFrame()

                out_path_pre = os.path.join(dir_pre, participant_id, f"{sheet_name}.xlsx")
                out_path_post = os.path.join(dir_post, participant_id, f"{sheet_name}.xlsx")

                # Save empty files
                df_pre.to_excel(out_path_pre, index=False)
                df_post.to_excel(out_path_post, index=False)
                continue 

            # Now safe to split:
            df_pre = data.iloc[:split_index]  # everything before the marker
            df_post = data.iloc[split_index + 1:]  # everything after

            out_path_pre = os.path.join(dir_pre, participant_id, f"{sheet_name}.xlsx")
            out_path_post = os.path.join(dir_post, participant_id, f"{sheet_name}.xlsx")

            df_pre.to_excel(out_path_pre, index=False)
            df_post.to_excel(out_path_post, index=False)

    #at end of file, check to see if dataframe_pre and datafram_post have successfully copied participant's data print (issue at file name)
    #move to next file
    #copy each dataframe to each respective directory
    print()


if __name__ == '__main__':
    # directory = 'GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    directory = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    create_directories(directory, 'pre_gemini_data', 'post_gemini_data')

    # total_dir = 'GeminiSummarizationStudy/analysis/participant_extractedmetrics/Participant100.xlsx'
    # pre_dir = 'pre_gemini_data/Participant100'
    # post_dir = 'post_gemini_data/Participant100'
    # # handle_outlier_participant(total_dir,pre_dir,post_dir)
    #/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/post_gemini_data/Participant100