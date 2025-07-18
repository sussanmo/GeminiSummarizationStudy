import os
import pandas as pd
import openpyxl
import os
import pandas as pd
from difflib import get_close_matches

import difflib
# create directories for pre v. post gemini data per participant/task
#mehtod assumes that pre/post are in each file 
def create_directories_initial(dir_input, dir_pre, dir_post, dir_total):
    # create directories
    os.makedirs(dir_pre, exist_ok=True)
    os.makedirs(dir_post, exist_ok=True)
    os.makedirs(dir_total, exist_ok=True)
    taskCounter=0
    #iterate through extracted data
    for pid_file in os.listdir(dir_input):
       
       if pid_file.endswith('.xlsx'):
        # Extract participant ID from filename
        participant_id = pid_file.replace('.xlsx', '')

        os.makedirs(os.path.join(dir_pre, participant_id), exist_ok=True)
        os.makedirs(os.path.join(dir_post, participant_id), exist_ok=True)
        os.makedirs(os.path.join(dir_total, participant_id), exist_ok=True)
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


def create_directories(dir_input, dir_pre, dir_post, dir_total):
    os.makedirs(dir_pre, exist_ok=True)
    os.makedirs(dir_post, exist_ok=True)
    os.makedirs(dir_total, exist_ok=True)

    for pid_file in os.listdir(dir_input):
        if not pid_file.endswith('.xlsx'):
            continue

        participant_id = pid_file.replace('.xlsx', '')
        print(f"\n🔍 Processing Participant: {participant_id}")

        os.makedirs(os.path.join(dir_pre, participant_id), exist_ok=True)
        os.makedirs(os.path.join(dir_post, participant_id), exist_ok=True)
        os.makedirs(os.path.join(dir_total, participant_id), exist_ok=True)

        file_path = os.path.join(dir_input, pid_file)

        try:
            excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        except Exception as e:
            print(f"❌ Error reading file {pid_file}: {e}")
            continue

        for sheet_name, data in excel_data.items():
            print(f"  📄 Sheet: {sheet_name}")
            marker_indices = []

            # Step 1: Find all rows where any cell starts with 'pre' or 'post'
            for idx, row in data.iterrows():
                row_strs = row.astype(str).str.lower()
                if row_strs.str.startswith("pre").any() or row_strs.str.startswith("post").any():
                    marker_indices.append(idx)

            # print(f"    ➤ Found markers at rows: {marker_indices}")

            if not marker_indices:
                print(f"    ⚠️ No 'pre' or 'post' markers found.")
                data.to_excel(os.path.join(dir_total, participant_id, f"{sheet_name}.xlsx"), index=False)
                pd.DataFrame().to_excel(os.path.join(dir_pre, participant_id, f"{sheet_name}.xlsx"), index=False)
                pd.DataFrame().to_excel(os.path.join(dir_post, participant_id, f"{sheet_name}.xlsx"), index=False)
                continue

            # Step 2: Define slices
            total_end = marker_indices[0]
            df_total = data.iloc[:total_end]
            # print(f"    ✅ Total section: rows 0 to {total_end - 1}")

            # Initialize
            df_pre = pd.DataFrame()
            df_post = pd.DataFrame()

            for i, marker_idx in enumerate(marker_indices):
                marker_row = data.iloc[marker_idx].astype(str).str.lower()
                label = None
                if marker_row.str.startswith("pre").any():
                    label = "pre"
                elif marker_row.str.startswith("post").any():
                    label = "post"
                else:
                    print(f"    ⚠️ Marker at row {marker_idx} not labeled correctly.")
                    continue

                start_idx = marker_idx + 1
                end_idx = marker_indices[i + 1] if i + 1 < len(marker_indices) else len(data)
                segment = data.iloc[start_idx:end_idx]

                # print(f"    ➤ Detected '{label}' segment: rows {start_idx} to {end_idx - 1} ({end_idx - start_idx} rows)")

                if label == "pre":
                    df_pre = segment
                elif label == "post":
                    df_post = segment

            # Save all
            df_total.to_excel(os.path.join(dir_total, participant_id, f"{sheet_name}.xlsx"), index=False, header=False)
            df_pre.to_excel(os.path.join(dir_pre, participant_id, f"{sheet_name}.xlsx"), index=False, header=False)
            df_post.to_excel(os.path.join(dir_post, participant_id, f"{sheet_name}.xlsx"), index=False, header=False)

            # Validation prints
            # print(f"    ✅ Saved total ({len(df_total)} rows), pre ({len(df_pre)} rows), post ({len(df_post)} rows)")

    print("\n🎉 All files processed.")



if __name__ == '__main__':
    # directory = 'GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    directory = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    create_directories(directory, 'pre_gemini_data', 'post_gemini_data', 'total_AOI_data')

    # total_dir = 'GeminiSummarizationStudy/analysis/participant_extractedmetrics/Participant100.xlsx'
    # pre_dir = 'pre_gemini_data/Participant100'
    # post_dir = 'post_gemini_data/Participant100'
    # # handle_outlier_participant(total_dir,pre_dir,post_dir)
    #/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/post_gemini_data/Participant100