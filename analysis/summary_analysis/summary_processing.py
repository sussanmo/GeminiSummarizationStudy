
# get ground truth summary from task_map json fil e

# create cleaned data of summaries from human eval (pre v. post)
# map back to participant file
import json
import pandas as pd
import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# nltk.download('punkt')

import re
import json
import pandas as pd

def parse_task_and_method(identifier: str):
    """
    Parses a string like '2.20', '1_2', '15-3' → ('2', '20') or ('1', '2')
    """
    parts = re.split(r'[._-]', str(identifier))
    if len(parts) == 2:
        task_num, method_num = parts
        return task_num, method_num
    return None, None

def get_method_index(method_number):
    """
    Maps any method number to 0, 1, or 2 using modulo logic.
    """
    try:
        num = int(method_number)
        mod = num % 3
        return mod - 1 if mod != 0 else 2
    except ValueError:
        return None

def get_method_description(task_map, task_num_str, method_num_str):
    """
    Retrieves the correct method description using task number and mod-3 logic on method number.
    """
    task_key = f"Task {task_num_str}"
    methods = task_map.get(task_key, [])
    index = get_method_index(method_num_str)
    if index is None or index >= len(methods):
        return None
    return methods[index].get('description', '').strip("'")

# Example usage on a DataFrame row:
def get_description_from_row_post(row, task_map, col_name):
    identifier = row[col_name]

    task_num, method_num = parse_task_and_method(identifier)
    if not task_num or not method_num:
        print(f"Row {row.name}: Missing task or method number from identifier '{identifier}'")
        return None

    description = get_method_description(task_map, task_num, method_num)
    if description is None:
        print(f"Row {row.name}: No description found for task {task_num}, method {method_num} (identifier '{identifier}')")
    return description

def extract_task_number(identifier):
    """Extract the task number from Gemini_Task or Task (e.g., 2.20 → 2)"""
    parts = re.split(r'[._-]', str(identifier))
    return parts[0] if parts else None

def parse_task_and_method(identifier):
    """
    Extracts task and method numbers as strings from identifiers like '20.2', '1_3', '7-1'
    Returns (task_num, method_num)
    """
    if not isinstance(identifier, str):
        identifier = str(identifier)
    match = re.match(r"^(\d+)[._-](\d+)$", identifier)
    if match:
        return match.group(1), match.group(2)
    else:
        return None, None
    
def get_description_from_row(row, task_map, cleaned_df):
    print(row)
    identifier = row['Gemini Task']
    
    try:
        gemini_part, method_part = str(identifier).strip().split('.')
        gemini_num = int(gemini_part)
        method_num = int(method_part)
    except Exception:
        print(f"❌ Could not parse identifier: {identifier}")
        return None

    # Find matching row in cleaned_df
    matching_row = cleaned_df[cleaned_df['Gemini'] == gemini_num]
    if matching_row.empty:
        print(f"❌ No matching Gemini in cleaned_df for {gemini_num}")
        return None

    task_str = matching_row.iloc[0]['Tasks']  # e.g., "Task 3"
    task_number = extract_task_number(task_str)

    try:
        method_list = task_map[task_number]
        return method_list[method_num - 1]['description']
    except Exception:
        print(f"❌ Could not find Task {task_number}, Method {method_num} in task_map (from identifier '{identifier}')")
        return None
def extract_task_number(task_str):
    """
    Given a string like 'Task 3', return 'Task 3' or just '3' depending on use.
    """
    match = re.search(r'Task\s*(\d+)', str(task_str))
    if match:
        return f"Task {match.group(1)}"
    return None


# Example usage
if __name__ == '__main__':
    # Load the JSON
    # with open('task_map.json', 'r') as f:
    #     task_list = json.load(f)  # This is a list of dicts

    # # Flatten into one dict
    # task_map = {}
    # for entry in task_list:
    #     task_map.update(entry)

    
    # # Example to check if it works
    # pre_example = {'Gemini_Task': '3.30'}
    # post_example = {'Task': '6-1'}

    # desc_pre = get_description_from_row(pre_example, task_map, 'Gemini_Task')
    # desc_post = get_description_from_row(post_example, task_map, 'Task')

    # print("PRE description:", desc_pre)
    # print("POST description:", desc_post)
        
    # add ground truth to the pre/psot summary file for later eval
    with open('task_map.json', 'r') as f:
        task_map = json.load(f)
        if isinstance(task_map, list):
            flat = {}
            for entry in task_map:
                flat.update(entry)
            task_map = flat

    # # Load pre and post summary files
    # # post_df = pd.read_excel('analysis/summary_analysis/post-gemini_summaries.xlsx', dtype={'Task': str})
    # pre_df = pd.read_excel('analysis/summary_analysis/Pre-Gemini Summary Annotations.xlsx', dtype={'Gemini Task': str})

    # # print(pre_df.columns)
    # # print(post_df.columns)
    
    
    # # Then in your main processing loop or apply function:
    # participant_df = pd.read_excel('analysis/CleanedParticipantData.xlsx')  # or .xlsx if needed

    # pre_df['GroundTruth'] = pre_df.apply(
    # lambda row: get_description_from_row(row, task_map, participant_df), axis=1)

    # # Save interim results
    # pre_df.to_excel('analysis/summary_analysis/pre_gemini_with_groundtruth.xlsx', index=False)
    # # post_df.to_excel('analysis/summary_analysis/post_gemini_with_groundtruth.xlsx', index=False)

    # print("✅ Added ground-truth descriptions to both files.")

    # # Load all files
    pre_df = pd.read_excel('analysis/summary_analysis/pre_gemini_with_groundtruth.xlsx')
    # post_df = pd.read_excel('analysis/summary_analysis/post_gemini_with_groundtruth_sum.xlsx')
    participant_df = pd.read_excel('analysis/CleanedParticipantData.xlsx')  # or .xlsx if needed


    # ===== Mapping participant to summary 
    # Convert to string and split on '.', take the first part, then convert back to int or str
    pre_df['TaskSuffix'] = pre_df['Gemini Task'].astype(str).str.split('.').str[1]

    # Get Gemini number (the number before the dot)
    pre_df['GeminiNum'] = pre_df['Gemini Task'].astype(str).str.split('.').str[0]
    participant_df['GeminiNum'] = participant_df['Gemini'].astype(str).str.split('.').str[0]

    # Map GeminiNum to TaskID
    gemini_to_taskid = dict(zip(participant_df['GeminiNum'], participant_df['Tasks']))
    # pre_df['TaskID'] = pre_df['GeminiNum'].map(gemini_to_taskid)

    pre_df['TaskID'] = pre_df['GeminiNum'].map(gemini_to_taskid).astype(str).str.replace('Task', '')
    print(pre_df['TaskID'].astype(str))
    # Combine to final Task column like "7.3"
    pre_df['Task'] = pre_df['TaskID'].astype(str) + '.' + pre_df['TaskSuffix'].astype(str)
   
    print(pre_df['Task'])

    # Clean up
    pre_df.drop(columns=['TaskSuffix', 'GeminiNum', 'TaskID'], inplace=True)


    # print(pre_df[['Gemini Task', 'Participant']].head())
    

    # print("Unique Gemini_Task values in pre_df:")
    # print(pre_df['Gemini Task'].unique())

    # print("Unique Gemini values in participant_df:")
    # print(participant_df['Gemini'].unique())

    # print("Unique Task values in post_df:")
    # print(post_df['Task'].unique())

    # print("Unique Tasks values in participant_df:")
    # print(participant_df['Tasks'].unique())

    
    # Save again
    # pre_df.to_excel('analysis/summary_analysis/pre_gemini_with_groundtruth.xlsx', index=False)
    # post_df.to_excel('analysis/summary_analysis/post_gemini_with_groundtruth.xlsx', index=False)
    
    # print(pre_df['Participant'])
    # print(pre_df[['Gemini Task', 'Participant', 'Task']].head())

    # pre_df = pd.read_excel('analysis/summary_analysis/pre_gemini_with_groundtruth.xlsx')
    # print(pre_df['Participant'])
    # print("✅ Participant IDs added to both pre and post summary files.")