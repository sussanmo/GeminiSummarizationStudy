import pandas as pd
import os
from collections import defaultdict
import numpy as np

import pandas as pd
import os
import pandas as pd
import os
import pandas as pd
from scipy.stats import zscore
from scipy.stats import ttest_ind, ttest_rel, shapiro, levene,  mannwhitneyu, wilcoxon

from statsmodels.stats.power import TTestPower, TTestIndPower

def compute_sample_size(effect_size, alpha=0.05, power=0.8, test_type='paired'):
    """
    Calculate required sample size for t-tests using statsmodels.
    
    Parameters:
    - effect_size: Cohen's d (float)
    - alpha: significance level (float)
    - power: desired power (float)
    - test_type: 'paired' or 'independent' (str)
    
    Returns:
    - required sample size (int)
    """
    if test_type == 'paired':
        analysis = TTestPower()
    elif test_type == 'independent':
        analysis = TTestIndPower()
    else:
        raise ValueError("test_type must be 'paired' or 'independent'")
    
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
    return int(sample_size)

def categorize_columns(columns):
    code_keywords = ['code']
    gemini_keywords = ['gemini']
    summary_keywords = ['summary']

    code, gemini, summary, other = [], [], [], []

    for col in columns:
        if pd.isna(col):
            continue  # Skip NaN column headers
        col_str = str(col).strip().lower().replace(" ", "")
        if col_str == "":
            continue  # Skip blank headers
        if any(k in col_str for k in code_keywords):
            code.append(col)
        elif any(k in col_str for k in gemini_keywords):
            gemini.append(col)
        elif any(k in col_str for k in summary_keywords):
            summary.append(col)
        else:
            other.append(col)

    return {"code": code, "gemini": gemini, "summary": summary, "other": other}


def extract_fixation_metrics_from_directory_old(base_dir, condition_name):
    all_records = []

    for participant_id in os.listdir(base_dir):
        participant_path = os.path.join(base_dir, participant_id)
        
        if not os.path.isdir(participant_path):
            continue

        for filename in os.listdir(participant_path):
            if not filename.endswith('.xlsx'):
                continue
            print(f"Reading {filename} for participant {participant_id}")


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
                # print(f"File: {filename} | AOI groups:")
                # for group_name, cols in grouped.items():
                #     print(f"  {group_name}: {cols}")

                def safe_mean(series, cols, metric = ""): 
                    try:
                        # print(f"Metric printing: {metric}")
                        values = pd.to_numeric(series[cols], errors='coerce')
                        values = values.dropna()
                        # print(f"🔎 Averaging values for columns {list(cols)}:")
                        # print(values.tolist())  # print as list for readability
                        # print(f"Mean of values: {values.mean()}")
                        return values.mean() if not values.empty else None
                    except Exception as e:
                        print(f"⚠️ Error computing mean: {e}")
                        return None
                    
                def safe_sum(series, cols, metric=""):
                    try:
                        values = pd.to_numeric(series[cols], errors='coerce').dropna()
                        total = values.sum() if not values.empty else None
                        # print(f"\n📌 Metric: {metric}")
                        # print(f"🔎 Columns: {list(cols)}")
                        # print(f"📊 Values: {values.tolist()}")
                        # print(f"🔢 Total: {total}")
                        return total
                    except Exception as e:
                        print(f"⚠️ Error computing sum for {metric}: {e}")
                        return None
                record = {
                    "participant": participant_id,
                    "task": filename,
                    "condition": condition_name,
                    #total fixation counts 
                    "total_fix_count_code": safe_sum(fixation_count, grouped["code"], metric="Fixation count"),
                    "total_fix_count_gemini": safe_sum(fixation_count, grouped["gemini"], metric="Fixation count"),
                    "total_fix_count_summary": safe_sum(fixation_count, grouped["summary"], metric="Fixation count"),
                    "total_fix_count_tokens": safe_sum(fixation_count, grouped["other"], metric="Fixation count"),
                    #avg fixation counts 
                    "avg_fix_count_tokens": safe_mean(fixation_count, grouped["other"], metric="Fixation count"),
                    #avg fixation duration 
                    "avg_fix_dur_code": safe_mean(fixation_duration, grouped["code"], metric="Fixation duration"),
                    "avg_fix_dur_gemini": safe_mean(fixation_duration, grouped["gemini"], metric="Fixation duration"),
                    "avg_fix_dur_summary": safe_mean(fixation_duration, grouped["summary"], metric="Fixation duration"),
                    "avg_fix_dur_other": safe_mean(fixation_duration, grouped["other"], metric="Fixation duration"),
                }

                all_records.append(record)

    return pd.DataFrame(all_records)

def extract_fixation_metrics_from_directory(base_dir, condition_name):
    all_records = []
    skipped_files = []

    for participant_id in os.listdir(base_dir):
        participant_path = os.path.join(base_dir, participant_id)
        
        if not os.path.isdir(participant_path):
            continue

        for filename in os.listdir(participant_path):
            if not filename.endswith('.xlsx'):
                continue

            # print(f"📄 Reading {filename} for participant {participant_id}")
            file_path = os.path.join(participant_path, filename)

            try:
                df = pd.read_excel(file_path, engine='openpyxl', header=None)
            except Exception as e:
                print(f"❌ Failed to read {file_path}: {e}")
                skipped_files.append((file_path, "read error"))
                continue

            if df.empty:
                print(f"⚠️ Empty file: {file_path}")
                skipped_files.append((file_path, "empty"))
                continue

            # Identify AOI row
            aoi_row_idx = None
            for i in range(min(5, len(df))):
                row = df.iloc[i].astype(str).str.lower().tolist()
                if any("code" in cell for cell in row) and any("gemini" in cell for cell in row):
                    aoi_row_idx = i
                    break

            if aoi_row_idx is None:
                print(f"❌ AOI row not found in {file_path}")
                skipped_files.append((file_path, "missing AOI row"))
                continue

            # Identify fixation rows
            fixation_count_row = fixation_dur_row = None
            for i in range(len(df)):
                val = str(df.iloc[i, 0]).lower()
                if "fixation count" in val:
                    fixation_count_row = i
                elif "fixation duration" in val:
                    fixation_dur_row = i

            if fixation_count_row is None or fixation_dur_row is None:
                print(f"❌ Fixation rows missing in {file_path}")
                skipped_files.append((file_path, "missing fixation rows"))
                continue

            try:
                column_names = df.iloc[aoi_row_idx]
                fixation_count = df.iloc[fixation_count_row][1:]
                fixation_duration = df.iloc[fixation_dur_row][1:]

                fixation_count.index = column_names[1:]
                fixation_duration.index = column_names[1:]

                grouped = categorize_columns(fixation_count.index)

                def safe_mean(series, cols): 
                    values = pd.to_numeric(series[cols], errors='coerce').dropna()
                    return values.mean() if not values.empty else None

                def safe_sum(series, cols):
                    values = pd.to_numeric(series[cols], errors='coerce').dropna()
                    return values.sum() if not values.empty else None

                record = {
                    "participant": participant_id,
                    "task": filename,
                    "condition": condition_name,
                    "total_fix_count_code": safe_sum(fixation_count, grouped["code"]),
                    "total_fix_count_gemini": safe_sum(fixation_count, grouped["gemini"]),
                    "total_fix_count_summary": safe_sum(fixation_count, grouped["summary"]),
                    "total_fix_count_tokens": safe_sum(fixation_count, grouped["other"]),
                    "avg_fix_count_tokens": safe_mean(fixation_count, grouped["other"]),
                    "avg_fix_dur_code": safe_mean(fixation_duration, grouped["code"]),
                    "avg_fix_dur_gemini": safe_mean(fixation_duration, grouped["gemini"]),
                    "avg_fix_dur_summary": safe_mean(fixation_duration, grouped["summary"]),
                    "avg_fix_dur_other": safe_mean(fixation_duration, grouped["other"]),
                }

                all_records.append(record)

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                skipped_files.append((file_path, str(e)))

    df_all = pd.DataFrame(all_records)
    print(f"\n✅ Successfully processed: {len(df_all)} files")
    print(f"❌ Skipped: {len(skipped_files)} files")
    return df_all



# basic descriptives
# avg pre gemini: code/summary - count and duration 
# avg post gemini: code/summary/gemini - count and duration 
# total fixation count and duration of all participants for basic stats? 
import pandas as pd

def get_avg_from_file(file_path, column):
   
    df = pd.read_csv(file_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in file.")

     # Compute stats
    mean_val = df[column].dropna().mean()
    std_val = df[column].dropna().std()
   
    print(f"Mean of {column}: {round(mean_val, 2)}")
    print(f"Std of {column}: {round(std_val, 2)}")

    return round(mean_val, 2), round(std_val, 2)

# get basic descriptives of participants
def descriptives (pidFile):
    try:
        excel_data = pd.read_excel(pidFile, engine='openpyxl')
    except Exception as e:
        print(f"❌ Error reading file {pidFile}: {e}")
        return
    
    print( excel_data.iloc[1:])
    print( excel_data.columns.to_list())
    #get avg age, number of experience gorup
    
    excel_data['Q9'] = pd.to_numeric(excel_data['Q9'], errors='coerce')
    age_mean = excel_data['Q9'].dropna().mean()
    print(f"Mean age: {excel_data['Q9'].describe()}")
    excel_data['Q9'] = pd.to_numeric(excel_data['Q9'], errors='coerce')
    #Q2	Q3	Q4	Q5	Q6
    # What is your major?	What is your classification?	How many years of programming experience do you have?	How would you rate your familiarity with Python?	How familiar are you with AI or large language models (LLMs) like ChatGPT or Gemini?
    # excel_data['Q2'] = pd.to_numeric(excel_data['Q9'], errors='coerce')
   
    # Q2 What is your major?
    print(excel_data['Q2'].value_counts())

    # Q3 What is your classification?
    print(excel_data['Q3'].value_counts())

    # Q4 How many years of programming experience do you have?	
    print(excel_data['Q4'].value_counts())
    
    # Q5 How would you rate your familiarity with Python?	
    print(excel_data['Q5'].value_counts())
    
    # Q6 How familiar are you with AI or large language models (LLMs) like ChatGPT or Gemini?
    print(excel_data['Q6'].value_counts())




# run basic stats on participants: 
def check_normality(data):
    stat, p = shapiro(data)
    print(f"Normality Shapiro Stat: p - {p}, stat - {stat}")
    return p > 0.05  # True if normal

def check_equal_variance(data1, data2):
    stat, p = levene(data1, data2)
    print(f"Equal Variance Levene: p - {p}, stat - {stat}")

    return p > 0.05  # True if variances equal
# run t-test on code: pre v. post gemini 

def independ_test(pre_gem, post_gem, metric):
    df_pre = pd.read_csv(pre_gem)
    df_post = pd.read_csv(post_gem)

    pre_data = df_pre[metric].dropna()
    post_data = df_post[metric].dropna()

    print(f"Testing metric: {metric}")
    pre_normal = check_normality(pre_data)
    post_normal = check_normality(post_data)
    equal_var = check_equal_variance(pre_data, post_data)

    print("Pre normal?", pre_normal)
    print("Post normal?", post_normal)
    print("Equal variance?", equal_var)

    print(f"Testing differences between : {metric}")
    # Choose test based on assumptions
    if pre_normal and post_normal:
        # Use t-test
        if equal_var:
            t_stat, p_val = ttest_ind(pre_data, post_data, equal_var=True)
            print(f"Student's t-test (equal var): t={t_stat:.3f}, p={p_val:.3f}")
        else:
            t_stat, p_val = ttest_ind(pre_data, post_data, equal_var=False)  # Welch's t-test
            print(f"Welch's t-test (unequal var): t={t_stat:.3f}, p={p_val:.3f}")
    else:
        # Non-parametric Mann-Whitney U test
        u_stat, p_val = mannwhitneyu(pre_data, post_data, alternative='two-sided')
        print(f"Mann-Whitney U test (non-parametric): U={u_stat:.3f}, p={p_val:.3f}")

#fixation duration 
#fixation count
# run test on experience v. code fixation count/duration group 
        
import pandas as pd

def merge_experience_with_fixation_(fixation_file, experience_file, experience_column='Q4'):
    """
    Merges fixation data and experience data on participant ID.

    Args:
        fixation_file (str): Path to CSV with fixation data (e.g., has 'participant' like 'participant 290,Sheet2.xlsx')
        experience_file (str): Path to CSV or Excel with clean participant numbers and experience columns.
        experience_column (str): The column in experience data containing experience levels (default = 'Q4').

    Returns:
        pd.DataFrame: Merged DataFrame ready for analysis.
    """

    # Load files
    df_fix = pd.read_csv(fixation_file)
    df_exp = pd.read_excel(experience_file, engine='openpyxl')  # or pd.read_csv if it's CSV

    # Extract numeric participant IDs from fixation file
    df_fix['participant'] = df_fix['participant'].astype(str).str.extract(r'(\d+)').astype(float)

    # Make sure experience participant column is float or str for matching
    if 'participant' not in df_exp.columns:
        raise ValueError("Experience file must have a 'participant' column with numeric IDs.")

    df_exp['participant'] = df_exp['participant'].astype(float)

    # Merge on participant ID
    df_merged = pd.merge(df_fix, df_exp[['participant', experience_column]], on='participant', how='inner')

    return df_merged

def rank_biserial_correlation(u, n1, n2):
    """Calculate rank biserial correlation from Mann-Whitney U test."""
    return 1 - (2 * u) / (n1 * n2)

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def experience_group_test(df, metric, experience_col='Q4'):
    """
    Compare fixation metrics between Low and High experience groups (aggregated per participant).

    Args:
        df (pd.DataFrame): The dataframe containing multiple rows per participant (e.g., per task).
        metric (str): The name of the metric column (e.g., 'fixation_count_code').
        experience_col (str): The name of the experience column (e.g., 'Q4').

    Prints:
        Group means, normality tests, variance test, and appropriate statistical test result.
    """
    # Define group mapping
    experience_mapping = {
        'Less than 1 year': 'Low',
        '1-2 years': 'Low',
        '2-4 years': 'High',
        '4+ years': 'High'
    }

    # Drop rows without participant ID or metric
    df = df.dropna(subset=['participant', metric])

    # Aggregate per participant
    per_participant = df.groupby('participant').agg({
        metric: 'mean',
        experience_col: 'first'  # Assuming experience doesn't change across tasks
    }).reset_index()

    # Map experience group
    per_participant['experience_group'] = per_participant[experience_col].map(experience_mapping)
    per_participant = per_participant[per_participant['experience_group'].notna()]

    low_data = per_participant[per_participant['experience_group'] == 'Low'][metric].dropna()
    high_data = per_participant[per_participant['experience_group'] == 'High'][metric].dropna()

    print(f"\nTesting experience effect on: {metric}")
    print(f"Low experience (n={len(low_data)}), High experience (n={len(high_data)})")
    print(f"Mean {metric} - Low experience: {low_data.mean():.3f}")
    print(f"Mean {metric} - High experience: {high_data.mean():.3f}")

    # Normality and variance checks
    low_normal = check_normality(low_data)
    high_normal = check_normality(high_data)
    equal_var = check_equal_variance(low_data, high_data)

    n_low = len(low_data)
    n_high = len(high_data)

    print("Low normality test?", low_normal)
    print("High normality test?", high_normal)
    print("Equal variance?", equal_var)

    # Choose appropriate test
    if low_normal and high_normal:
        if equal_var:
            t_stat, p_val = ttest_ind(low_data, high_data, equal_var=True)
            d = cohen_d(low_data, high_data)
            print(f"Independent's t-test (equal var): t={t_stat:.3f}, p={p_val:.3f}, Cohen's d={d:.3f}")

        else:
            t_stat, p_val = ttest_ind(low_data, high_data, equal_var=False)
            d = cohen_d(low_data, high_data)
            print(f"Welch's t-test (unequal var): t={t_stat:.3f}, p={p_val:.3f}, Cohen's d={d:.3f}")
    else:
        u_stat, p_val = mannwhitneyu(low_data, high_data, alternative='two-sided')
        rbc = rank_biserial_correlation(u_stat, n_low, n_high)
        print(f"Mann-Whitney U test: U={u_stat:.3f}, p={p_val:.3f}")
        print(f"Rank-biserial correlation: {rbc:.3f}")

def merge_experience_with_fixation(fixation_file, experience_file, condition_label, experience_column='Q4'):
    df_fix = pd.read_csv(fixation_file)
    df_exp = pd.read_excel(experience_file, engine='openpyxl')

    df_fix['participant_id'] = df_fix['participant'].astype(str).str.extract(r'(\d+)').astype(float)
    df_exp['participant_id'] = df_exp['participant'].astype(float)

    df_merged = pd.merge(df_fix, df_exp[['participant_id', experience_column]], on='participant_id', how='inner')
    df_merged['condition'] = condition_label  # pre or post
    return df_merged


def merge_difficulty_info(pre_post_df, experience_df):
    """
    Assigns task difficulty to pre_post_df rows based on the order of tasks listed per participant in experience_df.
    Assumes each participant has 6 rows (3 pre, 3 post) in pre_post_df.

    Returns:
        pre_post_df with a new 'difficulty' column.
    """
    difficulty_map = {}

    for _, row in experience_df.iterrows():
        pid = str(row['participant']).strip()
        difficulties = str(row['Task Difficulty']).split(',')  # e.g., 'hard, easy, medium'
        difficulties = [d.strip().lower() for d in difficulties]
        difficulty_map[pid] = difficulties

    pre_post_df = pre_post_df.copy()
    pre_post_df['difficulty'] = None

    # print(pre_post_df.columns)
    # Clean participant IDs in pre_post_df
    pre_post_df['participant_clean'] = pre_post_df['participant'].str.extract(r'(\d+)')
    # print(pre_post_df['participant_clean'])
    # Assign difficulty by order
    for pid, difficulties in difficulty_map.items():
        # print(pid)
        # print(pre_post_df['participant_clean'].dtype)
        # print(pre_post_df['participant_clean'].unique())

        # print(type(pid), pid)

        # Filter this participant
        pid_clean = str(float(pid)).split('.')[0]  # turns "320.0" → 320.0 (float) → "320" (string)
        participant_df = pre_post_df[pre_post_df['participant_clean'] == pid_clean]
        # print(participant_df)

        # print(participant_df)
        # print(participant_df[['participant', 'condition', 'difficulty', 'avg_fix_dur_code']])

        # Check for 3 pre and 3 post
        pre_df = participant_df[participant_df['condition'] == 'pre']
        post_df = participant_df[participant_df['condition'] == 'post']

        if len(pre_df) == 3 and len(post_df) == 3:
            # Assign difficulties to pre and post
            pre_post_df.loc[pre_df.index, 'difficulty'] = difficulties
            pre_post_df.loc[post_df.index, 'difficulty'] = difficulties
        else:
            print(f"Skipping participant {pid}: pre={len(pre_df)}, post={len(post_df)}")

    return pre_post_df.drop(columns=['participant_clean'])
def merge_method_info(pre_post_df, experience_df):
    """
    Assigns method names to pre_post_df rows by matching participants and assigning
    3 methods (from experience_df) to 3 pre and 3 post rows (same task, different condition).

    Assumes:
    - Each participant has 3 unique tasks, each shown in both 'pre' and 'post' conditions.
    - 'Methods' column in experience_df contains comma-separated method names.

    Returns:
        pre_post_df with a new 'method' column.
    """
    method_map = {}

    # Build a mapping: participant ID → list of 3 method names
    for _, row in experience_df.iterrows():
        pid = str(row['participant']).strip()
        methods = str(row['Methods']).split(',')
        methods = [m.strip() for m in methods]
        method_map[pid] = methods

    pre_post_df = pre_post_df.copy()
    pre_post_df['method'] = None

    # Normalize participant IDs
    pre_post_df['participant_clean'] = pre_post_df['participant'].astype(str).str.extract(r'(\d+)')[0]

    # Assign method names to each participant's rows
    for pid, methods in method_map.items():
        pid_clean = str(float(pid)).split('.')[0]

        participant_df = pre_post_df[pre_post_df['participant_clean'] == pid_clean]

        if len(participant_df) != 6:
            print(f"Skipping participant {pid}: expected 6 rows, got {len(participant_df)}")
            continue

        if len(methods) != 3:
            print(f"Skipping participant {pid}: expected 3 methods, got {len(methods)}")
            continue

        # Sort participant rows to ensure 3 pre come before 3 post
        participant_df = participant_df.sort_values(by='condition', key=lambda col: col.map({'pre': 0, 'post': 1}))

        # Assign each method to both pre and post condition
        pre_rows = participant_df[participant_df['condition'] == 'pre']
        post_rows = participant_df[participant_df['condition'] == 'post']

        if len(pre_rows) != 3 or len(post_rows) != 3:
            print(f"Skipping participant {pid}: expected 3 pre and 3 post rows, got {len(pre_rows)} pre and {len(post_rows)} post")
            continue

        # Match pre and post rows by index and assign the same method
        for method, pre_idx, post_idx in zip(methods, pre_rows.index, post_rows.index):
            pre_post_df.at[pre_idx, 'method'] = method
            pre_post_df.at[post_idx, 'method'] = method

    return pre_post_df.drop(columns=['participant_clean'])


from scipy.stats import ttest_rel

def run_paired_test_by_difficulty_old(df, metric):
    """
    Run paired t-tests for the given metric, comparing pre vs post across each difficulty level.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns: 'participant', 'condition', 'difficulty', and the metric.
        metric (str): The metric to test (e.g., 'avg_fix_dur_code').
    """
    from scipy.stats import ttest_rel

    for difficulty in df['difficulty'].dropna().unique():
        subset = df[df['difficulty'] == difficulty.lower()]
        pre = subset[subset['condition'] == 'pre']
        post = subset[subset['condition'] == 'post']

        # Get matched participants
        matched = set(pre['participant']) & set(post['participant'])

        pre_vals = []
        post_vals = []

        for pid in matched:
            pre_val = pre[pre['participant'] == pid][metric].values
            post_val = post[post['participant'] == pid][metric].values
            if len(pre_val) == 1 and len(post_val) == 1:
                pre_vals.append(pre_val[0])
                post_vals.append(post_val[0])

        if len(pre_vals) < 2:
            print(f"\nSkipping difficulty '{difficulty}' — not enough matched data.")
            continue

        t_stat, p_val = ttest_rel(pre_vals, post_vals)
        print(f"\nPaired t-test on '{metric}' for difficulty: {difficulty}")
        print(f"n={len(pre_vals)} | pre mean={np.mean(pre_vals):.2f}, post mean={np.mean(post_vals):.2f}")
        print(f"t = {t_stat:.3f}, p = {p_val:.3f}")

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

def run_paired_test_by_difficulty(df, metric):
    """
    Run paired tests for the given metric comparing pre vs post across each difficulty level,
    checking assumptions and choosing test accordingly.

    Args:
        df (pd.DataFrame): Must have columns 'participant', 'condition', 'difficulty', and metric.
        metric (str): The metric to test (e.g., 'avg_fix_dur_code').
    """
    for difficulty in df['difficulty'].dropna().unique():
        print(f"\n🟩 {difficulty.capitalize()}:")
        
        subset = df[df['difficulty'].str.lower() == difficulty.lower()]
        pre = subset[subset['condition'] == 'pre']
        post = subset[subset['condition'] == 'post']

        matched = set(pre['participant']) & set(post['participant'])

        pre_vals = []
        post_vals = []
        dropped_due_to_nan = []


        for pid in matched:
            pre_val = pre[pre['participant'] == pid][metric].values
            post_val = post[post['participant'] == pid][metric].values

            if len(pre_val) == 1 and len(post_val) == 1:
                if not np.isnan(pre_val[0]) and not np.isnan(post_val[0]):
                    pre_vals.append(pre_val[0])
                    post_vals.append(post_val[0])
                else:
                    dropped_due_to_nan.append(pid)

        unmatched_pre = set(pre['participant']) - matched
        unmatched_post = set(post['participant']) - matched

        print(f"  • Matched: {len(matched)} participants")
        print(f"  • Used in test: {len(pre_vals)}")
        print(f"  • Dropped due to NaN: {sorted(dropped_due_to_nan)}")
        print(f"  • Unmatched (pre only): {sorted(unmatched_pre)}")
        print(f"  • Unmatched (post only): {sorted(unmatched_post)}")

        if len(pre_vals) < 2:
            print("  ⚠️ Skipping test — not enough valid matched pairs.")
            continue

        # Check normality of the difference scores
        diffs = np.array(pre_vals) - np.array(post_vals)
        shapiro_stat, shapiro_p = shapiro(diffs)
        print(f"  🔍 Shapiro-Wilk test on difference scores: W={shapiro_stat:.3f}, p={shapiro_p:.3f}")

        if shapiro_p > 0.05:
            # Normal difference → paired t-test
            t_stat, p_val = ttest_rel(pre_vals, post_vals)
            print(f"  📊 Paired t-test on '{metric}':")
            print(f"    n={len(pre_vals)} | pre mean={np.mean(pre_vals):.2f}, post mean={np.mean(post_vals):.2f}")
            print(f"    t = {t_stat:.3f}, p = {p_val:.3f}")
        else:
            # Non-normal difference → Wilcoxon signed-rank test
            w_stat, p_val = wilcoxon(pre_vals, post_vals)
            print(f"  📊 Wilcoxon signed-rank test on '{metric}':")
            print(f"    n={len(pre_vals)} | pre median={np.median(pre_vals):.2f}, post median={np.median(post_vals):.2f}")
            print(f"    W = {w_stat:.3f}, p = {p_val:.3f}")


def run_paired_test_by_method(df, metric):
    """
    Run paired tests for the given metric comparing pre vs post across each method,
    checking assumptions and choosing test accordingly.

    Args:
        df (pd.DataFrame): Must have columns 'participant', 'condition', 'method', and metric.
        metric (str): The metric to test (e.g., 'avg_fix_dur_code').
    """
    for method in df['method'].dropna().unique():
        print(f"\n🟩 Method: {method.strip()}")

        subset = df[df['method'].str.strip() == method.strip()]
        pre = subset[subset['condition'] == 'pre']
        post = subset[subset['condition'] == 'post']

        matched = set(pre['participant']) & set(post['participant'])

        pre_vals = []
        post_vals = []
        dropped_due_to_nan = []

        
        for pid in matched:
            pre_val = pre[pre['participant'] == pid][metric].values
            post_val = post[post['participant'] == pid][metric].values

            if len(pre_val) == 1 and len(post_val) == 1:
                if not np.isnan(pre_val[0]) and not np.isnan(post_val[0]):
                    pre_vals.append(pre_val[0])
                    post_vals.append(post_val[0])
                else:
                    dropped_due_to_nan.append(pid)

        unmatched_pre = set(pre['participant']) - matched
        unmatched_post = set(post['participant']) - matched

        print(f"  • Matched: {len(matched)} participants")
        print(f"  • Used in test: {len(pre_vals)}")
        print(f"  • Dropped due to NaN: {sorted(dropped_due_to_nan)}")
        print(f"  • Unmatched (pre only): {sorted(unmatched_pre)}")
        print(f"  • Unmatched (post only): {sorted(unmatched_post)}")

        if len(pre_vals) < 2:
            print("  ⚠️ Skipping test — not enough valid matched pairs.")
            continue

        # Check normality of the difference scores
        diffs = np.array(pre_vals) - np.array(post_vals)
        shapiro_stat, shapiro_p = shapiro(diffs)
        print(f"  🔍 Shapiro-Wilk test on difference scores: W={shapiro_stat:.3f}, p={shapiro_p:.3f}")

        if shapiro_p > 0.05:
            # Normal difference → paired t-test
            t_stat, p_val = ttest_rel(pre_vals, post_vals)
            print(f"  📊 Paired t-test on '{metric}':")
            print(f"    n={len(pre_vals)} | pre mean={np.mean(pre_vals):.2f}, post mean={np.mean(post_vals):.2f}")
            print(f"    t = {t_stat:.3f}, p = {p_val:.3f}")
        else:
            # Non-normal difference → Wilcoxon signed-rank test
            w_stat, p_val = wilcoxon(pre_vals, post_vals)
            print(f"  📊 Wilcoxon signed-rank test on '{metric}':")
            print(f"    n={len(pre_vals)} | pre median={np.median(pre_vals):.2f}, post median={np.median(post_vals):.2f}")
            print(f"    W = {w_stat:.3f}, p = {p_val:.3f}")

if __name__ == '__main__':
    # directory = 'GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    # directory = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/participant_extractedmetrics'
    # create_directories(directory, 'pre_gemini_data', 'post_gemini_data')

    # testing one file 
    # pre_gem_dir = "/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/pre_gemini_data"

    # extract_fixation_metrics_from_directory(pre_gem_dir)

    # pre_gem_dir = "/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/pre_gemini_data"
    # post_gem_dir = "/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/analysis/post_gemini_data"

    # df_pre = extract_fixation_metrics_from_directory(pre_gem_dir, "pre")
    # df_pre.to_csv('pre_gemini_basic_fixations.csv', index=False)  # save without row index
    # print("Pre Gemini Data:")
    # print(df_pre.head())   # prints first 5 rows, change or remove .head() to see more

    # df_post = extract_fixation_metrics_from_directory(post_gem_dir, "post")
    # df_post.to_csv('post_gemini_basic_fixations.csv', index=False)  # save without row index
    # # # print("Post Gemini Data:")
    # print(df_post.head())

    # Combine
    # df_all = pd.concat([df_pre, df_post], ignore_index=True)
    # print("Combined Data:")
    # print(df_all.head())

    # # Normalize fixation metrics (z-score)
    # metrics_to_normalize = [col for col in df_all.columns if col.startswith("fix_")]
    # df_z = df_all.copy()
    # df_z[metrics_to_normalize] = df_all[metrics_to_normalize].apply(zscore)
    # participant,task,condition,
    #total_fix_count_code,total_fix_count_gemini,total_fix_count_summary,total_fix_count_tokens,
    #avg_fix_count_tokens,
    #avg_fix_dur_code,avg_fix_dur_gemini,avg_fix_dur_summary,avg_fix_dur_other

    # # avg pre gemini: code/summary - count and duration 
    # print("Fixation counts for pre-gemini")
    # get_avg_from_file('pre_gemini_basic_fixations.csv', 'total_fix_count_code')
    # get_avg_from_file('pre_gemini_basic_fixations.csv', 'total_fix_count_summary')
    # get_avg_from_file('pre_gemini_basic_fixations.csv', 'total_fix_count_tokens')
    # # get_avg_from_file('pre_gemini_basic_fixations.csv', 'avg_fix_count_tokens')

    # print("Fixation durations for pre gemini")

    
    # get_avg_from_file('pre_gemini_basic_fixations.csv', 'avg_fix_dur_code')
    # get_avg_from_file('pre_gemini_basic_fixations.csv', 'avg_fix_dur_summary')
    # get_avg_from_file('pre_gemini_basic_fixations.csv', 'avg_fix_dur_other')
   

    # print("Fixation counts for post-gemini")

    # # avg post gemini: code/summary/gemini - count and duration 
    # get_avg_from_file('post_gemini_basic_fixations.csv', 'total_fix_count_summary')
    # get_avg_from_file('post_gemini_basic_fixations.csv', 'total_fix_count_code')
    # get_avg_from_file('post_gemini_basic_fixations.csv', 'total_fix_count_gemini')

    # get_avg_from_file('post_gemini_basic_fixations.csv', 'total_fix_count_tokens')

    # print("Fixation duration for post-gemini")

    # get_avg_from_file('post_gemini_basic_fixations.csv', 'avg_fix_dur_code')
    # get_avg_from_file('post_gemini_basic_fixations.csv', 'avg_fix_dur_summary')
    # get_avg_from_file('post_gemini_basic_fixations.csv', 'avg_fix_dur_other')
    # get_avg_from_file('post_gemini_basic_fixations.csv', 'avg_fix_dur_gemini')


    # independ_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_code')
    # independ_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_summary')

    # independ_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_code')

    # independ_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_code')
    # independ_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_summary')

    # independ_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_other')

    # descriptives('CleanedParticipantData.xlsx')

    
    # df_merged = merge_experience_with_fixation(
    # fixation_file='post_gemini_basic_fixations.csv',
    # experience_file='CleanedParticipantData.xlsx', 
    # )

    # print(df_merged[['participant', 'participant', 'Q4']].head())


    # # Now run group analysis
    # experience_group_test(df_merged, 'total_fix_count_code')

    # Merge pre and post separately
    df_pre = merge_experience_with_fixation(
        fixation_file='pre_gemini_basic_fixations.csv',
        experience_file='CleanedParticipantData.xlsx',
        condition_label='pre'
    )

    df_post = merge_experience_with_fixation(
        fixation_file='post_gemini_basic_fixations.csv',
        experience_file='CleanedParticipantData.xlsx',
        condition_label='post'
    )

    # Combine them
    df_all = pd.concat([df_pre, df_post], ignore_index=True)

    # # Optional: confirm merging worked
    # print(df_all[['participant', 'participant_id', 'Q4', 'condition']].head())

    # print('--------------------------------------------------------------------------------')
    # # Compare experience group impact on code fixations count in post-Gemini only
    # experience_group_test(df_all[df_all['condition'] == 'post'], 'total_fix_count_code')

    # # And for pre-Gemini
    # experience_group_test(df_all[df_all['condition'] == 'pre'], 'total_fix_count_code')

    # print('--------------------------------------------------------------------------------')

    # print("testing fixation duration now")
    # # Compare experience group impact on code fixations duration in post-Gemini only
    # experience_group_test(df_all[df_all['condition'] == 'post'], 'avg_fix_dur_code')

    # # And for pre-Gemini
    # experience_group_test(df_all[df_all['condition'] == 'pre'], 'avg_fix_dur_code')

    # print('--------------------------------------------------------------------------------')

    # print("testing fixation count of summary now")
    # experience_group_test(df_all[df_all['condition'] == 'post'], 'total_fix_count_summary')

    # # And for pre-Gemini
    # experience_group_test(df_all[df_all['condition'] == 'pre'], 'total_fix_count_summary')

    # print('--------------------------------------------------------------------------------')

    # print("testing fixation duration of summary now")

    # experience_group_test(df_all[df_all['condition'] == 'post'], 'avg_fix_dur_summary')

    # And for pre-Gemini
    # experience_group_test(df_all[df_all['condition'] == 'pre'], 'avg_fix_dur_summary')
    experience_df = pd.read_excel('CleanedParticipantData.xlsx')

    # merged = merge_difficulty_info(df_all, experience_df)

    # print("PAIRED T_TEST for avg_fix_dur_code")
    # run_paired_test_by_difficulty(merged, metric='avg_fix_dur_code')

    # print("PAIRED T_TEST for total_fix_count_code")
    # run_paired_test_by_difficulty(merged, metric='total_fix_count_code')

    # print("PAIRED T_TEST for total_fix_count_summary")
    # run_paired_test_by_difficulty(merged, metric='total_fix_count_summary')

    # run_paired_test_by_difficulty(merged, metric='avg_fix_dur_summary')

    # by method: 
    # merged_by_methods = merge_method_info(df_all, experience_df)

    # print("PAIRED T_TEST for avg_fix_dur_code")
    # run_paired_test_by_method(merged_by_methods, metric='avg_fix_dur_code')


    from statsmodels.stats.power import TTestPower, TTestIndPower

    def compute_sample_size(effect_size, alpha, power, test_type='paired'):
        if test_type == 'paired':
            analysis = TTestPower()
        elif test_type == 'independent':
            analysis = TTestIndPower()
        else:
            raise ValueError("test_type must be 'paired' or 'independent'")

        sample_size = analysis.solve_power(effect_size=effect_size,
                                        alpha=alpha,
                                        power=power,
                                        alternative='two-sided')
        return sample_size

    # Example usage:
    d = 0.57  # moderate effect size (from Karas et al or Sharif et al)
    alpha = 0.05
    power = 0.8

    paired_n = compute_sample_size(d, alpha, power, 'paired')
    independent_n = compute_sample_size(d, alpha, power, 'independent')

    print(f"Required sample size (paired t-test): {paired_n:.1f}")
    print(f"Required sample size (independent t-test): {independent_n:.1f}")