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
import matplotlib.transforms as mtransforms

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
from scipy.stats import ttest_rel, wilcoxon

from scipy.stats import ttest_rel, wilcoxon
import pandas as pd
import numpy as np

def paired_test(pre_gem, post_gem, metric):
    df_pre = pd.read_csv(pre_gem)
    df_post = pd.read_csv(post_gem)

    pre_data = df_pre[metric].dropna()
    post_data = df_post[metric].dropna()

    # Align by index (drop rows where either is NaN)
    paired_df = pd.DataFrame({'pre': pre_data, 'post': post_data}).dropna()
    pre_data_aligned = paired_df['pre']
    post_data_aligned = paired_df['post']

    print(f"\n🔍 Testing metric: {metric}")
    pre_normal = check_normality(pre_data_aligned)
    post_normal = check_normality(post_data_aligned)

    print(f"Pre normal? {pre_normal}")
    print(f"Post normal? {post_normal}")

    mean_pre = pre_data_aligned.mean()
    mean_post = post_data_aligned.mean()
    print(f"Mean Pre: {mean_pre:.4f}")
    print(f"Mean Post: {mean_post:.4f}")

    # Paired difference
    diffs = post_data_aligned - pre_data_aligned
    diffs_normal = check_normality(diffs)

    if diffs_normal:
        t_stat, p_val = ttest_rel(pre_data_aligned, post_data_aligned)
        print(f"✅ Paired t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
        cohens_d_val = cohen_d(pre_data_aligned, post_data_aligned, paired=True)
        print(f"Effect size (Cohen's d): {cohens_d_val:.3f}")
    else:
        # w_stat, p_val = wilcoxon(pre_data_aligned, post_data_aligned)
        # Non-parametric: Wilcoxon signed-rank
        diffs = post_data_aligned - pre_data_aligned
        diffs_nonzero = diffs[diffs != 0]
        n = len(diffs_nonzero)

        w_stat, p_val = wilcoxon(pre_data_aligned, post_data_aligned)

        # Z approximation
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z_stat = (w_stat - mean_w) / std_w
        r = z_stat / np.sqrt(n)

        print(f"Wilcoxon signed-rank test: W = {w_stat:.3f}, p = {p_val:.3f}")
        print(f"Z = {z_stat:.3f}")
        print(f"Effect size (r = Z/sqrt(N)): {r:.3f}")


def wilcoxon_effect_size(pre_data, post_data):
    diffs = post_data - pre_data
    diffs = diffs[diffs != 0]  # Wilcoxon excludes zeros
    n = len(diffs)

    w_stat, p_val = wilcoxon(pre_data, post_data)
    
    # Approximate Z from W
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_stat - mean_w) / std_w

    r = z / np.sqrt(n)
    
    print(f"Wilcoxon signed-rank test: W = {w_stat:.3f}, p = {p_val:.3f}")
    print(f"Z = {z:.3f}")
    print(f"Effect size (r = Z/sqrt(N)): {r:.3f}")
# Example cohen_d adapted for paired data:
def cohen_d(x, y, paired=False):
    import numpy as np
    if paired:
        diff = y - x
        return diff.mean() / diff.std(ddof=1)
    else:
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std

import seaborn as sns
import matplotlib.pyplot as plt



# Set seaborn theme
sns.set(style="whitegrid", font_scale=1.2)

def violin_paired_plot_metrics(pre_df, post_df, metrics, save=False, save_dir="plots"):
    """
    Plots violin + swarm plots for paired pre/post data across multiple metrics.

    Parameters:
    - pre_df, post_df: DataFrames for pre and post condition
    - metrics: list of metric column names (strings)
    - save: if True, saves the plots as PNGs
    - save_dir: directory to save plots
    """
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for metric in metrics:
        # Prepare paired dataframe
        paired_df = pd.DataFrame({
            'Pre': pre_df[metric],
            'Post': post_df[metric]
        }).dropna()

        melted = paired_df.melt(var_name='Condition', value_name='Value')

        plt.figure(figsize=(7, 5))
        ax = sns.violinplot(x='Condition', y='Value', data=melted, inner='box', palette='pastel')
        sns.swarmplot(x='Condition', y='Value', data=melted, color='k', alpha=0.5, size=4)

        # Title and labels
        title = metric.replace('_', ' ').title()
        plt.title(f"{title}: Pre vs Post", fontsize=14, weight='bold')
        plt.xlabel("")
        plt.ylabel(title)

        plt.tight_layout()

        if save:
            filename = os.path.join(save_dir, f"{metric}_violin_plot.png")
            plt.savefig(filename, dpi=300)
            print(f"Saved plot: {filename}")

        plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

def grouped_violin_plot(pre_df, post_df, metrics, group_name='Fixation Counts', save=False, save_dir="plots"):
    """
    Create a grouped violin plot for a set of related metrics (e.g., fixation counts or durations)
    """
    plot_data = []
    for metric in metrics:
        paired_df = pd.DataFrame({
            'Pre': pre_df[metric],
            'Post': post_df[metric]
        }).dropna()
        melted = paired_df.melt(var_name='Condition', value_name='Value')
        melted['Metric'] = metric
        plot_data.append(melted)

    plot_df = pd.concat(plot_data, ignore_index=True)

    metric_rename = {
        'total_fix_count_code': 'Code',
        'total_fix_count_summary': 'Summary',
        'avg_fix_dur_code': 'Code',
        'total_fix_count_tokens': 'Method Tokens',
        'avg_fix_dur_summary': 'Summary',
        'avg_fix_dur_other': 'Method Tokens',
    }
    plot_df['Metric'] = plot_df['Metric'].map(metric_rename).fillna(plot_df['Metric'])
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        x='Metric',
        y='Value',
        hue='Condition',
        data=plot_df,
        inner='quartile',
        palette={"Pre": "#F5A623", "Post": "#4A90E2"},
        dodge=True
    )
    plt.title(f'{group_name} No AI Assistance vs With AI Assistance')
    plt.ylabel('Avg Duration (ms)' if 'Duration' in group_name else 'Avg Number of Fixations')
    plt.xlabel('')
    plt.xticks(rotation=30)

    # Change legend labels only
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['No AI Assistance' if label == 'Pre' else 'With AI Assistance' for label in labels]
    ax.legend(handles, new_labels, title='Condition')

    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{group_name.replace(' ', '_').lower()}_violin.png"))
    plt.show()

def paired_line_plot(pre_df, post_df, metric):
    paired_df = pd.DataFrame({
        'Pre': pre_df[metric],
        'Post': post_df[metric]
    }).dropna()

    plt.figure(figsize=(6, 4))
    for i in range(len(paired_df)):
        plt.plot(['Pre', 'Post'], paired_df.iloc[i], marker='o', color='gray', alpha=0.5)

    plt.title(f"Paired Change in {metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

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

# def cohen_d(x, y):
#     nx = len(x)
#     ny = len(y)
#     dof = nx + ny - 2
#     pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / dof)
#     return (np.mean(x) - np.mean(y)) / pooled_std

def experience_group_test(df, metric, experience_col='Q4'):
    experience_mapping = {
        'Less than 1 year': 'Low',
        '1-2 years': 'Low',
        '2-4 years': 'High',
        '4+ years': 'High'
    }

    df = df.dropna(subset=['participant', metric])

    per_participant = df.groupby('participant').agg({
        metric: 'mean',
        experience_col: 'first'
    }).reset_index()

    per_participant['experience_group'] = per_participant[experience_col].map(experience_mapping)
    per_participant = per_participant[per_participant['experience_group'].notna()]

    low_data = per_participant[per_participant['experience_group'] == 'Low'][metric].dropna()
    high_data = per_participant[per_participant['experience_group'] == 'High'][metric].dropna()

    # Compute SEM for both groups
    sem_low = low_data.sem()
    sem_high = high_data.sem()

    print(f"\nTesting experience effect on: {metric}")
    print(f"Low experience (n={len(low_data)}), High experience (n={len(high_data)})")
    print(f"Mean {metric} - Low experience: {low_data.mean():.3f}")
    print(f"Mean {metric} - High experience: {high_data.mean():.3f}")

    low_normal = check_normality(low_data)
    high_normal = check_normality(high_data)
    equal_var = check_equal_variance(low_data, high_data)

    n_low = len(low_data)
    n_high = len(high_data)

    print("Low normality test?", low_normal)
    print("High normality test?", high_normal)
    print("Equal variance?", equal_var)

    if low_normal and high_normal:
        if equal_var:
            stat, p_val = ttest_ind(low_data, high_data, equal_var=True)
            effect_size = cohen_d(low_data, high_data)
            test_used = "Student's t-test"
        else:
            stat, p_val = ttest_ind(low_data, high_data, equal_var=False)
            effect_size = cohen_d(low_data, high_data)
            test_used = "Welch's t-test"
    else:
        stat, p_val = mannwhitneyu(low_data, high_data, alternative='two-sided')
        effect_size = rank_biserial_correlation(stat, n_low, n_high)
        test_used = "Mann-Whitney U"

    return {
        'metric': metric,
        'mean_low': low_data.mean(),
        'mean_high': high_data.mean(),
        'sem_low': sem_low,
        'sem_high': sem_high,
        'n_low': n_low,
        'n_high': n_high,
        'p_val': p_val,
        'effect_size': effect_size,
        'test_used': test_used,
        'stat': stat
    }

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as N
import matplotlib.pyplot as plt

def plot_between_subject_metric(result_dict):
    """
    Plots a grouped bar chart for Low vs High experience groups with error bars,
    significance stars, and effect size annotation for one metric.

    Parameters:
    - result_dict: dict output from experience_group_test for a single metric
    """
    metric = result_dict['metric']
    means = [result_dict['mean_low'], result_dict['mean_high']]
    sems = [result_dict['sem_low'], result_dict['sem_high']]
    groups = ['Low Experience', 'High Experience']

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(groups, means, yerr=sems, capsize=5, color=['#F5A623', '#4A90E2'], alpha=0.8)

    # Add title and labels
    ax.set_title(f'{metric} by Experience Group (Post)')
    ax.set_ylabel('Mean Value')
    ax.set_ylim(0, max(means) + max(sems)*2)

    # Annotate significance if p < 0.05
    p_val = result_dict['p_val']
    effect_size = result_dict['effect_size']
    if p_val < 0.05:
        max_height = max(means) + max(sems)*1.5
        ax.plot([0,1], [max_height, max_height], color='black')
        ax.text(0.5, max_height + 0.05*max_height, '*', ha='center', va='bottom', fontsize=20)
    
    # Annotate effect size below bars
    ax.text(0, means[0] + sems[0] + 0.05*means[0], f"ES={effect_size:.2f}", ha='center', fontsize=10)
    ax.text(1, means[1] + sems[1] + 0.05*means[1], f"ES={effect_size:.2f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_metric_by_experience_and_condition(pre_dict, post_dict, fixation_type='count', metric_label=None):
    """
    Plots the same metric across two conditions (e.g., Pre vs Post) for Low and High experience groups.
    
    Parameters:
    - pre_dict: dict from experience_group_test (pre condition)
    - post_dict: dict from experience_group_test (post condition)
    - fixation_type: 'count' or 'duration' (for Y-axis label)
    - metric_label: optional custom label for display
    """
    import matplotlib.pyplot as plt
    import numpy as np

    groups = ['Low Experience', 'High Experience']
    x = np.arange(len(groups))
    width = 0.35

    # Means and SEMs
    means_pre = [pre_dict['mean_low'], pre_dict['mean_high']]
    sems_pre = [pre_dict['sem_low'], pre_dict['sem_high']]

    means_post = [post_dict['mean_low'], post_dict['mean_high']]
    sems_post = [post_dict['sem_low'], post_dict['sem_high']]

    capsize_val = 4
    err_mult = .96  # shorter space above error bar for significance

    fig, ax = plt.subplots(figsize=(8, 6))

    pos_pre = x - width / 2
    pos_post = x + width / 2

    # Plot bars
    bars1 = ax.bar(pos_pre, means_pre, width, yerr=sems_pre, capsize=capsize_val,
                   label='No AI Assistance', color='#7cb46b', alpha=0.85)
    bars2 = ax.bar(pos_post, means_post, width, yerr=sems_post, capsize=capsize_val,
                   label='AI Assistance', color='#51074a', alpha=0.85)

    ylabel = 'Avg Number of Fixations' if fixation_type == 'count' else 'Avg Duration (ms)'
    title = metric_label or pre_dict['metric']

    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} by Experience Group and Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(title='Condition')

    # Add significance star slightly above bar + SEM
    def annotate_star(pos, mean, sem, p_val):
        if p_val < 0.05:
            star_y = mean + sem * err_mult
            ax.text(pos, star_y, '*', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Apply annotations
    for i in range(2):
        annotate_star(pos_pre[i], means_pre[i], sems_pre[i], pre_dict['p_val'])
        annotate_star(pos_post[i], means_post[i], sems_post[i], post_dict['p_val'])

    plt.tight_layout()
    plt.show()


def grouped_bar_plot_pre_post_within(results_df, fixation_type='count'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'legend.fontsize': 12})

    count_metrics = ['total_fix_count_code', 'total_fix_count_summary']
    duration_metrics = ['avg_fix_dur_code', 'avg_fix_dur_summary']

    if fixation_type == 'count':
        selected_metrics = count_metrics
        ylabel = 'Average Number of Fixations'
        plot_title = 'Fixation Count'
    elif fixation_type == 'duration':
        selected_metrics = duration_metrics
        ylabel = 'Average Duration (ms)'
        plot_title = 'Fixation Duration'
    else:
        raise ValueError("fixation_type must be 'count' or 'duration'")

    metric_rename = {
        'total_fix_count_code': 'Code',
        'total_fix_count_summary': 'Summary',
        'avg_fix_dur_code': 'Code',
        'avg_fix_dur_summary': 'Summary',
    }

    df = results_df.copy()
    df = df[df['metric'].isin(selected_metrics)]
    df['condition'] = df['condition'].map({'pre': 'No AI Assistance', 'post': 'AI Assistance'})
    df['Metric_Label'] = df['metric'].map(metric_rename)
    # Create combined x-axis label: Metric + Condition
    df['X_Label'] = df['metric'].map(metric_rename) + '\n(' + df['condition'] + ')'

    # Define the order for x-axis categories
    x_order = []
    for metric in [metric_rename[m] for m in selected_metrics]:
        for cond in ['No AI Assistance', 'AI Assistance']:
            x_order.append(f"{metric}\n({cond})")

    hue_order = ['Low', 'High']
    palette = {'Low': '#d69cbc', 'High': '#769a6e'} 

    # Create figure and axis
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # Get unique x labels and hues in correct order
    x_labels = x_order
    n_x = len(x_labels)
    n_hues = len(hue_order)

    # Calculate bar width
    total_bar_width = 0.8
    bar_width = total_bar_width / n_hues

    # Create mapping for hatch based on condition
    hatch_map = {
        'No AI Assistance': '',
        'AI Assistance': '//'
    }

    # Plot bars manually so we can set hatch patterns
    for i, x_label in enumerate(x_labels):
        for j, hue in enumerate(hue_order):
            row = df[(df['X_Label'] == x_label) & (df['experience_group'] == hue)]
            if not row.empty:
                mean = row['mean'].values[0]
                sem = row['sem'].values[0]

                # Bar position with dodge
                x_pos = i - total_bar_width/2 + j*bar_width + bar_width/2

                # Extract condition from x_label string for hatch
                condition = x_label.split('\n(')[1][:-1]

                # Draw bar with color and hatch
                bar = ax.bar(
                    x=x_pos,
                    height=mean,
                    width=bar_width,
                    color=palette[hue],
                    label=hue if i == 0 else "",  # only label once for legend
                    hatch=hatch_map[condition],
                    edgecolor='black'
                )
                # Error bars
                ax.errorbar(
                    x=x_pos,
                    y=mean,
                    yerr=sem,
                    fmt='none',
                    c='black',
                    capsize=5,
                    lw=1
                )

    ax.set_xticks(range(n_x))
    ax.set_xticklabels(x_labels, rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{plot_title} by Condition and Experience Group')

    # Legend: combine color legend and hatch legend manually
    from matplotlib.patches import Patch
    color_patches = [Patch(facecolor=palette[h], edgecolor='black', label=h) for h in hue_order]
    hatch_patches = [Patch(facecolor='white', edgecolor='black', hatch=hatch_map[c] if hatch_map[c] else '', label=c) for c in ['No AI Assistance', 'AI Assistance']]

    first_legend = ax.legend(handles=color_patches, title='Experience Group', loc='upper center', bbox_to_anchor=(0.3, -0.3), ncol=2, frameon=False)
    ax.add_artist(first_legend)  # Add first legend manually
    ax.legend(handles=hatch_patches, title='Condition', loc='upper center', bbox_to_anchor=(0.7, -0.3), ncol=2, frameon=False)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def forest_plot_effect_size(results_df):
    metrics = results_df['metric']
    effect_sizes = results_df['effect_size']
    p_values = results_df['p_val']

    y_pos = np.arange(len(metrics))

    plt.figure(figsize=(8, 5))
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]

    plt.hlines(y_pos, 0, effect_sizes, color='lightgray')
    plt.scatter(effect_sizes, y_pos, color=colors, zorder=3, s=80)

    plt.yticks(y_pos, metrics.str.replace('_', ' ').str.title())
    plt.axvline(0, color='black', lw=0.8)
    plt.xlabel('Effect Size (Cohen\'s d or Rank Biserial)')
    plt.title('Effect Sizes of Experience Group on Metrics')

    # Annotate p-values
    for i, (es, p) in enumerate(zip(effect_sizes, p_values)):
        sig = '*' if p < 0.05 else ''
        plt.text(es + np.sign(es)*0.02, i, f"{es:.2f}{sig}", va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


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

    # Normalize participant IDs in experience_df as strings with digits only
    experience_df = experience_df.copy()
    experience_df['participant_clean'] = experience_df['participant'].astype(str).str.extract(r'(\d+)')[0]

    # Normalize participant IDs in pre_post_df similarly
    pre_post_df = pre_post_df.copy()
    pre_post_df['participant_clean'] = pre_post_df['participant'].astype(str).str.extract(r'(\d+)')[0]

    method_map = {}
    for _, row in experience_df.iterrows():
        pid = row['participant_clean']
        if pd.isna(pid):
            print(f"Skipping experience_df row with missing participant ID: {row['participant']}")
            continue

        methods = str(row['Methods']).split('def')
       
        methods = [m.strip() for m in methods if m.strip()]
        # print(f"Participant {pid} METHODS ({len(methods)}):")
        for m in methods:
            print(f"  - {m}")
        if len(methods) != 3:
            print(f"Skipping participant {pid}: expected 3 methods, got {len(methods)}")
            continue
        method_map[pid] = methods

    pre_post_df['method'] = None

    for pid, methods in method_map.items():
        participant_df = pre_post_df[pre_post_df['participant_clean'] == pid]

        if len(participant_df) != 6:
            print(f"Skipping participant {pid}: expected 6 rows, got {len(participant_df)}")
            continue

        pre_rows = participant_df[participant_df['condition'] == 'pre'].sort_index()
        post_rows = participant_df[participant_df['condition'] == 'post'].sort_index()

        if len(pre_rows) != 3 or len(post_rows) != 3:
            print(f"Skipping participant {pid}: expected 3 pre and 3 post rows, got {len(pre_rows)} pre and {len(post_rows)} post")
            continue

        for method, pre_idx, post_idx in zip(methods, pre_rows.index, post_rows.index):
            pre_post_df.at[pre_idx, 'method'] = method
            pre_post_df.at[post_idx, 'method'] = method
            
            # print(f"Participant {pid}: Assigned method '{method}' to pre index {pre_idx} and post index {post_idx}")

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

         # 🔍 Add method-level debug prints
        print(f"  • Pre rows: {len(pre)}")
        print(f"  • Post rows: {len(post)}")

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

    Assumes each participant has exactly one 'pre' and one 'post' row per method.
    """
    for method in df['method'].dropna().unique():
        method_clean = method.strip()
        print(f"\n🟩 Method: {method_clean}")

        subset = df[df['method'].str.strip() == method_clean]

        pre = subset[subset['condition'] == 'pre']
        post = subset[subset['condition'] == 'post']

        participants = set(pre['participant']) & set(post['participant'])
        pre_vals = []
        post_vals = []
        dropped_due_to_nan = []

        for pid in participants:
            pre_rows = pre[pre['participant'] == pid]
            post_rows = post[post['participant'] == pid]

            if len(pre_rows) != 1 or len(post_rows) != 1:
                raise ValueError(f"Participant {pid} does not have exactly one pre and one post row for method '{method_clean}'. "
                                 f"Found pre: {len(pre_rows)}, post: {len(post_rows)}")

            pre_val = pre_rows[metric].values[0]
            post_val = post_rows[metric].values[0]

            if np.isnan(pre_val) or np.isnan(post_val):
                dropped_due_to_nan.append(pid)
                continue

            pre_vals.append(pre_val)
            post_vals.append(post_val)

        print(f"  • Participants with valid data: {(pre_vals)}")
        if dropped_due_to_nan:
            print(f"  • Participants dropped due to NaN in {metric}: {sorted(dropped_due_to_nan)}")

        if len(pre_vals) < 2:
            print("  ⚠️ Skipping test — not enough valid matched pairs.")
            continue

        diffs = np.array(pre_vals) - np.array(post_vals)
        shapiro_stat, shapiro_p = shapiro(diffs)
        print(f"  🔍 Shapiro-Wilk test on difference scores: W={shapiro_stat:.3f}, p={shapiro_p:.3f}")

        if shapiro_p > 0.05:
            t_stat, p_val = ttest_rel(pre_vals, post_vals)
            print(f"  📊 Paired t-test on '{metric}':")
            print(f"    n={len(pre_vals)} | pre mean={np.mean(pre_vals):.2f}, post mean={np.mean(post_vals):.2f}")
            print(f"    t = {t_stat:.3f}, p = {p_val:.3f}")
        else:
            w_stat, p_val = wilcoxon(pre_vals, post_vals)
            print(f"  📊 Wilcoxon signed-rank test on '{metric}':")
            print(f"    n={len(pre_vals)} | pre median={np.median(pre_vals):.2f}, post median={np.median(post_vals):.2f}")
            print(f"    W = {w_stat:.3f}, p = {p_val:.3f}")


def condition_effect_test(df, metric, experience_group, experience_col='Q4'):
    # Map experience groups
    experience_mapping = {
        'Less than 1 year': 'Low',
        '1-2 years': 'Low',
        '2-4 years': 'High',
        '4+ years': 'High'
    }
    
    df = df.dropna(subset=['participant', metric])
    df['experience_group'] = df[experience_col].map(experience_mapping)
    
    # Filter for experience group
    df_exp = df[df['experience_group'] == experience_group]
    
    # Aggregate metric per participant per condition
    agg = df_exp.groupby(['participant', 'condition'])[metric].mean().unstack()
    
    # Ensure both pre and post exist for participant (paired)
    agg = agg.dropna(subset=['pre', 'post'])
    
    pre_vals = agg['pre']
    post_vals = agg['post']
    
    diffs = post_vals - pre_vals
    
    # Normality test of differences
    p_norm = shapiro(diffs).pvalue
    normal = p_norm > 0.05
    
    print(f"{experience_group} experience - {metric} change from Pre to Post:")
    print(f"Normality of differences p-value: {p_norm:.3f} -> {'Normal' if normal else 'Non-normal'}")
    
    if normal:
        # Paired t-test + Cohen's d
        t_stat, p_val = ttest_rel(post_vals, pre_vals)
        cohens_d_val = cohen_d(post_vals, pre_vals, paired=True)
        print(f"✅ Paired t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
        print(f"Effect size (Cohen's d): {cohens_d_val:.3f}")
        test_used = 'Paired t-test'
        stat = t_stat
        effect_size = cohens_d_val
    else:
        # Wilcoxon signed-rank test + r effect size
        diffs_nonzero = diffs[diffs != 0]
        n = len(diffs_nonzero)
        w_stat, p_val = wilcoxon(post_vals, pre_vals)
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z_stat = (w_stat - mean_w) / std_w
        r = z_stat / np.sqrt(n)
        print(f"Wilcoxon signed-rank test: W = {w_stat:.3f}, p = {p_val:.3f}")
        print(f"Z = {z_stat:.3f}")
        print(f"Effect size (r = Z/sqrt(N)): {r:.3f}")
        test_used = 'Wilcoxon signed-rank'
        stat = w_stat
        effect_size = r

    print('----------------------------------------')
    
    return {
        'experience_group': experience_group,
        'metric': metric,
        'test_used': test_used,
        'stat': stat,
        'p_val': p_val,
        'effect_size': effect_size,
        'normality_p': p_norm
    }

def build_within_subjects_plot_df(df_all, metrics, experience_col='Q4'):
    from scipy.stats import sem
    import pandas as pd
    
    experience_mapping = {
        'Less than 1 year': 'Low',
        '1-2 years': 'Low',
        '2-4 years': 'High',
        '4+ years': 'High'
    }

    results = []

    for exp_group in ['Low', 'High']:
        df_exp = df_all[df_all[experience_col].map(experience_mapping) == exp_group]

        for metric in metrics:
            # Drop NAs and calculate paired data
            df_exp_metric = df_exp.dropna(subset=['participant', metric])
            df_exp_metric['experience_group'] = exp_group
            agg = df_exp_metric.groupby(['participant', 'condition'])[metric].mean().unstack()
            agg = agg.dropna(subset=['pre', 'post'])

            # Run test
            test_result = condition_effect_test(df_all, metric, exp_group)

            # Collect means and SEMs
            for cond in ['pre', 'post']:
                mean = agg[cond].mean()
                sem_val = sem(agg[cond])
                results.append({
                    'metric': metric,
                    'condition': cond,
                    'experience_group': exp_group,
                    'mean': mean,
                    'sem': sem_val,
                    'p_val': test_result['p_val'],
                    'effect_size': test_result['effect_size']
                })

    return pd.DataFrame(results)

def grouped_bar_plot_combined_all_metrics(results_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    sns.set(style="whitegrid")

    # Rename mappings
    metric_rename = {
        'total_fix_count_code': ('Fixation Count', 'Code'),
        'total_fix_count_summary': ('Fixation Count', 'Summary'),
        'avg_fix_dur_code': ('Fixation Duration', 'Code'),
        'avg_fix_dur_summary': ('Fixation Duration', 'Summary'),
    }

    # Apply renaming
    results_df = results_df.copy()
    results_df['MetricGroup'] = results_df['metric'].map(lambda m: metric_rename[m][0])
    results_df['MetricLabel'] = results_df['metric'].map(lambda m: metric_rename[m][1])
    results_df['condition'] = results_df['condition'].map({'pre': 'No AI Assistance', 'post': 'AI Assistance'})

    # Sort order
    metric_order = ['Fixation Count', 'Fixation Duration']
    label_order = ['Code', 'Summary']
    condition_order = ['No AI Assistance', 'AI Assistance']
    exp_order = ['Low', 'High']

    # Create subplot for each MetricGroup
    for group in metric_order:
        df_group = results_df[results_df['MetricGroup'] == group]

        # Create a composite label for bar positions: (condition, metric)
        df_group['x_group'] = df_group['condition']
        df_group['bar_label'] = df_group['MetricLabel'] + ' (' + df_group['experience_group'] + ')'

        # Set order for bar categories
        bar_order = [f"{label} ({exp})" for label in label_order for exp in exp_order]

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=df_group,
            x='x_group',
            y='mean',
            hue='bar_label',
            hue_order=bar_order,
            palette={
                'Code (Low)': '#FFD700',
                'Code (High)': '#1E90FF',
                'Summary (Low)': '#FFA500',
                'Summary (High)': '#4169E1'
            },
            ci=None,
            dodge=True
        )

        # Add error bars
        bar_offsets = {
            'Code (Low)': -0.3, 'Code (High)': -0.1,
            'Summary (Low)': 0.1, 'Summary (High)': 0.3
        }

        for i, row in df_group.iterrows():
            x_pos = condition_order.index(row['condition']) + bar_offsets[row['bar_label']]
            ax.errorbar(
                x=x_pos,
                y=row['mean'],
                yerr=row['sem'],
                fmt='none',
                c='black',
                capsize=5,
                lw=1,
                zorder=10
            )

        # Add ES text for AI Assistance condition
        for label in bar_order:
            row = df_group[(df_group['condition'] == 'AI Assistance') & (df_group['bar_label'] == label)]
            if not row.empty:
                row = row.iloc[0]
                x_pos = 1 + bar_offsets[label]
                y = row['mean'] + row['sem'] + 0.05 * row['mean']
                sig = '*' if row['p_val'] < 0.05 else ''
                ax.text(x_pos, y, f"ES={row['effect_size']:.2f}{sig}", ha='center', fontsize=9, fontweight='bold')

        ax.set_title(f'{group} by Condition and Experience')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Avg Duration (ms)' if 'Duration' in group else 'Avg Number of Fixations')
        ax.set_xticklabels(condition_order, rotation=15)
        ax.legend(title='Metric (Experience)', loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()


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
    # # print(df_post.head())

    # # Combine
    # # df_all = pd.concat([df_pre, df_post], ignore_index=True)
    # # print("Combined Data:")
    # # print(df_all.head())

    # # # Normalize fixation metrics (z-score)
    # # metrics_to_normalize = [col for col in df_all.columns if col.startswith("fix_")]
    # # df_z = df_all.copy()
    # # df_z[metrics_to_normalize] = df_all[metrics_to_normalize].apply(zscore)
    # # participant,task,condition,
    # #total_fix_count_code,total_fix_count_gemini,total_fix_count_summary,total_fix_count_tokens,
    # #avg_fix_count_tokens,
    # #avg_fix_dur_code,avg_fix_dur_gemini,avg_fix_dur_summary,avg_fix_dur_other

    # # # avg pre gemini: code/summary - count and duration 
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


    # paired_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_code')
    # paired_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_summary')

    # paired_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_code')

    # paired_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_tokens')
    # paired_test('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_summary')


    # ===============================
    # Violin plot with individual data points
    pre_df = pd.read_csv('pre_gemini_basic_fixations.csv')
    post_df = pd.read_csv('post_gemini_basic_fixations.csv')

    # metric = 'total_fix_count_code'
    # # List of metrics to visualize
    # metrics = [
    #     'total_fix_count_code',
    #     'total_fix_count_summary',
    #     'avg_fix_dur_code',
    #     'total_fix_count_tokens',
    #     'avg_fix_dur_summary'
    # ]

    # # Run the plots for all metrics
    # violin_paired_plot_metrics(pre_df, post_df, metrics, save=True) 

    # count_metrics = ['total_fix_count_code', 'total_fix_count_summary', 'total_fix_count_tokens']
    # duration_metrics = ['avg_fix_dur_code', 'avg_fix_dur_summary', 'avg_fix_dur_other']


    # grouped_violin_plot(pre_df, post_df, count_metrics, group_name='Fixation Counts', save=True)
    # grouped_violin_plot(pre_df, post_df, duration_metrics, group_name='Fixation Durations', save=True)

    # violin_paired_plot(pre_df, post_df, metric)
    # violin_paired_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_summary')

    # violin_paired_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_code')

    # violin_paired_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_tokens')
    # violin_paired_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_summary')

    # =========
    # paired_line_plot(pre_df, post_df,metric )
    # paired_line_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_summary')

    # paired_line_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_code')

    # paired_line_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'total_fix_count_tokens')
    # paired_line_plot('pre_gemini_basic_fixations.csv', 'post_gemini_basic_fixations.csv', 'avg_fix_dur_summary')

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
    # Compare experience group impact on code fixations count in post-Gemini only
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

    # # And for pre-Gemini
    # experience_group_test(df_all[df_all['condition'] == 'pre'], 'avg_fix_dur_summary')


    
    # ====== Visualization for experienc e
    metrics = [
    'total_fix_count_code', 
    'avg_fix_dur_code', 
    'total_fix_count_summary', 
    'avg_fix_dur_summary'
    ]

    results = []

    # print("=== Between-Group Experience Tests (Low vs High) ===")
    # for condition in ['pre', 'post']:
    #     print(f"\nCondition: {condition.upper()}")
    #     df_cond = df_all[df_all['condition'] == condition]

    #     for metric in metrics:
    #         res = experience_group_test(df_cond, metric)
    #         res['condition'] = condition
    #         results.append(res)
    #         # Print summary for each test
    #         print(f"Metric: {metric}")
    #         print(f"  Groups: Low (n={res['n_low']}), High (n={res['n_high']})")
    #         print(f"  Means: Low={res['mean_low']:.3f}, High={res['mean_high']:.3f}")
    #         print(f"  Test used: {res['test_used']}, stat={res['stat']:.3f}, p={res['p_val']:.3f}")
    #         print(f"  Effect size: {res['effect_size']:.3f}")
    #         print("-" * 40)

    # print("\n=== Within-Subject Pre-vs-Post Tests by Experience Group ===")
    # for exp_group in ['Low', 'High']:
    #     print(f"\nExperience Group: {exp_group}")
    #     for metric in metrics:
    #         res = condition_effect_test(df_all, metric, exp_group)
    #         results.append(res)
    #         # Print summary for each test
    #         print(f"Metric: {metric}")
    #         print(f"  Test used: {res['test_used']}, stat={res['stat']:.3f}, p={res['p_val']:.3f}")
    #         print("-" * 40)


    # for condition in ['pre', 'post']:
    #     df_cond = df_all[df_all['condition'] == condition]
    #     for metric in metrics:
    #         res = experience_group_test(df_cond, metric)
    #     # Save or print res

    # Run within-group pre-vs-post tests
    # for exp_group in ['Low', 'High']:
    #     for metric in metrics:
    #         res = condition_effect_test(df_all, metric, exp_group)

    # results = []

    # for condition in ['pre', 'post']:
    #     print(f"--- Analyzing condition: {condition} ---")
    #     df_cond = df_all[df_all['condition'] == condition]

    #     for metric in metrics:
    #         res = experience_group_test(df_cond, metric)
    #         res['condition'] = condition  # keep track of pre/post
    #         results.append(res)

    # After collecting all results and making DataFrame
    # results_df = pd.DataFrame(results)
    # plot_df = build_within_subjects_plot_df(df_all, metrics)
    # grouped_bar_plot_pre_post_within(plot_df)
    # grouped_bar_plot_pre_post_within(plot_df)

    # # Pre- gemini v. Post-gemini
    results_pre = experience_group_test(df_all[df_all['condition'] == 'pre'], 'avg_fix_dur_summary')
    # plot_between_subject_metric(results_pre)

    results_post = experience_group_test(df_all[df_all['condition'] == 'post'], 'avg_fix_dur_summary')

    df_pre = pd.DataFrame([results_pre])
    df_post = pd.DataFrame([results_post])
    # plot_between_subject_metric(results_post)

    plot_metric_by_experience_and_condition(
    results_pre, 
    results_post, 
    fixation_type='duration',
    metric_label='Fixation Duration (Summary)'
    )

    

    # ============== Paired T-test for difficulty ================================
    # experience_df = pd.read_excel('CleanedParticipantData.xlsx')


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
    # run_paired_test_by_method(merged_by_methods, metric='total_fix_count_code')
    # run_paired_test_by_method(merged_by_methods, metric='total_fix_count_summary')
    # run_paired_test_by_method(merged_by_methods, metric='avg_fix_dur_summary')



    # code to get statistical power to report for methods section 
    # from statsmodels.stats.power import TTestPower, TTestIndPower

    # def compute_sample_size(effect_size, alpha, power, test_type='paired'):
    #     if test_type == 'paired':
    #         analysis = TTestPower()
    #     elif test_type == 'independent':
    #         analysis = TTestIndPower()
    #     else:
    #         raise ValueError("test_type must be 'paired' or 'independent'")

    #     sample_size = analysis.solve_power(effect_size=effect_size,
    #                                     alpha=alpha,
    #                                     power=power,
    #                                     alternative='two-sided')
    #     return sample_size

    # # Example usage:
    # d = 0.57  # moderate effect size (from Karas et al or Sharif et al)
    # alpha = 0.05
    # power = 0.8

    # paired_n = compute_sample_size(d, alpha, power, 'paired')
    # independent_n = compute_sample_size(d, alpha, power, 'independent')

    # print(f"Required sample size (paired t-test): {paired_n:.1f}")
    # print(f"Required sample size (independent t-test): {independent_n:.1f}")

    