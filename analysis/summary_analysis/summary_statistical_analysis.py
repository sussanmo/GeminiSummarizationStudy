import pandas as pd
from scipy.stats import ttest_rel
import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon
from scipy.stats import t  # for confidence intervals
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import scipy.stats as stats

def correlation_auto_vs_human(df, human_metric, auto_metrics, condition_suffix='_post', alpha=0.05):
    """
    Calculate Pearson and Spearman correlations between human evaluation and automatic metrics,
    and determine which is more appropriate based on normality of variables.

    Parameters:
    - df: DataFrame containing your data
    - human_metric: str, column name for the human evaluation metric (without suffix)
    - auto_metrics: list of str, column names for automatic metrics (without suffix)
    - condition_suffix: str, suffix for pre/post columns (default '_post')
    - alpha: float, significance level for Shapiro-Wilk normality test
    """
    human_col = f"{human_metric}{condition_suffix}"
    
    for metric in auto_metrics:
        auto_col = f"{metric}{condition_suffix}"
        paired = df[[human_col, auto_col]].dropna()
        
        if paired.empty:
            print(f"No data to compare {human_metric} and {metric}.\n")
            continue

        # Shapiro-Wilk tests for normality
        sw_human_stat, sw_human_p = stats.shapiro(paired[human_col])
        sw_auto_stat, sw_auto_p = stats.shapiro(paired[auto_col])

        is_human_normal = sw_human_p > alpha
        is_auto_normal = sw_auto_p > alpha

        pearson_r, pearson_p = stats.pearsonr(paired[human_col], paired[auto_col])
        spearman_r, spearman_p = stats.spearmanr(paired[human_col], paired[auto_col])

        print(f"\n📊 {metric} correlation with {human_metric} ({condition_suffix.strip('_')}):")
        print(f"  Normality (Shapiro-Wilk):")
        print(f"    {human_col}: W = {sw_human_stat:.3f}, p = {sw_human_p:.3f} → {'Normal' if is_human_normal else 'Non-normal'}")
        print(f"    {auto_col}:  W = {sw_auto_stat:.3f}, p = {sw_auto_p:.3f} → {'Normal' if is_auto_normal else 'Non-normal'}")

        print(f"\n  Pearson r = {pearson_r:.3f}, p = {pearson_p:.3f}")
        print(f"  Spearman rho = {spearman_r:.3f}, p = {spearman_p:.3f}")

        preferred = "Pearson" if is_human_normal and is_auto_normal else "Spearman"
        print(f"\n  ✅ Recommended: Use **{preferred}** correlation based on normality.\n")
        
def cohen_d_paired(x, y):
    """Compute Cohen's d for paired samples"""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def rank_biserial_effect_size(x, y):
    """Calculate rank biserial correlation effect size for Wilcoxon"""
    from scipy.stats import rankdata
    diff = x - y
    n = len(diff)
    r = wilcoxon(x, y).statistic
    # rank biserial = 1 - 2 * W / (n*(n+1)/2)
    rb = 1 - (2 * r) / (n * (n + 1) / 2)
    return rb

def rank_biserial_effect_size_experience(w_stat, n):
    """
    Calculate rank-biserial correlation from Wilcoxon signed-rank test statistic.
    w_stat: Wilcoxon test statistic (sum of ranks of the less frequent sign)
    n: number of paired samples
    Returns:
        float: rank-biserial correlation (effect size)
    """
    return 1 - (4 * w_stat) / (n * (n + 1))
def run_paired_t_tests_with_correction(merged_df, metrics, alpha=0.05, correction='fdr_bh'):

    # Step 1: Average by participant
    participant_avg = (
        merged_df.groupby('Participant')[
            [f'{metric}_pre' for metric in metrics] +
            [f'{metric}_post' for metric in metrics]
        ].mean().reset_index()
    )

    print("\n🔍 Overall Paired Tests with Normality Check & Effect Sizes:")
    
    p_vals = []
    test_results = []

    for metric in metrics:
        # paired_data = merged_df[[f'{metric}_pre', f'{metric}_post']].dropna()
        paired_data = participant_avg[[f'{metric}_pre', f'{metric}_post']].dropna()
        n = len(paired_data)
        if n < 2:
            print(f"❗ Not enough paired data for {metric}")
            continue

        pre_scores = paired_data[f'{metric}_pre']
        post_scores = paired_data[f'{metric}_post']
        diff = post_scores - pre_scores

         # Calculate means
        mean_pre = pre_scores.mean()
        std_pre = pre_scores.std()
        mean_post = post_scores.mean()
        std_post = post_scores.std()



        stat_norm, p_normality = shapiro(diff)
        normality = p_normality > 0.05

        if normality:
            t_stat, p_val = ttest_rel(post_scores, pre_scores)
            effect_size = cohen_d_paired(post_scores, pre_scores)
            test_name = "Paired t-test"
            effect_label = "Cohen's d"
        else:
            t_stat, p_val = wilcoxon(post_scores, pre_scores)
            effect_size = rank_biserial_effect_size(post_scores, pre_scores)
            test_name = "Wilcoxon signed-rank"
            effect_label = "Rank-biserial"

        # Save for correction
        p_vals.append(p_val)
        test_results.append({
            "metric": metric,
            "test": test_name,
            "stat": t_stat,
            "p": p_val,
            "effect": effect_size,
            "effect_label": effect_label,
            "normality": p_normality, 
            "mean_pre": mean_pre,
            "std_pre": std_pre,
            "mean_post": mean_post,
            "std_post": std_post
        })

    # Correction
    reject, p_corrected, _, _ = multipletests(p_vals, alpha=alpha, method=correction)

    print("\n📊 Corrected p-values using", correction.upper())
    for i, res in enumerate(test_results):
        print(f"\nMetric: {res['metric']}")
        print(f"Test: {res['test']} | Statistic = {res['stat']:.3f}")
        print(f"Mean Pre: {res['mean_pre']:.2f} ± {res['std_pre']:.2f} | Mean Post: {res['mean_post']:.2f} ± {res['std_post']:.2f}")

        print(f"Original p = {res['p']:.3f} | Corrected p = {p_corrected[i]:.3f} --> {'Significant' if reject[i] else 'Not significant'}")
        print(f"{res['effect_label']}: {res['effect']:.3f}")
        print(f"Normality of differences (Shapiro-Wilk p): {res['normality']:.3f}")

def run_paired_t_tests(merged_df, metrics):
    print("\n🔍 Overall Paired Tests with Normality Check & Effect Sizes:")
    for metric in metrics:
        paired_data = merged_df[[f'{metric}_pre', f'{metric}_post']].dropna()
        n = len(paired_data)
        if n < 2:
            print(f"❗ Not enough paired data for {metric}")
            continue
        
        pre_scores = paired_data[f'{metric}_pre']
        post_scores = paired_data[f'{metric}_post']
        diff = post_scores - pre_scores
        
        # Normality test on differences
        stat_norm, p_normality = shapiro(diff)
        normality = p_normality > 0.05
        
        # Descriptive stats
        mean_pre = np.mean(pre_scores)
        std_pre = np.std(pre_scores, ddof=1)
        mean_post = np.mean(post_scores)
        std_post = np.std(post_scores, ddof=1)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        se_diff = std_diff / np.sqrt(n)
        df = n - 1
        conf_interval = t.interval(0.95, df, loc=mean_diff, scale=se_diff)
        
        if normality:
            # Use paired t-test
            t_stat, p_val = ttest_rel(post_scores, pre_scores)
            d = cohen_d_paired(post_scores, pre_scores)
            test_name = "Paired t-test"
            effect_size_name = "Cohen's d"
            effect_size_val = d
        else:
            # Use Wilcoxon signed-rank test
            t_stat, p_val = wilcoxon(post_scores, pre_scores)
            rb = rank_biserial_effect_size(post_scores, pre_scores)
            test_name = "Wilcoxon signed-rank test"
            effect_size_name = "Rank-biserial correlation"
            effect_size_val = rb
        
        print(f"\nMetric: {metric}")
        print(f"Sample size: {n}")
        print(f"Normality of differences (Shapiro-Wilk): W = {stat_norm:.3f}, p = {p_normality:.3f} --> {'Normal' if normality else 'Non-normal'}")
        print(f"{test_name}: statistic = {t_stat:.3f}, p = {p_val:.3f}")
        print(f"{effect_size_name}: {effect_size_val:.3f}")
        print(f"Pre mean ± std: {mean_pre:.3f} ± {std_pre:.3f}")
        print(f"Post mean ± std: {mean_post:.3f} ± {std_post:.3f}")
        print(f"Mean difference ± std: {mean_diff:.3f} ± {std_diff:.3f}")
        print(f"95% CI of difference: [{conf_interval[0]:.3f}, {conf_interval[1]:.3f}]")

def load_and_prepare_data(pre_path, post_path):
    pre = pd.read_excel(pre_path)
    post = pd.read_excel(post_path)
    pre.columns = pre.columns.str.strip()
    post.columns = post.columns.str.strip()
    print(f"Pre Columns {pre.columns}")
    print(f"Post Columns {post.columns}")
    pre['Task'] = pre['Task'].astype(str)
    post['Task'] = post['Task'].astype(str)
    # Compute new Total Rating as average of the 3 sub-metrics
    human_metrics = ['Accuracy:', 'Readability:', 'Cohesion:']
    pre['Total Rating'] = pre[human_metrics].mean(axis=1)
    post['Total Rating'] = post[human_metrics].mean(axis=1)

    print(f"✅ Loaded and recalculated Total Rating for pre and post datasets.")
    
    return pre, post

def standardize_task_format(df, task_col='Task'):
    df[task_col] = df[task_col].astype(str).str.replace('_', '.', regex=False)
    return df

# Task
def merge_datasets(pre, post):
   
    # Create sets of unique keys for pre and post
    pre_keys = set(zip(pre['Participant'], pre['Task']))
    post_keys = set(zip(post['Participant'], post['Task']))

    # Keys only in pre, not in post
    pre_only = pre_keys - post_keys
    # Keys only in post, not in pre
    post_only = post_keys - pre_keys

    print(f"Unique pairs in pre: {len(pre_keys)}")
    print(f"Unique pairs in post: {len(post_keys)}")
    print(f"Pairs only in pre (not in post): {len(pre_only)}")
    print(f"Pairs only in post (not in pre): {len(post_only)}")

    # Optionally show some examples
    print(f"Examples of pairs only in pre (up to 5): {list(pre_only)[:5]}")
    print(f"Examples of pairs only in post (up to 5): {list(post_only)[:5]}")

    # Now do the merge (inner join)
    merged = pd.merge(pre, post, on=['Participant', 'Task'], suffixes=('_pre', '_post'))
    print(f"Merged dataset shape: {merged.shape}")
    return merged


def merge_experience(df, participant_info):
    # Assuming participant_info already has 'ExperienceRange' and 'ExperienceGroup'
    pid_to_exp_range = dict(zip(participant_info['Pid'], participant_info['ExperienceRange']))
    pid_to_exp_group = dict(zip(participant_info['Pid'], participant_info['ExperienceGroup']))
    
    df['ExperienceRange'] = df['Participant'].map(pid_to_exp_range)
    df['ExperienceGroup'] = df['Participant'].map(pid_to_exp_group)
    return df

def run_experience_based_tests(merged_df, metrics):
    print("\n🧠 Experience-Moderated Tests (Low vs. High):")

    if 'ExperienceGroup' not in merged_df.columns:
        print("⚠️ 'ExperienceGroup' column not found.")
        return

    for group_label in ['Low', 'High']:
        group_df = merged_df[merged_df['ExperienceGroup'] == group_label]
        print(f"\n🔹 Group: {group_label} (n = {len(group_df)})")

        for metric in metrics:
            paired_data = group_df[[f'{metric}_pre', f'{metric}_post']].dropna()
            n = len(paired_data)
            if n < 2:
                print(f"  ❗ Not enough data for {metric}")
                continue

            pre = paired_data[f'{metric}_pre']
            post = paired_data[f'{metric}_post']
            diff = post - pre

            # Normality check
            W, p_normal = shapiro(diff)
            normality = "Pass" if p_normal > 0.05 else "Fail"

            print(f"\n  Metric: {metric}")
            print(f"  Normality of diff: W = {W:.3f}, p = {p_normal:.3f} ({normality})")

            if normality == "Pass":
                t_stat, p_val = ttest_rel(post, pre)
                d = cohen_d_paired(post, pre)
                print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
                print(f"  Cohen's d: {d:.3f}")
            else:
                w_stat, p_val = wilcoxon(post, pre)
                r = rank_biserial_effect_size_experience(w_stat, n)
                print(f"  Wilcoxon test: W = {w_stat:.3f}, p = {p_val:.3f}")
                print(f"  Rank-biserial correlation: {r:.3f}")


def run_experience_based_tests_with_correction(merged_df, metrics, alpha=0.05, correction='fdr_bh'):
    print("\n🧠 Experience-Moderated Tests (Low vs. High):")

    if 'ExperienceGroup' not in merged_df.columns:
        print("⚠️ 'ExperienceGroup' column not found.")
        return

    # Average across tasks per participant before testing
    agg_df = (
        merged_df.groupby(['Participant', 'ExperienceGroup'])
        .agg({f'{m}_pre': 'mean' for m in metrics} | {f'{m}_post': 'mean' for m in metrics})
        .reset_index()
    )
    for group_label in ['Low', 'High']:
        # group_df = merged_df[merged_df['ExperienceGroup'] == group_label]
        group_df = agg_df[agg_df['ExperienceGroup'] == group_label]

        print(f"\n🔹 Group: {group_label} (n = {len(group_df)})")

        test_results = []
        p_vals = []

        for metric in metrics:
            paired_data = group_df[[f'{metric}_pre', f'{metric}_post']].dropna()
            n = len(paired_data)
            if n < 2:
                print(f"  ❗ Not enough data for {metric}")
                continue

            pre = paired_data[f'{metric}_pre']
            post = paired_data[f'{metric}_post']
            diff = post - pre
            mean_pre = pre.mean()
            std_pre = pre.std()
            mean_post = post.mean()
            std_post = post.std()


            W, p_normal = shapiro(diff)
            normal = p_normal > 0.05

            if normal:
                t_stat, p_val = ttest_rel(post, pre)
                effect = cohen_d_paired(post, pre)
                test_name = "Paired t-test"
                effect_name = "Cohen's d"
            else:
                t_stat, p_val = wilcoxon(post, pre)
                effect = rank_biserial_effect_size(post, pre)
                test_name = "Wilcoxon"
                effect_name = "Rank-biserial"

            p_vals.append(p_val)
            test_results.append({
                "metric": metric,
                "test": test_name,
                "stat": t_stat,
                "p_val": p_val,
                "effect": effect,
                "effect_name": effect_name,
                "normality_p": p_normal,
                "mean_pre": mean_pre,
                "std_pre": std_pre,
                "mean_post": mean_post,
                "std_post": std_post
            })

        # Multiple comparison correction
        reject, corrected_p_vals, _, _ = multipletests(p_vals, alpha=alpha, method=correction)

        print(f"\n📊 Corrected p-values using {correction.upper()}")
        for i, res in enumerate(test_results):
            print(f"\n  Metric: {res['metric']}")
            print(f"  Test: {res['test']} | Statistic = {res['stat']:.3f}")
            print(f"Mean Pre: {res['mean_pre']:.2f} ± {res['std_pre']:.2f} | Mean Post: {res['mean_post']:.2f} ± {res['std_post']:.2f}")

            print(f"  Original p = {res['p_val']:.3f} | Corrected p = {corrected_p_vals[i]:.3f} --> {'✅ Significant' if reject[i] else '❌ Not significant'}")
            print(f"  {res['effect_name']}: {res['effect']:.3f}")
            print(f"  Normality (Shapiro-Wilk): p = {res['normality_p']:.3f}")

def classify_experience(range_str):
    if pd.isna(range_str):
        return None
    if "Less" in range_str or "1–2" in range_str or "1-2" in range_str:
        return "Low"
    if "2–4" in range_str or "2-4" in range_str or "4+" in range_str:
        return "High"
    return None

def get_avg_from_file(file_path, column):
   
    df = pd.read_csv(file_path)


    # Exclude specific participants
    excluded = ['participant320', 'participant210']
    if 'participant' in df.columns:
        df = df[~df['participant'].isin(excluded)]
    elif 'pid' in df.columns:
        df = df[~df['pid'].isin(excluded)]

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in file.")

     # Compute stats
    mean_val = df[column].dropna().mean()
    std_val = df[column].dropna().std()
   
    print(f"Mean of {column}: {round(mean_val, 2)}")
    print(f"Std of {column}: {round(std_val, 2)}")

    return round(mean_val, 2), round(std_val, 2)

if __name__ == '__main__':

    pre, post = load_and_prepare_data('analysis/summary_analysis/pre_gemini_with_metrics.xlsx', 'analysis/summary_analysis/post_gemini_with_metrics.xlsx')
    
    pre = standardize_task_format(pre, 'Task')
    post = standardize_task_format(post, 'Task')

    # get_avg_from_file(pre, '')
    # Remove participants 210 and 320
    pre = pre[~pre['Participant'].isin([210, 320])]
    post = post[~post['Participant'].isin([210, 320])]

    merged = merge_datasets(pre, post)
    metrics = ['Accuracy:','Readability:','Cohesion:', 'Total Rating','BLEU', 'ROUGE-L', 'METEOR']
    #Accuracy:	Readability:	Cohesion:


    # run_paired_t_tests(merged, metrics)
    run_paired_t_tests_with_correction(merged, metrics)

    participant_info = pd.read_excel('analysis/CleanedParticipantData.xlsx')
    print(participant_info.columns)

    # Rename once outside, only if needed
    participant_info.rename(columns={'Q4': 'ExperienceRange'}, inplace=True)

    # print("Columns before creating ExperienceGroup:", participant_info.columns)
    participant_info['ExperienceGroup'] = participant_info['ExperienceRange'].apply(classify_experience)
    # print("Columns after creating ExperienceGroup:", participant_info.columns)



    # Attach experience info before merging
    pre = merge_experience(pre, participant_info)
    # print("Columns after merging ExperienceGroup in pre:", pre.columns)
    # print("Pre experience range", pre['ExperienceGroup'])
    post = merge_experience(post, participant_info)
    # print("Columns after merging ExperienceGroup in post:", post.columns)
    # print("Pre experience range", post['ExperienceGroup'])


    # Merge on participant & task
    merged = pd.merge(pre, post, on=['Participant', 'Task'], suffixes=('_pre', '_post'))


    
    
    
    # Consolidate ExperienceGroup and drop duplicates
    merged['ExperienceGroup'] = merged['ExperienceGroup_pre']


    merged.drop(columns=['ExperienceGroup_pre', 'ExperienceGroup_post', 'ExperienceRange_pre', 'ExperienceRange_post'], inplace=True)

    # metric = 'Total Rating'
    # matched = merged.dropna(subset=[f'{metric}_pre', f'{metric}_post'])  # for each metric or overall
    # print(f"Number of fully matched pairs: {len(matched)}")

    # Run experience moderated tests
    # run_experience_based_tests(merged, metrics)

    run_experience_based_tests_with_correction(merged, metrics)
    # 
    
    # from statsmodels.stats.multitest import multipletests

    # pvals = []
    # results = []

    # for metric in metrics:
    #     safe_metric = metric.replace(" ", "_").replace("-", "_")
    #     matched = merged.dropna(subset=[f'{metric}_pre', f'{metric}_post'])
        
    #     df_long = pd.melt(
    #         matched,
    #         id_vars=['Participant', 'Task', 'ExperienceGroup'],
    #         value_vars=[f'{metric}_pre', f'{metric}_post'],
    #         var_name='Assistance',
    #         value_name=metric
    #     )
    #     df_long['Assistance'] = df_long['Assistance'].str.replace(f'{metric}_', '')
    #     df_long.rename(columns={metric: safe_metric}, inplace=True) 
    #     model = smf.mixedlm(f"{safe_metric} ~ Assistance * ExperienceGroup",
    #                         data=df_long,
    #                         groups=df_long["Participant"])
    #     result = model.fit()
    #     print(f"\nMixed Effects Model Results for {metric}:")
    #     print(result.summary())

    #     # Store p-value for Assistance or interaction term
    #     pval = result.pvalues.get('Assistance[T.post]', np.nan)
    #     pvals.append(pval)
    #     results.append((metric, pval))

    # # Correction
    # reject, corrected_pvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # print("\n📊 Multiple Comparison Correction (FDR):")
    # for i, (metric, raw_p) in enumerate(results):
    #     print(f"{metric}: raw p = {raw_p:.4f}, corrected p = {corrected_pvals[i]:.4f} --> {'✅' if reject[i] else '❌'}")


    # relationship between 
    # correlation_auto_vs_human(merged, 'Total Rating', ['BLEU', 'ROUGE-L', 'METEOR'], '_post')
