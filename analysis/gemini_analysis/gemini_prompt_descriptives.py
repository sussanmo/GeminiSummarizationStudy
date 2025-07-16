import pandas as pd

import pandas as pd
from scipy.stats import chi2_contingency

# 1. Define your experience classification function
def classify_experience(range_str):
    if pd.isna(range_str):
        return None
    if "Less" in range_str or "1–2" in range_str or "1-2" in range_str:
        return "Low"
    if "2–4" in range_str or "2-4" in range_str or "4+" in range_str:
        return "High"
    return None

# 2. Apply experience classification
def apply_experience_classification(df, experience_col='Experience'):
    
    df['Experience'] = df[experience_col].apply(classify_experience)
    return df

def apply_category_mapping(df, category_col='Final'):
    df[category_col] = df[category_col].str.strip()  # clean whitespace
    df['MappedCategory'] = df[category_col].map(CATEGORY_MAP)
    # print(df['MappedCategory'])
    return df

# 3. Run chi-square test
def chi_square_experience_vs_category(df, experience_col='Experience', category_col='Final'):
    # Drop rows with missing values in required columns
    df_clean = df.dropna(subset=[experience_col, category_col])

    # Create a contingency table
    contingency_table = pd.crosstab(df_clean[category_col], df_clean[experience_col])
    print("\nContingency Table:\n", contingency_table)

    # Perform Chi-Square Test of Independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print("\nExpected Frequencies:\n", pd.DataFrame(expected, 
          index=contingency_table.index, 
          columns=contingency_table.columns))
    
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"p-value: {p:.4f}")
    
    return chi2, p, dof, contingency_table, expected

def aggregate_category_counts(df, participant_col='participant', experience_col='Experience', category_col='MappedCategory'):
    # Drop rows with missing experience or category
    df = df.dropna(subset=[experience_col, category_col, participant_col])
    
    # Remove duplicates so each participant-category-experience combo is counted once
    unique_rows = df[[participant_col, experience_col, category_col]].drop_duplicates()
    
    # Create contingency table: counts of unique participants by category (rows) and experience (columns)
    contingency_table = unique_rows.groupby([category_col, experience_col])[participant_col].nunique().unstack(fill_value=0)
    
    # print("Contingency Table (Participants per Category × Experience):")
    # print(contingency_table)
    
    return contingency_table


# Run chi-square test
def chi_square_test(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square test result:")
    print(f"Chi2 statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.4f}")
    return chi2, p, dof, expected

def get_most_common_category_per_participant(df, participant_col='participant', category_col='MappedCategory'):
    # Count how often each participant sees each category
    category_counts = df.groupby([participant_col, category_col]).size().reset_index(name='count')
    
    # Get the most frequent category for each participant
    idx = category_counts.groupby(participant_col)['count'].idxmax()
    most_common = category_counts.loc[idx].reset_index(drop=True)

    return most_common[[participant_col, category_col]]

def prepare_chi_square_dominant_categories(df):
    df = df.dropna(subset=['participant', 'Experience', 'MappedCategory'])
    dominant = get_most_common_category_per_participant(df)
    merged = pd.merge(dominant, df[['participant', 'Experience']].drop_duplicates(), on='participant')
    
    contingency_table = pd.crosstab(merged['MappedCategory'], merged['Experience'])
    return contingency_table

# get unique categories
def load_and_get_freq_data(llm_path):
   
    llm_prompts = pd.read_excel(llm_path)
    llm_prompts.columns = llm_prompts.columns.str.strip()
    
    # print(f"LLM Columns {llm_prompts.columns}")
   
    llm_prompts['Final'] = llm_prompts['Final'].astype(str)
    # print(f"LLM Columns {llm_prompts['Final'] }")
    # print(llm_prompts[['Final']].head(10))

    # ✅ Get frequency of each unique category in the Final column
    category_counts = llm_prompts['Final'].value_counts()
    print("\n🔢 Frequency of each category in 'Final':")
    print(category_counts)

    # Compute new Total Rating as average of the 3 sub-metrics
    # human_metrics = ['Accuracy:', 'Readability:', 'Cohesion:']
    # pre['Total Rating'] = pre[human_metrics].mean(axis=1)
    # post['Total Rating'] = post[human_metrics].mean(axis=1)

    print(f"✅")

# clean file and map to years of experience

def standardize_task_format(df, task_col='Task'):
    df[task_col] = df[task_col].astype(str).str.replace('_', '.', regex=False)
    return df
import pandas as pd

# Merge column by participant and expertise for later stats analysis 
def map_prompts_to_experience(participant_df_path, llm_prompt_df_path):
    # Load participant and prompt data
    participant_df = pd.read_excel(participant_df_path)
    llm_prompt_df  = pd.read_excel(llm_prompt_df_path)

    # Clean column names
    participant_df.columns = participant_df.columns.str.strip()
    llm_prompt_df.columns = llm_prompt_df.columns.str.strip()

    # print(participant_df.columns)
    # print(llm_prompt_df.columns)

    # print(participant_df['Gemini'])
    # print(llm_prompt_df['Task'])
    # Extract numeric task ID (e.g., "6_3" → "6")
    # Extract numeric task ID prefix from 'Task' (e.g., "6_3" -> "6")
    llm_prompt_df['Task'] = llm_prompt_df['Task'].astype(str).str.extract(r'(\d+)')[0]

    # Convert Gemini to int then str (e.g., 1.0 -> "1")
    participant_df['Gemini'] = participant_df['Gemini'].dropna().astype(int).astype(str)


    # Merge on Task_ID and Gemini columns
    merged_df = pd.merge(
        llm_prompt_df,
        participant_df[['Gemini', 'participant', 'Q4']],  # Replace with correct col names
        left_on='Task',
        right_on='Gemini',
        how='left'
    )

    # merged_df.rename(columns={'Q4': 'Experience'}, inplace=True)

    # Optional: Rename Q4 to something clearer like "Experience"
    merged_df.rename(columns={'Q4': 'Experience'}, inplace=True)

    # ✅ Check if the mapping worked correctly
    # print("Sample of merged experience (Q4 column):")
    # print(merged_df[['Task', 'Gemini', 'Experience']].drop_duplicates().head())

    return merged_df

def descriptive_by_experience_category(df, participant_col='participant', experience_col='Experience', category_col='MappedCategory'):
    # Drop missing values in relevant columns
    df = df.dropna(subset=[participant_col, experience_col, category_col])

    # Count of participants per experience group
    participant_counts = df.groupby(experience_col)[participant_col].nunique().sort_index()
    # print("Participants per Experience Group:")
    # print(participant_counts)
    # print()

    # Count of each category within each experience group (counting occurrences)
    category_counts = df.groupby([experience_col, category_col]).size().unstack(fill_value=0)
    # print("Category Counts per Experience Group:")
    # print(category_counts.to_string())
    # print()

    # Count of unique participants per category per experience group
    unique_participants = df.groupby([experience_col, category_col])[participant_col].nunique().unstack(fill_value=0)
    # print("Unique Participants per Category per Experience Group:")
    # print(unique_participants.to_string())

    print()

    return participant_counts, category_counts, unique_participants

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_dominant_category_by_experience(contingency_table):
    # Set seaborn style and pastel palette
    sns.set(style="whitegrid", font_scale=1.2)
    # Alternative: soft, distinct colors, great for print and presentations
    pastel_colors = [
    "#AEC6CF",  # pastel blue
    "#FAA0A0",  # pastel orange
    "#B5EAD7",  # pastel teal
    "#77DD77",  # pastel green
    "#F49AC2",  # pastel pink
    "#CBAACB",  # pastel purple
    "#FFD1DC",  # pastel coral

 
]
    # Limit to number of categories if needed
    pastel_colors = pastel_colors[:contingency_table.shape[0]]

    plt.figure(figsize=(12, 6))

    # Transpose for plotting (experience groups on x-axis)
    data = contingency_table.T
    data = data[sorted(data.columns)]  # Sort categories alphabetically

    # Plot
    ax = data.plot(
        kind='bar',
        stacked=True,
        color=pastel_colors,
        width=0.7,
        edgecolor='gray',
        figsize=(12, 6)
    )

    # Title and axes labels
    ax.set_title("Most Frequent Prompt Category by Experience Group", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Experience Group", fontsize=14)
    ax.set_ylabel("Number of Participants", fontsize=14)

    # Ticks
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # Legend inside plot (upper right corner)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Prompt Category",
        title_fontsize=12,
        fontsize=10,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )

    plt.tight_layout()
    plt.show()

def plot_total_category_descriptive():
    # Convert wide format to long format for plotting
    category_counts_long = (
        category_counts
        .reset_index()
        .melt(id_vars='Experience', var_name='Category', value_name='Count')
    )

    print(category_counts_long )

    # Use whitegrid style for clean background
    sns.set(style="whitegrid")

    # Define a pastel palette with distinct soft colors
    

    pastel_colors = [
    "#AEC6CF",  # pastel blue
    "#FAA0A0",  # pastel orange
    "#77DD77",  # pastel green
    "#B5EAD7",  # pastel teal
    "#F49AC2",  # pastel pink
    "#CBAACB",  # pastel purple
    "#FFD1DC",  # pastel coral
    
   
    ]
    # Plot
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(
        data=category_counts_long,
        x='Category',
        y='Count',
        hue='Experience',
        palette=pastel_colors
    )

    # Rotate x labels for readability
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45, ha='right')

    # Titles and axis labels with professional font sizes
    plt.title('Distribution of Category Counts Across Experience Levels', fontsize=16, weight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    plt.tight_layout()
    plt.show()

# calculate cohen kappa 

# run chi square on 
    

if __name__ == '__main__':

    load_and_get_freq_data('analysis/gemini_analysis/LLM_Interaction_Analysis.xlsx')
    # get unique categories
    # get frequencies for descriptive table in latex
    CATEGORY_MAP = {
    "Summary Request": "Summary Request",
    "Summary Request ": "Summary Request",  # with trailing space
    "Code Explanation": "Code Explanation",
    "Clarification snippet of code": "Clarification snippet of code",
    "Summary Revision": "Summary Revision",
    "Summary Feedback/Evaluation": "Summary Feedback/Evaluation",
    "Summary Feedback": "Summary Feedback/Evaluation",  # map to broader category
    "Request formatting output change": "Request formatting output change",
    "Code Generation from Summary": "Code Generation from Summary",
    "syntax clarification": "Syntax Clarification",
    "Syntax Clarification": "Syntax Clarification",
    "Task Clarification": "Task Clarification",
    "Python Syntax Explanation": "Syntax Clarification",  # similar meaning
    "nan": None,
    None: None,
    }

    # Step 1: Load and merge participant and LLM prompt data
    merged_df = map_prompts_to_experience(
        'analysis/CleanedParticipantData.xlsx',
        'analysis/gemini_analysis/LLM_Interaction_Analysis.xlsx'
    )

    # Step 2: Apply experience classification (Low vs High)
    # merged_df = apply_experience_classification(merged_df, experience_col='Experience')

    merged_df = apply_category_mapping(merged_df, category_col='Final')


    # Step 3: Apply standardized category mapping to prompt categories
    # merged_df = apply_category_mapping(merged_df, category_col='Final')

    # # Chi-square test
    # chi2, p, dof, table, _ = chi_square_experience_vs_category(
    #     merged_df,
    #     experience_col='Experience',
    #     category_col='MappedCategory'
    # )

    # print(merged_df[[ 'participant', 'Experience', 'MappedCategory']].dropna().head(20))
    
    # clean file and map to years of experience
    # contingency_table = aggregate_category_counts(
    # df=merged_df,  # your merged dataframe with experience and category columns
    # participant_col='participant',
    # experience_col='Experience',
    # category_col='MappedCategory'  # or 'Final' mapped to your final categories
    # )

    
    # chi2, p, dof, expected = chi2_contingency(contingency_table)
    # print(f"Chi-square statistic: {chi2}, p-value: {p}")


    # Usage example:
    participant_counts, category_counts, unique_participants = descriptive_by_experience_category(
        merged_df,
        participant_col='participant',
        experience_col='Experience',
        category_col='MappedCategory'
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    

    # # Prepare data for plotting unique participants
    # unique_participants_long = unique_participants.reset_index().melt(id_vars='Experience', 
    #                                                                 var_name='Category', 
    #                                                                 value_name='UniqueParticipants')

    # # Plot unique participants counts
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=unique_participants_long, x='Category', y='UniqueParticipants', hue='Experience')
    # plt.xticks(rotation=45, ha='right')
    # plt.title('Unique Participants per Category by Experience Group')
    # plt.tight_layout()
    # plt.show()

    merged_df = apply_category_mapping(merged_df, category_col='Final')

    contingency_table = prepare_chi_square_dominant_categories(merged_df)

    plot_total_category_descriptive()
    # Step 4: Run chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\nChi-square test for dominant category vs. experience group:")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.4f}")

    # Optional: visualize
    # proportions = contingency_table.div(contingency_table.sum(axis=0), axis=1)
    # print("Proportions of dominant categories per experience group:")
    # print(proportions.round(2))


    for group in contingency_table.columns:
        print(f"\nTop dominant categories for experience group '{group}':")
        top_cats = contingency_table[group].sort_values(ascending=False).head(5)
        print(top_cats)

    # print("Contingency Table (Participants per Category × Experience):")
    # print(contingency_table)

    plot_dominant_category_by_experience(contingency_table)


        