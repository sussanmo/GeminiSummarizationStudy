# Gemini Summarization Study

This repository contains data, scripts, and analyses for our study investigating how novice programmers engage with Large Language Models (LLMs) like Google Gemini during code summarization tasks. Specifically, we explore how AI assistance affects code reading behavior, visual attention patterns, and summarization quality.

## 📁 Repository Structure

- `analysis/`: Includes all post-processing analysis, such as visual attention analysis (`aoi_analysis/`), semantic category engagement (`semantic_analysis/`), Gemini interaction analysis (`gemini_analysis/`), and summary scoring (`summary_analysis/`). This also includes cleaned participant-level data in `CleanedParticipantData.xlsx`.
  
- `pre-processing-scripts/`: Scripts for preparing and filtering the dataset prior to analysis.
  - `ast_abstraction.py`: Extracts abstract syntax tree features using `srcML`.
  - `category_distribution.py`: Computes frequency of semantic categories.
  - `data_filtering.py`: Filters methods and participant responses based on task design constraints.
  
- `task_design_data/`: Contains all materials related to the design of the summarization tasks.
 - Includes filtered and final versions of code methods used in the study.
  - `semantic_categories_final.json`: The semantic categories used in analysis, derived from `srcML` annotations of Java methods (Karas et al., 2024).

- `gaze_data/`: Eye-tracking data captured during participant tasks, used to compute fixations and scanpaths.

- `code-docstring-corpus/`: Original source of code and natural language summaries. Based on the dataset by Barone and Sennrich (2017).
  - This dataset contains both parallel and code-only corpora used for task generation.

## References

- Karas, Z., Bansal, A., Zhang, Y., Li, T., McMillan, C., & Huang, Y. (2024). *A tale of two comprehensions? Analyzing student programmer attention during code summarization*. ACM Transactions on Software Engineering and Methodology, 33(7), 1–37.

- Barone, A. V. M., & Sennrich, R. (2017). *A parallel corpus of Python functions and documentation strings for automated code documentation and code generation*. arXiv preprint arXiv:1707.02275.

