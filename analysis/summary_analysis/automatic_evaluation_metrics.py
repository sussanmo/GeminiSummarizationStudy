

import nltk
# nltk.data.path.append('/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/venv/nltk_data')

# nltk.download('punkt', download_dir='/Users/-hm/nltk_data')

import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')  # For METEOR
from nltk.tokenize import word_tokenize

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd

from nltk.translate.meteor_score import meteor_score

# Functions
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
import nltk
import re
# # Download if needed
# nltk.download('punkt')
# nltk.download('wordnet')

def normalize_text(text):
    if not isinstance(text, str):
        return ''
    # Replace fancy quotes with normal ones
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    # Remove non-printable characters (optional)
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)
    return text.strip()

def safe_bleu(summary, reference, row_idx=None):
    try:
        ref_tokens = [word_tokenize(reference)]
        pred_tokens = word_tokenize(summary)
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        return score
    except Exception as e:
        print(f"❌ Row {row_idx}: BLEU error - {e}")
        return None

def safe_rouge(summary, reference, row_idx=None):
    try:
        score = rouge.score(summary, reference)['rougeL'].fmeasure
        return score
    except Exception as e:
        print(f"❌ Row {row_idx}: ROUGE error - {e}")
        return None

from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

def safe_meteor(summary, reference, row_idx=None):
    try:
        summary = normalize_text(summary)
        reference = normalize_text(reference)

        summary_tokens = word_tokenize(summary)
        reference_tokens = word_tokenize(reference)

        print(f"⚠️ Summary Tokens (row {row_idx}): {summary_tokens[:20]}")
        print(f"⚠️ Reference Tokens (row {row_idx}): {reference_tokens[:20]}")
        print(f"Reference input type: {type(reference_tokens)}")

        score = meteor_score([reference_tokens], summary_tokens)
        return score
    except Exception as e:
        print(f"❌ Row {row_idx}: METEOR error - {e}")
        return None

def compute_metrics(summary, reference, row_idx=None):
    # Strip quotes and whitespace to avoid tokenization issues
    if not isinstance(summary, str) or not isinstance(reference, str):
        print(f"⚠️ Row {row_idx}: Invalid summary or reference types.")
        return None, None, None

    summary = summary.strip("'\" ")
    reference = reference.strip("'\" ")

    if not summary or not reference:
        print(f"⚠️ Row {row_idx}: Empty summary or reference after stripping.")
        return None, None, None

    bleu = safe_bleu(summary, reference, row_idx)
    rouge_l = safe_rouge(summary, reference, row_idx)
    meteor = safe_meteor(summary, reference, row_idx)

    if None not in (bleu, rouge_l, meteor):
        print(f"\n📄 Row {row_idx} → BLEU: {bleu:.4f}, ROUGE-L: {rouge_l:.4f}, METEOR: {meteor:.4f}")

    return bleu, rouge_l, meteor

if __name__ == '__main__':
    # print(nltk.data.path)
    # text = "This is a test."
    # print(word_tokenize(text))

   
    # summary = "for every value in the 'values' collection, if the value starts with https:// and is debugged"
    # reference = "After settings, including DEBUG has loaded, see if we need to update CSP."

    # summary_tokens = word_tokenize(summary)
    # reference_tokens = word_tokenize(reference)

    # print("Summary tokens:", summary_tokens)
    # print("Reference tokens:", reference_tokens)

    # score = meteor_score([reference_tokens], summary_tokens)
    # print("METEOR score:", score)

    # Load your data
    predf = pd.read_excel('analysis/summary_analysis/pre_gemini_with_groundtruth.xlsx')

    # Initialize scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

   # Create empty lists to store metrics
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []

    # Loop over each row in the DataFrame
    for idx, row in predf.iterrows():
        
        summary = row['Summary']
        reference = row['GroundTruth']

        print(f"Summary - {summary}")
        print(f"Reference - {reference}")

        # break;
        bleu, rouge_l, meteor = compute_metrics(summary, reference, row_idx=idx)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)
        meteor_scores.append(meteor)

    # Assign the lists as new columns
    predf['BLEU'] = bleu_scores
    predf['ROUGE-L'] = rouge_scores
    predf['METEOR'] = meteor_scores
    # Save updated results
    predf.to_excel('analysis/summary_analysis/pre_gemini_with_metrics.xlsx', index=False)

    print("NLTK paths:", nltk.data.path)

    print("✅ Evaluation metrics computed and saved.")

    #-========= same for post gemini summaries 
    postdf = pd.read_excel('analysis/summary_analysis/post_gemini_with_groundtruth.xlsx')
    print(postdf.columns)
    # Initialize scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

   # Create empty lists to store metrics
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []

    # Loop over each row in the DataFrame
    for idx, row in postdf.iterrows():
        
        summary = row['Summary ']
        reference = row['GroundTruth']

        print(f"Summary - {summary}")
        print(f"Reference - {reference}")

        # break;
        bleu, rouge_l, meteor = compute_metrics(summary, reference, row_idx=idx)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_l)
        meteor_scores.append(meteor)

    # Assign the lists as new columns
    postdf['BLEU'] = bleu_scores
    postdf['ROUGE-L'] = rouge_scores
    postdf['METEOR'] = meteor_scores
    # Save updated results
    postdf.to_excel('analysis/summary_analysis/post_gemini_with_metrics.xlsx', index=False)

    print("NLTK paths:", nltk.data.path)

    print("✅ Evaluation metrics computed and saved.")
