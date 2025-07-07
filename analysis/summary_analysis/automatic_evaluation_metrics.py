

import nltk.data
# nltk.data.path.append('/Users/suadhm/nltk_data')

import nltk
nltk.download('punkt')
nltk.download('wordnet')  # For METEOR
# from nltk.tokenize import word_tokenize

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import pandas as pd


# Functions
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import pandas as pd
import nltk

# # Download if needed
# nltk.download('punkt')
# nltk.download('wordnet')


def compute_metrics(summary, reference, row_idx=None):
    if not isinstance(summary, str) or not isinstance(reference, str):
        print(f"⚠️ Row {row_idx}: Invalid summary or reference.")
        return None, None, None

    try:
        ref_tokens = [nltk.word_tokenize(reference)]
        pred_tokens = nltk.word_tokenize(summary)

        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        rouge_score = rouge.score(summary, reference)['rougeL'].fmeasure
        meteor = single_meteor_score(reference, summary)

        print(f"\n📄 Row {row_idx}")
        print(f"Summary: {summary[:100]}...")
        print(f"Reference: {reference[:100]}...")
        print(f"→ BLEU: {bleu:.4f}, ROUGE-L: {rouge_score:.4f}, METEOR: {meteor:.4f}")

        return bleu, rouge_score, meteor

    except Exception as e:
        print(f"❌ Row {row_idx}: Error computing metrics - {e}")
        return None, None, None



if __name__ == '__main__':
    # print(nltk.data.path)
    # text = "This is a test."
    # print(word_tokenize(text))

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

    print("✅ Evaluation metrics computed and saved.")
