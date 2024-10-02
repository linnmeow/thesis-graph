import json
import argparse
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

def calculate_ngram_overlap(original_text, summary, n=1):
    """
    Calculates the ratio of unique n-grams in the summary that are not present
    in the original text.
    
    Args:
        original_text (str): The original reference text.
        summary (str): The summary (can be generated or reference).
        n (int): The 'n' in n-gram (default is unigram).
    
    Returns:
        float: The originality score (ratio of unique n-grams).
    """
    # tokenize the text
    original_tokens = word_tokenize(original_text.lower())
    summary_tokens = word_tokenize(summary.lower())
    
    # generate n-grams for original and summary text
    original_ngrams = list(ngrams(original_tokens, n))
    summary_ngrams = list(ngrams(summary_tokens, n))
    
    # find unique n-grams in summary (not in original)
    unique_ngrams = [ngram for ngram in summary_ngrams if ngram not in original_ngrams]
    
    # calculate the originality score as the ratio of unique n-grams to total n-grams in the summary
    if len(summary_ngrams) == 0:
        return 0  # Avoid division by zero
    
    originality_score = len(unique_ngrams) / len(summary_ngrams)
    
    return originality_score

def evaluate_summaries(json_file_path, n=1):
    """
    Evaluates the originality of generated and reference summaries in a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file with original texts and summaries.
        n (int): The 'n' in n-gram (default is unigram).
    
    Returns:
        tuple: (average originality score for generated summaries, average originality score for reference summaries).
    """
    # load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    generated_originality_scores = []
    reference_originality_scores = []
    
    # iterate over each entry in the JSON file
    for entry in data:
        original_text = entry['source_text']
        generated_summary = entry['generated_summary']
        reference_summary = entry['reference_summary']
        
        # calculate originality score for generated summary
        generated_score = calculate_ngram_overlap(original_text, generated_summary, n)
        generated_originality_scores.append(generated_score)
        
        # calculate originality score for reference summary (if it exists)
        if reference_summary:
            reference_score = calculate_ngram_overlap(original_text, reference_summary, n)
            reference_originality_scores.append(reference_score)
    
    # calculate the average originality score for generated summaries
    avg_generated_originality_score = sum(generated_originality_scores) / len(generated_originality_scores)
    
    # calculate the average originality score for reference summaries, if available
    if reference_originality_scores:
        avg_reference_originality_score = sum(reference_originality_scores) / len(reference_originality_scores)
    else:
        avg_reference_originality_score = 0  # Return 0 if no reference summaries are available
    
    return avg_generated_originality_score, avg_reference_originality_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate the originality of generated and reference summaries by n-gram overlap.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing original texts and summaries.")
    parser.add_argument("--ngram", type=int, default=1, help="The 'n' in n-gram (default: 1 for unigram).")
    
    args = parser.parse_args()
    
    avg_generated_score, avg_reference_score = evaluate_summaries(args.json_file, args.ngram)
    
    print(f"Average originality score for generated summaries: {avg_generated_score:.4f}")
    print(f"Average originality score for reference summaries: {avg_reference_score:.4f}")

if __name__ == "__main__":
    main()
