import json
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

# nltk.download('punkt')

def compute_fragments(source_text, summary_text):
    # tokenize the source document and summary
    source_tokens = word_tokenize(source_text.lower())
    summary_tokens = word_tokenize(summary_text.lower())
    
    # create token-to-index mappings for quick lookup
    source_token_indices = {token: idx for idx, token in enumerate(source_tokens)}
    
    # initialize fragment length lists
    fragment_lengths = []
    current_fragment_length = 0
    
    for token in summary_tokens:
        if token in source_token_indices:
            if current_fragment_length == 0:
                current_fragment_length = 1
            else:
                current_fragment_length += 1
        else:
            if current_fragment_length > 0:
                fragment_lengths.append(current_fragment_length)
                current_fragment_length = 0
    
    if current_fragment_length > 0:
        fragment_lengths.append(current_fragment_length)
    
    return fragment_lengths

def compute_coverage_and_density(source_text, summary_text):
    fragment_lengths = compute_fragments(source_text, summary_text)
    
    # total number of words in the summary
    summary_word_count = len(word_tokenize(summary_text.lower()))
    
    # coverage: Percentage of words in the summary that are part of an extractive fragment
    coverage = sum(fragment_lengths) / summary_word_count if summary_word_count > 0 else 0
    
    # density: Average length of the extractive fragment squared
    density = (sum(length ** 2 for length in fragment_lengths) / summary_word_count
               if summary_word_count > 0 else 0)
    
    return coverage, density

def calculate_average_coverage_and_density(json_file_path):
    coverage_list = []
    density_list = []
    
    # load the dataset from the JSON file
    with open(json_file_path, 'r') as file:
        dataset = json.load(file)
    
    # process each instance in the dataset
    for instance in dataset:
        source_text = instance.get('text', '')
        summary_text = instance.get('summary', '')

        # source_text = instance.get('article', '')
        # summary_text = instance.get('highlights', '')
        
        coverage, density = compute_coverage_and_density(source_text, summary_text)
        
        coverage_list.append(coverage)
        density_list.append(density)
    
    # calculate average coverage and density
    avg_coverage = sum(coverage_list) / len(coverage_list) if coverage_list else 0
    avg_density = sum(density_list) / len(density_list) if density_list else 0
    
    return avg_coverage, avg_density

average_coverage, average_density = calculate_average_coverage_and_density('/Users/lynn/Desktop/thesis/booksum_paragraph-level-summary-alignments/article_collections/train_article.json')
print(f"Average Coverage: {average_coverage:.2f}")
print(f"Average Density: {average_density:.2f}")
