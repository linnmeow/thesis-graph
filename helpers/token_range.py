import json
import os
from transformers import AutoTokenizer
import argparse
from collections import Counter

def token_count_range(directory, model_name): 
    files = ["train_article.json", "test_article.json", "valid_article.json"]
    document_token_counts = []
    summary_token_counts = []
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for filename in files:
        file_path = os.path.join(directory, filename)
        
        if not os.path.isfile(file_path):
            print(f"File {filename} does not exist in the directory.")
            continue
        
        with open(file_path, 'r') as file:
            data = json.load(file)
            for item in data:
                # cnn/dm
                document = item.get("article", "") 
                summary = item.get("highlights", "") 

                # # xsum
                # document = item.get("document", "") 
                # summary = item.get("summary", "") 
                # index = item.get("id", "")

                # check if the document or summary is empty and print the filename and index
                if not document or not summary:
                    print(f"Empty document or summary found in file {filename}")
                    continue
                
                # tokenize the document and summary
                document_encoding = tokenizer(document, return_tensors=None, truncation=True, padding=False)
                summary_encoding = tokenizer(summary, return_tensors=None, truncation=True, padding=False)
                
                # access the 'input_ids' from the tokenization output
                document_tokens = document_encoding['input_ids']
                summary_tokens = summary_encoding['input_ids']
                
                # ensure that token_ids is a list of lists
                if isinstance(document_tokens[0], int):
                    document_tokens = [document_tokens]
                if isinstance(summary_tokens[0], int):
                    summary_tokens = [summary_tokens]
                
                # flatten the token IDs if necessary and filter out special tokens
                document_token_count = len([token_id for token_id in document_tokens[0] if token_id not in tokenizer.all_special_ids])
                summary_token_count = len([token_id for token_id in summary_tokens[0] if token_id not in tokenizer.all_special_ids])
                
                document_token_counts.append(document_token_count)
                summary_token_counts.append(summary_token_count)
    
    document_token_range = (min(document_token_counts), max(document_token_counts)) if document_token_counts else (0, 0)
    summary_token_range = (min(summary_token_counts), max(summary_token_counts)) if summary_token_counts else (0, 0)

    # compute token length distribution
    def compute_distribution(token_counts):
        distribution = Counter()
        for count in token_counts:
            if count <= 512:
                distribution['0-512'] += 1
            elif count <= 768:
                distribution['513-768'] += 1
            elif count <= 1024:
                distribution['769-1024'] += 1
            else:
                distribution['1025+'] += 1
        return distribution
    
    document_token_distribution = compute_distribution(document_token_counts)
    summary_token_distribution = compute_distribution(summary_token_counts)

    return {
        "document_token_range": document_token_range,
        "summary_token_range": summary_token_range,
        "document_token_distribution": document_token_distribution,
        "summary_token_distribution": summary_token_distribution
    }

def main():
    parser = argparse.ArgumentParser(description="Process JSON files to calculate token ranges and distributions.")
    parser.add_argument('directory', type=str, help="Directory containing JSON files")
    
    args = parser.parse_args()
    
    directory_path = args.directory
    token_ranges = token_count_range(directory_path, model_name="facebook/bart-large")
    
    print(f"Document Token Range: {token_ranges['document_token_range']}")
    print(f"Summary Token Range: {token_ranges['summary_token_range']}")
    print(f"Document Token Distribution: {token_ranges['document_token_distribution']}")
    print(f"Summary Token Distribution: {token_ranges['summary_token_distribution']}")

if __name__ == "__main__":
    main()
