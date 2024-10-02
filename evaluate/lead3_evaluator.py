import argparse
import json
import os
import nltk
from nltk.tokenize import sent_tokenize
from evaluate import load
from evaluator import evaluate_generated_summaries  

def lead_three(text):
    return "\n".join(sent_tokenize(text)[:3])

def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_article.json")
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

def evaluate_baseline(dataset, metric_name):
    # generate baseline summaries by taking the lead three sentences
    summaries = [lead_three(entry["article"]) for entry in dataset]
    
    # replace the "generated_summary" with lead summaries and "reference_summary" with highlights
    baseline_dataset = [
        {"generated_summary": summary, "reference_summary": entry["highlights"]}
        for summary, entry in zip(summaries, dataset)
    ]
    
    score = evaluate_generated_summaries(baseline_dataset, metric_name)
    return score

def main(data_dir, file_prefix, metric_name):
    data = open_json_file(data_dir, file_prefix)
    score = evaluate_baseline(data, metric_name)
    print(score)

if __name__ == "__main__":

    # nltk.download('punkt')
    parser = argparse.ArgumentParser(description="Evaluating baseline summaries with different metrics")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--file_prefix', type=str, required=True, help='Prefix for the dataset files')
    parser.add_argument('--metric_name', type=str, required=True, choices=["rouge", "bleu", "meteor", "bertscore", "fenice"], help='Evaluation metric to use')

    args = parser.parse_args()

    main(args.data_dir, args.file_prefix, args.metric_name)
