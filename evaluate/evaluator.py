import argparse
import json
import os
from nltk.tokenize import sent_tokenize
from evaluate import load
import nltk
import evaluate

# nltk.download('punkt')

def lead_three(text):
    return "\n".join(sent_tokenize(text)[:3])

def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_article.json")
    with open(input_path, "r") as f:
        data = json.load(f)
        # print(data)
    return data

def evaluate_baseline(dataset, metric_name):
    # initialize the evaluation metric
    metric = evaluate.load(metric_name, trust_remote_code=True)
    
    # extract summaries and references from the dataset
    summaries = [lead_three(entry["article"]) for entry in dataset]
    references = [entry["highlights"] for entry in dataset]
    
    # compute the metric score
    if metric_name == "rouge":
        score = metric.compute(predictions=summaries, references=references)
        print(score)
        rouge_dict = {key: round(value * 100, 2) for key, value in score.items()}
        return rouge_dict
    elif metric_name == "bertscore":
        score = metric.compute(predictions=summaries, references=references, lang="en")
        # BERTScore returns precision, recall, and F1 as separate lists, average them
        bertscore_dict = {
            "precision": round(sum(score["precision"]) / len(score["precision"]) * 100, 2),
            "recall": round(sum(score["recall"]) / len(score["recall"]) * 100, 2),
            "f1": round(sum(score["f1"]) / len(score["f1"]) * 100, 2),
        }
        return bertscore_dict
    else:
        score = metric.compute(predictions=summaries, references=references)
        return score

def main(data_dir, file_prefix, metric_name):
    data = open_json_file(data_dir, file_prefix)
    score = evaluate_baseline(data, metric_name)
    print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating baseline summaries with different metrics")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--file_prefix', type=str, required=True, help='Prefix for the dataset files')
    parser.add_argument('--metric_name', type=str, required=True, choices=["rouge", "bleu", "meteor", "bertscore"], help='Evaluation metric to use')

    args = parser.parse_args()

    main(args.data_dir, args.file_prefix, args.metric_name)
