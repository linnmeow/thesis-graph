import argparse
import json
import os
from evaluate import load
import evaluate
from FENICE import FENICE  

# function to open and load the JSON file
def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_article.json")
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

# function to evaluate generated summaries with the specified metric
def evaluate_generated_summaries(dataset, metric_name):
    # initialize the evaluation metric
    if metric_name == "fenice":
        fenice = FENICE()  # Initialize FENICE directly
    else:
        metric = evaluate.load(metric_name, trust_remote_code=True)

    # extract generated summaries and reference summaries from the dataset
    summaries = [entry["generated_summary"] for entry in dataset]
    references = [entry["reference_summary"] for entry in dataset]

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
    
    elif metric_name == "fenice":
        # FENICE uses a different structure, create a batch for each sample
        fenice_scores = []
        for summary, reference in zip(summaries, references):
            batch = [{"document": reference, "summary": summary}]
            score = fenice.score_batch(batch)
            fenice_scores.append({
                'score': score[0]['score'],
                'alignments': score[0]['alignments']
            })
        return fenice_scores
    
    else:
        score = metric.compute(predictions=summaries, references=references)
        return score

# main function to handle argument parsing and evaluation
def main(data_dir, file_prefix, metric_name):
    data = open_json_file(data_dir, file_prefix)
    score = evaluate_generated_summaries(data, metric_name)
    print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating generated summaries with different metrics")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--file_prefix', type=str, required=True, help='Prefix for the dataset files')
    parser.add_argument('--metric_name', type=str, required=True, choices=["rouge", "bleu", "meteor", "bertscore", "fenice"], help='Evaluation metric to use')

    args = parser.parse_args()

    main(args.data_dir, args.file_prefix, args.metric_name)
