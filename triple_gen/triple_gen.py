import subprocess
import json
import os
from pycorenlp import StanfordCoreNLP

def generate_triples(article):

    command = ["java",
    "-mx32g",
    "-cp",
    "/Users/lynn/desktop/thesis/stanford-corenlp-4.5.6/stanford-corenlp-4.5.6.jar:/Users/lynn/desktop/thesis/stanford-corenlp-4.5.6/stanford-corenlp-4.5.6-models.jar:/Users/lynn/desktop/thesis/stanford-corenlp-4.5.6/CoreNLP-to-HTML.xsl:/Users/lynn/desktop/thesis/stanford-corenlp-4.5.6/slf4j-api.jar:/Users/lynn/desktop/thesis/stanford-corenlp-4.5.6/slf4j-simple.jar",
    "edu.stanford.nlp.naturalli.OpenIE",
    "-threads",
    "8",
    "-resolve_coref",
    "true",
    "-ssplit.newlineIsSentenceBreak",
    "always"]
    # "-format",
    # "reverb"]

    # execute the command using subprocess
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    input_text = f"{article}"
    output, _ = p.communicate(input_text)

    # parse the output and filter triples with confidence scores under 0.6
    triples = []
    lines = output.strip().split('\n')
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 4:
            confidence_score = parts[0]
            subject = parts[1]
            relation = parts[2]
            object = parts[3]
            triples.append((confidence_score, subject, relation, object))
    return triples

def process_dataset(dataset):
    """
    append generated triples with the original articles
    """
    for data in dataset:
        article = data["article"]
        triples = generate_triples(article)
        data["triples"] = triples
    return dataset

def read_and_save_datasets(data_dir, file_prefixes):
    """
    reads train, validation, and test datasets from JSON files, processes them,
    and saves the processed datasets to separate JSON files.
    
    parameters:
    - data_dir (str): directory where the data files are located.
    - file_prefixes (list of str): list of prefixes for the train, val, and test files.
    - process_function (function): function to process the datasets.
    """
    for prefix in file_prefixes:
        # construct file paths
        input_path = os.path.join(data_dir, f"{prefix}_article.json")
        output_path = os.path.join(data_dir, f"{prefix}_processed.json")
        
        # read data from JSON file
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # process the dataset
        processed_data = process_dataset(data)
        
        # save the processed dataset to a JSON file
        with open(output_path, "w") as f:
            json.dump(processed_data, f, indent=4)


if __name__ == "__main__":

    data_directory = "/Users/lynn/desktop/thesis/cnn_dm4openie_extraction/article_collections_large" 
    prefixes = ["train", "valid", "test"]  
    
    read_and_save_datasets(
        data_dir=data_directory,
        file_prefixes=prefixes
    )

