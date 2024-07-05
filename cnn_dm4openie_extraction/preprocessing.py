import json
import os

def filter_by_confidence_score(triples, threshold):
    return [triple for triple in triples if float(triple[0]) >= threshold]

def filter_by_argument_length(triples, min_length):
    return [triple for triple in triples if len(triple[1]) >= min_length and len(triple[2]) >= min_length and len(triple[3]) >= min_length]

def filter_overlapping(triples):
    def is_overlapping(triple1, triple2):
        return (triple1[1] == triple2[1] and triple1[2] == triple2[2]) or \
               (triple1[1] == triple2[1] and triple1[3] == triple2[3]) or \
               (triple1[2] == triple2[2] and triple1[3] == triple2[3])

    sorted_triples = sorted(triples, key=lambda x: (x[1], x[2], x[3]))

    filtered_triples = []
    for i in range(len(sorted_triples)):
        if i == 0 or not is_overlapping(sorted_triples[i], sorted_triples[i - 1]):
            filtered_triples.append(sorted_triples[i])
        elif len(sorted_triples[i][1] + sorted_triples[i][2] + sorted_triples[i][3]) > len(sorted_triples[i - 1][1] + sorted_triples[i - 1][2] + sorted_triples[i - 1][3]):
            filtered_triples[-1] = sorted_triples[i]

    return filtered_triples

def process_entry(entry, confidence_threshold, argument_length_threshold):
    """
    processes a single dataset entry by applying the filtering functions to its triples.
    
    parameters:
    - entry (dict): a single dataset entry containing 'article', 'highlights', 'id', and 'triples'.
    - confidence_threshold (float): the confidence score threshold.
    - argument_length_threshold (int): the minimum length of the arguments.
    
    returns:
    - dict: the processed dataset entry.
    """
    triples = entry['triples']
    triples = filter_by_confidence_score(triples, confidence_threshold)
    triples = filter_by_argument_length(triples, argument_length_threshold)
    triples = filter_overlapping(triples)
    entry['triples'] = triples
    return entry

def process_dataset(data, confidence_threshold, argument_length_threshold):
    """
    processes the entire dataset by applying the filtering functions to each entry's triples.
    
    parameters:
    - data (list of dict): The dataset containing multiple entries.
    - confidence_threshold (float): The confidence score threshold.
    - argument_length_threshold (int): The minimum length of the arguments.
    
    returns:
    - list of dict: The processed dataset.
    """
    return [process_entry(entry, confidence_threshold, argument_length_threshold) for entry in data]

def preprocessing(data_dir, file_prefixes, confidence_threshold, argument_length_threshold):
    """
    reads train, validation, and test datasets from JSON files, processes them,
    and saves the processed datasets to separate JSON files.
    
    parameters:
    - data_dir (str): directory where the data files are located.
    - file_prefixes (list of str): list of prefixes for the train, val, and test files.
    - confidence_threshold (float): the confidence score threshold.
    - argument_length_threshold (int): the minimum length of the arguments.
    """
    for prefix in file_prefixes:
        # construct file paths
        input_path = os.path.join(data_dir, f"{prefix}_processed.json")
        output_path = os.path.join(data_dir, f"{prefix}_filtered.json")
        
        # read data from JSON file
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # process the dataset
        processed_data = process_dataset(data, confidence_threshold, argument_length_threshold)
        
        # save the processed dataset to a JSON file
        with open(output_path, "w") as f:
            json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    data_directory = "/Users/lynn/desktop/thesis/cnn_dm4openie_extraction/original_triples"  
    prefixes = ["train", "valid", "test"]  
    
    preprocessing(
        data_dir=data_directory,
        file_prefixes=prefixes,
        confidence_threshold=0.6,
        argument_length_threshold=3
    )
