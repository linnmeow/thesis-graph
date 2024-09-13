import json
import argparse
import os

def calculate_total_triples(file_path):
    """Calculates the total number of triples and the total number of entries in a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0  # returning zero to avoid affecting the final average calculation

    total_triples = 0
    num_entries = len(data)

    # calculate total number of triples
    for entry in data:
        triples = entry.get("triples", [])
        total_triples += len(triples)

    return total_triples, num_entries

def main():
    parser = argparse.ArgumentParser(description="Calculate the average number of triples across multiple JSON files.")
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='JSON files to process')
    args = parser.parse_args()

    combined_total_triples = 0
    combined_num_entries = 0

    for file_path in args.files:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        total_triples, num_entries = calculate_total_triples(file_path)
        combined_total_triples += total_triples
        combined_num_entries += num_entries

    # Calculate the combined average
    if combined_num_entries == 0:
        print("No entries found in the provided files.")
    else:
        combined_average_triples = combined_total_triples / combined_num_entries
        print(f"Combined average number of triples across all files: {combined_average_triples:.2f}")

if __name__ == "__main__":
    main()
