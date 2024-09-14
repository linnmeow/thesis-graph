import json
import argparse

def convert_json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        data = json.load(infile)
        for idx, entry in enumerate(data):
            jsonl_entry = {
                "id": str(idx + 1),
                "text": entry['source_text'],
                "claim": entry['generated_summary']
            }
            outfile.write(json.dumps(jsonl_entry) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert JSON file to JSONL format.')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('output_file', type=str, help='Path to the output JSONL file')
    args = parser.parse_args()

    convert_json_to_jsonl(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
