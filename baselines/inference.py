import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from koila import lazy
from baseline_cnn import DocumentEncoder, SummaryDecoder, Seq2Seq  

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]['article']
        highlights = self.data[idx]['highlights']

        inputs = self.tokenizer(article, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        outputs = self.tokenizer(highlights, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        (inputs, outputs) = lazy(inputs, outputs, batch=0)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': outputs['input_ids'].squeeze(),
            'article': article,
            'highlights': highlights
        }

def data_collator(features):
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
    batch['labels'] = torch.stack([f['labels'] for f in features])
    batch['article'] = [f['article'] for f in features]
    batch['highlights'] = [f['highlights'] for f in features]
    return batch

def load_trained_model(hidden_size, output_size, device, model_path):
    """Load the trained model from the checkpoint."""
    print(f"Loading model from {model_path}...")
    encoder = DocumentEncoder('roberta-base', hidden_size).to(device)
    decoder = SummaryDecoder(hidden_size, output_size).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def test_model(model, dataloader, tokenizer, device, output_file):
    """Test the model and save the generated summaries."""
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Generate summaries
            summary_tokens = model.generate_summary(input_ids, attention_mask)
            generated_summary = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)

            # Collect the results
            for i in range(input_ids.size(0)):
                reference_summary = batch['highlights'][i]
                source_text = batch['article'][i]
                results.append({
                    'source_text': source_text,
                    'generated_summary': generated_summary,
                    'reference_summary': reference_summary
                })

            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f"Batch {batch_idx+1}/{len(dataloader)} processed.")

    # Save the results to a file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_article.json")
    print(f"Loading data from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    hidden_size = 768
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    output_size = tokenizer.vocab_size

    model_path = '/home1/s5734436/model_checkpoints/best_model.pth'  
    data_directory = "/home1/s5734436/thesis-graph/cnn_dm4openie_extraction/article_collections_large"  

    # Load the trained model
    model = load_trained_model(hidden_size, output_size, device, model_path)

    test_data = open_json_file(data_directory, "test")
    test_dataset = CustomDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

    output_file = os.path.join(data_directory, "generated_summaries.json")
    results = test_model(model, test_dataloader, tokenizer, device, output_file)

    for i in range(5):
        print(f"Generated Summary {i+1}: {results[i]['generated_summary']}")
        print(f"Reference Summary {i+1}: {results[i]['reference_summary']}")
