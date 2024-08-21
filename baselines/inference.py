import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import Dataset

class CNN_DM(Dataset):
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

def load_trained_model(model_path, bart_model_name, device):
    model = BartForConditionalGeneration.from_pretrained(bart_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # Use strict=False
    model.to(device)
    model.eval()
    return model

def generate_summaries(model, dataloader, tokenizer, device, output_file):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            summary_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                min_length=56,
                max_length=142,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            for i in range(input_ids.size(0)):
                generated_summary = tokenizer.decode(summary_tokens[i], skip_special_tokens=True)
                reference_summary = batch['highlights'][i]
                source_text = batch['article'][i]

                results.append({
                    'source_text': source_text,
                    'generated_summary': generated_summary,
                    'reference_summary': reference_summary
                })

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)} processed.")

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
    
    bart_model_name = 'facebook/bart-large'
    tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    
    model_save_path = '/local/linye/thesis-graph/baselines/model_checkpoints/best_model.pth'
    data_directory = "/local/linye/thesis-graph/cnn_dm4openie_extraction/article_collections_large/"
    output_file = os.path.join(data_directory, "generated_summaries.json")

    model = load_trained_model(model_save_path, bart_model_name, device)

    test_data = open_json_file(data_directory, "test")
    test_dataset = CNN_DM(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)

    results = generate_summaries(model, test_dataloader, tokenizer, device, output_file)

    for i in range(5):
        print(f"Generated Summary {i+1}: {results[i]['generated_summary']}")
        print(f"Reference Summary {i+1}: {results[i]['reference_summary']}")