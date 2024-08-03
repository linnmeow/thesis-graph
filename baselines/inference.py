import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torch.nn as nn
from model.summ import Seq2SeqSumm
from utils import END

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

def load_model(model_path, device):
    emb_dim = 768
    hidden_size = 768
    n_layer = 2
    bidirectional = True
    dropout = 0.2
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vocab_size = tokenizer.vocab_size
    
    model = Seq2SeqSumm(vocab_size, emb_dim, hidden_size, bidirectional, n_layer, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def test_model(model, dataloader, tokenizer, device, output_file):
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            art_lens = (attention_mask != 0).sum(dim=1)  # Calculate article lengths

            # Generate summaries
            summary_tokens_batch, _ = model.batch_decode(input_ids, art_lens, tokenizer.cls_token_id, tokenizer.sep_token_id, max_len=512)

            # summary_tokens_batch has shape (max_len, batch_size)
            # Transpose to get (batch_size, max_len)
            summary_tokens_batch = list(map(list, zip(*summary_tokens_batch)))

            # Iterate over the batch
            for i in range(len(summary_tokens_batch)): 
                summary_tokens = summary_tokens_batch[i]  # Get the tokens for this batch element
                generated_summary = tokenizer.decode(summary_tokens, skip_special_tokens=True)
                reference_summary = batch['highlights'][i]
                source_text = batch['article'][i]
                results.append({
                    'source_text': source_text,
                    'generated_summary': generated_summary,
                    'reference_summary': reference_summary
                })

            if batch_idx % 10 == 0:
                print(f"Test Batch {batch_idx}/{len(dataloader)}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_article.json")
    print(f"Loading data from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model_path = '/home1/s5734436/best_model_baseline.pth'
    model = load_model(model_path, device)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    data_directory = "/home1/s5734436/thesis-graph/cnn_dm4openie_extraction/article_collections" 

    test_data = open_json_file(data_directory, "test")
    test_dataset = CustomDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)

    output_file = os.path.join(data_directory, "generated_summaries_baseline.json")
    results = test_model(model, test_dataloader, tokenizer, device, output_file)

    for i in range(5):
        print(f"Source Text {i+1}: {results[i]['source_text']}")
        print(f"Generated Summary {i+1}: {results[i]['generated_summary']}")
        print(f"Reference Summary {i+1}: {results[i]['reference_summary']}")

if __name__ == "__main__":
    main()
