import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torch.nn as nn
import torch.optim as optim
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
            'labels': outputs['input_ids'].squeeze()
        }

def data_collator(features):
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
    batch['labels'] = torch.stack([f['labels'] for f in features])
    return batch

def initialize_model(vocab_size, emb_dim, n_hidden, bidirectional, n_layer, dropout, device):
    print("Initializing model...")
    model = Seq2SeqSumm(vocab_size, emb_dim, n_hidden, bidirectional, n_layer, dropout).to(device)
    return model

def partial_train_model(model, train_loader, optimizer, criterion, num_steps, device):
    model.train()
    for step in range(num_steps):
        batch_idx, batch = next(enumerate(train_loader))
        input_ids, attention_mask, target_summary = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        art_lens = (attention_mask != 0).sum(dim=1)  # Calculate article lengths

        optimizer.zero_grad()
        output = model(input_ids, art_lens, target_summary)

        # Check for out-of-bounds indices
        vocab_size = output.size(-1)
        max_target_index = target_summary.max().item()
        if max_target_index >= vocab_size:
            raise ValueError(f"Target index out of bounds: {max_target_index} >= {vocab_size}")

        loss = criterion(output.view(-1, output.size(-1)), target_summary.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")

def debug_decoding(model, dataloader, tokenizer, device, max_len=512):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_summaries = batch['labels']
            art_lens = (attention_mask != 0).sum(dim=1)  # Calculate article lengths

            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention Mask shape: {attention_mask.shape}")
            print(f"Article lengths: {art_lens}")

            try:
                # Perform the batch decoding
                summary_tokens_batch, _ = model.batch_decode(input_ids, art_lens, tokenizer.cls_token_id, tokenizer.sep_token_id, max_len=max_len)
                
                for i in range(len(summary_tokens_batch)):
                    generated_summary = tokenizer.decode(summary_tokens_batch[i], skip_special_tokens=True)
                    reference_summary = tokenizer.decode(target_summaries[i], skip_special_tokens=True)

                    print(f"Generated Summary {i+1}: {generated_summary}")
                    print(f"Reference Summary {i+1}: {reference_summary}")

                if batch_idx % 10 == 0:
                    print(f"Test Batch {batch_idx}/{len(dataloader)}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                raise

def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_article.json")
    print(f"Loading data from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    emb_dim = 768
    hidden_size = 768
    n_layer = 2
    bidirectional = True
    dropout = 0.2
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vocab_size = tokenizer.vocab_size
    num_partial_steps = 10  # Train for a few steps
    learning_rate = 0.001
    batch_size = 4

    model = initialize_model(vocab_size, emb_dim, hidden_size, bidirectional, n_layer, dropout, device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    data_directory = "/Users/lynn/desktop/thesis/cnn_dm4openie_extraction/article_collections"

    train_data = open_json_file(data_directory, "train")
    test_data = open_json_file(data_directory, "test")

    train_dataset = CustomDataset(train_data, tokenizer)
    test_dataset = CustomDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    # Partially train the model
    partial_train_model(model, train_dataloader, optimizer, criterion, num_partial_steps, device)

    # Debug the decoding part
    debug_decoding(model, test_dataloader, tokenizer, device)

if __name__ == "__main__":
    main()
