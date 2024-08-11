import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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

def initialize_model(vocab_size, emb_dim, n_hidden, bidirectional, n_layer, dropout, device):
    print("Initializing model...")
    model = Seq2SeqSumm(vocab_size, emb_dim, n_hidden, bidirectional, n_layer, dropout).to(device)
    return model

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, log_dir, model_save_path, patience=3):
    writer = SummaryWriter(log_dir)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, target_summary = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            art_lens = (attention_mask != 0).sum(dim=1).to(device)  # Calculate article lengths

            optimizer.zero_grad()
            output = model(input_ids, art_lens, target_summary.to(device))

            # Check for out-of-bounds indices
            vocab_size = output.size(-1)
            max_target_index = target_summary.max().item()
            if max_target_index >= vocab_size:
                raise ValueError(f"Target index out of bounds: {max_target_index} >= {vocab_size}")

            loss = criterion(output.view(-1, output.size(-1)), target_summary.view(-1).to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, criterion, device)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # save the model
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved, model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve, epochs_no_improve: {epochs_no_improve}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    writer.close()

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, target_summary = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            art_lens = (attention_mask != 0).sum(dim=1).to(device)  # Calculate article lengths

            output = model(input_ids, art_lens, target_summary.to(device))
            loss = criterion(output.view(-1, output.size(-1)), target_summary.view(-1).to(device))
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_model(model, dataloader, tokenizer, device, output_file):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            art_lens = (attention_mask != 0).sum(dim=1).to(device)  # Calculate article lengths

            # Generate summaries
            summary_tokens_batch, _ = model.batch_decode(input_ids, art_lens, tokenizer.cls_token_id, tokenizer.sep_token_id, max_len=512)

            # summary_tokens_batch has shape (max_len, batch_size)
            # Transpose to get (batch_size, max_len)
            summary_tokens_batch = list(map(list, zip(*summary_tokens_batch)))

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
    emb_dim = 768
    hidden_size = 768
    n_layer = 2
    bidirectional = True
    dropout = 0.2
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vocab_size = tokenizer.vocab_size
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 8
    log_dir = './logs'
    model_save_path = './best_model_baseline.pth'
    patience = 3   

    model = initialize_model(vocab_size, emb_dim, hidden_size, bidirectional, n_layer, dropout, device)
    # wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
    data_directory = "/home1/s5734436/thesis-graph/cnn_dm4openie_extraction/article_collections_large" 


    train_data = open_json_file(data_directory, "train")
    valid_data = open_json_file(data_directory, "valid")
    test_data = open_json_file(data_directory, "test")

    train_dataset = CustomDataset(train_data, tokenizer)
    valid_dataset = CustomDataset(valid_data, tokenizer)
    test_dataset = CustomDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, num_epochs, device, log_dir, model_save_path, patience)

    # Load the best model before testing
    model.load_state_dict(torch.load(model_save_path))

    output_file = os.path.join(data_directory, "generated_summaries.json")
    results = test_model(model, test_dataloader, tokenizer, device, output_file)

    for i in range(5):
        print(f"Generated Summary {i+1}: {results[i]['generated_summary']}")
        print(f"Reference Summary {i+1}: {results[i]['reference_summary']}")

if __name__ == "__main__":
    main()
