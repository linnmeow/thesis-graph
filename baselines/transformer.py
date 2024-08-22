import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast


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

class Seq2SeqModel(nn.Module):
    def __init__(self, bart_model_name):
        super(Seq2SeqModel, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained(bart_model_name)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )

    def generate_summary(self, input_ids, attention_mask, num_beams, min_length, max_length, no_repeat_ngram_size):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True
        )

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, log_dir, model_save_path, early_stopping_patience=3):
    writer = SummaryWriter(log_dir)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scaler = GradScaler()

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids, attention_mask, labels)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, criterion, device)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            print(f"Validation loss improved, saving model to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered. Stopping training.")
            break

    writer.close()

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_model(model, dataloader, tokenizer, device, output_file, num_beams, min_length, max_length, no_repeat_ngram_size):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # check if model is wrapped in DataParallel
            summary_tokens = model.module.generate_summary(input_ids, attention_mask, num_beams, min_length, max_length, no_repeat_ngram_size) if hasattr(model, 'module') else model.generate_summary(input_ids, attention_mask, input_ids, attention_mask, num_beams, min_length, max_length, no_repeat_ngram_size)
            
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
    
    num_epochs = 15
    learning_rate = 0.0001
    batch_size = 4
    patience = 3
    model_save_path = './model_checkpoints'
    log_dir = './logs'

    # hyperparameters for decoding
    num_beams = 4
    min_length = 56
    max_length = 142
    no_repeat_ngram_size = 3

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model = Seq2SeqModel(bart_model_name).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    data_directory = "/content/drive/MyDrive/cnn_dm4openie_extraction/article_collections_large/"
    train_data = open_json_file(data_directory, "train")
    valid_data = open_json_file(data_directory, "valid")
    test_data = open_json_file(data_directory, "test")

    train_dataset = CNN_DM(train_data, tokenizer)
    valid_dataset = CNN_DM(valid_data, tokenizer)
    test_dataset = CNN_DM(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, num_epochs, device, log_dir, model_save_path, patience)

    output_file = os.path.join(data_directory, "generated_summaries.json")
    results = test_model(model, test_dataloader, tokenizer, device, output_file, num_beams, min_length, max_length, no_repeat_ngram_size)

    for i in range(5):
        print(f"Generated Summary {i+1}: {results[i]['generated_summary']}")
        print(f"Reference Summary {i+1}: {results[i]['reference_summary']}")
