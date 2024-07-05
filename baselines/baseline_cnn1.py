import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.tensorboard import SummaryWriter

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

        # tokenize inputs and outputs
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

class DocumentEncoder(nn.Module):
    def __init__(self, roberta_model_name, hidden_size):
        super(DocumentEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
        self.bilstm = nn.LSTM(self.roberta.config.hidden_size, 
                              hidden_size // 2,  # BiLSTM has half the hidden size per direction
                              num_layers=1,
                              bidirectional=True)

    def forward(self, input_ids, attention_mask):     
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        
        lstm_output, _ = self.bilstm(last_hidden_states)
        
        return lstm_output, attention_mask

class SummaryDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SummaryDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1)
        self.attention_document = nn.Linear(hidden_size, hidden_size)
        self.u1 = nn.Parameter(torch.randn(hidden_size))
        self.Wout = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_token, hidden_state, document_output, document_mask):
        # LSTM expects input of shape (seq_len, batch_size, input_size)
        input_token = input_token.unsqueeze(0)  # Add seq_len dimension (1 here)
        
        lstm_output, hidden_state = self.lstm(input_token, hidden_state)

        # Assuming you are only interested in the last output for simplicity
        lstm_output = lstm_output[-1]  # [batch_size, hidden_size]

        document_scores = torch.tanh(self.attention_document(lstm_output.unsqueeze(1)) + document_output)
        document_scores = torch.matmul(document_scores, self.u1)

        # Convert document_mask to torch.bool
        document_mask = document_mask.to(torch.bool)

        document_scores = document_scores.masked_fill(~document_mask, float('-inf'))
        document_attention_weights = F.softmax(document_scores, dim=1)

        document_context_vector = torch.sum(document_attention_weights.unsqueeze(2) * document_output, dim=1)

        combined_context = torch.cat((lstm_output, document_context_vector), dim=1)
        output = self.Wout(combined_context)

        return output, hidden_state



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_ids, attention_mask, target_summary=None, teacher_forcing_ratio=0.5):
        document_output, document_mask = self.encoder(input_ids, attention_mask)
        document_output = document_output.to(self.device)
        document_mask = document_mask.to(self.device)

        batch_size = document_output.size(0)
        decoder_input = torch.zeros((batch_size, self.decoder.hidden_size)).to(self.device)
        hidden_state = None

        outputs = []
        for t in range(target_summary.size(1)):
            output, hidden_state = self.decoder(decoder_input, hidden_state, document_output, document_mask)
            outputs.append(output.unsqueeze(1))

            # teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            # if teacher_force:
            #     print(f"Time step {t}: Using teacher forcing")
            # else:
            #     print(f"Time step {t}: Not using teacher forcing")

            # decoder_input = target_summary[:, t, :] if teacher_force else output

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def generate_summary(self, document_text, max_length=50):
        self.eval()
        with torch.no_grad():
            document_output, document_mask = self.encoder(document_text)
            document_output = document_output.to(self.device)
            document_mask = document_mask.to(self.device)

            batch_size = document_output.size(0)
            decoder_input = torch.zeros((batch_size, self.decoder.hidden_size)).to(self.device)
            hidden_state = None

            summary_tokens = []
            for _ in range(max_length):
                output, hidden_state = self.decoder(decoder_input, hidden_state, document_output, document_mask)
                summary_tokens.append(output.argmax(dim=1).unsqueeze(1))

                decoder_input = output

            summary_tokens = torch.cat(summary_tokens, dim=1)
        return summary_tokens

def initialize_model(hidden_size, output_size, device):
    encoder = DocumentEncoder('roberta-base', hidden_size).to(device)
    decoder = SummaryDecoder(hidden_size, output_size).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, log_dir):
    writer = SummaryWriter(log_dir)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, target_summary = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, target_summary)
            loss = criterion(output.view(-1, output.size(-1)), target_summary.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    writer.close()

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, target_summary = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            output = model(input_ids, attention_mask, target_summary)
            loss = criterion(output.view(-1, output.size(-1)), target_summary.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_model(model, dataloader, tokenizer, device, output_file):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_summary = batch['labels']

            summary_tokens = model.generate_summary(input_ids)
            generated_summary = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)

            reference_summary = tokenizer.decode(target_summary[0], skip_special_tokens=True)

            results.append({
                'generated_summary': generated_summary,
                'reference_summary': reference_summary
            })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_article.json")
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = 768
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    output_size = tokenizer.vocab_size
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 2
    log_dir = './logs'

    model = initialize_model(hidden_size, output_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    data_directory = "/Users/lynn/desktop/thesis/cnn_dm4openie_extraction/article_collections" 
    train_data = open_json_file(data_directory, "train")
    valid_data = open_json_file(data_directory, "valid")
    test_data = open_json_file(data_directory, "test")

    train_dataset = CustomDataset(train_data, tokenizer)
    valid_dataset = CustomDataset(valid_data, tokenizer)
    test_dataset = CustomDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, num_epochs, device, log_dir)

    output_file = os.path.join(data_directory, "generated_summaries.json")
    results = test_model(model, test_dataloader, tokenizer, device, output_file)

    for i in range(5):
        print(f"Generated Summary {i+1}: {results[i]['generated_summary']}")
        print(f"Reference Summary {i+1}: {results[i]['reference_summary']}")
