import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.tensorboard import SummaryWriter
from koila import lazy

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

class DocumentEncoder(nn.Module):
    def __init__(self, roberta_model_name, hidden_size):
        super(DocumentEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
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
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.u1 = nn.Parameter(torch.randn(hidden_size))
        self.Wout = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_token, hidden_state, document_output, document_mask):
        # lstm processing
        lstm_output, hidden_state = self.lstm(input_token.unsqueeze(1), hidden_state)

        # expand lstm output to match the shape of document output
        lstm_output_tiled = lstm_output.expand(-1, document_output.size(1), -1) # batch_size, 1, hidden_size -> batch_size, seq_len, hidden_size
        # attention mechanism over document encoder outputs
        transformed_lstm_output = self.W1(lstm_output_tiled)  # shape: [batch_size, seq_len, hidden_size]
        transformed_document_output = self.W2(document_output)  # shape: [batch_size, seq_len, hidden_size]
  
        # combine and apply non-linear activation
        combined = torch.tanh(transformed_lstm_output + transformed_document_output)  # shape: [batch_size, seq_len, hidden_size]
        # project to scalar with u1
        document_scores = torch.matmul(combined, self.u1)  # shape: [batch_size, seq_len]
        # mask the scores and calculate attention weights
        document_scores = document_scores.masked_fill(~document_mask.bool(), float('-inf')) # shape: [batch_size, seq_len]
        document_attention_weights = F.softmax(document_scores, dim=1) # shape: [batch_size, seq_len]
        # compute the context vector 
        document_context_vector = torch.sum(document_attention_weights.unsqueeze(2) * document_output, dim=1)  # shape: [batch_size, hidden_size]    

        # combine LSTM output with document context vector
        combined_context = torch.cat((lstm_output.squeeze(1), document_context_vector), dim=1) # shape: [batch_size, hidden_size * 2]
        output = self.Wout(combined_context) # shape: [batch_size, output_size]

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
            outputs.append(output.unsqueeze(1))  # append the decoder's output for the current time step to the list of outputs

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def generate_summary(self, input_ids, attention_mask, max_length=50):
        self.eval()
        with torch.no_grad():
            document_output, document_mask = self.encoder(input_ids, attention_mask)
            document_output = document_output.to(self.device)
            document_mask = document_mask.to(self.device)

            batch_size = document_output.size(0)
            decoder_input = torch.zeros((batch_size, self.decoder.hidden_size)).to(self.device)
            hidden_state = None

            summary_tokens = []
            for _ in range(max_length):
                output, hidden_state = self.decoder(decoder_input, hidden_state, document_output, document_mask)
                
                # Get the predicted token (output is of size [batch_size, output_size])
                predicted_token_id = output.argmax(dim=1)
                summary_tokens.append(predicted_token_id.unsqueeze(1))
                
                # Convert the predicted token ID into an embedding for the next LSTM input
                decoder_input = self.encoder.roberta.embeddings.word_embeddings(predicted_token_id)

            summary_tokens = torch.cat(summary_tokens, dim=1)
        return summary_tokens
    
def initialize_model(hidden_size, output_size, device):
    print("Initializing model...")
    encoder = DocumentEncoder('roberta-base', hidden_size).to(device)
    decoder = SummaryDecoder(hidden_size, output_size).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, log_dir, model_save_path, early_stopping_patience=1):
    writer = SummaryWriter(log_dir)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = torch.cuda.amp.GradScaler()  # initialize GradScaler

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        total_loss = 0

        print(f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_summary = batch['labels'].to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  
                output = model(input_ids, attention_mask, target_summary)
                loss = criterion(output.view(-1, output.size(-1)), target_summary.view(-1))
            
            scaler.scale(loss).backward()  # scale the loss and backpropagate
            scaler.step(optimizer)         # update optimizer
            scaler.update()               # update scaler
            
            total_loss += loss.item()

            if batch_idx % 100 == 0:  # print every 100 batches
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))  # Save the best model
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
            input_ids, attention_mask, target_summary = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            output = model(input_ids, attention_mask, target_summary)
            loss = criterion(output.view(-1, output.size(-1)), target_summary.view(-1))
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
            # target_summary = batch['labels']

            summary_tokens = model.generate_summary(input_ids, attention_mask)
            generated_summary = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)

            reference_summary = batch['highlights'][i]
            source_text = batch['article'][i]
            results.append({
                'source_text': source_text,
                'generated_summary': generated_summary,
                'reference_summary': reference_summary
            })

            if batch_idx % 100 == 0:  # print every 100 batches
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

def print_dataloader_samples(dataloader, num_batches=1):
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch {i + 1}:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"{key}: {value.shape} - {value[:2]}")
            else:
                print(f"{key}: {value[:2]}")
        print("\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    hidden_size = 768
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    output_size = tokenizer.vocab_size
    num_epochs = 15
    learning_rate = 0.001
    batch_size = 2
    patience = 1
    model_save_path = './model_checkpoints'
    log_dir = './logs'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model = initialize_model(hidden_size, output_size, device)
    
    # check for multiple GPUs and wrap the model in DataParallel if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

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

    output_file = os.path.join(data_directory, "generated_summaries.json")
    results = test_model(model, test_dataloader, tokenizer, device, output_file)

    for i in range(5):
        print(f"Generated Summary {i+1}: {results[i]['generated_summary']}")
        print(f"Reference Summary {i+1}: {results[i]['reference_summary']}")
