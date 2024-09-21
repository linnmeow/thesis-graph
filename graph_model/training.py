from graph_module.dataset_graph import CNN_DM_Graph, load_data, custom_collate_fn
from graph_module.get_graph_embeddings import embed_graph, BiLSTM
from graph_module.levi_transformation import perform_levi_transformation, prune_small_subgraphs, open_json_file
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
import json
import os
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from graph_module.seq2seq import Seq2SeqModel


def train_model(train_loader, val_loader, model, tokenizer, optimizer, num_epochs, model_save_path, early_stopping_patience):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            encoder_input_ids = batch['encoder_input_ids']
            encoder_attention_mask = batch['encoder_attention_mask']
            decoder_input_ids = batch['decoder_input_ids']
            decoder_attention_mask = batch['decoder_attention_mask']

            # shift decoder_input_ids to create labels
            labels = decoder_input_ids[:, 1:]
            pad_token_id = tokenizer.pad_token_id
            pad_tensor = decoder_input_ids.new_full((decoder_input_ids.size(0), 1), pad_token_id)

            labels = torch.cat([labels, pad_tensor], dim=1) # 1 padding token
            
            graph_data = batch['graph_batch']
            graph_node_features = graph_data.x if graph_data is not None else None
            edge_index = graph_data.edge_index if graph_data is not None else None

            logits = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                graph_node_features=graph_node_features,
                edge_index=edge_index
            )

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_model(val_loader, model, tokenizer)

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

def validate_model(val_loader, model, tokenizer):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            encoder_input_ids = batch['encoder_input_ids']
            encoder_attention_mask = batch['encoder_attention_mask']
            decoder_input_ids = batch['decoder_input_ids']
            # decoder_attention_mask = batch['decoder_attention_mask']

            # shift decoder_input_ids to create labels
            labels = decoder_input_ids[:, 1:]
            # add padding token at the end of the sequence to ensure seq length consistency
            pad_token_id = tokenizer.pad_token_id
            pad_tensor = decoder_input_ids.new_full((decoder_input_ids.size(0), 1), pad_token_id)

            # append the padding token to the end of the labels
            labels = torch.cat([labels, pad_tensor], dim=1)

            graph_data = batch['graph_batch'] 
            graph_node_features = graph_data.x if graph_data is not None else None
            edge_index = graph_data.edge_index if graph_data is not None else None

            # forward pass
            logits = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=None,  # not needed during validation
                graph_node_features=graph_node_features,
                edge_index=edge_index
            )

            # calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def test_model(model, dataloader, tokenizer, output_file, min_length, max_length, ngram_size):
    model.eval()
    results = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['encoder_input_ids']
            attention_mask = batch['encoder_attention_mask']
            graph_node_features = batch['graph_batch'].x
            edge_index = batch['graph_batch'].edge_index

            generate_func = (
                model.module.generate_summary 
                if hasattr(model, 'module') 
                else model.generate_summary
            )

            summary_tokens = generate_func(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_node_features=graph_node_features,
                edge_index=edge_index,
                min_length=min_length,
                max_length=max_length,
                # beam_size=beam_size,
                ngram_size=ngram_size
            )

            for i in range(input_ids.size(0)):
                # decode generated tokens
                generated_summary = tokenizer.decode(summary_tokens[i], skip_special_tokens=True)

                # get reference summary and source text
                reference_summary = batch['highlights'][i]
                source_text = batch['article'][i]

                # append results
                results.append({
                    'source_text': source_text,
                    'generated_summary': generated_summary,
                    'reference_summary': reference_summary
                })

            # compute loss for this batch
            decoder_input_ids = batch['decoder_input_ids']
            labels = decoder_input_ids[:, 1:]
            pad_token_id = tokenizer.pad_token_id
            pad_tensor = decoder_input_ids.new_full((decoder_input_ids.size(0), 1), pad_token_id)
            labels = torch.cat([labels, pad_tensor], dim=1)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=None,  
                graph_node_features=graph_node_features,
                edge_index=edge_index
            )

            # calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Testing Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    
    # save results to a file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    return avg_loss, results

def main():

    config = {
        "data_directory": "/Users/lynn/Desktop/thesis/cnn_dm/original_triples/",
        "output_file": "/Users/lynn/Desktop/thesis/cnn_dm/original_triples/generated_summaries.json",
        "model_save_path": "/Users/lynn/Desktop/thesis/cnn_dm/",
        "batch_size": 4,
        "num_epochs": 1,
        "patience": 2,
        "learning_rate": 1e-5,
        "bart_model_name": 'facebook/bart-base',
        "vocab_size": None,  # set dynamically from tokenizer
        "embedding_dim": 64,
        "hidden_dim": 128,
        "gat_in_channels": 128,
        "gat_out_channels": 96,
        "gat_heads": 8,
        "dropout": 0.6, # GAT dropout
        "initialization_scheme": 'kaiming',
        "min_length": 9,
        "max_length": 751,
        "beam_size": 3,
        "ngram_size": 3,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(config['model_save_path']):
        os.makedirs(config['model_save_path'])

    # load tokenizer
    tokenizer = BartTokenizer.from_pretrained(config["bart_model_name"])
    config['vocab_size'] = len(tokenizer.get_vocab()) 

    # load BiLSTM model
    bilstm_model = BiLSTM(config['vocab_size'], config['embedding_dim'], config['hidden_dim'])
    bilstm_model.to(device)
    bilstm_model.eval()

    # load dataset
    train_data = load_data('examples.json')
    valid_data = load_data('valid.json')
    test_data = load_data('test.json')

    # create datasets and dataloaders
    train_dataset = CNN_DM_Graph(train_data, tokenizer, max_length=1024, model=bilstm_model, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    valid_dataset = CNN_DM_Graph(valid_data, tokenizer, max_length=1024, model=bilstm_model, device=device)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
    test_dataset = CNN_DM_Graph(test_data, tokenizer, max_length=1024, model=bilstm_model, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)  

    # custom encoder and decoder configurations
    encoder_config = BartConfig(
        d_model=768,
        encoder_layers=config['encoder_layers'],
        encoder_attention_heads=config['encoder_attention_heads'],
        dropout=0.1
    )

    decoder_config = BartConfig(
        d_model=768,
        decoder_layers=config['decoder_layers'],
        decoder_attention_heads=config['decoder_attention_heads'],
        dropout=0.1
    )

    model = Seq2SeqModel(     
        bart_model_name=config['bart_model_name'],
            gat_in_channels=config['gat_in_channels'],
            gat_out_channels=config['gat_out_channels'],
            gat_heads=config['gat_heads'],
            dropout=config['dropout'],
            initialization_scheme=config['initialization_scheme'],
            encoder_config=encoder_config,
            decoder_config=decoder_config
        ).to(device)
    
    # wrap the model in DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    train_model(train_dataloader, valid_dataloader, model, tokenizer, optimizer, config['num_epochs'], config['model_save_path'], config['patience'])

    # decoding
    print("Testing the model...")
    # summary range cnn_dm: 9-751, xsum: 1-90, fiction: 5-321
    # num_beams = 4 (cnn_dm), 6 (xsum)
    avg_loss, test_results = test_model(
        model=model,
        dataloader=test_dataloader,
        tokenizer=tokenizer,
        output_file=config['output_file'],
        min_length=config['min_length'],  
        max_length=config['max_length'],
        # beam_size=config['beam_size'], 
        ngram_size=config['ngram_size']
    )

    print(f"Testing completed. Results saved to {config['output_file']}")

    # print test results
    print("\nTest Results:")
    for result in test_results:
        print(f"Source Text: {result['source_text'][:200]}...")  # print first 200 characters of source text
        print(f"Generated Summary: {result['generated_summary']}")
        print(f"Reference Summary: {result['reference_summary']}\n")

if __name__ == "__main__":
    main()
