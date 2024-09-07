from graph_module.dataset_graph import CNN_DM_Graph, load_data, custom_collate_fn
from graph_module.get_graph_embeddings import embed_graph, BiLSTM
from graph_module.levi_transformation import perform_levi_transformation, prune_small_subgraphs
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from graph_module.decoder import Seq2SeqModel

    
def train_model(train_loader, model, tokenizer, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    for batch in train_loader:
        optimizer.zero_grad()

        # move text data to device
        encoder_input_ids = batch['encoder_input_ids'].to(device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)  
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)

        # shift decoder_input_ids to create labels
        labels = decoder_input_ids[:, 1:]
        # add padding token at the end of the sequence to ensure seq length consistency
        pad_token_id = tokenizer.pad_token_id
        pad_tensor = decoder_input_ids.new_full((decoder_input_ids.size(0), 1), pad_token_id)

        # append the padding token to the end of the labels [batch_size, seq_len]
        labels = torch.cat([labels, pad_tensor], dim=1).to(device) 

        # move graph data to device
        graph_data = batch['graph_batch']
        
        graph_node_features = graph_data.x.to(device) if graph_data is not None else None
        edge_index = graph_data.edge_index.to(device) if graph_data is not None else None
       
        # forward pass
        # logits: [batch_size, seq_len, vocab_size]
        logits = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            graph_node_features=graph_node_features,
            edge_index=edge_index,
            is_testing=False  
        ) 

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Training Loss: {total_loss/len(train_loader)}")

def validate_model(val_loader, model, tokenizer, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in val_loader:
            # move text data to device
            encoder_input_ids = batch['encoder_input_ids'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)  
            # decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            # shift decoder_input_ids to create labels
            labels = decoder_input_ids[:, 1:]
            # add padding token at the end of the sequence to ensure seq length consistency
            pad_token_id = tokenizer.pad_token_id
            pad_tensor = decoder_input_ids.new_full((decoder_input_ids.size(0), 1), pad_token_id)

            # append the padding token to the end of the labels
            labels = torch.cat([labels, pad_tensor], dim=1).to(device)

            # move graph data to device
            graph_data = batch['graph_batch']
            
            graph_node_features = graph_data.x.to(device) if graph_data is not None else None
            edge_index = graph_data.edge_index.to(device) if graph_data is not None else None
        
            # forward pass
            logits = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=None,  # not needed during validation
                graph_node_features=graph_node_features,
                edge_index=edge_index,
                is_testing=False
            )

            # calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss

def test_model(model, dataloader, tokenizer, device, output_file, min_length, max_length):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            input_ids = batch['encoder_input_ids'].to(device)
            attention_mask = batch['encoder_attention_mask'].to(device)
             # reshape attention mask to 2D
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.view(attention_mask.size(0), -1)

            graph_node_features = batch['graph_batch'].x.to(device)
            edge_index = batch['graph_batch'].edge_index.to(device)

            # generate summaries
            summary_tokens = model.generate_summary(
                input_ids=input_ids,
                # attention_mask=attention_mask,
                attention_mask=None, # set to None to avoid dimension mismatch
                graph_node_features=graph_node_features,
                edge_index=edge_index,
                min_length=min_length,
                max_length=max_length
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

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)} processed.")

    # save results to a file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = load_data("./examples.json")
    valid_data = load_data("./valid.json")
    test_data = load_data("./test.json")


    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    vocab_size = len(tokenizer.get_vocab())
    embedding_dim = 256
    hidden_dim = 512  
    bilstm_model = BiLSTM(vocab_size, embedding_dim, hidden_dim)
    bilstm_model.eval()

    train_dataset = CNN_DM_Graph(train_data, tokenizer, max_length=1024, model=bilstm_model)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    valid_dataset = CNN_DM_Graph(valid_data, tokenizer, max_length=1024, model=bilstm_model)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    test_dataset = CNN_DM_Graph(test_data, tokenizer, max_length=1024, model=bilstm_model)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    model = Seq2SeqModel(     
        bart_model_name='facebook/bart-large',
            gat_in_channels=512,
            gat_out_channels=256,
            gat_heads=4,
            dropout=0.2
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # Training and validation loop
    num_epochs = 1
    for epoch in range(num_epochs):  
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train model
        train_model(train_dataloader, model, tokenizer, optimizer, device)
        
        # Validate model
        val_loss = validate_model(valid_dataloader, model, tokenizer, device)
        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss}")

    # Test the model
    print("Testing the model...")
    output_file = "test_results.json"
    test_results = test_model(
        model=model,
        dataloader=test_dataloader,
        tokenizer=tokenizer,
        device=device,
        output_file=output_file,
        min_length=30,  
        max_length=100  
    )

    print(f"Testing completed. Results saved to {output_file}")

    # Print test results
    print("\nTest Results:")
    for result in test_results:
        print(f"Source Text: {result['source_text'][:200]}...")  # Print first 200 characters of source text
        print(f"Generated Summary: {result['generated_summary']}")
        print(f"Reference Summary: {result['reference_summary']}\n")


if __name__ == "__main__":
    main()
