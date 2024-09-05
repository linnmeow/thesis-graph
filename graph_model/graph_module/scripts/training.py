from graph_module.dataset_graph import CNN_DM_Graph, load_data, custom_collate_fn
from graph_module.get_graph_embeddings import embed_graph, BiLSTM
from graph_module.levi_transformation import perform_levi_transformation, prune_small_subgraphs
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset

from graph_module.decoder import Seq2SeqModel

    
def train_model(train_loader, model, tokenizer, optimizer, device):
    model.train()
    total_loss = 0
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

        # append the padding token to the end of the labels
        labels = torch.cat([labels, pad_tensor], dim=1).to(device)

        # move graph data to device
        graph_data = batch['graph_batch']
        
        graph_node_features = graph_data.x.to(device) if graph_data is not None else None
        edge_index = graph_data.edge_index.to(device) if graph_data is not None else None
       
        # if decoder_attention_mask is None:
        #     decoder_attention_mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).to(device)

        # forward pass
        outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            graph_node_features=graph_node_features,
            edge_index=edge_index,
            labels=labels
        )
        logits, loss = outputs
        
        # Calculate loss (use the combined outputs for the loss)
        logits, loss = outputs
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Training Loss: {total_loss/len(train_loader)}")




def main():
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

   data = load_data("./examples.json")

   tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
   vocab_size = len(tokenizer.get_vocab())
   embedding_dim = 256
   hidden_dim = 512  
   bilstm_model = BiLSTM(vocab_size, embedding_dim, hidden_dim)
   bilstm_model.eval()

   dataset = CNN_DM_Graph(data, tokenizer, max_length=1024, model=bilstm_model)
   dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

   model = Seq2SeqModel(     
      bart_model_name='facebook/bart-large',
        gat_in_channels=512,
        gat_out_channels=256,
        gat_heads=4,
        dropout=0.2
    ).to(device)

   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
   for epoch in range(3):  
       print(f"Epoch {epoch + 1}")
       train_model(dataloader, model, tokenizer, optimizer, device)


if __name__ == "__main__":
    main()
