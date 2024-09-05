import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, AdamW, BartConfig
import sentencepiece as spm
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartDecoderLayer

from graph_module.dataset_graph import CNN_DM_Graph, custom_collate_fn, load_data
from graph_module.encoders import GraphAttentionNetwork, EncoderDocument, EncoderGraph
from graph_module.get_graph_embeddings import BiLSTM
    
class DecoderLayerWithDualCrossAttention(BartDecoderLayer):
    def __init__(self, config):
        super().__init__(config)

        # define the self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # define the second cross-attention for graph embeddings
        self.graph_cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # cross-attention over encoder outputs
        self.encoder_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # embedding layer for the decoder input tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # layer normalization layers
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.graph_attn_layer_norm = nn.LayerNorm(config.d_model)

        # dropout and activation
        self.dropout = config.dropout
        self.activation_fn = nn.ReLU()
        self.activation_dropout = config.activation_dropout

    def prepare_attention_mask(self, attention_mask, num_heads):
        # convert to boolean if not already
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # expand mask for each attention head
        batch_size, seq_len = attention_mask.shape
        # expand to [batch_size, 1, seq_len, seq_len] and then repeat for num_heads
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len)
        # reshape to [batch_size * num_heads, seq_len, seq_len]
        attention_mask = attention_mask.reshape(batch_size * num_heads, seq_len, seq_len)
        
        return attention_mask
    
    def add_batch_dimension(self, graph_embeddings, batch_size):
        # graph_embeddings is of shape [num_nodes, embedding_dim]
        num_nodes, embedding_dim = graph_embeddings.shape
        # add batch dimension
        graph_embeddings = graph_embeddings.unsqueeze(0).expand(batch_size, num_nodes, embedding_dim)
        return graph_embeddings
    
    def forward(
        self,
        decoder_input_ids=None,  
        hidden_states=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        graph_embeddings=None,
        graph_attention_mask=None,
        labels=None,
        output_attentions=False,
        use_cache=False,
    ):
        # ensure graph_embeddings has batch dimension
        batch_size = hidden_states.size(0)
        if graph_embeddings is not None:
            graph_embeddings = self.add_batch_dimension(graph_embeddings, batch_size)
        
        # if hidden_states are not provided, use the embedding layer
        if hidden_states is None and decoder_input_ids is not None:
            hidden_states = self.embed_tokens(decoder_input_ids)

        # prepare the attention masks
        num_heads = self.self_attn.num_heads  # number of attention heads
        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, num_heads)

        if encoder_attention_mask is not None:
            encoder_attention_mask = self.prepare_attention_mask(encoder_attention_mask, num_heads)

        if graph_attention_mask is not None:
            graph_attention_mask = self.prepare_attention_mask(graph_attention_mask, num_heads)
        
        # standard self-attention
        residual = hidden_states
        # batch_size = hidden_states.shape[0]
        # attention_mask = attention_mask.expand(batch_size, -1, -1)  # Dynamically handle batch size
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,  
            hidden_states,  
            hidden_states,  
            attn_mask=attention_mask,  
            need_weights=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # cross-attention over encoder outputs
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states,  
                encoder_hidden_states,  
                encoder_hidden_states,  
                attn_mask=encoder_attention_mask,  
                need_weights=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # cross-attention over graph embeddings
        if graph_embeddings is not None:
            residual = hidden_states
            hidden_states, graph_attn_weights = self.graph_cross_attn(
                query=hidden_states,
                key=graph_embeddings,
                value=graph_embeddings,
                attn_mask=graph_attention_mask,  
                need_weights=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.graph_attn_layer_norm(hidden_states)

        # feed-forward network
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if use_cache:
            present_key_value_state = (self_attn_weights, cross_attn_weights)  
        else:
            present_key_value_state = None

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights, graph_attn_weights)

        if use_cache:
            outputs += (present_key_value_state,)

        return outputs
    
class Seq2SeqModel(nn.Module):
    def __init__(self, bart_model_name, gat_in_channels, gat_out_channels, gat_heads, dropout):
        super(Seq2SeqModel, self).__init__()
        self.bart = BartModel.from_pretrained(bart_model_name)
        self.encoder_document = EncoderDocument(bart_model_name)
        self.encoder_graph = EncoderGraph(gat_in_channels, gat_out_channels, gat_heads, dropout)
        self.decoder = nn.ModuleList([
            DecoderLayerWithDualCrossAttention(self.bart.config) for _ in range(self.bart.config.decoder_layers)
        ])
        self.output_proj = nn.Linear(self.bart.config.d_model, self.bart.config.vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, graph_node_features, edge_index, labels=None):
        # get encoder outputs
        encoder_hidden_states = self.encoder_document(input_ids, attention_mask)
        # get graph embeddings
        graph_embeddings = self.encoder_graph(graph_node_features, edge_index)
        
        # initialize decoder hidden states
        hidden_states = self.bart.get_input_embeddings()(decoder_input_ids)

        # apply custom decoder layers
        for layer in self.decoder:
            hidden_states = layer(
                decoder_input_ids=decoder_input_ids,
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                graph_embeddings=graph_embeddings
            )[0]

        # project to vocabulary size
        logits = self.output_proj(hidden_states)

        # calculate loss if labels are provided
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

    
# def train_model(train_loader, model, tokenizer, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()

#         # move text data to device
#         encoder_input_ids = batch['encoder_input_ids'].to(device)
#         encoder_attention_mask = batch['encoder_attention_mask'].to(device)
#         decoder_input_ids = batch['decoder_input_ids'].to(device)  
#         decoder_attention_mask = batch['decoder_attention_mask'].to(device)

#         # shift decoder_input_ids to create labels
#         labels = decoder_input_ids[:, 1:]
#         # add padding token at the end of the sequence to ensure seq length consistency
#         pad_token_id = tokenizer.pad_token_id
#         pad_tensor = decoder_input_ids.new_full((decoder_input_ids.size(0), 1), pad_token_id)

#         # append the padding token to the end of the labels
#         labels = torch.cat([labels, pad_tensor], dim=1).to(device)

#         # move graph data to device
#         graph_data = batch['graph_batch']
        
#         graph_node_features = graph_data.x.to(device) if graph_data is not None else None
#         edge_index = graph_data.edge_index.to(device) if graph_data is not None else None
    
#         # forward pass
#         outputs = model(
#             input_ids=encoder_input_ids,
#             attention_mask=encoder_attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             graph_node_features=graph_node_features,
#             edge_index=edge_index,
#             labels=labels
#         )
#         logits, loss = outputs
        
#         # calculate loss (use the combined outputs for the loss)
#         logits, loss = outputs
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Training Loss: {total_loss/len(train_loader)}")


# if __name__ == "__main__":

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     data = load_data("./examples.json")

#     tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
#     vocab_size = len(tokenizer.get_vocab())
#     embedding_dim = 256
#     hidden_dim = 512  
#     bilstm_model = BiLSTM(vocab_size, embedding_dim, hidden_dim)
#     bilstm_model.eval()

#     dataset = CNN_DM_Graph(data, tokenizer, max_length=1024, model=bilstm_model)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

#     model = Seq2SeqModel(
#         bart_model_name='facebook/bart-large',
#         gat_in_channels=512,
#         gat_out_channels=256,
#         gat_heads=4,
#         dropout=0.2
#     ).to(device)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#     for epoch in range(3):  
#         print(f"Epoch {epoch + 1}")
#         train_model(dataloader, model, tokenizer, optimizer, device)
