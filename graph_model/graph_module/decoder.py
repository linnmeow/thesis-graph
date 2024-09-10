import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, AdamW, BartConfig
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
        self.activation_fn = nn.LeakyReLU()
        self.activation_dropout = config.activation_dropout

    def prepare_attention_mask(self, attention_mask, num_heads):
        # convert to boolean mask if not already
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # get batch size and sequence length
        batch_size, seq_len = attention_mask.shape

        # during training, expand to [batch_size, 1, 1, seq_len] for broadcasting across heads
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)  # [batch_size, num_heads, seq_len, seq_len]
        
        # reshape to [batch_size * num_heads, seq_len, seq_len] for multi-head attention
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
        use_cache=False
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

        # Debugging output after self-attention normalization
        print("Post self-attention LayerNorm:", hidden_states.mean().item(), hidden_states.std().item())
    

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

            # Debugging output after encoder attention normalization
            print("Post encoder attention LayerNorm:", hidden_states.mean().item(), hidden_states.std().item())

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

            # Debugging output after graph attention normalization
            print("Post graph attention LayerNorm:", hidden_states.mean().item(), hidden_states.std().item())
    
        # feed-forward network
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # Debugging output after feed-forward network normalization
        print("Post feed-forward LayerNorm:", hidden_states.mean().item(), hidden_states.std().item())

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

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, graph_node_features, edge_index):
        # get encoder outputs
        encoder_hidden_states = self.encoder_document(input_ids, attention_mask)

        # get graph embeddings if not all subgraphs were pruned
        if graph_node_features is not None and edge_index is not None:
            graph_embeddings = self.encoder_graph(graph_node_features, edge_index)
        else:
            graph_embeddings = None  

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

        return logits
   
    def generate_summary(self, input_ids, attention_mask, graph_node_features, edge_index, max_length, min_length):
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # initialize decoder input with BOS token
        generated_ids = torch.full((batch_size, 1), self.bart.config.bos_token_id, dtype=torch.long, device=device)

        for step in range(max_length):
            # forward pass
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=generated_ids,
                decoder_attention_mask=None,  # no mask needed during inference
                graph_node_features=graph_node_features,
                edge_index=edge_index,
                is_testing=True
            )

            # get the logits of the last generated token
            logits = outputs[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # ensure min_length constraint is met before allowing <eos> token to break the loop
            if step >= min_length and next_token_id.item() == self.bart.config.eos_token_id:
                break

            # append the predicted token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        return generated_ids

    def generate_summary_with_beam_search(
        self,
        input_ids,
        attention_mask,
        graph_node_features,
        edge_index,
        max_length,
        min_length,
        beam_size
    ):
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # initialize the beam search
        beams = [(torch.full((batch_size, 1), self.bart.config.bos_token_id, dtype=torch.long, device=device), 0)]  # (sequence, score)

        # list to store the final hypotheses
        final_beams = []

        for step in range(max_length):
            new_beams = []

            # iterate over all beams
            for seq, score in beams:
                # forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=seq,
                    decoder_attention_mask=None,  # no mask needed during inference
                    graph_node_features=graph_node_features,
                    edge_index=edge_index
                )

                # get logits and scores for the last token
                logits = outputs[:, -1, :]
                probs = F.log_softmax(logits, dim=-1)  # get log probabilities

                # expand the beams
                for token_id in range(logits.size(-1)):
                    new_seq = torch.cat([seq, torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)], dim=1)
                    new_score = score + probs[:, token_id].item()

                    # ensure that the <eos> token is not selected before reaching min_length
                    if step + 1 < min_length and token_id == self.bart.config.eos_token_id:
                        continue  # skip adding this sequence if it ends with <eos> before min_length

                    new_beams.append((new_seq, new_score))

            # prune beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # check if end-of-sequence token reached
            final_beams.extend([beam for beam in new_beams if beam[0][0, -1].item() == self.bart.config.eos_token_id and step + 1 >= min_length])
            new_beams = [beam for beam in new_beams if beam[0][0, -1].item() != self.bart.config.eos_token_id or step + 1 < min_length]

            if not new_beams and final_beams:
                # if no more beams but some final sequences, break early
                break

            beams = new_beams

        # get the best sequence
        if not final_beams:
            final_beams = beams

        best_seq = max(final_beams, key=lambda x: x[1])[0]

        return best_seq

