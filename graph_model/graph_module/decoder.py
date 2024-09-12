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
from graph_module.initialize_weight import xavier_initialization, kaiming_initialization, normal_initialization
    
class DecoderLayerWithDualCrossAttention(BartDecoderLayer):
    def __init__(self, config, initialization_scheme='kaiming'):
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

        # apply initialization
        if initialization_scheme == 'xavier':
            self.apply(xavier_initialization)
        elif initialization_scheme == 'kaiming':
            self.apply(kaiming_initialization)
        elif initialization_scheme == 'normal':
            self.apply(normal_initialization)

    def prepare_attention_mask(self, attention_mask, num_heads):
        # convert to boolean mask where 1 -> False (do not mask) and 0 -> True (mask)
        attention_mask = (attention_mask == 0)  # Convert: 0 -> True, 1 -> False
        
        # ensure the mask is in boolean format
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        # get batch size and sequence length
        batch_size, seq_len = attention_mask.shape

        # expand to [batch_size, 1, 1, seq_len] for broadcasting across heads
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
        num_heads = self.self_attn.num_heads  
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
    
