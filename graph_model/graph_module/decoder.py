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
    def __init__(self, decoder_config=None, initialization_scheme='kaiming', pad_token_id=0):
        super().__init__(config=decoder_config)

        self.pad_token_id = pad_token_id

        # define the self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=decoder_config.d_model,
            num_heads=decoder_config.decoder_attention_heads,
            dropout=decoder_config.attention_dropout,
            batch_first=True
        )

        # define the second cross-attention for graph embeddings
        self.graph_cross_attn = nn.MultiheadAttention(
            embed_dim=decoder_config.d_model,
            num_heads=decoder_config.decoder_attention_heads,
            dropout=decoder_config.attention_dropout,
            batch_first=True
        )

        # cross-attention over encoder outputs
        self.encoder_attn = nn.MultiheadAttention(
            embed_dim=decoder_config.d_model,
            num_heads=decoder_config.decoder_attention_heads,
            dropout=decoder_config.attention_dropout,
            batch_first=True
        )

        # embedding layer for the decoder input tokens
        self.embed_tokens = nn.Embedding(decoder_config.vocab_size, decoder_config.d_model)

        # layer normalization layers
        self.self_attn_layer_norm = nn.LayerNorm(decoder_config.d_model)
        self.encoder_attn_layer_norm = nn.LayerNorm(decoder_config.d_model)
        self.graph_attn_layer_norm = nn.LayerNorm(decoder_config.d_model)

        # dropout and activation
        self.dropout = decoder_config.dropout
        self.activation_fn = nn.LeakyReLU()
        self.activation_dropout = decoder_config.activation_dropout

        # apply initialization
        if initialization_scheme == 'xavier':
            self.apply(xavier_initialization)
        elif initialization_scheme == 'kaiming':
            self.apply(kaiming_initialization)
        elif initialization_scheme == 'normal':
            self.apply(normal_initialization)

    def create_key_padding_mask(self, padded_sequences):
        """
        Create a binary key padding mask for padded sequences.
        
        Args:
            padded_sequences (torch.Tensor): Tensor of shape (batch_size, seq_length) with padded sequences.
            padding_value (int, optional): The value used for padding in the sequences (default is 0).
            
        Returns:
            torch.Tensor: A binary mask of shape (batch_size, seq_length) with True for padding positions.
        """
        # create a mask where padding positions are True
        # check pad token id 
        key_padding_mask = (padded_sequences == 0).float()
        # print(f'key_padding_mask: {key_padding_mask.tolist()}') 
        return key_padding_mask
    
    def add_batch_dimension(self, graph_embeddings, batch_size):
        # graph_embeddings is of shape [num_nodes, embedding_dim]
        num_nodes, embedding_dim = graph_embeddings.shape
        # add batch dimension
        graph_embeddings = graph_embeddings.unsqueeze(0).expand(batch_size, num_nodes, embedding_dim)
        return graph_embeddings
    
    def generate_causal_mask(self, seq_length, batch_size, num_heads):
        # generate the causal mask with shape [seq_length, seq_length]
        mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
        mask = mask.float() # ensure mask is float 
        
        # expand the mask to shape [batch_size, num_heads, seq_length, seq_length]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
        mask = mask.expand(batch_size, num_heads, seq_length, seq_length)  # [batch_size, num_heads, seq_length, seq_length]
        
        # reshape to [batch_size * num_heads, seq_length, seq_length]
        mask = mask.reshape(batch_size * num_heads, seq_length, seq_length)  # [batch_size * num_heads, seq_length, seq_length]
        return mask

    def forward(
        self,
        decoder_input_ids=None,  
        hidden_states=None,
        attention_mask=None,  # mask for self-attention/ decoder input
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        graph_embeddings=None,
        graph_attention_mask=None,
        labels=None,
        output_attentions=False,
        use_cache=False
    ):
        # ensure graph_embeddings has batch dimension
        batch_size = decoder_input_ids.size(0)
        seq_length = decoder_input_ids.size(1)
        num_heads = self.self_attn.num_heads 

        if graph_embeddings is not None:
            graph_embeddings = self.add_batch_dimension(graph_embeddings, batch_size)
        
        # intialize hidden states with decoder input tokens 
        if hidden_states is None and decoder_input_ids is not None:
            hidden_states = self.embed_tokens(decoder_input_ids)
     
        # prepare the key attention masks and causal attention mask for self attention
        key_attention_mask = self.create_key_padding_mask(attention_mask) if attention_mask is not None else None
        causal_mask = self.generate_causal_mask(seq_length, batch_size, num_heads).to(decoder_input_ids.device)
        # print(f'causal_mask: {causal_mask.tolist()}')
        key_encoder_attention_mask = self.create_key_padding_mask(encoder_attention_mask) if encoder_attention_mask is not None else None

        # standard self-attention
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,  # query
            hidden_states,  # key
            hidden_states,  # value
            key_padding_mask=key_attention_mask,
            attn_mask=causal_mask,
            is_causal=True,  
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
                key_padding_mask=key_encoder_attention_mask,
                # attn_mask=encoder_attention_mask,  
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
                key_padding_mask=None, # no padding in graph embeddings
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
    
