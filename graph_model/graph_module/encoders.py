import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from transformers import BartTokenizer, BartModel, BartConfig
from graph_module.get_graph_embeddings import BiLSTM
from graph_module.dataset_graph import CNN_DM_Graph, custom_collate_fn, load_data
from graph_module.initialize_weight import xavier_initialization, kaiming_initialization, normal_initialization

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout):
        super(GraphAttentionNetwork, self).__init__()
        # GATConv expects [num_nodes, num_features], not [batch_size, num_nodes, num_features] 
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

class EncoderDocument(nn.Module):
    def __init__(self, bart_model_name, encoder_config=None):
        super().__init__()
        if encoder_config is None:
            # load default config from the pretrained model
            self.bart = BartModel.from_pretrained(bart_model_name)
        else:
            # use custom config
            config = BartConfig.from_pretrained(bart_model_name)
            config.update(encoder_config.to_dict())  # update with custom values
            self.bart = BartModel(config)

    def forward(self, input_ids, attention_mask):
        bart_outputs = self.bart.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return bart_outputs.last_hidden_state
    
class EncoderGraph(nn.Module):
    def __init__(self, gat_in_channels, gat_out_channels, gat_heads, dropout):
        super(EncoderGraph, self).__init__()
        self.gat = GraphAttentionNetwork(
            in_channels=gat_in_channels,  
            out_channels=gat_out_channels,         
            heads=gat_heads,
            dropout=dropout
        )

    def forward(self, graph_node_features, edge_index):
        gat_output = self.gat(graph_node_features, edge_index) # [num_nodes, out_channels * heads]
        return gat_output