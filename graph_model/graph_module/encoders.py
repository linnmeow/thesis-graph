import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from transformers import BartTokenizer, BartModel
from graph_module.get_graph_embeddings import BiLSTM
from graph_module.dataset_graph import CNN_DM_Graph, custom_collate_fn, load_data

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout):
        super(GraphAttentionNetwork, self).__init__()
        # GATConv expects [num_nodes, num_features], not [batch_size, num_nodes, num_features] 
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)
    
class EncoderDocument(nn.Module):
    def __init__(self, bart_model_name):
        super(EncoderDocument, self).__init__()
        self.bart = BartModel.from_pretrained(bart_model_name)

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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and prepare tokenizer and models
    data = load_data("./examples.json")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    vocab_size = len(tokenizer.get_vocab())
    embedding_dim = 64
    hidden_dim = 128  
    bilstm_model = BiLSTM(vocab_size, embedding_dim, hidden_dim).to(device)
    bilstm_model.eval()

    # Load dataset and dataloader
    dataset = CNN_DM_Graph(data, tokenizer, max_length=1024, model=bilstm_model, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    # Initialize models and optimizer
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartModel.from_pretrained(model_name).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Initialize the encoders
    encoder_document = EncoderDocument(model_name).to(device)
    encoder_graph = EncoderGraph(
        gat_in_channels=128,
        gat_out_channels=96,
        gat_heads=8,
        dropout=0.6
    ).to(device)
   
    for batch in dataloader:
        
        encoder_input_ids = batch['encoder_input_ids']
        encoder_attention_mask = batch['encoder_attention_mask']
        graph_data = batch['graph_batch'] if 'graph_batch' in batch else None
        
        graph_node_features = graph_data.x if graph_data is not None else None
        edge_index = graph_data.edge_index if graph_data is not None else None

        # print(f"Input IDs Shape: {encoder_input_ids.shape}") #  [1, 1024], [batch_size, sequence_length]
        # print(f"Attention Mask Shape: {encoder_attention_mask.shape}") # [1, 1024]
        # print(f"Graph Node Features Shape: {graph_node_features.shape}") # [17, 256 * 4], [num_nodes, out_channels * heads]
        # print(f"Edge Index Shape: {edge_index.shape}") # [2, 14], [2, num_edges], source and target nodes (2 rows)

        # Pass through the document encoder
        with torch.no_grad():
            document_outputs = encoder_document(encoder_input_ids, encoder_attention_mask)
            print(f"Document Encoder Output Shape: {document_outputs.shape}")  # [1, 1024, 1024], [batch_size, sequence_length, hidden_size]
            # print(document_outputs)

        # Pass through the graph encoder
        with torch.no_grad():
            if graph_node_features is not None and edge_index is not None:
                graph_outputs = encoder_graph(graph_node_features, edge_index)
                print(f"Graph Encoder Output Shape: {graph_outputs.shape}")  # [17, 1024], [num_nodes, out_channels]
                # print(graph_outputs)