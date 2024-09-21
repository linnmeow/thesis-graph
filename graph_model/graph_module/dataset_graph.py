import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from transformers import BartTokenizer, BartModel
import os

from graph_module.levi_transformation import perform_levi_transformation, prune_small_subgraphs
from graph_module.get_graph_embeddings import get_embeddings, embed_graph, BiLSTM

class CNN_DM_Graph(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, model=None, device=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        article = example['article']
        highlights = example['highlights']
        doc_id = example.get('id', idx)  # use document ID or index as default

        # tokenize the text inputs
        encoder_inputs = self.tokenizer(article, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        decoder_inputs = self.tokenizer(highlights, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        item = {
            'encoder_input_ids': encoder_inputs['input_ids'].squeeze().to(self.device),   # [batch_size, seq_length]
            'encoder_attention_mask': encoder_inputs['attention_mask'].squeeze().to(self.device),   # [batch_size, seq_length]
            'decoder_input_ids': decoder_inputs['input_ids'].squeeze().to(self.device),
            'decoder_attention_mask': decoder_inputs['attention_mask'].squeeze().to(self.device),
            'article': article,
            'highlights': highlights,
            'doc_id': doc_id
        }

        # handle graph data
        if 'triples' in example:
            triples = example['triples']
            G = perform_levi_transformation(triples)
            G = prune_small_subgraphs(G)
            
            if len(G.nodes) > 0:
                # embed the graph
                graph_data = embed_graph(list(G.nodes), list(G.edges), self.tokenizer, self.model, device=self.device)
                
            # # normalize graph data features
            # if graph_data.x is not None:
            #     graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / graph_data.x.std(dim=0)
            
                graph_item = Data(x=graph_data.x, edge_index=graph_data.edge_index).to(self.device)

                # include graph data in the item
                item['graph_data'] = graph_item
            else:
                print(f"All subgraphs were pruned for Entry ID: {example['id']}")

        return item

def custom_collate_fn(batch):
    text_data = {
        'encoder_input_ids': torch.stack([item['encoder_input_ids'] for item in batch]),
        'encoder_attention_mask': torch.stack([item['encoder_attention_mask'] for item in batch]),
        'decoder_input_ids': torch.stack([item['decoder_input_ids'] for item in batch]),
        'decoder_attention_mask': torch.stack([item['decoder_attention_mask'] for item in batch]),
        'doc_ids': [item['doc_id'] for item in batch],  # keep track of document IDs
        'article': [item['article'] for item in batch],
        'highlights': [item['highlights'] for item in batch]
    }
    
    graph_data = [item['graph_data'] for item in batch if 'graph_data' in item]

    # batch the graph data if it exists
    if graph_data:
        graph_batch = Batch.from_data_list(graph_data)
        text_data['graph_batch'] = graph_batch

    return text_data

def load_data(data_path):
    with open(data_path, "r") as f:
        return json.load(f)
    
def check_zeros_in_tensor(tensor):
    # Check for zeros in the tensor
    zero_mask = (tensor == 0)
    num_zeros = zero_mask.sum().item()  # Total number of zeros in the tensor
    zero_positions = zero_mask.nonzero(as_tuple=True)  # Positions where zeros are found
    
    if num_zeros > 0:
        print(f"Found {num_zeros} zeros in the tensor.")
        print(f"Positions of zeros: {zero_positions}")
    else:
        print("No zeros found in the tensor.")
    
    return num_zeros > 0, num_zeros, zero_positions
    
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_data("examples.json")

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    vocab_size = len(tokenizer.get_vocab())
    embedding_dim = 256
    hidden_dim = 512  

    # Move BiLSTM model to the correct device
    bilstm_model = BiLSTM(vocab_size, embedding_dim, hidden_dim).to(device)
    bilstm_model.eval()

    dataset = CNN_DM_Graph(data, tokenizer, max_length=1024, model=bilstm_model, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    # [BOS]: 0, [EOS]: 2, [PAD]: 1

    for batch in dataloader:
        if 'graph_batch' in batch:
            print(batch['graph_batch'].x.tolist())
            check_zeros_in_tensor(batch['graph_batch'].x)

        # print(batch['decoder_input_ids'].size())
        # print(batch['encoder_attention_mask'].tolist()) # 1 for real tokens, 0 for padding
        # print(batch['encoder_input_ids'].tolist())
        # print(batch['decoder_attention_mask'].tolist()) # 1 for real tokens, 0 for padding
        # print(batch['decoder_input_ids'].tolist()) # 0 for [BOS], 2 for [EOS], 1 for [PAD]
        # print("Batch keys:", batch.keys())
        break
