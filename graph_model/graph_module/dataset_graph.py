import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from transformers import BartTokenizer, BartModel

from graph_module.levi_transformation import perform_levi_transformation, prune_small_subgraphs
from graph_module.get_graph_embeddings import get_embeddings, embed_graph, BiLSTM

class CNN_DM_Graph(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, model=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model

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
            'encoder_input_ids': encoder_inputs['input_ids'].squeeze(),   # [batch_size, seq_length]
            'encoder_attention_mask': encoder_inputs['attention_mask'].squeeze(),   # [batch_size, seq_length]
            'decoder_input_ids': decoder_inputs['input_ids'].squeeze(),
            'decoder_attention_mask': decoder_inputs['attention_mask'].squeeze(),
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
                graph_data = embed_graph(list(G.nodes), list(G.edges), self.tokenizer, self.model)
                
                graph_item = Data(x=graph_data.x, edge_index=graph_data.edge_index)

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
    # print the type and dtype of the attention mask
    # print(f"Attention Mask - Type: {type(text_data['attention_mask'])}, Dtype: {text_data['attention_mask'].dtype}")
    # Type: <class 'torch.Tensor'>, Dtype: torch.int64
    
    graph_data = [item['graph_data'] for item in batch if 'graph_data' in item]

    # batch the graph data if it exists
    if graph_data:
        graph_batch = Batch.from_data_list(graph_data)
        text_data['graph_batch'] = graph_batch

    return text_data

def load_data(data_path):
    with open(data_path, "r") as f:
        return json.load(f)
    
if __name__ == "__main__":

    data = load_data("examples.json")

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # # Retrieve special token IDs
    # start_token_id = tokenizer.bos_token_id  # 0, <s>
    # end_token_id = tokenizer.eos_token_id    # 2, </s>
    # pad_token_id = tokenizer.pad_token_id    # 1, <pad>

    # # Convert token IDs to token names
    # start_token = tokenizer.convert_ids_to_tokens(start_token_id) if start_token_id is not None else None
    # end_token = tokenizer.convert_ids_to_tokens(end_token_id) if end_token_id is not None else None
    # pad_token = tokenizer.convert_ids_to_tokens(pad_token_id) if pad_token_id is not None else None

    # print(f"Start Token ID: {start_token_id}, Token: {start_token}") 
    # print(f"End Token ID: {end_token_id}, Token: {end_token}")       
    # print(f"Pad Token ID: {pad_token_id}, Token: {pad_token}")   


    vocab_size = len(tokenizer.get_vocab())
    embedding_dim = 256
    hidden_dim = 512  
    bilstm_model = BiLSTM(vocab_size, embedding_dim, hidden_dim)
    bilstm_model.eval()

    dataset = CNN_DM_Graph(data, tokenizer, max_length=1024, model=bilstm_model)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

    for batch in dataloader:
        print(batch['decoder_input_ids'])
        print(batch['encoder_input_ids'])
        print("Batch keys:", batch.keys())
        break