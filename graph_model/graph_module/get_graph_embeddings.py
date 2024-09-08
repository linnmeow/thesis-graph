import torch
import networkx as nx
from transformers import BartTokenizer, BartModel
from torch_geometric.utils import from_networkx
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # 2 * hidden_dim due to bidirectionality

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  
        return self.fc(lstm_out)


def get_embeddings(text, tokenizer, model):
    """
    Generate embeddings for a given text using a pre-trained BiLSTM model.
    
    Args:
        text (str): The text to embed.
        tokenizer (Tokenizer): The tokenizer for text to token indices.
        model (BiLSTM): The pre-trained BiLSTM model.

    Returns:
        torch.Tensor: The embedding for the text.
    """
    # tokenize and convert to indices
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
    input_ids = tokenized['input_ids'].squeeze(0)  # remove batch dimension
    
    with torch.no_grad():
        embedding = model(input_ids.unsqueeze(0))  # add batch dimension
    return embedding.squeeze()  # remove batch dimension


def embed_graph(nodes, edges, tokenizer, model):
    """
    Create an undirected graph with nodes and edges, and embed the nodes using BART embeddings.
    
    Args:
        nodes (list): A list of node texts.
        edges (list): A list of tuples representing edges between nodes.
        tokenizer (BartTokenizer): The tokenizer.
        model (BartModel): The pre-trained BART model.

    Returns:
        torch_geometric.data.Data: The graph data object with embedded node features.
    """
  
    G = nx.Graph()  

    for i, node in enumerate(nodes):
        embedding = get_embeddings(node, tokenizer, model)
        G.add_node(i, text=node, embedding=embedding)
    
    for edge in edges:
        source = nodes.index(edge[0])  # find index of the source node
        target = nodes.index(edge[1])  # find index of the target node
        G.add_edge(source, target)

    # convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(G)
    
    # extract node features from the embeddings
    node_features = torch.stack([G.nodes[n]['embedding'] for n in G.nodes])
    data.x = node_features  # Dimension: [num_nodes, num_node_features]

    return data

if __name__ == "__main__":
    # Sample data
    nodes = ['budget prices', 'is from', 'International Airport', 'France', 'boutique airline company', 
             'same trip', 'Paris', 'is with', 'Hotel Matignon', 'Newark', 'bus', 'company in', 
             'is in', 'board luxury aircraft', 'Charles de Gaulle Airport', 'to make', 'Trying', 
             'exclusively business class flights', 'Forget', 'has launched', 'founder', "L'Avion", 
             'Frantz Yvelin', 'is by', 'is founder of', 'choice of', 'is', 'generation of working business travelers', 
             'French', 'La Compagnie', 'of', 'flights', 'passengers', 'Yvelin', 'offers', 'says', 'airline La Compagnie']
    
    edges = [('budget prices', 'is with'), ('is from', 'Charles de Gaulle Airport'), ('is from', 'Paris'),
             ('International Airport', 'is in'), ('France', 'is in'), ('France', 'company in'), ('France', 'is with'),
             ('boutique airline company', 'is in'), ('same trip', 'is with'), ('Paris', 'is in'),
             ('is with', 'board luxury aircraft'), ('Hotel Matignon', 'is in'), ('Newark', 'is in'),
             ('bus', 'company in'), ('to make', 'Trying'), ('Trying', 'Forget'), 
             ('exclusively business class flights', 'has launched'), ('Forget', 'has launched'), 
             ('founder', 'is by'), ("L'Avion", 'is founder of'), ('Frantz Yvelin', 'is by'),
             ('Frantz Yvelin', 'is founder of'), ('choice of', 'La Compagnie'), 
             ('choice of', 'generation of working business travelers'), ('is', 'La Compagnie'), 
             ('is', 'French'), ('La Compagnie', 'of'), ('of', 'flights'), ('passengers', 'offers'),
             ('Yvelin', 'says'), ('offers', 'airline La Compagnie'), ('says', 'airline La Compagnie')]

    # load the tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    vocab_size = len(tokenizer.get_vocab())
    embedding_dim = 256
    hidden_dim = 512  
    bilstm_model = BiLSTM(vocab_size, embedding_dim, hidden_dim)
    bilstm_model.eval()

    # Embed the graph
    graph_data = embed_graph(nodes, edges, tokenizer, bilstm_model)

    # Print the dimensions of the node features
    print(f"Node Features (x) dimensions: {graph_data.x.size()}")  # [num_nodes, num_node_features]
    print(f"Edge Index (edge_index) dimensions: {graph_data.edge_index.size()}")  # [2, num_edges]

    print("Node Features (x):", graph_data.x)  # size: [num_nodes, num_node_features]
    print("Edge Index (edge_index):", graph_data.edge_index)  # size: [2, num_edges]
