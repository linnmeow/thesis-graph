import torch
import networkx as nx
from transformers import BartTokenizer, BartModel
from torch_geometric.utils import from_networkx

def get_embeddings(text, tokenizer, model):
    """
    Generate embeddings for a given text using a pre-trained BART model.
    
    Args:
        text (str): The text to embed.
        tokenizer (BartTokenizer): The tokenizer.
        model (BartModel): The pre-trained BART model.

    Returns:
        torch.Tensor: The average embedding for the text.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    # get the average embedding for the entire sequence
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    
    # # print out the dimensions of the embedding
    # print(f"Embedding dimensions for '{text}': {embedding.size()}")
    
    return embedding

def embed_graph(nodes, edges, tokenizer, model):
    """
    Create a directed graph with nodes and edges, and embed the nodes using BART embeddings.
    
    Args:
        nodes (list): A list of node texts.
        edges (list): A list of tuples representing edges between nodes.
        tokenizer (BartTokenizer): The tokenizer.
        model (BartModel): The pre-trained BART model.

    Returns:
        torch_geometric.data.Data: The graph data object with embedded node features.
    """
    G = nx.DiGraph()
    
    # add nodes with their embeddings
    for i, node in enumerate(nodes):
        G.add_node(i, text=node, embedding=get_embeddings(node, tokenizer, model))
    
    # add edges to the graph, edge dim = [2, num_edges]
    for edge in edges:
        source = nodes.index(edge[0])  # find index of the source node
        target = nodes.index(edge[1])  # find index of the target node
        G.add_edge(source, target) 
    
    # convert the NetworkX graph to a PyTorch Geometric Data object
    data = from_networkx(G)
    
    # extract node features from the embeddings
    node_features = torch.stack([G.nodes[n]['embedding'] for n in G.nodes])
    data.x = node_features  # dimension: [num_nodes, num_node_features]
    
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
    model = BartModel.from_pretrained('facebook/bart-large')
    model.eval()

    # embed the graph
    graph_data = embed_graph(nodes, edges, tokenizer, model)

    # print the dimensions of the node features
    print(f"Node Features (x) dimensions: {graph_data.x.size()}")  # [num_nodes, num_node_features]
    print(f"Edge Index (edge_index) dimensions: {graph_data.edge_index.size()}")  # [2, num_edges]

    print("Node Features (x):", graph_data.x)  # size: [num_nodes, num_node_features]
    print("Edge Index (edge_index):", graph_data.edge_index)  # size: [2, num_edges]
