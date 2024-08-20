import torch
import networkx as nx
from transformers import BartTokenizer, BartModel
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# nodes and edges from train id: ca8d3c7918fe30a1ba4218a3f4105a5ddb1e29dd
nodes = ['budget prices', 'is from', 'International Airport', 'France', 'boutique airline company', 'same trip', 
         'Paris', 'is with', 'Hotel Matignon', 'Newark', 'bus', 'company in', 'is in', 'board luxury aircraft', 
         'Charles de Gaulle Airport', 'to make', 'Trying', 'exclusively business class flights', 'Forget', 
         'has launched', 'founder', "L'Avion", 'Frantz Yvelin', 'is by', 'is founder of', 'choice of', 'is', 
         'generation of working business travelers', 'French', 'La Compagnie', 'of', 'flights', 'passengers', 
         'Yvelin', 'offers', 'says', 'airline La Compagnie']

edges = [('budget prices', 'is with'), ('is from', 'Charles de Gaulle Airport'), ('is from', 'Paris'),
         ('International Airport', 'is in'), ('France', 'is in'), ('France', 'company in'), ('France', 'is with'),
         ('boutique airline company', 'is in'), ('same trip', 'is with'), ('Paris', 'is in'),
         ('is with', 'board luxury aircraft'), ('Hotel Matignon', 'is in'), ('Newark', 'is in'),
         ('bus', 'company in'), ('to make', 'Trying'), ('Trying', 'Forget'), ('exclusively business class flights', 'has launched'),
         ('Forget', 'has launched'), ('founder', 'is by'), ("L'Avion", 'is founder of'), ('Frantz Yvelin', 'is by'),
         ('Frantz Yvelin', 'is founder of'), ('choice of', 'La Compagnie'), ('choice of', 'generation of working business travelers'),
         ('is', 'La Compagnie'), ('is', 'French'), ('La Compagnie', 'of'), ('of', 'flights'), ('passengers', 'offers'),
         ('Yvelin', 'says'), ('offers', 'airline La Compagnie'), ('says', 'airline La Compagnie')]

# initialize the tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large')
model.eval()

# function to get embeddings for text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the average embedding for the entire sequence
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# create a NetworkX graph and add nodes with their embeddings
G = nx.DiGraph()
for i, node in enumerate(nodes):
    G.add_node(i, text=node, embedding=get_embedding(node))

# create edges in the NetworkX graph
for edge in edges:
    source = nodes.index(edge[0])  # Find index of the source node
    target = nodes.index(edge[1])  # Find index of the target node
    G.add_edge(source, target)

# convert NetworkX graph to PyTorch Geometric Data object
data = from_networkx(G)

# extract node features from the embeddings
node_features = torch.stack([G.nodes[n]['embedding'] for n in G.nodes])
data.x = node_features

print("Node Features (x):", data.x) # size: [num_nodes, num_node_features]
print("Edge Index (edge_index):", data.edge_index) # size: [2, num_edges]
