import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def perform_levi_transformation(triples):
    G = nx.Graph()
    for triple in triples:
        if isinstance(triple[0], str):  # check if the first element is a string (assuming it's a label)
            s, r, o = triple[1], triple[2], triple[3]  # adjust indices based on your actual data structure
        else:
            s, r, o = triple[0], triple[1], triple[2]  # otherwise, use the first element as subject
            
        G.add_node(s)
        G.add_node(r)
        G.add_node(o)
        G.add_edge(s, r)
        G.add_edge(r, o)
    return G

def load_data(data_path):
    with open(data_path, "r") as f:
        return json.load(f)

def visualize_graph(G, entry_id):
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # positions for all nodes
    # nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=15, font_weight="bold", 
            edge_color="grey", width=2.0)
    plt.title(f"Knowledge Graph for Entry ID: {entry_id}")
    plt.show()

if __name__ == "__main__":
    data_folder = "/Users/lynn/desktop/thesis/cnn_dm4openie_extraction/filtered_conf_0.6_length_5"  
    prefixes = ["train", "valid", "test"]  
    
    for prefix in prefixes:
        data_path = os.path.join(data_folder, f"{prefix}_filtered.json")
        
        # load data
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # process each entry in the dataset
        for entry in data:
            if 'triples' in entry:
                triples = entry['triples']
                
                # perform Levi transformation
                G = perform_levi_transformation(triples)
                
                # visualize the graph for the current entry
                visualize_graph(G, entry['id']) 