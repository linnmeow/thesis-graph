import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def perform_levi_transformation(triples):
    G = nx.Graph()
    for triple in triples:
        if isinstance(triple[0], str):  # check if the first element is a string
            s, r, o = triple[1], triple[2], triple[3]  # if so, use the second element as subject
        else:
            s, r, o = triple[0], triple[1], triple[2]  # otherwise, use the first element as subject
            
        G.add_node(s)
        G.add_node(r)
        G.add_node(o)
        G.add_edge(s, r)
        G.add_edge(r, o)
    return G

def create_directed_graph(triples):
    G = nx.DiGraph()  # use directed graph
    for triple in triples:
        if isinstance(triple[0], str):  # check if the first element is a string (assuming it's a label)
            s, r, o = triple[1], triple[2], triple[3] 
        else:
            s, r, o = triple[0], triple[1], triple[2]  # otherwise, use the first element as subject
            
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, predicate=r)  # add edge with predicate as attribute
    return G

def load_data(data_path): 
    with open(data_path, "r") as f:
        return json.load(f)

def visualize_graph(G, entry_id):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.35)  # positions for all nodes with further increased spacing

    # Drawing the graph with adjusted parameters for better visualization
    nx.draw(G, pos, with_labels=True, node_size=200, node_color="skyblue", font_size=10, font_weight="bold",
            edge_color="grey", width=1.0, alpha=0.6)

    if isinstance(G, nx.DiGraph):
        edge_labels = nx.get_edge_attributes(G, 'predicate')  # get edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)  # draw edge labels
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='grey', width=2.0)  # draw directed edges

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
                # print(G.nodes)
                # print(G.edges)

                # create normal directed graph
                # G = create_directed_graph(triples)
                
                # visualize the graph for the current entry
                visualize_graph(G, entry['id']) 