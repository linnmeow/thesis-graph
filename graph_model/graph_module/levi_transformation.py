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

def prune_small_subgraphs(G):

    if isinstance(G, nx.DiGraph):
        # for directed graphs, use strongly connected components
        subgraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        # filter out small subgraphs (fewer than 3 nodes)
        filtered_subgraphs = [sg for sg in subgraphs if len(sg.nodes) >= 3]
    else:
        # for undirected graphs, use connected components
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        # filter out small subgraphs (fewer than 4 nodes)
        filtered_subgraphs = [sg for sg in subgraphs if len(sg.nodes) >= 4]
   
    
    # create a new graph combining the filtered subgraphs
    pruned_G = nx.DiGraph() if isinstance(G, nx.DiGraph) else nx.Graph()
    for sg in filtered_subgraphs:
        pruned_G = nx.compose(pruned_G, sg)
    
    return pruned_G


def load_data(data_path): 
    with open(data_path, "r") as f:
        return json.load(f)

def open_json_file(data_dir, file_prefix):
    input_path = os.path.join(data_dir, f"{file_prefix}_processed.json")
    print(f"Loading data from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

def visualize_graph(G, entry_id):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.35)  

    nx.draw(G, pos, with_labels=True, node_size=200, node_color="skyblue", font_size=10, font_weight="bold",
            edge_color="grey", width=1.0, alpha=0.6)

    if isinstance(G, nx.DiGraph):
        edge_labels = nx.get_edge_attributes(G, 'predicate')  
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8) 
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='grey', width=2.0) 

    plt.title(f"Knowledge Graph for Entry ID: {entry_id}")
    plt.show()


if __name__ == "__main__":
    data_folder = "/Users/lynn/desktop/thesis/cnn_dm/original_triples/"  
    prefixes = ["train", "valid", "test"]  

    target_entry_id = 'ca8d3c7918fe30a1ba4218a3f4105a5ddb1e29dd'
    
    for prefix in prefixes:
        data_path = os.path.join(data_folder, f"{prefix}_processed.json")
    
    # data_path = "examples.json" # toy dataset
    # target_entry_id = None
    # load data
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # process each entry in the dataset
    for entry in data:
        if 'triples' in entry:
            triples = entry['triples']
            
            # perform Levi transformation
            G = perform_levi_transformation(triples)

            # create normal directed graph
            # G = create_directed_graph(triples)
            
            G = prune_small_subgraphs(G)
            # print(G.nodes)
            # print(G.edges)

            if target_entry_id is None or entry['id'] == target_entry_id:
                print(f"Nodes for Entry ID {entry['id']}:")
                print(G.nodes)
                print("\nEdges for Entry ID {entry['id']}:")
                print(G.edges(data=True))  # the data=True flag includes edge attributes (like predicates)
                print("\n")
            
            if len(G.nodes) > 0:
                visualize_graph(G, entry['id'])
            else:
                print(f"All subgraphs were pruned for Entry ID: {entry['id']}")