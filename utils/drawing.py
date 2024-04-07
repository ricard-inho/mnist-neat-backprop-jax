import networkx as nx
import matplotlib.pyplot as plt

import numpy as np


def draw_graph(trained_model_state, cfg):
    """
    Draw a graph representing the trained model architecture and weights.

    Parameters:
    trained_model_state: Trained model state.
    cfg: Configuration object containing network parameters.
    """

    G = nx.DiGraph()
    num_elements = cfg.network.num_inputs
    start_nodes = [f'input_node_{i+1}' for i in range(min(num_elements, 6))]

    for i, node in enumerate(start_nodes):
        y_pos = i - (len(start_nodes) - 1) / 2 
        G.add_node(node, pos=(0, y_pos))


    prev_layer_nodes = start_nodes
    layer_distance = 0.5
    max_nodes = 0 

    for layer, value in trained_model_state.params['params'].items():
        if 'kernel' in value:
            num_prev_nodes = min(len(value['kernel']), 6) 
            num_nodes = min(len(value['kernel'][0]), 6) 
            max_nodes = max(max_nodes, num_nodes)

        for node_i, node in enumerate(range(num_nodes)):
            current_node = f"{layer}_node_{node}"
            y_pos = node_i - (num_nodes - 1) / 2
            G.add_node(current_node, pos=(layer_distance, y_pos))
            


            for weight in range(num_prev_nodes):
                prev_node = prev_layer_nodes[weight]
                weight_value = value['kernel'][weight][node]
                G.add_edge(prev_node, current_node, weight=weight_value)

        prev_layer_nodes = [f"{layer}_node_{node}" for node in range(num_nodes)]


        if len(value['kernel']) > 6:
            # If there are more than 5 nodes in the layer, add three dots in the middle
            num_nodes = 3
            middle_y = (num_prev_nodes - 1) / 2 - 2.5
            G.add_node(np.random.randint(10000), pos=(layer_distance - 0.5, middle_y - 0.12), shape='dot')
            G.add_node(np.random.randint(10000), pos=(layer_distance - 0.5, middle_y), shape='dot')
            G.add_node(np.random.randint(10000), pos=(layer_distance - 0.5, middle_y + 0.12), shape='dot')

        layer_distance += 0.5

    # Plotting
    # pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')

    distance = 0.5
    count = 1
    height_text = 2
    for i, (layer, value) in enumerate(trained_model_state.params['params'].items()):
        num_nodes = min(len(value['kernel'][0]), 6) 

        if num_nodes <= 4:
            height_text = 2
        elif num_nodes == 5:
            height_text = 2.5
        else:
            height_text = 3

        if i < len(trained_model_state.params['params'].items()) - 1:
            plt.text(distance - 0.08, height_text, f"HL {count}", fontsize=12, color='black')
            plt.text(distance - 0.08, height_text - 0.2, f"num nodes {len(value['kernel'].T)}", fontsize=12, color='black')
            count +=1
            distance += 0.5
        else:
            plt.text(distance - 0.07, height_text, f"Output Layer", fontsize=12, color='black')
    
    plt.text(-0.05, height_text, f"Input Layer", fontsize=12, color='black')


    # Calculate edge widths based on absolute value of weights
    edge_widths = [abs(d['weight']) for u, v, d in G.edges(data=True)]
    edge_widths_normalized = [w / max(edge_widths) * 3 for w in edge_widths]
    
    nx.draw(G, pos, with_labels=False, node_size=1500, node_color='#C70039', font_size=10, font_weight='bold', nodelist=[n for n, d in G.nodes(data=True) if not 'shape' in d])
    nx.draw(G, pos, with_labels=False, node_size=30, node_color='grey', nodelist=[n for n, d in G.nodes(data=True) if 'shape' in d and d['shape'] == 'dot'])

    nx.draw_networkx_edges(G, pos, width=edge_widths_normalized)
    
    
    