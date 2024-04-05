import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(trained_model_state):
    G = nx.DiGraph()

    start_nodes = ['input_node_1', 'input_node_2', 'input_node_3', 'input_node_4']
    for i, node in enumerate(start_nodes):
        y_pos = i - (len(start_nodes) - 1) / 2 
        G.add_node(node, pos=(0, y_pos))

    prev_layer_nodes = start_nodes
    layer_distance = 0.5
    max_nodes = 0 

    for layer, value in trained_model_state.params['params'].items():
        if 'kernel' in value:
            num_prev_nodes = len(value['kernel'])
            num_nodes = len(value['kernel'][0])
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
        layer_distance += 0.5

    # Plotting
    # pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')

    # Calculate edge widths based on absolute value of weights
    edge_widths = [abs(d['weight']) for u, v, d in G.edges(data=True)]
    edge_widths_normalized = [w / max(edge_widths) * 3 for w in edge_widths]
    
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10, font_weight='bold')
    node_labels = nx.get_node_attributes(G, 'value')
    nx.draw_networkx_edges(G, pos, width=edge_widths_normalized)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    