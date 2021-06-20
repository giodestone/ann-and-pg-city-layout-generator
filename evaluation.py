from math import sqrt
import random
from networkx.algorithms.shortest_paths.generic import shortest_path_length
from networkx.exception import NetworkXNoPath

from tensorflow import keras
from tensorflow.python.keras.utils.vis_utils import plot_model
import encoder_decoder_rnn_roads
import get_nodes_networkx
import networkx
import matplotlib, matplotlib.pyplot
import numpy
import scipy, scipy.stats
import progressbar

def evaluate_graphs(test_set_name:str, graphs:list, graph_titles:list, ground_truth_cartesian:networkx.graph, comparison_graphs:list, comparison_graph_titles:list, post_append="", display_figures=False):
    FIGURE_DPI = 500

    # Road Length
    road_length = matplotlib.pyplot.boxplot([calculate_road_lengths(graph) for graph in graphs], labels=[title for title in graph_titles], notch=True, vert=True)
    matplotlib.pyplot.title("Road Lengths" + " " + test_set_name)
    if display_figures:
        matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("Road Lengths" + " " + test_set_name + post_append, dpi=FIGURE_DPI)
    matplotlib.pyplot.clf()

    # Junction Connectivity
    junction_connectivity = matplotlib.pyplot.boxplot([calculate_road_connectivity(graph) for graph in graphs], labels=[title for title in graph_titles], notch=True, vert=True, autorange=True, manage_ticks=True)
    matplotlib.pyplot.title("Junction Connectivity" + " " + test_set_name)
    if display_figures:
        matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("Junction Connectivity" + " " + test_set_name + post_append, dpi=FIGURE_DPI)
    matplotlib.pyplot.clf()

    # Transport convenience
    transport_conv = matplotlib.pyplot.boxplot([transport_convenience(graph) for graph in graphs], labels=[title for title in graph_titles], notch=True, vert=True, autorange=True, manage_ticks=True)
    matplotlib.pyplot.title("Transport Convenience" + " " + test_set_name)
    if display_figures:
        matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("Transport Convenience" + " " + test_set_name + post_append, dpi=FIGURE_DPI)
    matplotlib.pyplot.clf()

    # Graph Density
    graph_density = matplotlib.pyplot.bar([title for title in graph_titles], [road_density(graph) for graph in graphs])
    matplotlib.pyplot.title("Graph Density" + " " + test_set_name)
    if display_figures:
        matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("Graph Density" + " " + test_set_name + post_append, dpi=FIGURE_DPI)
    matplotlib.pyplot.clf()

    # Diversity Metric

    for i in range(len(comparison_graphs)):
        matplotlib.pyplot.bar(comparison_graph_titles[i], diversity_metric(ground_truth_cartesian, comparison_graphs[i]))

    matplotlib.pyplot.title("Diversity Metric" + " " + test_set_name)
    matplotlib.pyplot.savefig("Diversity Metric" + " " + test_set_name + post_append, dpi=FIGURE_DPI)
    if display_figures:
        matplotlib.pyplot.show()
    matplotlib.pyplot.clf()


def calculate_road_lengths(graph:networkx.Graph):
    road_lengths = []
    
    for edge in graph.edges:
        frmpos = graph.nodes[edge[0]]['position']
        topos = graph.nodes[edge[1]]['position']
        distance = sqrt(pow(topos[0] - frmpos[0], 2) + pow(topos[1] - frmpos[1], 2))
        road_lengths.append(distance)

    return road_lengths

def calculate_road_connectivity(graph:networkx.Graph):
    num_connections = []

    for node in graph.nodes:
        successors = list(networkx.bfs_successors(graph, node, 1))
        num_conns = len(successors[0][1])
        num_connections.append(num_conns)

    return num_connections

def road_density(graph:networkx.Graph):
    return networkx.density(graph)

def diversity_metric(ground_truth_cartesian:networkx.Graph, generated_graph:networkx.Graph):

    # TODO: This method is silly inefficient (worst case = O(a.n * b.n)), could benefit from sorting all of the nodes in sectors based on position and fixing them.
    num_nodes_overlapping = 0

    num_generated_nodes_processed = 0

    bar = progressbar.ProgressBar(widgets=["Calculating Diversity Metric... ", progressbar.Timer(), "  ", progressbar.ETA(), "  ", progressbar.Bar(), " ", progressbar.Percentage()], max_value=generated_graph.number_of_nodes())
    bar.start()

    for generated_node in generated_graph.nodes:
        bar.update(num_generated_nodes_processed)
        for gt_node in ground_truth_cartesian.nodes:
            bar.update()
            if check_if_in_radius(ground_truth_cartesian, generated_graph, gt_node, generated_node):
                num_nodes_overlapping += 1
                continue
        
        num_generated_nodes_processed += 1
    
    bar.finish()
    
    # Sometimes when generated graph and ground truth are subsets of eachother the number increases too much.
    num_nodes_overlapping = (float(num_nodes_overlapping) / float(generated_graph.number_of_nodes())) * 100
    if num_nodes_overlapping > 100:
        num_nodes_overlapping = 100
    return num_nodes_overlapping


def check_if_in_radius(graph_a:networkx.Graph, graph_b:networkx.Graph, node_a, node_b, a_radius_meters=10, b_radius_meters=0.5):
    """Check if `node a` (in `graph_a`) is in `node_b`'s radius.

    Args:
        graph_a (networkx.Graph): Graph which contains `node_a`. Must be cartesian.
        graph_b (networkx.Graph): Graph which contains `node_b`. Must be cartesian.
        node_a: Node in `graph_a` which contains a position.
        node_b: Node in `graph_b` which contains a position.
        a_radius_meters (int, optional): `node_a`'s radius. Defaults to 10.
        b_radius_meters (float, optional): `node_b`'s radius. Defaults to 0.5.

    Returns:
        bool: True if contained, False if not.
    """

    circle_1_pos = graph_a.nodes[node_a]['position']
    circle_2_pos = graph_b.nodes[node_b]['position']
    circle_1_radius = a_radius_meters
    circle_2_radius = b_radius_meters

    dx = circle_1_pos[0] - circle_2_pos[0]
    dy = circle_1_pos[1] - circle_2_pos[1]

    distance = sqrt(pow(dx, 2) + pow(dy, 2))

    return distance < circle_1_radius + circle_2_radius

def transport_convenience(graph:networkx.Graph):
    PERCENT_RANDOM_DESTINATION_NODES = 0.25 # How many nodes should be destination nodes

    start_node_pos = None # TODO - may need to convert existing graphs to cartesian to actually make this work.
    furthest_node = None

    all_nodes = list(graph.nodes)
    start_nodes = random.sample(all_nodes, int(PERCENT_RANDOM_DESTINATION_NODES * len(all_nodes)))

    distances_travelled = []
    
    for start_node in start_nodes:
        for node_id in all_nodes:
            shortest_path_len = None
            
            try:
                shortest_path_len = shortest_path_length(graph, start_node, node_id)
            except NetworkXNoPath:
                continue

            distances_travelled.append(shortest_path_len)

    return distances_travelled

def get_AABB_of_cartesian_graph(graph:networkx.Graph):
    """Get Axis Aligned Bounding Box (AABB) of a `graph`.

    Args:
        graph (networkx.Graph): Graph with a cartesian positions.

    Returns:
        float, float, float, float: x min, y min, x max, y max positions, respectively
    """

    # Find min/max bounding box for the graph
    x_min, y_min = float('inf')
    x_max, y_max = float('-inf')
    
    for node in graph.nodes:
        pos = graph.nodes[node]['position']
        
        x_min = min(pos[0], x_min)
        y_min = min(pos[1], y_min)

        x_max = max(pos[0], x_max)
        y_max = max(pos[1], y_max)

    return x_min, y_min, x_max, y_max

def get_width_height_of_AABB(x_min:float, y_min:float, x_max:float, y_max:float):
    return (abs(x_max) - abs(x_min), abs(y_max) - abs(y_min))

def get_area(width:float, height:float):
    return width*height

def get_area_of_graph(graph:networkx.Graph):
    return get_area(get_width_height_of_AABB(get_AABB_of_cartesian_graph(get_nodes_networkx.convert_lat_lon_graph_to_cartesian_graph(graph))))

def translate_graph_to_centre(graph:networkx.Graph):
    # Find min/max bounding box for the graph
    x_min, y_min, x_max, y_max = get_AABB_of_cartesian_graph(graph)

    # Find centre of bounding box
    x_centre = ((x_max - x_min) / 2) + x_min
    y_centre = ((y_max - y_min) / 2) + y_min

    for node in graph.nodes:
        node_pos = graph.nodes[node]['position']

        graph.nodes[node]['position'] = (node_pos[0] - x_centre, node_pos[1] - y_centre)

    return graph

def train_model_and_save(query_string:str, prefix:str, num_epochs:int, num_units:int, num_samples:int, max_num_paths_per_individual_node:int, num_features:int, max_num_input_paths:int, max_num_input_nodes:int, max_num_output_nodes:int, num_dimensions:int):
    graph = get_nodes_networkx.get_map_graph_from_bounding_box(query_string)

    training_sequence_non_encoded = encoder_decoder_rnn_roads.generate_training_sequence_for_graph(num_samples, max_num_paths_per_individual_node, max_num_input_paths, num_dimensions, graph, max_num_input_nodes, get_nodes_networkx.get_signed_distance_between_nodes)
    training_sequence, prediction_sequence = encoder_decoder_rnn_roads.encode_training_sequence(training_sequence_non_encoded, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)

    encoder_decoder_rnn_roads.save_training_sequences(prefix + " (num samples {}) (max incoming paths per node {}) (num input paths {}) (num input nodes {})".format(num_samples, max_num_paths_per_individual_node, max_num_input_paths, max_num_input_nodes), training_sequence_non_encoded, training_sequence, prediction_sequence, should_overwrite_existing=False)

    model = encoder_decoder_rnn_roads.define_models(max_num_input_nodes, max_num_output_nodes, num_dimensions, num_features, num_units)
    
    for _ in range(num_epochs): # epochs
        for i in range(len(training_sequence)):
            model.fit(x=training_sequence[i], y=prediction_sequence[i], epochs=1, batch_size=max_num_input_paths)

        if (_ + 1) % 10 == 0:
            model.save("model " + prefix + " (num epochs {})".format(_ + 1) + " (max incoming paths per node {}) (num input paths {}) (num input nodes {})".format(num_samples, max_num_paths_per_individual_node, max_num_input_paths, max_num_input_nodes))

def load_model(prefix:str, num_epochs:int, num_units:int, num_samples:int, max_num_paths_per_individual_node:int, num_features:int, max_num_input_paths:int, max_num_input_nodes:int, max_num_output_nodes:int, num_dimensions:int):
    name = "model " + prefix + " (num epochs {})".format(num_epochs) + " (max incoming paths per node {}) (num input paths {}) (num input nodes {})".format(num_samples, max_num_paths_per_individual_node, max_num_input_paths, max_num_input_nodes)
    return keras.models.load_model(name)

def generate_prediction_graph(model:keras.Model, ground_truth:networkx.Graph, max_num_nodes_to_generate:int=500, start_depth:int=3, start_node_id=None):

    road_graph, unvisited_nodes = encoder_decoder_rnn_roads.get_starting_map(ground_truth, start_depth, get_nodes_networkx.get_signed_distance_between_nodes, start_node_id)
    
    for i in range (max_num_nodes_to_generate):
        current_node_id = unvisited_nodes.pop(0)
        position_changes = []

        bar = progressbar.ProgressBar(widgets=['Generating prediction sequence...  ', progressbar.AnimatedMarker(), '  ', progressbar.Timer(), '  ', progressbar.AdaptiveETA()])
        bar.start()
        encoder_decoder_rnn_roads.generate_paths_for_node(1, position_changes, road_graph, current_node_id, max_num_input_nodes, max_num_paths_per_individual_node, max_num_input_paths, encoder_decoder_rnn_roads.get_cartesian_distance_between_nodes, bar)
        x, y = encoder_decoder_rnn_roads.encode_training_sequence(position_changes, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
        bar.finish()

        prediction = model.predict(x[-1])
        prediction = encoder_decoder_rnn_roads.decode_prediction(prediction, num_features)

        for new_node in prediction:
            if not (new_node[0] == 0 and new_node[1] == 0):
                new_node_id = str(i)
                
                road_graph.add_node(new_node_id, position=(road_graph.nodes[current_node_id]['position'][0] +  new_node[0], road_graph.nodes[current_node_id]['position'][1] +  new_node[1]))
                road_graph.add_edge(current_node_id, new_node_id)
                unvisited_nodes.append(new_node_id)

        if len(unvisited_nodes) == 0:
            break

    return road_graph


num_samples = 1000
num_epochs = 30
num_units = 500

max_num_paths_per_individual_node = 20

num_features = 200 # aka cardinality
max_num_input_paths = 50
max_num_input_nodes = 20
max_num_output_nodes = 6
num_dimensions = 2


# Dundee Around Central Mosque 56.459124,-2.985106,56.464221,-2.977424
# Berwick Dr, Dundee 56.485423,-2.946804,56.496866,-2.905970
# New York Times Square, 40.752247,-73.990324,40.759269,-73.982642

# ===== Train 30 epochs of Dundee Central, Betwrick Dr, and New York + save training sequences =====
train_model_and_save('way["highway"](56.459124,-2.985106,56.464221,-2.977424); (._;>;); out body;', "Dundee Central Mosque", num_epochs, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
train_model_and_save('way["highway"](40.752247,-73.990324,40.759269,-73.982642); (._;>;); out body;', "Times Square", num_epochs, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
train_model_and_save('way["highway"](56.485423,-2.946804,56.496866,-2.905970); (._;>;); out body;', "Berwick Dr Dundee", num_epochs, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)

# ===== Show Generated Graphs of 30 Epoch Models ====
# Generate dundee central graph.
# dundee_central_source_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](56.459124,-2.985106,56.464221,-2.977424); (._;>;); out body;')
# get_nodes_networkx.show_graph(dundee_central_source_graph)

# dundee_central_30_epochs_model = load_model("Dundee Central Mosque", 30, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
# dundee_central_30_epochs_model_predicited_graph = generate_prediction_graph(dundee_central_30_epochs_model, dundee_central_source_graph, start_node_id='45350941')
# get_nodes_networkx.show_graph(dundee_central_30_epochs_model_predicited_graph)


# new_york_source_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](40.752247,-73.990324,40.759269,-73.982642); (._;>;); out body;')
# get_nodes_networkx.show_graph(new_york_source_graph)

# new_york_30_epochs_model = load_model("Times Square", 10, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
# new_york_30_epochs_model_predicited_graph = generate_prediction_graph(new_york_30_epochs_model, new_york_source_graph, start_node_id='42430344')
# get_nodes_networkx.show_graph(new_york_30_epochs_model_predicited_graph)


# baxter_park_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](56.467837,-2.958262,56.474190,-2.951138); (._;>;); out body;')
# get_nodes_networkx.show_graph(baxter_park_graph)

# baxter_park = keras.models.load_model("Big Baxter Park Area Model 20 epochs (50 inpt paths), (20 inpt nodes), (20 paths per individual node), (500 unit enc")
# baxter_park_prediction_graph = generate_prediction_graph(baxter_park, baxter_park_graph)
# get_nodes_networkx.show_graph(baxter_park_prediction_graph)


# Evaluate Dundee Central

# dundee_central_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](56.459124,-2.985106,56.464221,-2.977424); (._;>;); out body;')
# evaluate_graphs("Baxter Park", [dundee_central_graph, dundee_central_graph, dundee_central_graph], ["Ground Truth", "Generated Graph 10 Epochs", "Generated Graph 20 Epochs"])

#Evaluation of all graphs
dundee_central_source_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](56.459124,-2.985106,56.464221,-2.977424); (._;>;); out body;')
get_nodes_networkx.show_graph(dundee_central_source_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Dundee Central Source Graph")

dundee_central_30_epochs_model = load_model("Dundee Central Mosque", 30, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
dundee_central_30_epochs_model_predicited_graph = generate_prediction_graph(dundee_central_30_epochs_model, dundee_central_source_graph, start_node_id='45350941')
get_nodes_networkx.show_graph(dundee_central_30_epochs_model_predicited_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Dundee 30 Epoch Prediction")

dundee_central_20_epochs_model = load_model("Dundee Central Mosque", 20, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
dundee_central_20_epochs_model_predicited_graph = generate_prediction_graph(dundee_central_20_epochs_model, dundee_central_source_graph, start_node_id='45350941')
#get_nodes_networkx.show_graph(dundee_central_20_epochs_model_predicited_graph)

dundee_central_10_epochs_model = load_model("Dundee Central Mosque", 10, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
dundee_central_10_epochs_model_predicited_graph = generate_prediction_graph(dundee_central_10_epochs_model, dundee_central_source_graph, start_node_id='45350941')
#get_nodes_networkx.show_graph(dundee_central_10_epochs_model_predicited_graph)

evaluate_graphs("Dundee Central", [get_nodes_networkx.convert_lat_lon_graph_to_cartesian_graph(dundee_central_source_graph), dundee_central_10_epochs_model_predicited_graph, dundee_central_20_epochs_model_predicited_graph, dundee_central_30_epochs_model_predicited_graph], ["Ground Truth", "10 Epochs", "20 Epochs", "30 Epochs"], get_nodes_networkx.convert_lat_lon_graph_to_cartesian_graph(dundee_central_source_graph), [dundee_central_10_epochs_model_predicited_graph, dundee_central_20_epochs_model_predicited_graph, dundee_central_30_epochs_model_predicited_graph], ["10 Epochs", "20 Epochs", "30 Epochs"], post_append="grad show")

# Evaluate New York

new_york_source_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](40.752247,-73.990324,40.759269,-73.982642); (._;>;); out body;')
get_nodes_networkx.show_graph(new_york_source_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="New York Source Graph")
new_york_30_epochs_model = load_model("Times Square", 30, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
new_york_30_epochs_model_predicited_graph = generate_prediction_graph(new_york_30_epochs_model, new_york_source_graph, start_node_id='42430344')
get_nodes_networkx.show_graph(new_york_30_epochs_model_predicited_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="New York 30 Epoch Prediction")

new_york_20_epochs_model = load_model("Times Square", 20, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
new_york_20_epochs_model_predicited_graph = generate_prediction_graph(new_york_20_epochs_model, new_york_source_graph, start_node_id='42430344')
# get_nodes_networkx.show_graph(new_york_20_epochs_model_predicited_graph)

new_york_10_epochs_model = load_model("Times Square", 10, num_units, num_samples, max_num_paths_per_individual_node, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
new_york_10_epochs_model_predicited_graph = generate_prediction_graph(new_york_10_epochs_model, new_york_source_graph, start_node_id='42430344')
# get_nodes_networkx.show_graph(new_york_10_epochs_model_predicited_graph)

evaluate_graphs("New York", [get_nodes_networkx.convert_lat_lon_graph_to_cartesian_graph(new_york_source_graph), new_york_10_epochs_model_predicited_graph, new_york_20_epochs_model_predicited_graph, new_york_30_epochs_model_predicited_graph], ["Ground Truth", "10 Epochs", "20 Epochs", "30 Epochs"], get_nodes_networkx.convert_lat_lon_graph_to_cartesian_graph(new_york_source_graph), [new_york_10_epochs_model_predicited_graph, new_york_20_epochs_model_predicited_graph, new_york_30_epochs_model_predicited_graph], ["10 Epochs", "20 Epochs", "30 Epochs"], post_append="grad show")

# model = encoder_decoder_rnn_roads.define_models(max_num_input_nodes, max_num_output_nodes, num_dimensions, num_features, 500)
# plot_model(model)