from imp import source_from_cache
from math import ceil, log10
from random import randint
from threading import active_count
import networkx
from networkx.classes.function import nodes
import numpy
import numpy.doc
from numpy import array
from numpy import argmax
from numpy import array_equal
from numpy.lib.function_base import append
import progressbar
import tensorflow
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow.python.keras.layers.core import RepeatVector
import get_nodes_networkx
from get_nodes_networkx import convert_path_to_changes_in_distance, getBearing, get_all_paths_to_node_depth_limited, get_all_simple_paths_from_node, get_destination_point, get_nodes_sorted_by_bfs_rel_pos, get_signed_distance_between_nodes, show_graph

import os

from sklearn.model_selection import train_test_split
import pickle

import time
from threading import Thread
from ThreadSafeBool import ThreadSafeBoolRef

import random

import cProfile
import pstats
from pstats import SortKey

SAVED_LIST_FILE_EXTENSION = ".pkl"
SAVED_NUMPY_LIST_FILE_EXTENSION = ".npy"
TRAINING_SEQUENCE_NON_ENCODED_NAME = 'training sequence non encoded'
TRAINING_SEQUENCE_NAME = 'training sequence'
PREDICT_SEQUENCE_NAME = 'prediction sequence'

def define_models(num_input, num_output, num_dimensions, cardinality, num_units):

    # https://vel.life/%E9%98%85%E3%80%8Along-short-term-memory-networks-with-python%E3%80%8B/long-short-term-memory-networks-with-python.pdf chapter 9

    model = keras.Sequential()
    model.add(keras.layers.GRU(num_units, input_shape=(num_input * num_dimensions, cardinality), name="encoder"))
    model.add(keras.layers.RepeatVector(num_output * num_dimensions))
    model.add(keras.layers.GRU(num_units, return_sequences=True, name="decoder"))
    model.add(keras.layers.TimeDistributed(Dense(units=cardinality, activation='softmax'))) # output of (12, cardinality)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    return model

def generate_training_sequence_for_graph(max_sample_size:int, max_num_paths_per_node:int, max_num_previous_paths:int, num_dimensions:int, graph:networkx.Graph, depth_limit:int, distance_function):
    """ Generate a training sequence.

    Args:
        graph (networkx.Graph): Graph to generate the training sequence from.
        depth_limit: Maximum length of a path. Minimum of 1.

    Returns:
        list([tuple], [tuple]): Training sequence. [0] are the distances of the path (relative to eachother) leading up to the node, (ordered with first elements being closesest to the node); [1] are the distances to the node(s) connecting to the node.
    """
    nodes_to_process = min(max_sample_size, len(graph.nodes))
    
    bar = progressbar.ProgressBar(widgets=['Generating Training Sequence From Nodes...  ', progressbar.Timer(), '  ', progressbar.AdaptiveETA(), '  ', progressbar.Bar(), '  ', progressbar.Percentage()], max_value=nodes_to_process)
    bar.start()
    training_data = []
    nodes_processed = 0
    
    for node_id in graph.nodes:
        generate_paths_for_node(max_sample_size, training_data, graph, node_id, depth_limit, max_num_previous_paths, max_num_paths_per_node, distance_function, bar)
        nodes_processed += 1

        if nodes_processed >= nodes_to_process:
            break

        bar.update(nodes_processed)
        
    bar.finish()
    return training_data

def generate_paths_for_node(max_sample_size:int, training_data, graph:networkx.graph, node_id, depth_limit:int, max_num_paths_leading_to_node:int, max_num_paths_per_node:int, distance_function, bar:progressbar.ProgressBar):
    paths = get_all_simple_paths_from_node(graph, node_id, depth_limit, max_num_paths_leading_to_node, max_num_paths_per_node, bar)

    # Find all unique prediction nodes and put them into individual arrays. This must be done as paths predicting separate nodes must be passed in differently.
    grouped_paths = []

    unique_nodes = list() # a tuple is not hashable using a set(), need to use list() instead
    for path in paths:
        if (path[0], path[1]) not in unique_nodes:
            unique_nodes.append((path[0], path[1])) 

    bar.update()

    for unique_node in unique_nodes:
        new_list = []
        new_list = [x for x in paths if x[0] == unique_node[0] and x[1] == unique_node[1]]
        grouped_paths.append(new_list)

    for path_group in grouped_paths:
        bar.update()

        path_group_training_data = []
        for path in path_group:
            
            bar.update()

            prediction_node_ids = path[0]
            current_node_id = path[1]
            previous_node_ids = path[2]
            
            incoming_path_relative_distances = convert_path_to_changes_in_distance(graph, previous_node_ids, distance_function)

            prediction_nodes_relative_distances = []
            for prediction_node_id in prediction_node_ids:
                prediction_nodes_relative_distances.append(distance_function(graph, current_node_id, prediction_node_id))

            path_group_training_data.append((incoming_path_relative_distances, prediction_nodes_relative_distances))
        
        training_data.append(path_group_training_data)

        if len(training_data) >= max_sample_size:
            return

    return


def encode_training_sequence(training_sequence:list, cardinality:int, max_num_previous_paths:int, max_num_input_nodes:int, max_num_output_nodes:int, num_dimensions:int):    
    # cardinality should be the size of the total range (including negative)
    # up to six incoming nodes, and up to 40 outgoing nodes.

    # suggested output
    # needs to be padded at the start with empty paths [0,0, 0,0, 0,0, 0,0, 0,0, ...] if under 
    # [x, y,  x, y,  x, y,  x, y,  x, y,  x, y,  x, y]
    # all x,y need to be zero if they don't exist.

    # suggested output
    # needs to be padded with zeros with paths which have been made empty.
    # [x, y,  x, y,  ...  x, y]
    # all x,y that are empty should be zero.

    empty_tuple = (0, 0)

    empty_incoming_path = []
    for _ in range(max_num_input_nodes):
        empty_incoming_path.append(empty_tuple)

    empty_prediction = []
    for _ in range(max_num_output_nodes):
        empty_prediction.append(empty_tuple)

    # Split paths that have more than 100 paths into new paths.
    bar = progressbar.ProgressBar(widgets=["Ensuring that there are a maximum of {} previous paths. This step can take a while...  ".format(max_num_previous_paths), progressbar.AnimatedMarker(), '  ', progressbar.Timer(), '  ', progressbar.AdaptiveETA()])
    bar.start()
    training_sequences_to_remove = []
    training_sequences_to_add = []
    for paths in training_sequence:
        bar.update()
        if len(paths) > 5: # replace 2 with max_num_input_paths
            training_sequences_to_remove.append(paths)
            training_sequences_to_add.extend(split_list_into_chunks(paths, 5)) # replace 2 with max_num_input_paths

    for ts_to_remove in training_sequences_to_remove:
        bar.update()
        training_sequence.remove(ts_to_remove)
    
    bar.update()
    training_sequence.extend(training_sequences_to_add)
    bar.update()

    training_sequences_to_add.clear()
    training_sequences_to_remove.clear()

    bar.finish()

    # Set up x+y shapes.
    input_shape = (len(training_sequence), max_num_previous_paths, max_num_input_nodes * num_dimensions, cardinality) # x shape
    output_shape = (len(training_sequence), max_num_previous_paths, max_num_output_nodes * num_dimensions, cardinality) # y shape

    # faster than zeroing out anything.
    x = numpy.empty(input_shape, dtype='float32')
    y = numpy.empty(output_shape, dtype='float32')
    
    # Encode training sequence.
    bar = progressbar.ProgressBar(widgets=['Encoding training sequence...  ', progressbar.Timer(), '  ', progressbar.AdaptiveETA(), '  ', progressbar.Bar(), ' ', progressbar.Percentage()], max_value=len(training_sequence))
    bar.start()
    train_sequence_index = 0
    for paths in training_sequence:
        # Pad with empty paths at the start to bring the length up to max_num_input_previous_paths.
        for _ in range(max_num_previous_paths - len(paths)):
            paths.insert(0, (empty_incoming_path, empty_prediction))

        if len(paths) > max_num_previous_paths:
            raise Exception("Error: The training sequence has too many paths! The number of paths must be shortened to max_num_previous_paths.")
        
        path_index = 0
        for path in paths:
            # Pad incoming path with empty distances at the start to bring the length up to max_num_input_nodes.
            for _ in range(max_num_input_nodes - len(path[0])):
                path[0].insert(0, empty_tuple)

            if len(path[0]) > max_num_input_nodes:
                raise Exception("Error: The input path is too long! The number of input paths must be shortened to max_num_input_nodes.") # They should already be a max of max_num_input_nodes if generate_training_sequence_for_graph was called correctly.
                
            # Pad prediction with empty tuples at the end to bring up the length to max_num_output_nodes.
            for _ in range(max_num_output_nodes - len(path[1])):
                path[1].append(empty_tuple)

            if len(path[1]) > max_num_output_nodes:
                raise Exception("Error: Too many prediction nodes! The number of predictions must be shortened to max_num_output_nodes.") # They should already be a max of max_num_output_nodes if generate_training_sequence_for_graph was called correctly.
            
            # Encode input and move the previous path into x
            input_path_node_index = 0
            for input_path_node in path[0]:
                dist_x = input_path_node[0]
                dist_y = input_path_node[1]

                # Shift up as negative values are not encoded. Remember when decoding to move back!
                dist_x += cardinality / 2
                dist_y += cardinality / 2

                dist_x_enc = keras.utils.to_categorical(dist_x, cardinality)
                dist_y_enc = keras.utils.to_categorical(dist_y, cardinality)

                x[train_sequence_index, path_index, input_path_node_index] = dist_x_enc
                x[train_sequence_index, path_index, input_path_node_index + 1] = dist_y_enc

                input_path_node_index += 2
            
            # Encode prediction and set values in y
            prediction_node_index = 0
            for prediction_node in path[1]:
                dist_x = prediction_node[0]
                dist_y = prediction_node[1]

                # Shift up as negative values are not encoded. Remember when decoding to move back!
                dist_x += cardinality / 2
                dist_y += cardinality / 2

                dist_x_enc = keras.utils.to_categorical(dist_x, cardinality)
                dist_y_enc = keras.utils.to_categorical(dist_y, cardinality)

                y[train_sequence_index, path_index, prediction_node_index] = dist_x_enc
                y[train_sequence_index, path_index, prediction_node_index + 1] = dist_y_enc

                prediction_node_index += 2
            
            path_index += 1

        train_sequence_index += 1
        bar.update(train_sequence_index)

    bar.finish()

    return x, y

def split_list_into_chunks(l:list, max_size:int):
    """Split a list into chunks of maximum length max_size. If the list doesn't evenly divide the last list will be smaller than max_size/

    Args:
        l (list): List to split into chunks
        max_size (int): Maximum length of a chunk.

    Returns:
        list: A list of lists which have a maximum length of max_size.
    """

    new_list = []
    max_index = len(l) - 1
    for i in range(0, len(l), max_size):
        if i + max_size > max_index:
            new_list.append(l[i:max_index])
            break
        else:
            new_list.append(l[i:i + max_size])
    
    return new_list

def decode_prediction(pred:list, cardinality:int):
    prediction = pred[-1]

    decoded_predictions = []

    for node in prediction:
        # round biggest value up to 1
        max_value = 0
        max_value_index = 0
        
        for i in range(len(node)):
            if node[i] > max_value:
                max_value = node[i]
                max_value_index = i
        
        final_val = max_value_index - (cardinality / 2) # reverse encoding

        decoded_predictions.append(final_val)

        # node[max_value_index] = 1

        # node_val = one_hot_decode(node)

    prediction_sequence_decoded = []

    for i in range(0, len(decoded_predictions) - 1, 2):
        prediction_sequence_decoded.append((decoded_predictions[i], decoded_predictions[i + 1]))

    return prediction_sequence_decoded

def get_cartesian_distance_between_nodes(graph: networkx.Graph, from_node_id, to_node_id):
    """Get the distance between nodes, assuming that the positions are stored in a cartesian format.

    Args:
        graph (networkx.Graph): Graph which nodes are in.
        from_node_id ([type]): [description]
        to_node_id ([type]): [description]

    Returns:
        tuple: [description]
    """
    from_node_pos = graph.nodes[from_node_id]['position']
    to_node_pos = graph.nodes[to_node_id]['position']

    x = to_node_pos[0] - from_node_pos[0]
    y = to_node_pos[1] - from_node_pos[1]
    
    return (x, y)

def encode_path_into_cartesian_coordinates(graph:networkx.Graph, path:list, max_num_input_nodes:int, num_dimensions:int, cardinality:int):
    """For encoding a single path (whose positions are actually in cartesian coordinates) as denoted by a sequence of IDs into
    changes in distance from eachother. Pads the beginning of the distances with zeros.

    Args:
        graph (networkx.Graph): [description]
        path (list): IDs of items contained in graph. Requires a minimum of two IDs.
        max_num_input (int):  Number of input nodes.
        num_dimensions (int): Number of position dimensions that a node has.
        cardinality (int): [description]

    Returns:
        [type]: [description]
    """
    path = path[2]

    if len(path) > max_num_input_nodes:
        path = path[0:max_num_input_nodes] # make sure the input is max 40

    # Convert to distances.
    path_changes_in_distance = []
    for i in range(len(path) - 1):
        pos_from = graph.nodes[path[i]]['position']
        pos_to = graph.nodes[path[i + 1]]['position']

        change_in_x = pos_to[0] - pos_from[0]
        change_in_y = pos_to[1] - pos_from[1]

        path_changes_in_distance.append((change_in_x, change_in_y))
    
    # Fill in start with zeros.
    for i in range(max_num_input_nodes - len(path_changes_in_distance)):
        path_changes_in_distance.insert(0, (0, 0)) 
    
    # Encode and store in array.
    out_path = []
    for i in range(len(path_changes_in_distance)):
        change_in_x = path_changes_in_distance[i][0]
        change_in_y = path_changes_in_distance[i][1]

        change_in_x += cardinality / 2
        change_in_y += cardinality / 2

        change_in_x = keras.utils.to_categorical(change_in_x, cardinality)
        change_in_y = keras.utils.to_categorical(change_in_y, cardinality)

        out_path.append(change_in_x)
        out_path.append(change_in_y)

    return out_path

def save_list_to_disk(l, prefix:str, list_name:str, should_overwrite_existing:bool=True):
    """Save a list (or variable) to file, prefixed with prefix, then list_name. If l is numpy.ndarray, it will be saved using numpy.save() function; pickle.dump() otherwise.

    Args:
        l (any): list to save.
        prefix (str): first part of save string.
        list_name (str): name of the list
        should_overwrite_existing (bool): Whether to overwrite an existing file if it exists.
    """

    final_file_name = prefix + list_name

    if should_overwrite_existing and (os.path.isfile(final_file_name + SAVED_LIST_FILE_EXTENSION) or os.path.isfile(final_file_name + SAVED_NUMPY_LIST_FILE_EXTENSION)):
        return

    if type(l) is numpy.ndarray:
        with open(final_file_name + SAVED_NUMPY_LIST_FILE_EXTENSION, "wb") as f:
            numpy.save(f, l)
    else:
        with open(final_file_name + SAVED_LIST_FILE_EXTENSION, "wb") as f:
            pickle.dump(l, f)

def unknown_time_taken_progressbar_update(bar:progressbar.ProgressBar, should_stop:ThreadSafeBoolRef, update_period_seconds:int=0.1):
    """Function for thread. Thread should have daemon=True. Calls .update() on bar every update_period_seconds.

    Args:
        bar (progressbar.ProgressBar): ProgressBar to call update() on. Must have .start() called before. Functions on bar should not be called before thread is terminated as there is no mutex protections.
        update_period_seconds (int, optional): How often to call update() on bar. Defaults to 0.1.
        should_stop (ThreadsafeBool): Whether to stop.
    """

    while should_stop.get_value():
        bar.update()
        time.sleep(update_period_seconds)

def save_training_sequences(prefix:str, training_seq_non_enc:list, training_seq:numpy.ndarray, pred_seq:numpy.ndarray, should_overwrite_existing:bool):
    """Save training sequences to disk using json format. Creates a progressbar to indicate progress, which is updated by
    a separate thread.

    Args:
        prefix (str): What to append before the name of the list.
        training_seq_non_enc (list): Non encoded training sequence.
        training_seq (numpy.ndarray): Training sequence ready to be used with the neural network. Will be converted to list first.
        pred_seq (numpy.ndarray): Prediction sequence ready to be used with the neural network. Will be converted to list first.
        should_overwrite_existing (bool): Whether to to overwrite existing saved sequences. Does not check if file will match.
    """
    bar = progressbar.ProgressBar(widgets=["Saving training data to file...  ", progressbar.AnimatedMarker(), '  ', progressbar.Timer(), '  ', progressbar.AdaptiveETA()])
    bar.start() # DO NOT CALL ELSEWHERE, NO MUTEX PROTECTION

    should_stop_thread = ThreadSafeBoolRef(False)
    bar_update_thread = Thread(target=unknown_time_taken_progressbar_update, args=(bar,should_stop_thread,), daemon=True)
    bar_update_thread.start()

    save_list_to_disk(training_seq_non_enc, prefix, TRAINING_SEQUENCE_NON_ENCODED_NAME, should_overwrite_existing)
    save_list_to_disk(training_seq, prefix, TRAINING_SEQUENCE_NAME, should_overwrite_existing)
    save_list_to_disk(pred_seq, prefix, PREDICT_SEQUENCE_NAME, should_overwrite_existing)

    should_stop_thread.set_value(True)
    bar_update_thread.join()
    bar.finish()

def load_list(prefix: str, list_name: str):
    """Load a list with a prefix and list_name and a file extension as defined by SAVED_LIST_FILE_EXTENSION and SAVED_NUMPY_LIST_FILE_EXTENSION.
    Will check whether the file saved using Numpy or Pickle. Path is formatted as (prefix + last_name + EXTENSION).

    Args:
        prefix (str): Prefix of the path.
        list_name (str): Name of list saved.

    Raises:
        FileNotFoundError: If file is not found with either extension.

    Returns:
        any: Either a list or numpy.narray (or whatever was found)
    """
    file_name = prefix + list_name

    if os.path.isfile(file_name + SAVED_LIST_FILE_EXTENSION):
        with open(file_name + SAVED_LIST_FILE_EXTENSION, 'rb') as f:
            return pickle.load(f)
    elif os.path.isfile(file_name + SAVED_NUMPY_LIST_FILE_EXTENSION):
        with open(file_name + SAVED_NUMPY_LIST_FILE_EXTENSION, 'rb') as f:
            return numpy.load(f)
    else:
        raise FileNotFoundError()

def load_training_sequences(prefix: str):
    """Load training sequences from string, if they exist, otherwise the relevant sequence will be none. If the file is not found, this function
    returns None for the relevant array.

    Args:
        prefix (str): Prefix before the files.

    Returns:
        list, numpy.ndarray, numpy.ndarray: The arrays in the following order - training sequence non encoded, training sequence, prediction sequence. They will be None if the file was not found.
    """
    
    train_seq_non_enc = None
    train_seq = None
    predict_seq = None
    
    try:
        train_seq_non_enc = load_list(prefix, TRAINING_SEQUENCE_NON_ENCODED_NAME)
    except FileNotFoundError:
        train_seq_non_enc = None

    try:
        train_seq = load_list(prefix, TRAINING_SEQUENCE_NAME)
    except FileNotFoundError:
        train_seq = None

    try:
        predict_seq = load_list(prefix, PREDICT_SEQUENCE_NAME)
    except FileNotFoundError:
        predict_seq = None
    
    return train_seq_non_enc, train_seq, predict_seq


def get_starting_map(graph: networkx.Graph, depth: int, distance_function, start_node_id=None):
    """Get a starting map to begin the generation off of.

    Args:
        graph (networkx.Graph): The graph with the source nodes used for training.
        depth (int): How many nodes to perform a breadth first search on.
        distance_function (callable): A function that converts the stored position of the graph into a changes in deistance.
        start_node_id (any, optional): ID of the starting node. Defaults to None.

    Returns:
        networkx.Graph, list: A graph which contains some of the starting nodes, with cartesian coordinates; the list of unvisited nodes.
    """


    if start_node_id is None:
        start_node_id = min(graph.nodes)
    
    successors = list(networkx.bfs_successors(graph, start_node_id, depth_limit=depth))

    starting_map = networkx.Graph()
    unvisited_nodes = []

    for successor in successors:
        current_node_id = successor[0]
        if starting_map.number_of_nodes() == 0 or (not starting_map.has_node(current_node_id)):
            starting_map.add_node(current_node_id, position=(0,0))

        for next_node_id in successor[1]:
            change_in_pos = distance_function(graph, current_node_id, next_node_id)

            x = starting_map.nodes[current_node_id]['position'][0] + change_in_pos[0]
            y = starting_map.nodes[current_node_id]['position'][1] + change_in_pos[1]

            starting_map.add_node(next_node_id, position=(x, y))
            starting_map.add_edge(current_node_id, next_node_id)
            
            if next_node_id not in unvisited_nodes:
                unvisited_nodes.append(next_node_id)
    
    return starting_map, unvisited_nodes



# max_num_paths_per_individual_node = 20

# num_features = 200 # aka cardinality
# max_num_input_paths = 50
# max_num_input_nodes = 20
# max_num_output_nodes = 6
# num_dimensions = 2

# graph = graph_example_osm()

# # profiler = cProfile.Profile()
# # profiler.enable()
# # generate_training_sequence_for_graph(1, max_num_input_paths, num_dimensions, graph, max_num_input_nodes, get_signed_distance_between_nodes)
# # profiler.disable()
# # stats = pstats.Stats(profiler).sort_stats('tottime')
# # stats.print_stats()

# training_sequence_non_encoded = generate_training_sequence_for_graph(1000, max_num_paths_per_individual_node, max_num_input_paths, num_dimensions, graph, max_num_input_nodes, get_signed_distance_between_nodes)
# training_sequence, prediction_sequence = encode_training_sequence(training_sequence_non_encoded, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)

# # # Reshape the training sequence and prediction sequence to simply be a long list of paths, which are not seperated into their own indices (i.e. 4d > 3d)
# # training_sequence = numpy.reshape(training_sequence, (training_sequence.shape[0] * training_sequence.shape[1], training_sequence.shape[2], training_sequence.shape[3]))
# # prediction_sequence = numpy.reshape(prediction_sequence, (prediction_sequence.shape[0] * prediction_sequence.shape[1], prediction_sequence.shape[2], training_sequence.shape[3]))

# prefix = "Bounding box around a park "
# save_training_sequences(prefix, training_sequence_non_encoded, training_sequence, prediction_sequence, False)
# tsnc, ts, ps = load_training_sequences(prefix)
# training_sequence = ts
# prediction_sequence = ps

# if not numpy.array_equal(ts, training_sequence):
#     print("err")
# if not numpy.array_equal(ps, prediction_sequence):
#     print("err")
# if training_sequence_non_encoded != tsnc:
#     print("err")



# # training_sequence, validate_training_sequence, prediction_sequence, validate_prediction_sequence = train_test_split(training_sequence, prediction_sequence, test_size=0.2)

# model = define_models(max_num_input_nodes, max_num_output_nodes, num_dimensions, num_features, 500)

# keras.utils.plot_model(model, show_shapes=True, show_dtype=True)

# for _ in range(20): # epochs
#     for i in range(len(training_sequence)):
#         model.fit(x=training_sequence[i], y=prediction_sequence[i], epochs=1, batch_size=max_num_input_paths)


# # training_sequence = numpy.resize(training_sequence, (len(training_sequence) * max_num_input_paths, max_num_input_nodes * num_dimensions, num_features)) # not enough memory
# # prediction_sequence = numpy.resize(prediction_sequence, (len(prediction_sequence) * max_num_input_paths, max_num_output_nodes * num_dimensions, num_features)) # not enough memory

# # model.fit(x=training_sequence, y=prediction_sequence, epochs=1, batch_size=max_num_input_paths)

# model.save("Big Baxter Park Area Model 20 epochs")
# model = keras.models.load_model("Big Baxter Park Area Model 20 epochs")

# # simple evaluate model
# # loss, accuracy = model.evaluate(validate_training_sequence, validate_prediction_sequence, verbose=0)
# #print('Loss: %f, Accuracy: %f' % (loss, accuracy*100))

# # tell it to generate something - positions are stored in x,y not coordinates so we have to come up with new way of encoding path
# road_graph = networkx.Graph()

# road_graph, unvisited_nodes = get_starting_map(graph, 3, distance_function=get_signed_distance_between_nodes)

# for i in range (100):
#     current_node_id = unvisited_nodes.pop(0)
#     position_changes = []

#     bar = progressbar.ProgressBar(widgets=['Generating prediction sequence...  ', progressbar.AnimatedMarker(), '  ', progressbar.Timer(), '  ', progressbar.AdaptiveETA()])
#     bar.start()
#     generate_paths_for_node(1, position_changes, road_graph, current_node_id, max_num_input_nodes, max_num_paths_per_individual_node, max_num_input_paths, get_cartesian_distance_between_nodes, bar)
#     x, y = encode_training_sequence(position_changes, num_features, max_num_input_paths, max_num_input_nodes, max_num_output_nodes, num_dimensions)
#     bar.finish()

#     prediction = model.predict(x[-1])
#     prediction = decode_prediction(prediction, num_features)

#     for new_node in prediction:
#         if not (new_node[0] == 0 and new_node[1] == 0):
#             new_node_id = str(i)
            
#             road_graph.add_node(new_node_id, position=(road_graph.nodes[current_node_id]['position'][0] +  new_node[0], road_graph.nodes[current_node_id]['position'][1] +  new_node[1]))
#             road_graph.add_edge(current_node_id, new_node_id)
#             unvisited_nodes.append(new_node_id)

#     if len(unvisited_nodes) == 0:
#         break

# show_graph(road_graph)
  

# # VERSION BEFORE INSTALLING TENSORFLOW 2.4
# # tf-estimator-nightly   2.4.0.dev2020102301
# # tf-nightly             2.5.0.dev20201110