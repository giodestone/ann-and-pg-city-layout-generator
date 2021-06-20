from time import time
from typing import MappingView, Tuple
import OSMPythonTools
from networkx.algorithms.dag import ancestors, is_directed_acyclic_graph
from networkx.algorithms.shortest_paths.unweighted import predecessor, single_source_shortest_path
from networkx.algorithms.simple_paths import all_simple_paths
from networkx.algorithms.traversal.breadth_first_search import bfs_edges, bfs_predecessors, bfs_successors
from networkx.algorithms.traversal.edgebfs import edge_bfs
from networkx.drawing.layout import kamada_kawai_layout
from numpy.core.fromnumeric import ndim
from numpy.lib.function_base import append
import progressbar
import requests
from OSMPythonTools.element import Element
from OSMPythonTools.api import Api
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from OSMPythonTools.nominatim import Nominatim
import networkx as nx
import matplotlib.pyplot as plt
import logging
import numpy as np
from numpy import doc
from math import asin, degrees, log, pi, sin, cos, tan, sqrt, atan2, radians, ceil
from pyproj import Transformer
import random

import logging
logging.getLogger('OSMPythonTools').setLevel(logging.ERROR)

def get_map_graph_from_bounding_box(ways_query:str):
    """Get map that is converted to a graph format with nodes. Distances between nodes that are over 100m are split up.

    Args:
        ways_query (str): The query string to get the ways in the relevant area. Should be a bounding box.

    Raises:
        Exception: If no ways are found in the queried area.

    Returns:
        networkx.Graph: Graph containing the nodes.
    """

    graph = nx.Graph()
    overpass = Overpass()
    
    ways = overpass.query(ways_query)

    if (ways.ways() is None or ways.countWays() == 0): 
        raise Exception("There are no ways inside of the queried area. Check your query and try again.")

    print("SUCCESS: Got {0} ways. If this step is taking a while then API is being queried. Converting them to a graph format...".format(ways.countWays()))
    
    bar = progressbar.ProgressBar(widgets=['Processing Ways... ', progressbar.Timer(), '  ', progressbar.AdaptiveETA(), ' ', progressbar.Bar(), ' ', progressbar.Percentage()], max_value=ways.countWays())
    bar.start()
    ways_processed = 0
    for way in ways.ways():
        bar.update(ways_processed)
        add_highway_nodes_to_graph_max_dist(graph, way)
        ways_processed += 1

    verify_all_ids_are_str(graph)

    bar.finish()
    print("SUCCESS: Converted to graph format!")

    return graph

def add_highway_nodes_to_graph_max_dist(graph:nx.Graph, highway:Element, max_dist_before_splitting_node=100):
    previous_node = None
    for node in highway.nodes():
        
        if (previous_node is not None):
            
            frm = (previous_node.lon(), previous_node.lat())
            to = (node.lon(), node.lat())

            is_longer_than_max_dist = False
            dist_between_nodes = convert_lat_long_to_distance(frm, to)
            
            # Need to check if the actual distance is greater but also if a straight line is too long - as it is possible to have a distance below when going frm to to; but going down then along may not be (i.e. taking the horizontal and vertical sides of a triangle).

            # perform cheap check first
            if dist_between_nodes > max_dist_before_splitting_node:
                is_longer_than_max_dist = True
            # perform expensive check next
            else:
                dist_x_from_coords = frm
                dist_x_to_coords = (to[0], frm[1])
                dist_x = convert_lat_long_to_distance(dist_x_from_coords, dist_x_to_coords)

                dist_y_from_coords = (frm[0], to[1])
                dist_y_to_coords = frm
                dist_y = convert_lat_long_to_distance(dist_y_from_coords, dist_y_to_coords)

                is_longer_than_max_dist = dist_x > max_dist_before_splitting_node or dist_y > max_dist_before_splitting_node


            if is_longer_than_max_dist:
                # find out how many nodes to split it into
                num_to_split_into = ceil(dist_between_nodes / max_dist_before_splitting_node)

                # create a string of nodes between previous node and node, distances of which are seperated evenly.
                prevAddedNodeID = str(previous_node.id())
                
                bearing = getBearing(frm, to)
                distanceIncrements = dist_between_nodes / num_to_split_into
                curDistanceCovered = distanceIncrements
                
                for i in range(num_to_split_into - 1):
                    newPos = get_destination_point(frm, bearing, curDistanceCovered / 1000)

                    addedNodeID = str(previous_node.id()) + '-split-{}'.format(str(i + 1))

                    # if previous_node.id() == 224438439:
                    #     print("watch oot")
                    #     plotMap(graph, "uuuuuuurgh {}".format(i + 1))

                    # Due to how the highway is iterated, part of the roadway would've been split before, this can happen any number of times (especially for big roads).
                    if graph.has_node(addedNodeID):
                        while (graph.has_node(addedNodeID)):
                            addedNodeID += (' dupe')

                    graph.add_node(addedNodeID, position=(newPos[0], newPos[1]))

                    graph.add_edge(prevAddedNodeID, addedNodeID)

                    prevAddedNodeID = addedNodeID
                    curDistanceCovered += distanceIncrements
                    pass
                
                # add final node
                graph.add_node(str(node.id()), position=(node.lon(), node.lat()))
                graph.add_edge(prevAddedNodeID, str(node.id()))
            else:
                graph.add_node(str(node.id()), position=(node.lon(), node.lat()))
            
                graph.add_edge(str(previous_node.id()), str(node.id()))
        else:
            graph.add_node(str(node.id()), position=(node.lon(), node.lat()))

        previous_node = node
    pass

def getBearing(frm,to):
    #φ = lat
    #λ = lon
    #Δλ = diff in lon

    lon1 = radians(frm[0])
    lon2 = radians(to[0])

    lat1 = radians(frm[1])
    lat2 = radians(to[1])

    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos (lat2) * cos (lon2 - lon1)
    θ = atan2(y,  x)
    # bearing = (θ * 180.0 / pi + 360.0) % 360
    bearing = degrees(θ)
    return bearing

def get_destination_point(start, bearing, distance):
    # const φ2 = Math.asin( Math.sin(φ1)*Math.cos(d/R) +
    #                       Math.cos(φ1)*Math.sin(d/R)*Math.cos(brng) );
    # const λ2 = λ1 + Math.atan2(Math.sin(brng)*Math.sin(d/R)*Math.cos(φ1),
    #                            Math.cos(d/R)-Math.sin(φ1)*Math.sin(φ2));

    R = 6373.0 # earth radius

    bearing = radians(bearing) # convert to radians

    lon = radians(start[0])
    lat = radians(start[1])

    lonResult = 0.0
    latResult = 0.0

    latResult = asin(sin(lat) * cos(distance / R) + cos(lat) * sin(distance/R) * cos(bearing))
    lonResult = lon + atan2(sin(bearing) * sin(distance/R) * cos(lat), cos(distance/R) - sin(lat) * sin(latResult))

    return (degrees(lonResult), degrees(latResult))


def verify_all_ids_are_str(graph:nx.Graph):
    """Verify that all IDs are string in the graph

    Args:
        graph (nx.Graph): graph that contains nodes with an .id() function that the type can be checked against.
    """
    for node in graph.nodes:
        if type(node) is not str:
            raise "Not all node ID's are strings, make sure all added node id's are converted into strings, which can be easily found using control+f '.id()' and see where 'str()' is lacking!"

def get_positions_from_nodes(graph:nx.Graph):
    pos = {}
    for node in graph.nodes:
        pos[node] = graph.nodes[node]["position"]
    return pos

def get_nodes_sorted_by_bfs_rel_pos(graph:nx.Graph, num_previous_nodes:int, num_prediction_nodes:int, max_dataset_size=100000):
    node_ids_in_graph = list(graph.nodes.keys())
    shape = (max_dataset_size, num_previous_nodes + num_prediction_nodes, 3)
    dataset = np.zeros(shape, dtype=int)
    #dataset[:] = np.zeros

    bar = progressbar.ProgressBar(widgets=['Creating training sequence... ', progressbar.Timer(), '  ', progressbar.AdaptiveETA(), ' ', progressbar.Bar(), ' ', progressbar.Percentage()], max_value=max_dataset_size)
    has_value_been_added = False
    current_dataset_index = 0 # Which training sequence index ([i,x,x]) data is being inserted into. On the last cycle value is always num inserted + 1.
    while current_dataset_index < max_dataset_size and current_dataset_index < len(node_ids_in_graph) - 1:
        has_value_been_added, dataset[current_dataset_index] = get_successor_of_node_by_bfs_rel_pos(graph, node_ids_in_graph[current_dataset_index], num_previous_nodes=num_previous_nodes, num_prediction_nodes=num_prediction_nodes)
        current_dataset_index += 1

    bar.finish()

    if (has_value_been_added is False): # len(node_list) doesn't work because its preallocated (i.e. all entries are there.)
        raise Exception("ERROR: No nodes added to `dataset`. Indicates that there were no edges in the graph passed in (`graph`).")

    # Shrink the node_list to make it match the number of nodes that it was iterated on.
    if (current_dataset_index < max_dataset_size):
        print("INFO: Shrinking dataset as the number of nodes is {}, which is under the specified target of {}.".format(current_dataset_index, max_dataset_size))
        shape_new = (current_dataset_index, shape[1], shape[2])
        dataset = np.resize(dataset, shape_new)
    
    return current_dataset_index, dataset

def get_successor_of_node_by_bfs_rel_pos(graph:nx.Graph, node_id, cardinality:int=200, num_previous_nodes:int=40, num_prediction_nodes:int=1, exclude_starting_node:bool=False):
    shape = (num_previous_nodes + num_prediction_nodes, 3)
    dataset = np.zeros(shape, dtype=int)

    has_value_been_added = False

    bfs_result = list(bfs_predecessors(graph, node_id, depth_limit=num_previous_nodes + num_prediction_nodes))

    # (30/03/2021 18:10) ok what were we working on: currently there is a problem with how the nodes are sorted. these nodes are actually incorrectly sorted because we should be getting what lead to them. the problem is getting the nodes it should predict, the ones that go on from the item. I should look the if the paper has something on it.
    matchessearchnode = [x for x in bfs_result if (x[0] == node_id or x[1] == node_id)]
    indexofmatches = []
    for i in range(len(bfs_result)):
        if (matchessearchnode.count(bfs_result[i]) > 0):
            indexofmatches.append(i)

    # Remove first entry
    if (exclude_starting_node and len(bfs_result) > 1):
        bfs_result.pop()
    
    current_entry_index = 0
    for entry in bfs_result:
        has_value_been_added = True

        if current_entry_index >= num_prediction_nodes + num_previous_nodes:
            continue

        source_id = entry[0]
        source_position = graph.nodes[source_id]['position']

        for result_id in entry[1]:
            result_position = graph.nodes[result_id]['position']
            
            distXFrom = source_position
            distXTo = (result_position[0], source_position[1])

            distYFrom = (source_position[0], result_position[1])
            distYTo = source_position

            distX = convert_lat_long_to_distance(distXFrom, distXTo) # TODO: PROBLEM HERE - THIS DOES NOT CONSIDER NEGATIVE DISTANCES, WHEREAS IT SHOULD. NEED TO CALCULATE IF A SPECIFIC X,Y COMPONENT SHOULD BE NEGATIVE BASED ON BEARING (90-270 SHOULD BE -Y, 180-359.99 SHOULD BE -X), ALSO EXPAND CARDINALITY TO 200
            distY = convert_lat_long_to_distance(distYFrom, distYTo)
            distance_negative_multiplier = sign_distance(source_position, result_position)

            distX *= distance_negative_multiplier[0]
            distY *= distance_negative_multiplier[1]

            if (round(distX) > cardinality or round(distY) > cardinality):
                raise "One of the distances in X or Y are larger than cardinality! Check whether the inserted nodes were inserted correctly (split if too big)."
        
        dataset[current_entry_index][0] = round(distX)
        dataset[current_entry_index][1] = round(distY)
        dataset[current_entry_index][2] = len(entry[1])
        current_entry_index += 1
    return has_value_been_added, dataset


def get_all_paths_to_node_depth_limited(graph:nx.Graph, source_node_id, depth_limit):

    return list(single_source_shortest_path(graph, source_node_id, depth_limit))

def convert_lat_long_to_distance(fromPos:tuple, toPos:tuple):
    """Calulcates the distance between `from` and `to` and returns the distance in meters.
    Adapted from https://stackoverflow.com/a/19412565

    Args:
        fromPos (tuple): [0] should be lon, [1] should be lat.
        toPos (tuple): [0] should be lon, [1] should be lat.

    Returns:
        float: Distance between `from` and `to` in meters.
    """
    R = 6373.0 # earth radius

    lat1 = radians(fromPos[1])
    lon1 = radians(fromPos[0])
    lat2 = radians(toPos[1])
    lon2 = radians(toPos[0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance * 1000

def calculate_bearing(frm:tuple, to:tuple):
    # adapted from https://www.movable-type.co.uk/scripts/latlong.html

    # φ is latitude, λ is longitude
    λ1 = frm[0]
    λ2 = to[0]

    φ1 = frm[1]
    φ2 = to[1]

    y = sin(λ2-λ1) * cos(φ2);
    x = cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(λ2-λ1);
    θ = atan2(y, x);
    brng = (θ*180/pi + 360) % 360; # in degrees

    return brng

def sign_distance(frm:tuple, to:tuple):
    """Determines whether the x, and/or y distance should be made negative.

    Args:
        frm (tuple): From coordinates.
        to (tuple): To coordinates.

    Returns:
        tuple: Contains -1 or 1 in the [0] or [1] index, depending on whether x or y should be negative.
    """
    bearing = calculate_bearing(frm, to)
    multiplication = (1, 1)

    # (90-270 SHOULD BE -Y, 180-359.99 SHOULD BE -X)
    if (bearing >= 180 and bearing <= 359.99):
        multiplication = (-1, multiplication[1])
    
    if (bearing >= 90 and bearing <= 270):
        multiplication = (multiplication[0], -1)

    return multiplication

def show_graph(graph:nx.Graph, node_size=3, edge_width=1, show_labels=True, font_size=7, should_display=True, save_to_disk=False, file_name=None):
    """Show the graph on screen in a window.

    Args:
        graph (nx.Graph): Graph to display.
        node_size (int, optional): Size of node circles. Defaults to 3.
        edge_width (float, optional): Width of the line that represents the edges. Defaults to 1.
        show_labels (bool, optional): Whether to the node_id's. Defaults to True.
        font_size (int, optional): Size of the label font. `show_labels` must be `True` for them to display. Defaults to 7.
        should_display (bool, optional): Whether a window should be displayed. Defaults to True.
        save_to_disk (bool, optional): Whether to save the graph to disk. Defaults to False.
        file_name (str, optional): Name of the file. Must be defined if `save_to_disk` is `True`. Defaults to None.
    """
    
    # Build pos dict.
    pos_dict = dict()
    for nodeID in graph.nodes:
        if len(graph.nodes[nodeID]) == 0:
            pass
        else:
            pos = graph.nodes[nodeID]['position'] # If the position is empty or invalid, nx.draw will give error!"
            pos_dict[nodeID] = pos

    nx.draw(graph, pos=pos_dict, with_labels=show_labels, node_size=node_size, width=edge_width, font_size=font_size)
    
    if save_to_disk:
        if file_name is None:
            raise Exception("File name is not set.")
        plt.savefig(file_name, dpi=500)

    if should_display:
        plt.show()

    plt.clf()

def get_surrounding_nodes(graph:nx.Graph, node_id):
    """Get the immediate surrounding (adjacent) nodes `node_id` using `networkx.bfs_successors()`.

    Args:
        graph (nx.Graph): Graph which `node_id` is located in.
        node_id (any): Node ID to find adjacent nodes of.

    Returns:
        list: List of adjacent nodes. May be empty if none found.
    """
    return list(bfs_successors(graph, node_id, 1))[0][1]

def convert_lat_lon_graph_to_cartesian_graph(graph:nx.Graph):
    """Convert a graph which uses lat/lon to determine the nodes' position into a cartesian graph which represents coordinates as x and y on a flat plane.
    The x, y is determined by using the `get_nodes_networkx.get_signed_distance_between_nodes()`.

    Remarks:
        **Will disregard disconnected areas.**

    Args:
        graph (networkx.Graph): Graph to convert.

    Returns:
        networkx.Graph: The graph with the ['position'] attribute being in 
    """
    start_node_id = min(graph.nodes)
    
    successors = nx.bfs_successors(graph, start_node_id)

    cartesian_graph = nx.Graph()

    is_start_added = False

    for successor in successors:
        current_node_id = successor[0]
        if (cartesian_graph.number_of_nodes() == 0 or (not cartesian_graph.has_node(current_node_id))) and not is_start_added:
            cartesian_graph.add_node(current_node_id, position=(0,0))
            if is_start_added:
                print("INFO: The graph is not entirely attached to each other. The other detached bit was ignored.")
                break;
            else:
                is_start_added = True


        for next_node_id in successor[1]:
            change_in_pos = get_signed_distance_between_nodes(graph, current_node_id, next_node_id)

            x = cartesian_graph.nodes[current_node_id]['position'][0] + change_in_pos[0]
            y = cartesian_graph.nodes[current_node_id]['position'][1] + change_in_pos[1]

            cartesian_graph.add_node(next_node_id, position=(x, y))
            cartesian_graph.add_edge(current_node_id, next_node_id)

    return cartesian_graph

def get_all_simple_paths_from_node(graph:nx.Graph, start_node_id, depth_limit, max_num_paths_per_node:int, max_length_of_each_path_leading_to_node:int, bar:progressbar.ProgressBar):
    """Get all possible simple paths between all reachable nodes from start_node_id, until depth_limit.

    Args:
        graph (nx.Graph): Graph to read from.
        start_node_id (any): ID of the node to read out from
        depth_limit (int): Maximum depth/how far away to go from start_node_id.
        max_num_paths_leading_to_node (int): Maximum number of paths that can lead to the node. The more paths the longer this calculation will take and probably cause an overrepresentation of well connected nodes.
        max_length_of_each_path_leading_to_node (int): Maximum number of node id's in a path leading up to start_node_id.
        bar (progressBar.ProgressBar): Progressbar to update on progress, as this function takes a while.

    Returns:
        list: [0] = y/predict (next node IDs, going away from start node, excluding ones in [2]), [1] = start_node_id, [2] = path of node IDs leading away from start_node_id, including start_node_id at the start.
    """
    
    all_reachable_nodes = list(single_source_shortest_path(graph, start_node_id, depth_limit))
    all_reachable_nodes = list(filter(lambda a: a != start_node_id, all_reachable_nodes))  # Get rid of self from list
    random.shuffle(all_reachable_nodes)  # This is so random nodes get selected as they are in a depth flow out pattern.

    sequence = [] # formatted as [0] = y/predict, [1] = pivot node id, [2] = x/train

    for i in range(len(all_reachable_nodes)):
        bar.update()

        # 1. get path
        paths_from_start_node_id_to_reachable_node_iterator = all_simple_paths(graph, start_node_id, all_reachable_nodes[i], depth_limit)
        paths_from_start_node_id_to_reachable_node = []
        
        num_simple_paths_added = 0
        for simple_path in paths_from_start_node_id_to_reachable_node_iterator:
            paths_from_start_node_id_to_reachable_node.append(simple_path)
            num_simple_paths_added += 1
            if num_simple_paths_added > max_length_of_each_path_leading_to_node:
                break
        
        for path in paths_from_start_node_id_to_reachable_node:
            # 2. get surrounding nodes
            surrounding_nodes = get_all_paths_to_node_depth_limited(graph, start_node_id, 1)
            surrounding_nodes = list(filter(lambda n_id: n_id != start_node_id, surrounding_nodes)) # remove self

            # 3.1 remove any paths that cross over surrounding nodes twice.
            if len(set(surrounding_nodes) & set(path)) > 1:
                continue;

            # 3. remove surrounding nodes contained in paths
            surrounding_nodes = list(filter(lambda a: a not in path, surrounding_nodes))

            sequence.append((surrounding_nodes, start_node_id, path))

        if len(sequence) > max_num_paths_per_node:
            break

    return sequence

def get_paths_to_all_nodes_to_other_nodes(graph:nx.Graph, depth_limit:int, max_num_paths_per_node:int):
    """Get paths from all nodes in the graph, to the other nodes in the graph.

    Args:
        graph (nx.Graph): Graph to get nodes from.
        depth_limit (int): Maximum depth/distance a node will be from another. Will increase number of total paths.
        max_num_paths_per_node (int): Maximum paths leading to a node. The larger this number the longer the calculations will take.
    
    Returns:
        list: First element of each element is the id of the originating node, the second path is an array of the node id's taken to get there.
    """
    path_to_path = []
    for node in graph.nodes:
        paths = get_all_simple_paths_from_node(graph, node, depth_limit)
        for path in paths:
            path_to_path.append(path)

    return path_to_path

def get_signed_distance_between_nodes(graph:nx.Graph, from_node_id, to_node_id):
    """Get the signed distance between two nodes as change in x, and y in meters.

    Args:
        graph (nx.Graph): Graph that the nodes are in.
        from_node_id ([type]): From node ID.
        to_node_id ([type]): To node ID.

    Returns:
        tuple: Change in x and y between the nodes.
    """
    current_node = graph.nodes[from_node_id]
    next_node = graph.nodes[to_node_id]

    current_node_pos = current_node['position']
    next_node_pos = next_node['position']

    dist_x_from_coords = current_node_pos
    dist_x_to_coords = (next_node_pos[0], current_node_pos[1])

    dist_y_from_coords = (current_node_pos[0], next_node_pos[1])
    dist_y_to_coords = current_node_pos

    distance_between_two = (convert_lat_long_to_distance(dist_x_from_coords, dist_x_to_coords), convert_lat_long_to_distance(dist_y_from_coords, dist_y_to_coords))
    
    distance_signs = sign_distance(current_node_pos, next_node_pos)

    distance_between_two = (distance_between_two[0] * distance_signs[0], distance_between_two[1] * distance_signs[1])

    return distance_between_two

def convert_path_to_changes_in_distance(graph:nx.Graph, path:list, distance_function):
    """Convert the path, which contains the ID's of the nodes in a path format.

    Args:
        graph (nx.Graph): Graph that the nodes are in.
        path (list): A list of IDs that are ordered in succession to eachother. The first node should be connected to the second one, third to fourth, etc. Minimum of two are needed.

    Returns:
        list: Signed changes in distance between nodes in meters.
    """
    distances = []
    
    for i in range(0, len(path) - 1):
        distances.append(distance_function(graph, path[i], path[i + 1])) # If you're getting error here, the incoming path probably doesn't have enough nodes.

    return distances
