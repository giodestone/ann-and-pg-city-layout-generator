import enum
from math import acos, atan, atan2, cosh, sqrt
import math
import networkx
from networkx.algorithms.shortest_paths.generic import all_shortest_paths, shortest_path
from networkx.algorithms.traversal.breadth_first_search import bfs_successors
from networkx.algorithms.cycles import cycle_basis, simple_cycles
from networkx.exception import NetworkXNoPath
from networkx.generators.lattice import hexagonal_lattice_graph
from numpy.lib.polynomial import poly
import progressbar
import get_nodes_networkx
from get_nodes_networkx import get_surrounding_nodes, show_graph

def find_polygons_dijkstra(graph:networkx.Graph):
    """Find polygons for a graph using the modified dijkstra method.

    Args:
        graph (networkx.Graph): Graph to find polygons in.

    Returns:
        list: All found polygons, represented as arrays of node IDs themselves.
    """

    bar = progressbar.ProgressBar(widgets=['Finding Polygons...  ', progressbar.Timer(), '  ', progressbar.AdaptiveETA(), '  ', progressbar.Bar(), ' ', progressbar.Percentage()], max_value=graph.number_of_nodes())
    bar.start()

    polygons = []
    num_nodes_processed = 0
    for node_id in graph.nodes:
        bar.update(num_nodes_processed)
        visited_surrounding_nodes = []
        surr_nodes_of_start = get_surrounding_nodes(graph, node_id)

        for sur_node in surr_nodes_of_start:
            bar.update()
            polys = polygon_walk_dijkstra(graph, node_id, visited_surrounding_nodes)
            visited_surrounding_nodes.append(sur_node)
            
            if polys is None:
                continue

            for p in polys:
                if p is None or is_poly_present(p, polygons):
                    continue
                polygons.append(p)
        
        num_nodes_processed += 1

    bar.finish()

    return polygons

def polygon_walk_dijkstra(graph:networkx.Graph, start_node, already_visisted_surrounding_nodes):
    """Find polygons using the Dijkstra algorithm, and by modifying the `graph` temporarily to exclude the obvious shortest path.

    Args:
        graph (networkx.Graph): Graph which `start_node` and `already_visisted_surrounding_nodes` are contained in.
        start_node (any): Node to start with.
        already_visisted_surrounding_nodes: A collection of immediately adjacent nodes to `start_node` which have already been visited.

    Returns:
        list: All shortest paths that could be used to reach `start_node` from an unvisited adjacent node. `None` if no paths are found.
    """

    path = []
    path.append(start_node)
    visited = set(already_visisted_surrounding_nodes)
    visited.add(start_node)

    # Check that there is a node to go to.
    initial_surrounding_nodes = get_surrounding_nodes(graph, start_node)
    start_surrounding_nodes = [x for x in initial_surrounding_nodes if x not in already_visisted_surrounding_nodes]
    if len(start_surrounding_nodes) == 0: # No nodes to visit
        return None

    graph.remove_edge(start_node, start_surrounding_nodes[0]) # simulate obstacle.
    
    try:
        shortest_paths = list(all_shortest_paths(graph, start_node, start_surrounding_nodes[0], weight=1, method='dijkstra'))
    except NetworkXNoPath:
        return None # no paths found!    
    
    graph.add_edge(start_node, start_surrounding_nodes[0])

    return shortest_paths

def is_poly_present(poly:list, polygons):
    """ Check if a `poly` is present in `polygons` by rotating the list to check if every ordered permutation exists in the list, including if it is reversed.

    Args:
        poly (list(tuple)): Poly to check whether its in `polygons`
        polygons (list(tuple)]): Set of existing polygons

    Returns:
        bool: True if `poly` is present in `polygons` at any rotation even when reveresed; False if `poly` is not.
    """

    poly_shifted = poly.copy()
    for i in range(len(poly) + 1):
        poly_shifted_reversed = poly_shifted.copy()
        poly_shifted_reversed.reverse()

        if poly_shifted in polygons or poly_shifted_reversed in polygons:
            return True
        else:
            poly_shifted = poly[i:] + poly[:i]
    
    return False

def grid_2d_graph_with_positions(num_quads_x:int, num_quads_y:int):
    """Get a 2d quad with a number of quads.

    Args:
        num_quads_x (int): Number of points in x direction.
        num_quads_y (int): Number of points in y direction.

    Example:
        Generate a graph with a single quad:
        >>> grid_2d_graph_with_positions(2, 2)
        
    Returns:
        networkx.Graph: Graph with quads and position variables set.
    """

    quad_graph = networkx.grid_2d_graph(num_quads_x, num_quads_y)

    for node in quad_graph.nodes:
        quad_graph.nodes[node]['position'] = (node[0], node[1])

    return quad_graph

def generate_hex_graph(m:int, n:int):
    """Generate a hex lattice graph with positions ready for rendering. Uses NetworkX function.

    Args:
        m (int): The number of rows of hexagons in the lattice.
        n (int): The number of columns of hexagons in the lattice.

    Returns:
        networkx.Graph: The m by n hexagonal lattice graph with fixed positions for drawing.
    """

    graph = hexagonal_lattice_graph(m, n)
    for n in graph.nodes:
        graph.nodes[n]['position'] = graph.nodes[n]['pos']

    return graph

def plot_polygons(graph:networkx.Graph, polygons:list, graph_to_add_polys_to:networkx.Graph):
    """Plot the polygons on a graph.

    Args:
        graph (networkx.Graph): Graph where the nodes on the polygon are added to.
        polygons (list): Polygons to add.
        graph_to_add_polys_to (networkx.Graph): The graph to which the polygons should be added.
    """

    for poly in polygons:

        prev_node = None
        for cur_node in poly:
            if prev_node is None:
                prev_node = cur_node
                continue
            
            if not graph_to_add_polys_to.has_node(prev_node):
                graph_to_add_polys_to.add_node(prev_node, position=graph.nodes[prev_node]['position'])
            if not graph_to_add_polys_to.has_node(cur_node):
                graph_to_add_polys_to.add_node(cur_node, position=graph.nodes[cur_node]['position'])

            graph_to_add_polys_to.add_edge(prev_node, cur_node)
            prev_node = cur_node

        graph_to_add_polys_to.add_edge(poly[0], poly[-1])

def run_simple_tests():
    print("===== Quad Graph =====")
    quad_graph = grid_2d_graph_with_positions(4, 4)
    show_graph(quad_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Quad Source Graph")
    
    quad_polys = find_polygons_dijkstra(quad_graph)
    print(quad_polys)
    print("Found Polygons: {}.".format(len(quad_polys)))
    show_graph(quad_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Quad Found Polygons Graph")
    
    # quad_poly_graph = networkx.Graph()
    # plot_polygons(quad_graph, quad_polys, quad_poly_graph)
    # show_graph(quad_poly_graph)

    # print("===== Hex Graph =====")
    # hex_graph = generate_hex_graph(3, 3)
    # hex_polys = find_polygons_dijkstra(hex_graph)
    # print(hex_polys)
    # print("Found Polygons: {}.".format(len(hex_polys)))
    # show_graph(hex_graph)

    # hex_poly_graph = networkx.Graph()
    # plot_polygons(hex_graph, hex_polys, hex_poly_graph)
    # show_graph(hex_poly_graph)

    # # To demonstrate example of polygon being generated with a road inside.
    # print("===== Plot with interior =====") # Gold Coast, Australia -27.949549,153.367397,-27.946592,153.372896
    # small_plot = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](-27.949194,153.369382,-27.947303,153.372622); (._;>;); out body;')
    # small_plot_polys = find_polygons_dijkstra(small_plot)
    # print(small_plot_polys)
    # print("Found Polygons: {}.".format(len(small_plot_polys)))
    # show_graph(small_plot)
    
    # small_plot_poly_graph = networkx.Graph()
    # plot_polygons(small_plot, small_plot_polys, small_plot_poly_graph)
    # show_graph(small_plot_poly_graph)

    # # To demonstrate functionality on a real-world function.
    # print("===== City Graph Scottish Parliment =====") # Around Scottish Parliament, Edinburgh 55.949506,-3.181583,55.953068,-3.171573
    # city_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](55.949506,-3.181583,55.953068,-3.171573); (._;>;); out body;')
    # city_polys = find_polygons_dijkstra(city_graph)
    # print(city_polys)
    # print("Found Polygons: {}.".format(len(city_polys)))
    # show_graph(city_graph)
    
    # city_poly_graph = networkx.Graph()
    # plot_polygons(city_graph, city_polys, city_poly_graph)
    # show_graph(city_poly_graph)
    # pass

    print("===== City Graph Cumin Place =====") # Around Cumin Place, Edinburgh 55.932760,-3.188041,55.936739,-3.179866
    city_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](55.932760,-3.188041,55.936739,-3.179866); (._;>;); out body;')
    show_graph(city_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Cumin Place Source Graph")
    city_polys = find_polygons_dijkstra(city_graph)
    print(city_polys)
    print("Found Polygons: {}.".format(len(city_polys)))
    
    city_poly_graph = networkx.Graph()
    plot_polygons(city_graph, city_polys, city_poly_graph)
    show_graph(city_poly_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Cumin Space Plot Graph")
    pass

    print("===== City Graph Marchmont Road =====") # Around Marchmont Road, Edinburgh 55.936132,-3.198298,55.939653,-3.192515
    city_graph = get_nodes_networkx.get_map_graph_from_bounding_box('way["highway"](55.936132,-3.198298,55.939653,-3.192515); (._;>;); out body;')
    show_graph(city_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Marchmont Road Source Graph")
    city_polys = find_polygons_dijkstra(city_graph)
    print(city_polys)
    print("Found Polygons: {}.".format(len(city_polys)))
    
    city_poly_graph = networkx.Graph()
    plot_polygons(city_graph, city_polys, city_poly_graph)
    show_graph(city_poly_graph, show_labels=False, should_display=False, save_to_disk=True, file_name="Marchmont Road Plot Graph")
    pass

run_simple_tests()