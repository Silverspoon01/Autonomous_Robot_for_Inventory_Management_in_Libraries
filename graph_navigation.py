import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

# Define the graph (library)
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'D': 5, 'C': 1},
    'C': {'A': 4, 'B': 1, 'E': 3},
    'D': {'B': 5, 'E': 2, 'F': 6},
    'E': {'C': 3, 'D': 2, 'F': 1},
    'F': {'D': 6, 'E': 1}
}

# Define robot state
robot_state = {
    'current_location': 'A'
}

# Dijkstra's algorithm
def dijkstra(graph, start, end):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous_nodes = {}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    path = []
    current = end
    while current:
        path.insert(0, current)
        current = previous_nodes.get(current)
    return path

def update_orientation(turn):
    current_idx = directions.index(robot_state['orientation'])
    if turn == 'Left':
        new_idx = (current_idx - 1) % len(directions)
    elif turn == 'Right':
        new_idx = (current_idx + 1) % len(directions)
    robot_state['orientation'] = directions[new_idx]
    print(f"Robot turned {turn}. New orientation: {robot_state['orientation']}")

def move_to_location(location):
    print(f"Moving from {robot_state['current_location']} to {location}")
    robot_state['current_location'] = location
    time.sleep(1)

def draw_graph_with_path(path, title):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.show()

def run_robot_scenario():
    shelf_locations = {
        'book1': 'C',
        'book2': 'D',
        'book3': 'E',
        'book4': 'F'
    }

    for book, location in shelf_locations.items():
        print(f"\nAssigning shelf for {book} at location {location}")
        path = dijkstra(graph, robot_state['current_location'], location)
        draw_graph_with_path(path, f"Path to Shelf for {book}")
        for step in path[1:]:
            move_to_location(step)

    print("\nReturning to home base...")
    home_path = dijkstra(graph, robot_state['current_location'], 'A')
    draw_graph_with_path(home_path, "Path Back to Home")
    for step in home_path[1:]:
        move_to_location(step)

run_robot_scenario()
