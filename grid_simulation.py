import matplotlib.pyplot as plt
import numpy as np
import heapq
import time
from random import random

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

def dijkstra(graph, start, end):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    shortest_path = {}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == end:
            break

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
                shortest_path[neighbor] = current_node

    path = []
    while end:
        path.insert(0, end)
        end = shortest_path.get(end)

    return path

directions = {
    'axis1': (1, 0),
    'axis2': (0, 1),
    'axis3': (-1, 0),
    'axis4': (0, -1)
}

headingDirection = 'axis1'

def moveForward():
    print(f"Moving forward along {headingDirection}")
    time.sleep(1)

def updateHeading(new_direction):
    global headingDirection
    if headingDirection != new_direction:
        print(f"Changing heading from {headingDirection} to {new_direction}")
        headingDirection = new_direction
        time.sleep(1)

def detect_obstacle():
    return random() < 0.2

def dynamic_replan_path(current_position, destination):
    print("Replanning path due to obstacle...")
    return dijkstra(graph, current_position, destination)

def assign_shelf(book_rfid):
    shelf_location = get_shelf_location_from_rfid(book_rfid)
    print(f"Shelf assigned for book {book_rfid}: {shelf_location}")
    return shelf_location

def get_shelf_location_from_rfid(rfid):
    shelf_mapping = {
        'book1': 'A',
        'book2': 'B',
        'book3': 'C',
        'book4': 'D'
    }
    return shelf_mapping.get(rfid, 'Unknown Shelf')

def simulate_bookshelf_scanning(book_id):
    distance_traveled = []
    sensed_distance = []

    for step in range(70):
        distance_traveled.append(step)
        
        if book_id == 'book1':
            sensed_distance.append(40 if step < 10 else (5 if step < 30 else 10))
        elif book_id == 'book2':
            sensed_distance.append(35 if step < 15 else (8 if step < 25 else 15))
        elif book_id == 'book3':
            sensed_distance.append(30 if step < 20 else (7 if step < 35 else 20))
        else:
            sensed_distance.append(50)

        if detect_obstacle():
            print("Obstacle detected! Replanning...")
            dynamic_replan_path('A', 'D')

    plt.figure(figsize=(10, 6))
    plt.plot(distance_traveled, sensed_distance, marker='o', linestyle='-', label=f'{book_id} Scan')
    plt.title('Bookshelf Scanning')
    plt.xlabel('Distance Traveled by Robot')
    plt.ylabel('Distance Between Robot and Shelf')
    plt.grid()
    plt.axvline(x=10, color='red', linestyle='--', label='Bookshelf detected line')
    plt.axvline(x=30, color='orange', linestyle='--', label='Space detected line')
    plt.annotate('Bookshelf detected', xy=(10, 40), xytext=(15, 35),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Space detected', xy=(30, 5), xytext=(35, 15),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend()
    plt.show()

grid_size = (10, 10)
grid = np.zeros(grid_size)

def draw_grid_with_path(path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='Blues', extent=(0, grid_size[1], 0, grid_size[0]), alpha=0.5)
    
    if path:
        for (x, y) in path:
            plt.plot(x + 0.5, y + 0.5, marker='o', color='red')
    
    plt.xticks(np.arange(0, grid_size[1] + 1, 1))
    plt.yticks(np.arange(0, grid_size[0] + 1, 1))
    plt.grid(which='both', color='black', linestyle='-', linewidth=2)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

def plan_path(start, end):
    path = []
    x_start, y_start = start
    x_end, y_end = end
    
    if x_start == x_end:
        for y in range(min(y_start, y_end), max(y_start, y_end) + 1):
            path.append((x_start, y))
    elif y_start == y_end:
        for x in range(min(x_start, x_end), max(x_start, x_end) + 1):
            path.append((x, y_start))
    else:
        for y in range(min(y_start, y_end), max(y_start, y_end) + 1):
            path.append((x_start, y))
        for x in range(min(x_start, x_end), max(x_start, x_end) + 1):
            path.append((x, y_end))

    return path

def run_robot_scenario():
    rfid_books = ['book1', 'book2', 'book3', 'book4']
    for rfid in rfid_books:
        shelf_location = assign_shelf(rfid)
        simulate_bookshelf_scanning(rfid)

    home_position = (0, 0)
    destination_1 = (4, 4)
    destination_2 = (8, 2)
    destination_3 = (4, 8)

    path_home_to_dest1 = plan_path(home_position, destination_1)
    path_dest1_to_dest2 = plan_path(destination_1, destination_2)
    path_dest2_to_dest3 = plan_path(destination_2, destination_3)
    path_dest3_to_home = plan_path(destination_3, home_position)

    draw_grid_with_path(path_home_to_dest1, "Home Position to Destination 1")
    draw_grid_with_path(path_dest1_to_dest2, "Destination 1 to Destination 2")
    draw_grid_with_path(path_dest2_to_dest3, "Destination 2 to Destination 3")
    draw_grid_with_path(path_dest3_to_home, "Destination 3 to Home Position")

run_robot_scenario()
