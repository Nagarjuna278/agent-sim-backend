import random
from queue import Queue

def get_shortest_path_action(grid, start, target):
    """Calculate the next best move for catcher using Breadth-First Search (BFS) pathfinding"""
    def get_neighbors(pos):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        neighbors = []
        for idx, (dx, dy) in enumerate(moves):
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if (0 <= new_x < grid.shape[0] and
                0 <= new_y < grid.shape[1] and
                grid[new_x, new_y] != 1):  # not an obstacle
                neighbors.append((new_x, new_y, idx))
        return neighbors

    # BFS implementation
    frontier = Queue() # Use Queue for BFS
    frontier.put(start)
    came_from = {start: None} # Store predecessor for each position

    while not frontier.empty():
        current = frontier.get()

        if current == target:
            break

        for next_pos, next_pos_y, action in get_neighbors(current):
            next_node = (next_pos, next_pos_y)
            if next_node not in came_from: # Check if not visited
                came_from[next_node] = (current, action) # Store where we came from and action
                frontier.put(next_node)

    # Backtrack to find first action
    if target in came_from:
        current = target
        while current != start:
            prev_node_action = came_from[current]
            if prev_node_action is None:
                print("Error: Incomplete path in came_from during backtracking (BFS)!")
                return random.randint(0, 3)
            prev_node, action = prev_node_action
            if prev_node == start:
                return action
            current = prev_node

    # If no path found, or path reconstruction failed, move randomly
    return random.randint(0, 3)