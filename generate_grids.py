import numpy as np
import h5py
import time
from queue import Queue

def is_valid_path(grid, start, end):
    """Check if a path exists from start to end using BFS"""
    if tuple(start) == tuple(end):
        return True

    rows, cols = grid.shape
    visited = set()
    q = Queue()
    q.put(tuple(start))
    visited.add(tuple(start))

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while not q.empty():
        current = q.get()
        for dx, dy in directions:
            new_x, new_y = current[0] + dx, current[1] + dy
            if (0 <= new_x < rows and 0 <= new_y < cols and 
                grid[new_x, new_y] != 1 and  # Not an obstacle
                (new_x, new_y) not in visited):

                if (new_x, new_y) == tuple(end):
                    return True

                visited.add((new_x, new_y))
                q.put((new_x, new_y))

    return False

def generate_valid_grid():
    """Generate a valid 10x10 grid ensuring a path from catcher to runner"""
    size = 10
    
    while True:  # Keep trying until we generate a valid grid
        grid = np.zeros((size, size))
        
        # Place runner (value 3)
        runner_pos = (np.random.randint(0, size), np.random.randint(0, size))
        grid[runner_pos] = 3
        
        # Place catcher (value 2)
        while True:
            catcher_pos = (np.random.randint(0, size), np.random.randint(0, size))
            if catcher_pos != runner_pos:
                grid[catcher_pos] = 2
                break

        # Place obstacles (value 1)
        num_obstacles = np.random.randint(1, 26)  # 1 to 25 obstacles
        obstacle_positions = set()

        while len(obstacle_positions) < num_obstacles:
            pos = (np.random.randint(0, size), np.random.randint(0, size))
            if pos not in [runner_pos, catcher_pos]:  # Avoid placing on runner or catcher
                obstacle_positions.add(pos)

        for pos in obstacle_positions:
            grid[pos] = 1  # Set obstacle

        # Validate grid only once
        if is_valid_path(grid, catcher_pos, runner_pos):
            return grid  # Return only valid grids

def grid_to_tuple(grid):
    """Convert grid to tuple for hashable type"""
    return tuple(map(tuple, grid))

def main():
    num_grids = 10000000
    batch_size = 1000  # Save in batches for performance
    
    print("Starting grid generation...")
    start_time = time.time()
    
    unique_grids = set()

    with h5py.File('valid_grids.h5', 'w') as f:
        dset = f.create_dataset('grids', 
                                shape=(num_grids, 10, 10),
                                dtype='uint8',
                                compression='gzip',
                                compression_opts=9,
                                chunks=(batch_size, 10, 10))
        
        batch = []
        i = 0
        while i < num_grids:
            if i % 10000 == 0:
                print(f"Generated {i} unique grids... ({(i/num_grids)*100:.1f}%)")

            grid = generate_valid_grid()
            grid_tuple = grid_to_tuple(grid)

            if grid_tuple not in unique_grids:
                unique_grids.add(grid_tuple)
                batch.append(grid)
                
                if i % 9999 == 0:
                    print("\nSample grid:")
                    print(grid)
                    print()

                i += 1

                if len(batch) == batch_size:
                    dset[i-batch_size:i] = batch
                    batch = []

        if batch:
            dset[num_grids-len(batch):num_grids] = batch

    end_time = time.time()
    print(f"\nGeneration complete! Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
