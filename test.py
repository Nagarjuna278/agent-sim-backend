import numpy as np
import torch
import torch.nn as nn
import pygame
import sys
import time # Import time for delays

# Assume environment.py and network.py are in the same directory
import environment as env
import policy_network as net # Assuming network.py contains the Network class

# --- Pygame Initialization ---
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)    # Agent color
RED = (255, 0, 0)      # Goal color
BLUE = (0, 0, 255)     # Obstacle color
GRAY = (150, 150, 150) # Grid line color

# Cell size for visualization
CELL_SIZE = 60 # Pixels per grid cell

# --- Drawing Function ---
def draw_grid(screen, environment, cell_size, colors):
    """
    Draws the current state of the grid environment using Pygame.

    Args:
        screen (pygame.Surface): The Pygame surface to draw on.
        environment (env.Environment): The environment object with grid, agent_pos, goal, and obstacles.
        cell_size (int): The size of each grid cell in pixels.
        colors (dict): A dictionary mapping color names (str) to RGB tuples.
    """
    screen.fill(colors['white']) # Fill background

    grid = environment.grid
    grid_size = environment.size

    for r in range(grid_size):
        for c in range(grid_size):
            cell_x = c * cell_size
            cell_y = r * cell_size
            cell_rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)

            # Draw cell background based on type
            if (r, c) == environment.goal:
                pygame.draw.rect(screen, colors['red'], cell_rect)
            elif (r, c) in environment.obstacles:
                pygame.draw.rect(screen, colors['blue'], cell_rect)
            else:
                 pygame.draw.rect(screen, colors['white'], cell_rect) # Default empty cell color

            # Draw grid lines
            pygame.draw.rect(screen, colors['gray'], cell_rect, 1) # Draw border

    # Draw agent
    agent_r, agent_c = environment.agent_pos
    agent_center_x = agent_c * cell_size + cell_size // 2
    agent_center_y = agent_r * cell_size + cell_size // 2
    agent_radius = int(cell_size * 0.35)
    pygame.draw.circle(screen, colors['green'], (agent_center_x, agent_center_y), agent_radius)

    pygame.display.flip() # Update the full display Surface to the screen


# --- Testing Function ---
def run_test_episode(env_size=8, obstacles=None, agent_start=(0, 0), goal=None, model_path='NetworkStep3M.pth', delay=0.2):
    """
    Runs a single test episode using a loaded policy network and visualizes it.

    Args:
        env_size (int): Size of the square maze grid.
        obstacles (list): List of (row, col) tuples representing obstacle positions.
        agent_start (tuple): (row, col) tuple for the agent's starting position.
        goal (tuple): (row, col) tuple for the goal position.
        model_path (str): Path to the saved PyTorch model state dictionary file.
        delay (float): Time delay in seconds between steps for visualization.
    """
    # Setup Environment
    # Use the provided obstacles, start, and goal for the test environment
    environment = env.Environment(size=env_size, obstacles=obstacles, agent_start=agent_start, goal=goal)

    # Setup Network
    # Instantiate the network architecture. It must match the training network.
    policy_network = net.GridMapNetwork()

    # Load trained weights
    try:
        # Load the state dictionary
        state_dict = torch.load(model_path)
        # Ensure the loaded state_dict keys match the current model keys, ignoring prefixes if any
        # This can sometimes be an issue if model was saved/loaded with DataParallel, etc.
        # Basic load:
        policy_network.load_state_dict(state_dict)
        print(f"Successfully loaded model state dictionary from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run the training script first.")
        pygame.quit()
        sys.exit()
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        # You might get errors here if the network architecture in network.py doesn't match
        # the architecture that saved the state_dict file.
        print("Please ensure network.py defines the same model architecture that was trained.")
        pygame.quit()
        sys.exit()


    # Set the network to evaluation mode
    # This is important for layers like Dropout or BatchNorm if they were used
    policy_network.eval()

    # --- Pygame Setup ---
    screen_width = env_size * CELL_SIZE
    screen_height = env_size * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Policy Gradient Agent Test")

    colors = {
        'black': BLACK, 'white': WHITE, 'green': GREEN,
        'red': RED, 'blue': BLUE, 'gray': GRAY
    }

    # Possible actions mapping (needs to match the one in environment and trainer)
    # Down, Right, Up, Left
    possible_actions = [(1,0),(0,1),(-1,0),(0,-1)]

    # --- Run Episode ---
    # Reset environment for the test episode
    state = environment.reset()
    done = False
    total_reward = 0
    step_count = 0
    # Use the same step limit logic as in training
    step_limit = environment.size * environment.size * 4

    print(f"Starting test episode from {environment.agent_pos} to {environment.goal}...")

    # Initial draw
    draw_grid(screen, environment, CELL_SIZE, colors)
    time.sleep(1) # Initial pause

    # Episode loop
    while not done and step_count < step_limit:
        step_count += 1

        # --- Action Selection (Deterministic for testing) ---
        # Get state representation (the grid)
        # Convert to tensor, add batch dimension, and flatten, matching training input
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Get policy output (action probabilities)
        # Use torch.no_grad() to disable gradient calculation during testing
        with torch.no_grad():
            # The network outputs probabilities directly if Softmax is in the forward pass
            # Or logits if Softmax is applied later.
            # Assuming network output is pre-Softmax logits based on common PG implementation
            # If your network has Softmax in forward(), remove the softmax line below.
            # Let's adjust based on the likelihood that your network.py returns logits
            # and Softmax/Categorical is applied in the select_action of the trainer.
            # If network.py *does* have Softmax in forward(), the output is probs.
            # Based on your trainer select_action using Categorical *after* network call,
            # let's assume network returns logits.
            action_logits = policy_network(state_tensor)


        # Apply valid action mask
        valid_actions = environment.actions() # Get valid actions from current env state
        # Create a mask where invalid action indices have a very low value (or -inf for logits)
        # This ensures they are not chosen after softmax
        action_mask = torch.full_like(action_logits, float('-inf')) # Use -inf for logits

        for i, action_tuple in enumerate(possible_actions):
             if action_tuple in valid_actions:
                 action_mask[0, i] = 0 # Set mask to 0 for valid action logits

        # Add mask to logits. Invalid actions will have very low logits.
        masked_logits = action_logits + action_mask

        # Convert logits to probabilities using Softmax over the valid actions
        action_probs = torch.softmax(masked_logits, dim=1).squeeze()

        # Select the action with the highest probability (deterministic for testing)
        # argmax finds the index of the maximum value
        action_index = torch.argmax(action_probs).item()

        # Get the actual action tuple
        action = possible_actions[action_index]
        # --- End Action Selection ---


        # --- Take Step ---
        # Check for Pygame quit event before taking step
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        next_state, reward, done = environment.step(action)
        total_reward += reward
        state = next_state # Update state for next iteration

        # --- Visualization ---
        draw_grid(screen, environment, CELL_SIZE, colors)

        # Pause for visualization
        time.sleep(delay)

    print(f"Episode finished after {step_count} steps.")
    if environment.agent_pos == environment.goal:
        print("Goal reached!")
    elif step_count >= step_limit:
        print("Step limit reached.")
    else:
        # This case means 'done' became true for another reason (e.g. hit wall)
        print("Episode terminated for another reason (e.g., hit wall).")
    print(f"Total Reward: {total_reward}")


    # --- Keep window open after episode finishes ---
    print("Episode complete. Close the window to exit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Keep the final state drawn (in case window was minimized/covered)
        draw_grid(screen, environment, CELL_SIZE, colors) # Redraw

        # Small delay to prevent the loop from consuming too much CPU
        time.sleep(0.05)

    pygame.quit()
    sys.exit()

# --- Main Execution ---
if __name__ == "__main__":
    # Define environment parameters for testing
    test_env_size = 8
    # Define obstacles for the test environment
    test_obstacles = [(0,3),(1,3),(3,1),(2,5),(3,6),(4,6),(4,2),(4,4),(5,3),(6,2)]
    # Define start and goal for the test environment
    test_agent_start = (0, 0)
    test_goal = (test_env_size - 1, test_env_size - 1)


    # Run the test episode with visualization
    run_test_episode(env_size=test_env_size,
                     obstacles=test_obstacles,
                     agent_start=test_agent_start,
                     goal=test_goal,
                     model_path='Multi_GridMapNetworkStep3M.pth', # Path to your trained model file
                     delay=0.1) # Delay in seconds between steps (adjust as needed)