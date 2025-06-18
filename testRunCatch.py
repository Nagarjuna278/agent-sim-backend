import torch
import environment as env # Assuming this exists and defines Environment based on user code
import policy_network as net # Assuming this exists and defines GridMapNetwork
import pygame
import time
import numpy as np
import sys # For exiting pygame properly
import os # To check if image files exist
import random # Added import for random.choice fallback

# --- Pygame Configuration ---
pygame.init()

# Define colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)      # Obstacles
FALLBACK_RUNNER_COLOR = (0, 0, 255) # Blue fallback
FALLBACK_CATCHER_COLOR = (255, 0, 0) # Red fallback
GREY = (200, 200, 200) # Grid lines

# Define cell size for drawing
CELL_SIZE = 60 # Adjust as needed for screen size

# --- Image Filenames ---
# IMPORTANT: Make sure these PNG files exist in the same directory as the script!
RUNNER_IMG_FILENAME = "runner.png"
CATCHER_IMG_FILENAME = "catcher.png"
# ---------------------------


class TestRunCatchPygame:
    """
    Class to test trained Runner and Catcher models with Pygame visualization.
    Uses the provided Environment structure where 'agent_pos' is the catcher
    and 'goal' is the runner. Displays runner and catcher using PNG images.
    Includes enhanced error checking for image loading and refined drawing order.
    """
    def __init__(self, runner_model_path, catcher_model_path, grid_size=8):
        """
        Initializes the environment, loads models, sets up Pygame display,
        and loads agent images.

        Args:
            runner_model_path (str): Path to the saved runner model state_dict.
            catcher_model_path (str): Path to the saved catcher model state_dict.
            grid_size (int): The size of the grid (e.g., 8 for 8x8).
        """
        print("Initializing testing environment...")
        # Initialize environment
        try:
            self.env = env.Environment(size=grid_size)
        except Exception as e:
            print(f"FATAL ERROR: Could not initialize Environment: {e}")
            print("Ensure environment.py contains the Environment class as provided.")
            sys.exit()

        self.grid_size = grid_size

        # Load models
        print("Loading trained models...")
        # Ensure the network class exists and matches the definition used for training
        try:
            self.runner_network = net.GridMapNetwork()
            self.catcher_network = net.GridMapNetwork()
        except AttributeError:
             print("FATAL ERROR: GridMapNetwork class not found in policy_network.py.")
             sys.exit()
        except Exception as e:
             print(f"FATAL ERROR: Could not initialize network models: {e}")
             sys.exit()

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            self.runner_network.load_state_dict(torch.load(runner_model_path, map_location=device))
            print(f"Successfully loaded runner model from {runner_model_path}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Runner model file not found at {runner_model_path}")
            pygame.quit()
            sys.exit()
        except Exception as e:
            print(f"FATAL ERROR loading runner model state_dict: {e}")
            pygame.quit()
            sys.exit()
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.catcher_network.load_state_dict(torch.load(catcher_model_path, map_location=device))
            print(f"Successfully loaded catcher model from {catcher_model_path}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Catcher model file not found at {catcher_model_path}")
            pygame.quit()
            sys.exit()
        except Exception as e:
            print(f"FATAL ERROR loading catcher model state_dict: {e}")
            pygame.quit()
            sys.exit()

        self.runner_network.eval()
        self.catcher_network.eval()
        print("Models loaded and set to evaluation mode.")

        # --- Pygame Display Setup ---
        print("Setting up Pygame display...")
        self.screen_width = self.grid_size * CELL_SIZE
        self.screen_height = self.grid_size * CELL_SIZE
        try:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Run and Catch - Test Visualization")
        except pygame.error as e:
            print(f"FATAL ERROR initializing Pygame display: {e}")
            sys.exit()

        # --- Load Agent Images ---
        print("Loading agent images...")
        # Initialize image attributes to None first
        self.runner_img = None
        self.catcher_img = None
        self.runner_img = self.load_and_scale_image(RUNNER_IMG_FILENAME)
        self.catcher_img = self.load_and_scale_image(CATCHER_IMG_FILENAME)
        # Check if loading actually succeeded (load_and_scale_image now returns None on failure)
        if not self.runner_img:
             print(f"WARNING: Failed to load runner image '{RUNNER_IMG_FILENAME}'. Will use fallback color.")
        if not self.catcher_img:
             print(f"WARNING: Failed to load catcher image '{CATCHER_IMG_FILENAME}'. Will use fallback color.")

        print("Initialization complete.")

    def load_and_scale_image(self, filename):
        """Loads an image, scales it, and optimizes for display. Returns None on failure."""
        # Construct absolute path relative to the script file
        # __file__ is the path to the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)

        # --- ADDED PRINT ---
        print(f"Attempting to load image from: {filepath}")
        # -------------------

        if not os.path.exists(filepath):
             print(f"ERROR: Image file not found: {filepath}")
             return None
        try:
            # Load the image
            image = pygame.image.load(filepath)
            # Scale the image to fit the cell size
            image = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
            # Optimize image format for faster drawing (handles transparency)
            image = image.convert_alpha()
            print(f"Successfully loaded and scaled {filename}")
            return image
        except pygame.error as e:
            print(f"ERROR loading or scaling image {filename}: {e}")
            return None
        except Exception as e:
             print(f"An unexpected error occurred while processing image {filename}: {e}")
             return None


    def select_action_test(self, state, agent_type):
        """
        Selects an action for the agent during testing (deterministic).
        Uses the loaded model and chooses the action with the highest probability.
        Handles invalid actions by masking.

        Args:
            state (np.array): The current environment state grid.
            agent_type (str): "runner" or "catcher".

        Returns:
            tuple: The selected action (e.g., (dx, dy)).
        """
        # Prepare state tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Ensure model and tensor are on the same device
        try:
            device = next(self.runner_network.parameters()).device
        except StopIteration:
             print("ERROR: Could not determine model device (no parameters found?). Using CPU.")
             device = torch.device('cpu')

        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            try:
                if agent_type == "runner":
                    network = self.runner_network
                    valid_actions = self.env.goalactions()
                else: # catcher
                    network = self.catcher_network
                    valid_actions = self.env.actions()
                # Move network to the correct device if necessary (paranoid check)
                network.to(device)
                logits = network(state_tensor)
            except Exception as e:
                 print(f"ERROR during network forward pass for {agent_type}: {e}")
                 return (0,0) # Return 'stay' action on error

            action_probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()

        possible_actions = [(1,0), (0,1), (-1,0), (0,-1)] # Down, Right, Up, Left

        # Masking invalid actions
        mask = torch.zeros_like(action_probabilities)
        valid_action_indices = []
        if not valid_actions: # Handle case where agent is trapped
             # Get current position for logging
             current_pos = self.env.goal if agent_type == 'runner' else self.env.agent_pos
             print(f"Warning: No valid actions available for {agent_type} at pos {current_pos}!")
             action_index = torch.argmax(action_probabilities).item()
             print(f"  Returning highest probability (potentially invalid) action index: {action_index}")
             if 0 <= action_index < len(possible_actions):
                 return possible_actions[action_index]
             else:
                 print("  Error: Argmax index out of bounds. Returning (0,0).")
                 return (0,0)

        for i, action in enumerate(possible_actions):
            if action in valid_actions:
                mask[i] = 1.0
                valid_action_indices.append(i)

        if not valid_action_indices:
             current_pos = self.env.goal if agent_type == 'runner' else self.env.agent_pos
             print(f"Error: No valid action indices derived for {agent_type} at {current_pos} from valid_actions: {valid_actions}. Returning (0,0).")
             return (0,0)


        masked_action_probabilities = action_probabilities * mask
        sum_probs = masked_action_probabilities.sum()

        if sum_probs < 1e-8:
            print(f"Warning: All valid actions for {agent_type} have near-zero probability. Choosing uniformly from valid actions.")
            chosen_valid_index = random.choice(valid_action_indices)
            action_index = chosen_valid_index
        else:
            normalized_action_probabilities = masked_action_probabilities / sum_probs
            action_index = torch.argmax(normalized_action_probabilities).item()

        if 0 <= action_index < len(possible_actions):
             if action_index in valid_action_indices:
                 return possible_actions[action_index]
             else:
                 print(f"Warning: Argmax index {action_index} doesn't correspond to a known valid index {valid_action_indices}. Choosing first valid action.")
                 first_valid_action_index = valid_action_indices[0]
                 return possible_actions[first_valid_action_index]
        else:
             print(f"  Error: Calculated action_index {action_index} is out of bounds. Choosing first valid action.")
             first_valid_action_index = valid_action_indices[0]
             return possible_actions[first_valid_action_index]


    def visualize_step_pygame(self, grid, runner_pos, catcher_pos, step, agent_turn):
        """
        Visualizes the current grid state using Pygame, drawing agent images or fallbacks.
        Refined drawing order: Draw static elements first, then agents on top.

        Args:
            grid (np.array): The base environment grid (showing obstacles).
            runner_pos (tuple): (row, col) of the runner (env.goal).
            catcher_pos (tuple): (row, col) of the catcher (env.agent_pos).
            step (int): The current step number in the episode.
            agent_turn (str): Indicates whose turn it is/just was.
        """
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Pygame window closed by user.")
                pygame.quit()
                sys.exit()

        # --- Draw Static Elements ---
        self.screen.fill(WHITE)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Draw grid line
                pygame.draw.rect(self.screen, GREY, rect, 1)
                # Draw obstacle if present
                try:
                    if grid[r, c] == -1: # Assuming -1 is obstacle
                        pygame.draw.rect(self.screen, BLACK, rect)
                except IndexError:
                     print(f"Warning: Grid index [{r},{c}] out of bounds during drawing.")
                     continue # Skip drawing this cell

        # --- Draw Agents on Top ---
        # Calculate top-left corner for blitting/drawing
        catcher_draw_pos = (catcher_pos[1] * CELL_SIZE, catcher_pos[0] * CELL_SIZE)
        runner_draw_pos = (runner_pos[1] * CELL_SIZE, runner_pos[0] * CELL_SIZE)

        # Draw Catcher (Image or Fallback)
        if self.catcher_img:
             try:
                 self.screen.blit(self.catcher_img, catcher_draw_pos)
             except Exception as e:
                  print(f"ERROR blitting catcher image: {e}")
                  catcher_rect = pygame.Rect(catcher_draw_pos[0], catcher_draw_pos[1], CELL_SIZE, CELL_SIZE)
                  pygame.draw.ellipse(self.screen, FALLBACK_CATCHER_COLOR, catcher_rect)
        else: # Fallback if image wasn't loaded
             catcher_rect = pygame.Rect(catcher_draw_pos[0], catcher_draw_pos[1], CELL_SIZE, CELL_SIZE)
             pygame.draw.ellipse(self.screen, FALLBACK_CATCHER_COLOR, catcher_rect)

        # Draw Runner (Image or Fallback) - Drawn AFTER catcher
        if self.runner_img:
             try:
                self.screen.blit(self.runner_img, runner_draw_pos)
             except Exception as e:
                  print(f"ERROR blitting runner image: {e}")
                  runner_rect = pygame.Rect(runner_draw_pos[0], runner_draw_pos[1], CELL_SIZE, CELL_SIZE)
                  pygame.draw.ellipse(self.screen, FALLBACK_RUNNER_COLOR, runner_rect)
        else: # Fallback if image wasn't loaded
             runner_rect = pygame.Rect(runner_draw_pos[0], runner_draw_pos[1], CELL_SIZE, CELL_SIZE)
             pygame.draw.ellipse(self.screen, FALLBACK_RUNNER_COLOR, runner_rect)
        # -------------------------

        # Update Display and Pause
        pygame.display.set_caption(f"Run and Catch - Step: {step}, Turn: {agent_turn}")
        pygame.display.flip()
        pygame.time.wait(200) # Pause for 0.2 seconds


    def run_test_episode(self, max_steps=100):
        """
        Runs a single test episode with step-by-step Pygame visualization.

        Args:
            max_steps (int): Maximum number of steps allowed for the episode.
        """
        print("\n--- Starting Test Episode ---")
        try:
            state = self.env.reset()
            runner_pos = self.env.goal
            catcher_pos = self.env.agent_pos
            print(f"Initial State: Runner (Goal) at {runner_pos}, Catcher (Agent) at {catcher_pos}")
            if not isinstance(state, np.ndarray) or state.shape != (self.grid_size, self.grid_size):
                 print(f"ERROR: Environment reset did not return a valid {self.grid_size}x{self.grid_size} numpy array.")
                 pygame.quit()
                 sys.exit()

        except Exception as e:
            print(f"Error resetting environment: {e}")
            pygame.quit()
            sys.exit()

        done = False
        step = 0
        current_turn = "Runner"

        # Visualize initial state
        try:
            self.visualize_step_pygame(self.env.grid, runner_pos, catcher_pos, step, "Start")
        except Exception as e:
             print(f"Error during initial visualization: {e}")

        # Main Test Loop
        while not done and step < max_steps:
            step += 1

            # Handle Pygame events FIRST
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Pygame window closed by user during episode.")
                    done = True
                    break
            if done: continue

            # --- Agent Actions ---
            action_taken = False
            turn_for_viz = current_turn # Store whose turn it is for visualization

            if current_turn == "Runner":
                runner_action = self.select_action_test(state, "runner")
                try:
                    next_state, _, runner_done = self.env.runner_step(runner_action)
                    new_runner_pos = self.env.goal # Get new position
                    # Check for immediate catch after runner moves
                    if new_runner_pos == catcher_pos:
                        print("  Runner stepped onto Catcher!")
                        runner_done = True
                    runner_pos = new_runner_pos # Update position *after* check
                    action_taken = True
                except Exception as e:
                    print(f"  Error during runner step execution: {e}")
                    done = True # End episode on error
                if not done: # Only update state etc. if no error/done
                     state = next_state
                     done = runner_done
                     if not done: current_turn = "Catcher"

            elif current_turn == "Catcher":
                catcher_action = self.select_action_test(state, "catcher")
                try:
                    next_state, _, catcher_done = self.env.step(catcher_action)
                    catcher_pos = self.env.agent_pos # Update position *after* step
                    # env.step should set catcher_done if catcher_pos == runner_pos
                    action_taken = True
                except Exception as e:
                    print(f"  Error during catcher step execution: {e}")
                    done = True # End episode on error
                if not done: # Only update state etc. if no error/done
                    state = next_state
                    done = catcher_done
                    if not done: current_turn = "Runner"

            # Visualize only if an action was successfully taken and game not ended by error
            if action_taken and not done:
                try:
                    # Pass the correct grid state (which is 'state' after the update)
                    self.visualize_step_pygame(state, runner_pos, catcher_pos, step, turn_for_viz)
                except Exception as e:
                    print(f"Error during visualization update: {e}")
            elif done:
                 break # Exit the loop immediately if done flag is set


        # --- End of Episode ---
        print("\n--- Episode Finished ---")
        print(f"Total Steps: {step}")
        final_outcome = "Max steps reached"
        turn_for_viz = "End"
        if runner_pos == catcher_pos:
             final_outcome = "Catcher caught the Runner!"
             turn_for_viz = "Caught!"
        elif step >= max_steps:
             final_outcome = "Max steps reached"
             turn_for_viz = "Max Steps"
        print(f"Outcome: {final_outcome}")

        # Final visualization update (if Pygame hasn't quit)
        if pygame.display.get_init():
            try:
                # Use the very last known grid state ('state') and positions
                self.visualize_step_pygame(state, runner_pos, catcher_pos, step, final_outcome)
                print("Displaying final state. Close the Pygame window to exit.")
            except Exception as e:
                print(f"Error during final visualization: {e}")

            # Keep window open
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                pygame.time.wait(50)

        pygame.quit()
        print("Pygame closed.")


if __name__ == "__main__":
    # Configuration
    RUNNER_MODEL_PATH = "Runner_GridMapStep2.5M.pth"
    CATCHER_MODEL_PATH = "Catcher_GridMapStep2.5M.pth"
    MAX_EPISODE_STEPS = 50
    GRID_SIZE = 8

    # Prerequisite Check
    print("--- Prerequisite Checks ---")
    # Check for python files and classes
    try:
        print("Checking for environment.py and Environment class...")
        import environment as env
        if not hasattr(env, 'Environment'): raise ImportError("Environment class not found")
        print("Checking for policy_network.py and GridMapNetwork class...")
        import policy_network as net
        if not hasattr(net, 'GridMapNetwork'): raise ImportError("GridMapNetwork class not found")
        print("Required Python files and classes found.")
    except ImportError as e:
        print(f"FATAL ERROR: Cannot import required file or class: {e}")
        sys.exit()
    except Exception as e:
        print(f"FATAL ERROR during import/check: {e}")
        sys.exit()

    # Check for image files
    print(f"Checking for runner image: {RUNNER_IMG_FILENAME}...")
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get script directory
    runner_img_path = os.path.join(script_dir, RUNNER_IMG_FILENAME)
    if not os.path.exists(runner_img_path):
         print(f"  WARNING: Runner image file not found at '{runner_img_path}'. Will use fallback.")
    else:
         print("  Runner image found.")

    print(f"Checking for catcher image: {CATCHER_IMG_FILENAME}...")
    catcher_img_path = os.path.join(script_dir, CATCHER_IMG_FILENAME)
    if not os.path.exists(catcher_img_path):
         print(f"  WARNING: Catcher image file not found at '{catcher_img_path}'. Will use fallback.")
    else:
         print("  Catcher image found.")
    print("--- End Prerequisite Checks ---")


    # Run the Test
    tester = TestRunCatchPygame(RUNNER_MODEL_PATH, CATCHER_MODEL_PATH, grid_size=GRID_SIZE)
    tester.run_test_episode(max_steps=MAX_EPISODE_STEPS)

    print("Test script finished.")
