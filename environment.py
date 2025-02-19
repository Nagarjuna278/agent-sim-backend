import random
import numpy as np
import torch
from dataset import fetch_grid


class RunnerCatcherEnv:
    def __init__(self, grid_size=10, min_obstacles=1):
        self.grid_size = grid_size
        self.min_obstacles = min_obstacles
        self.action_space = 5  # [NO_MOVE, UP, DOWN, LEFT, RIGHT]
        self.action_map = {
            0: (0, 0),   # NO_MOVE
            1: (-1, 0),  # UP
            2: (1, 0),   # DOWN
            3: (0, -1),  # LEFT
            4: (0, 1)    # RIGHT
        }
        self.reset()

    def reset(self,episode=0):
        self.grid = fetch_grid(episode)
        
        self.num_obstacles = np.sum(self.grid == 1)
        self.runner_pos = np.argwhere(self.grid == 3)[0]
        self.catcher_pos = np.argwhere(self.grid == 2)[0]
        
        
        return self._get_state()

    def _get_state(self):
        """Convert grid to 3-channel state."""
        state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: Obstacles
        state[0] = (self.grid == 1).astype(np.float32)
        # Channel 1: Runner
        state[1] = (self.grid == 3).astype(np.float32)
        # Channel 2: Catcher
        state[2] = (self.grid == 2).astype(np.float32)
        
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def step(self, runner_action, catcher_action):
        """Execute one step of the environment."""
        # Store old positions
        old_runner_pos = self.runner_pos
        old_catcher_pos = self.catcher_pos
        
        # Calculate new positions
        runner_move = self.action_map[runner_action]
        catcher_move = self.action_map[catcher_action]
        
        new_runner_pos = (
            old_runner_pos[0] + runner_move[0],
            old_runner_pos[1] + runner_move[1]
        )
        new_catcher_pos = (
            old_catcher_pos[0] + catcher_move[0],
            old_catcher_pos[1] + catcher_move[1]
        )
        
        # Check validity and update positions
        if self._is_valid_move(new_runner_pos):
            self.grid[old_runner_pos] = 0
            self.grid[new_runner_pos] = 3
            self.runner_pos = new_runner_pos
        
        if self._is_valid_move(new_catcher_pos):
            self.grid[old_catcher_pos] = 0
            self.grid[new_catcher_pos] = 2
            self.catcher_pos = new_catcher_pos
        
        # Check if caught
        caught = self._is_caught()
        
        # Calculate rewards
        runner_reward = -1.0 if np.any(caught) else 0.1  # Penalty for being caught, small reward for surviving
        catcher_reward = 1.0 if np.any(caught) else -0.1  # Reward for catching, small penalty for not catching
        
        # Get new state
        next_state = self._get_state()
        
        return next_state, runner_reward, catcher_reward, np.any(caught)

    def _is_valid_move(self, pos):
        """Check if a move is valid."""
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        return self.grid[pos] != 1  # Can move if not obstacle

    def _is_caught(self):
        """Check if runner is caught."""
        return self.runner_pos == self.catcher_pos

    def render(self):
        """Render the grid."""
        symbols = {0: '.', 1: '#', 2: 'C', 3: 'R'}
        for row in self.grid:
            print(' '.join(symbols[int(cell)] for cell in row))
        print()
