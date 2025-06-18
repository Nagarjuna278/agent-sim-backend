
# Multi-Agent Run and Catch Game: Policy Gradient Training

This project implements a Policy Gradient (PG) reinforcement learning algorithm to train an AI agent to navigate a grid-based environment. While the `TrainerPG` class is designed for a single agent (likely the Catcher, given the `env.step` and `env.actions` calls), the broader context implies it's part of a "Run and Catch" game.

This `TrainerPG` script focuses on training one of the agents, likely the Catcher, to reach a goal (which in a run-and-catch scenario would be the Runner's position).

## Features

* **Policy Gradient Algorithm:** Trains a neural network policy using the REINFORCE algorithm.
* **Grid-based Environment:** Uses a customizable grid world with obstacles.
* **PyTorch Implementation:** Leverages PyTorch for building and training the neural network.
* **TensorBoard Logging:** Tracks training progress (total reward, steps per episode, etc.) using TensorBoard.
* **Model Saving:** Periodically saves the trained model's state dictionary.
* **Action Masking:** Ensures the agent only attempts valid moves within the environment.
* **Return Clipping:** Applies clipping to discounted returns for training stability.

## Prerequisites

Before running the training script, ensure you have the following installed:

* **Python 3.x**
* **PyTorch:** `pip install torch`
* **NumPy:** `pip install numpy`
* **TensorBoard:** `pip install tensorboard`

You also need the following files in the same directory as the main script:

1.  **`environment.py`**: This file must define the `Environment` class, which handles the game logic, grid, agent positions, and valid actions.
2.  **`policy_network.py`**: This file must define the `GridMapNetwork` class, which is the neural network architecture used by the agent.

## Project Structure

```
.
├── trainer_pg.py             # Main script for training the agent
├── environment.py            # Defines the game environment
├── policy_network.py         # Defines the neural network architecture
└── runs/                     # Directory for TensorBoard logs
    └── Multi_GridMapNetworkStep3M_1  # Example log directory
```

## How to Run

1.  **Place all necessary files** (`trainer_pg.py`, `environment.py`, `policy_network.py`) into the same directory.
2.  **Open your terminal or command prompt.**
3.  **Navigate to the directory** where you saved the files.
4.  **Run the training script** using Python:

    ```bash
    python trainer_pg.py
    ```

### Viewing Training Progress with TensorBoard

While the script is running (or after it has completed), you can monitor the training progress using TensorBoard.

1.  **Open a new terminal or command prompt.**
2.  **Navigate to the project's root directory.**
3.  **Run TensorBoard:**

    ```bash
    tensorboard --logdir=runs
    ```

4.  **Open your web browser** and go to the address provided by TensorBoard (usually `http://localhost:6006`).

## Configuration

You can modify the following parameters in the `if __name__ == "__main__":` block and within the `TrainerPG` class constructor in `trainer_pg.py`:

* **`env_size`**: The size of the square grid (e.g., `8` for an 8x8 grid).
* **`obstacles`**: A list of `(row, col)` tuples representing fixed obstacle positions on the grid.
* **`agent_start`**: The starting `(row, col)` position for the agent.
* **`goal`**: The target `(row, col)` position for the agent. In a multi-agent setting, this might dynamically change (e.g., to the Runner's position).
* **`learning_rate`**: The learning rate for the AdamW optimizer (default: `0.0001`).
* **`num_episodes`**: The total number of training episodes to run (default: `3,000,000`).
* **`gamma`**: The discount factor for future rewards (default: `0.99`).
* **`weight_decay`**: L2 regularization strength for the optimizer (default: `1e-2`).
* **`clip_return_value`**: Value to clip the advantage/return at (default: `15`). This helps prevent large gradients.
* **`step_limit`**: The maximum number of steps allowed per episode to prevent infinite loops (calculated as `env_size * env_size * 4`).
* **`log_dir`**: The directory where TensorBoard logs will be saved (e.g., `runs/Multi_GridMapNetworkStep3M_1`). Change this for new training runs to avoid mixing logs.
* **`torch.save(...)` filename**: The name of the file where the model's state dictionary will be saved (e.g., `'Multi_GridMapNetworkStep3M.pth'`).

## Model Architecture (`policy_network.py`)

The `policy_network.py` file is expected to contain a PyTorch `nn.Module` class named `GridMapNetwork`. This network should be designed to:

* Accept a state representation from the `Environment` (likely a grid or flattened grid).
* Output logits for the possible actions. The actions are expected to be represented by indices: `0` for `(1,0)` (Down), `1` for `(0,1)` (Right), `2` for `(-1,0)` (Up), `3` for `(0,-1)` (Left).

## Environment (`environment.py`)

The `environment.py` file is expected to contain an `Environment` class with the following methods and attributes:

* `__init__(self, size, obstacles, agent_start, goal)`: Constructor to set up the grid, obstacles, and initial agent/goal positions.
* `reset(self)`: Resets the environment to a new starting state (or a fixed one if configured) and returns the initial state observation.
* `step(self, action)`: Takes an action for the *agent being trained by this script* (likely the Catcher), updates its position, calculates the reward, and returns `(next_state, reward, done)`.
* `actions(self)`: Returns a list of valid actions (tuples like `(dx, dy)`) that the current agent (Catcher) can take from its current position.
* `goalactions(self)`: (If present, likely for a Runner agent) Returns valid actions for the 'goal' (Runner). This method is not directly used by *this specific `TrainerPG`'s `select_action`*, but it's essential if `environment.py` supports both agents.
* `grid`: A NumPy array representing the game grid, where obstacles might be represented by a specific value (e.g., `-1`).
* `agent_pos`: The current `(row, col)` position of the agent being trained.
* `goal`: The current `(row, col)` position of the target (e.g., the Runner in a "Run and Catch" game).
* `distance_to_goal(self, pos1, pos2)`: A utility method to calculate the distance between two positions.

## Important Notes

* **Multi-Agent Training:** This `trainer_pg.py` script is primarily set up to train *one* agent. For a full "Run and Catch" game with two independent AI agents learning simultaneously (Runner and Catcher), you would typically need two separate training loops or a more complex multi-agent reinforcement learning setup (e.g., using A2C or PPO for both agents, or an adversarial training approach). This script likely represents the training for the Catcher to reach a static or pre-defined moving goal.
* **Model Naming:** The saved model is named `Multi_GridMapNetworkStep3M.pth`. Adjust this if you are training different versions or agents.
* **TensorBoard Log Directory:** Make sure the `log_dir` for `SummaryWriter` is unique for each distinct training run if you want to compare results easily in TensorBoard.
