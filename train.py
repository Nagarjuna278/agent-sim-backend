import numpy as np
import environment as env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import policy_network as net # Assuming network.py contains the PolicyNetwork class
from torch.utils.tensorboard import SummaryWriter

class TrainerPG:
    def __init__(self, env_size=8, obstacles=None, agent_start=None, goal=None, learning_rate=0.0001, num_episodes=3000000, gamma=0.99):
        """
        Initializes the Policy Gradient Trainer.

        Args:
            env_size (int): Size of the square maze grid (e.g., 8 for 8x8).
            obstacles (list): List of (row, col) tuples representing obstacle positions.
            agent_start (tuple): (row, col) tuple for the agent's starting position.
            goal (tuple): (row, col) tuple for the goal position.
            learning_rate (float): Learning rate for the optimizer.
            num_episodes (int): Total number of training episodes.
            gamma (float): Discount factor for future rewards.
        """
        self.env = env.Environment(size=env_size, obstacles=obstacles, agent_start=agent_start,goal=goal)
        print("inside init")
        # Instantiate the PolicyNetwork class from your network.py file
        # Make sure the class name in network.py is PolicyNetwork sign 
        self.network = net.GridMapNetwork()
        # Uncomment the lines below if you want to load a pre-trained model
        # try:
        #     self.network.load_state_dict(torch.load('policy_network_state_dict.pth'))
        #     print("Loaded saved model state dictionary.")
        # except FileNotFoundError:
        #     print("No saved model state dictionary found, starting from scratch.")

        self.network.train() # Set network to training mode (enables dropout, batchnorm if used)
        self.weight_decay = 1e-2 # L2 regularization to prevent overfitting
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.rewards = [] # List to store rewards collected during an episode
        self.log_probs = [] # List to store log probabilities of actions taken
        self.env_size = env_size
        self.clip_return_value = 15 # Define clip value for advantage/return

    def select_action(self, state):
        """
        Selects an action based on the current state using the policy network.

        Args:
            state (np.ndarray): The current state representation from the environment (8x8 grid).

        Returns:
            tuple: The chosen action (e.g., (1, 0) for move down).
        """
        # Convert the state (numpy array) to a torch tensor
        # Add a batch dimension at the beginning (unsqueeze(0))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # --- Flatten the state tensor before passing to the network ---
        # The network expects input shape (batch_size, input_size)
        # We reshape the (1, 8, 8) tensor to (1, 64)
        # state_tensor_flattened = state_tensor.view(state_tensor.size(0), -1)
        # Alternatively, using flatten: state_tensor_flattened = state_tensor.flatten(start_dim=1)
        # -------------------------------------------------------------

        # Pass the correctly shaped tensor through the network
        # The network outputs action probabilities directly due to Softmax in forward()
        logits = self.network(state_tensor)

        action_probabilities = torch.softmax(logits,dim=-1)

        # Ensure action_probabilities is 1D for Categorical if batch size is 1
        # Squeeze removes dimensions of size 1, so (1, 4) becomes (4,)
        action_probabilities = action_probabilities.squeeze(0)

        # Get valid actions from the environment
        valid_actions = self.env.actions()
        # Define the mapping from action index to action tuple
        possible_actions = [(1,0),(0,1),(-1,0),(0,-1)] # Corresponds to indices 0, 1, 2, 3


        # --- FIX: Correctly mask invalid actions by setting their probability to zero and re-normalizing ---
        # Create a mask tensor initialized to zeros (will keep valid action probabilities)
        mask = torch.zeros_like(action_probabilities)

        # Set the mask to 1.0 for valid actions
        for i, action in enumerate(possible_actions):
            if action in valid_actions:
                mask[i] = 1.0

        # Apply the mask: multiply probabilities by the mask (0 for invalid, 1 for valid)
        masked_action_probabilities = action_probabilities * mask

        # Re-normalize the probabilities so they sum to 1 over the valid actions
        # Avoid division by zero if all actions are invalid (shouldn't happen in a solvable maze)
        sum_probs = masked_action_probabilities.sum()
        if sum_probs == 0:
            # This case should ideally not be reached in a valid maze state where actions are possible.
            # If it happens, it means no valid action has a non-zero probability from the network.
            # You might want to handle this edge case, e.g., by assigning uniform probability
            # to valid actions if any exist, or raising an error if no valid actions exist.
            # For simplicity, let's assume at least one valid action has >0 probability initially.
            # If sum is 0, it might indicate a problem with training or environment setup.
            # As a fallback, we could assign uniform probability to valid actions here.
            # For now, let's proceed assuming sum_probs > 0 for states with valid actions.
             print("Warning: Sum of masked probabilities is zero. Check environment or network output.")
             # Fallback: Assign uniform probability to valid actions if sum is zero
             num_valid_actions = sum(1 for action in valid_actions)
             if num_valid_actions > 0:
                 uniform_prob = 1.0 / num_valid_actions
                 for i, action in enumerate(possible_actions):
                     if action in valid_actions:
                         masked_action_probabilities[i] = uniform_prob
                 sum_probs = 1.0 # Reset sum for normalization below
             else:
                 # If no valid actions exist, this state shouldn't be reachable or needs special handling
                 raise ValueError("Agent in a state with no valid actions.")


        # Normalize by the sum of the remaining probabilities
        # Add a small epsilon to the denominator to prevent division by zero if sum_probs is close to 0
        normalized_action_probabilities = masked_action_probabilities / (sum_probs + 1e-8)
        # ----------------------------------------------------------------------------------------

        # Sample an action from the normalized probability distribution
        # Use the normalized probabilities to ensure we only sample valid actions
        m = Categorical(normalized_action_probabilities)
        action_index = m.sample()

        # Store the log probability of the chosen action for training
        # It's important to use the log probability from the distribution
        # we actually sampled from (the normalized one).
        self.log_probs.append(m.log_prob(action_index))

        # Return the actual action tuple corresponding to the sampled index
        return possible_actions[action_index.item()]

    def finish_episode(self):
        """
        Calculates the policy loss for the episode and updates the network weights.
        """
        R = 0 # Initialize discounted return
        policy_loss = [] # List to store loss for each step in the episode
        returns = [] # List to store discounted returns for each step

        # Calculate discounted returns (Gt) for each step
        # Iterate backwards through the rewards
        for r in self.rewards[::-1]:
            R = r + self.gamma * R # Gt = rt + gamma * Gt+1
            returns.insert(0, R) # Insert at the beginning to get returns in chronological order

        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns (optional but often helpful for stability)
        # Avoid division by zero if std is very small
        if returns.std() > 1e-7:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        else:
            # If std is effectively zero, just center the returns around zero
            returns = (returns - returns.mean())

        # Apply Clipping to returns/advantages
        if self.clip_return_value is not None:
            returns = torch.clamp(returns, -self.clip_return_value, self.clip_return_value)

        # Calculate policy loss for each step: -log_prob * advantage
        # The 'advantage' here is approximated by the discounted return (R)
        # Ensure log_probs and returns have the same length
        if len(self.log_probs) != len(returns):
             print(f"Warning: Mismatch between log_probs ({len(self.log_probs)}) and returns ({len(returns)}). Skipping episode loss calculation.")
             self.rewards = []
             self.log_probs = []
             return # Skip loss calculation for this episode

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Perform backpropagation and update weights
        self.optimizer.zero_grad() # Clear previous gradients
        # Stack the individual loss tensors and sum them to get the total loss for the episode
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward() # Compute gradients
        self.optimizer.step() # Update network parameters

        # Clear episode data for the next episode
        self.rewards = []
        self.log_probs = []

    def get_state_representation(self):
        """
        Retrieves the current state representation from the environment.
        This method is not directly used in the training loop but can be useful.
        """
        # Assuming your environment's state is a 2D grid (numpy array)
        return self.env.grid.copy()

    def train(self):
        """
        Runs the main training loop for the policy gradient agent.
        """
        # Initialize TensorBoard writer for logging training progress
        log_dir = f"runs/Multi_GridMapNetworkStep3M_1" # Unique directory name for this run
        writer = SummaryWriter(log_dir)

        print(f"Starting training for {self.num_episodes} episodes...")

        # Main training loop
        for episode in range(self.num_episodes):
            # Reset environment and get initial state
            state = self.env.reset()

            local_start = self.env.agent_pos # Store start pos for logging
            local_goal = self.env.goal # Store goal pos for logging

            done = False # Flag to indicate if the episode is finished
            total_reward = 0 # Accumulate reward for the episode
            total_steps = 0 # Count steps in the episode
            # Set a reasonable step limit to prevent infinite loops in difficult mazes
            step_limit = self.env.size * self.env.size * 4 # Example limit

            # Episode loop
            while not done and total_steps < step_limit:
                total_steps += 1
                # Select an action using the policy network
                action = self.select_action(state)

                # Take the action in the environment
                next_state, reward, done = self.env.step(action)

                # Store reward for training
                self.rewards.append(reward)
                total_reward += reward # Accumulate reward for logging

                # Update state for the next step
                state = next_state

            # Finish the episode: calculate loss and update network weights
            self.finish_episode()

            # --- Logging Metrics ---
            # Calculate distance-based reward for logging (optional, for tracking progress)
            # This is different from the step-based reward used for training
            distance_before = ((self.env.distance_to_goal(local_start, local_goal)-1)*0.4) + 10
            # Positive value if the agent got closer to the goal during the episode
            distance_change_reward = distance_before - total_reward

            # Log metrics to TensorBoard
            writer.add_scalar('Training/Episode Total Reward', total_reward, episode)
            writer.add_scalar('Training/Episode Steps', total_steps, episode)
            writer.add_scalar('Training/Distance Change (Start to End)', distance_change_reward, episode)
            # Log whether the goal was reached (1 if reached, 0 otherwise)
            writer.add_scalar('Training/Reached Goal', 1 if done and self.env.agent_pos == local_goal else 0, episode)


            # Print progress and save model periodically
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward}, Steps: {total_steps}")
                # Save the state dictionary (weights and biases) of the network
                # This is generally preferred over saving the whole model as it's more flexible
                torch.save(self.network.state_dict(), 'Multi_GridMapNetworkStep3M.pth')
                # Optionally, save the entire model (includes architecture)
                # torch.save(self.network, 'policy_network_full_model.pth')
                print(f"Model state dictionary saved at episode {episode + 1}")

        print("Training finished.")
        writer.close() # Close the TensorBoard writer

if __name__ == "__main__":
    # Example usage:
    env_size = 8
    obstacles = [(0,3),(1,3),(3,1),(2,5),(3,6),(4,6),(4,2),(4,4),(5,3),(6,2)]
    # Ensure agent_start and goal are within bounds and not on obstacles if possible
    trainer_pg = TrainerPG(env_size=env_size, obstacles=obstacles, agent_start=(0, 0), goal=(env_size-1, env_size-1))
    trainer_pg.train()
