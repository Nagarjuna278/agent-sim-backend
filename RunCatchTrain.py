import torch
import torch.optim as optim
import torch.distributions as distributions # Changed alias for clarity
import policy_network as net # Assuming this file exists and defines GridMapNetwork
import environment as env    # Assuming this file exists and defines Environment
from torch.utils.tensorboard import SummaryWriter

class RunCatchTrain:
    def __init__(self, gamma=0.99, clip_return_value=None, runner_lr=0.0001, catcher_lr=0.0001):
        """
        Initializes the training environment, networks, optimizers, and storage.

        Args:
            gamma (float): Discount factor for future rewards.
            clip_return_value (float, optional): Value to clamp normalized returns. Defaults to None.
            runner_lr (float): Learning rate for the runner's optimizer.
            catcher_lr (float): Learning rate for the catcher's optimizer.
        """
        self.env = env.Environment()

        self.runner_network = net.GridMapNetwork()
        self.catcher_network = net.GridMapNetwork()

        # Load pre-trained weights for the catcher if desired
        try:
            # Use strict=False if the state dict might have missing/extra keys
            self.catcher_network.load_state_dict(torch.load("Catcher_GridMapStep2.5M.pth"))
            print("Loaded pre-trained weights for catcher.")
        except FileNotFoundError:
            print("Pre-trained catcher weights not found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading catcher weights: {e}. Starting from scratch.")

        try:
            # Use strict=False if the state dict might have missing/extra keys
            self.runner_network.load_state_dict(torch.load("Runner_GridMapStep2.5M.pth"))
            print("Loaded pre-trained weights for catcher.")
        except FileNotFoundError:
            print("Pre-trained catcher weights not found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading catcher weights: {e}. Starting from scratch.")

        self.runner_network.train()  # Set networks to training mode
        self.catcher_network.train()

        self.runner_optimizer = optim.AdamW(self.runner_network.parameters(), lr=runner_lr)
        self.catcher_optimizer = optim.AdamW(self.catcher_network.parameters(), lr=catcher_lr)

        # --- Initialize missing attributes ---
        self.gamma = gamma
        self.clip_return_value = clip_return_value

        # --- Initialize lists for storing episode data ---
        self.runner_rewards = []
        self.catcher_rewards = []
        self.log_runner_probs = []
        self.log_catcher_probs = []
        # It might also be useful to store states if needed later, but not for basic REINFORCE

    def select_action(self, state, agent_type):
        """
        Selects an action for the given agent based on the current state.

        Args:
            state: The current environment state.
            agent_type (str): "runner" or "catcher".

        Returns:
            tuple: (action (int), log_prob (torch.Tensor))
                   The selected action index and its log probability.
        """
        # Ensure state is a FloatTensor, add batch and channel dimensions if needed
        # Assuming state is already a numpy array like grid
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # --- Flatten the state tensor before passing to the network ---
        # The network expects input shape (batch_size, input_size)
        # We reshape the (1, 8, 8) tensor to (1, 64)
        # state_tensor_flattened = state_tensor.view(state_tensor.size(0), -1)
        # Alternatively, using flatten: state_tensor_flattened = state_tensor.flatten(start_dim=1)
        # -------------------------------------------------------------

        # Pass the correctly shaped tensor through the network
        # The network outputs action probabilities directly due to Softmax in forward()
        if agent_type == "runner":
            logits = self.runner_network(state_tensor)
            valid_actions = self.env.goalactions()
        else:
            logits = self.catcher_network(state_tensor)
            valid_actions = self.env.actions()


        action_probabilities = torch.softmax(logits,dim=-1)

        # Ensure action_probabilities is 1D for Categorical if batch size is 1
        # Squeeze removes dimensions of size 1, so (1, 4) becomes (4,)
        action_probabilities = action_probabilities.squeeze(0)

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
        m = distributions.Categorical(normalized_action_probabilities)
        action_index = m.sample()

        # Store the log probability of the chosen action for training
        # It's important to use the log probability from the distribution
        # we actually sampled from (the normalized one).
        if agent_type == "runner":
            self.log_runner_probs.append(m.log_prob(action_index))
        else:
            self.log_catcher_probs.append(m.log_prob(action_index))

        # Return the actual action tuple corresponding to the sampled index
        return possible_actions[action_index.item()]

    def finish_episode(self):
        """
        Calculates discounted returns and performs the policy gradient update
        for both agents at the end of an episode.
        """
        # --- Check for data consistency ---
        if not self.log_runner_probs or not self.log_catcher_probs:
            #  print("Warning: No log probabilities recorded for one or both agents. Skipping update.")
             self._clear_episode_data()
             return

        # Ensure rewards and log_probs have the same length for each agent
        # Note: In turn-based, lengths might differ by 1 if episode ends after runner's move.
        # Handle this asymmetry if necessary (e.g., pad or truncate).
        # For simplicity here, we assume they *should* match or the shorter one dictates the update length.


        # --- Calculate Discounted Returns (Gt) ---
        R_runner = 0
        runner_returns = []
        for r in reversed(self.runner_rewards): # Use reversed() for iterator
            R_runner = r + self.gamma * R_runner
            runner_returns.insert(0, R_runner) # Insert at the beginning

        R_catcher = 0 # Reset R for catcher calculation
        catcher_returns = []
        for r in reversed(self.catcher_rewards):
            R_catcher = r + self.gamma * R_catcher
            catcher_returns.insert(0, R_catcher)

        # Convert returns to tensors
        runner_returns = torch.tensor(runner_returns, dtype=torch.float32)
        catcher_returns = torch.tensor(catcher_returns, dtype=torch.float32)

        # --- Normalize Returns (optional but recommended) ---
        # Add small epsilon to std dev calculation for numerical stability
        eps = 1e-7

        if runner_returns.std() > eps:
            runner_returns = (runner_returns - runner_returns.mean()) / (runner_returns.std() + eps)
        else:
            runner_returns = (runner_returns - runner_returns.mean())

        if catcher_returns.std() > eps:
            catcher_returns = (catcher_returns - catcher_returns.mean()) / (catcher_returns.std() + eps)
        else:
            catcher_returns = (catcher_returns - catcher_returns.mean())

        if len(runner_returns) != len(self.log_runner_probs) or len(catcher_returns) != len(self.log_catcher_probs):
            print(f"Warning: Mismatch in lengths. Runner (R:{len(self.runner_rewards)}, LP:{len(self.log_runner_probs)}), "
                  f"Catcher (R:{len(self.catcher_rewards)}, LP:{len(self.log_catcher_probs)}). Truncating to min len: {min_len}")
            # Truncate lists to the minimum consistent length found
            self._clear_episode_data()
            return

        # --- Clip Returns (optional) ---
        if self.clip_return_value is not None:
            runner_returns = torch.clamp(runner_returns, -self.clip_return_value, self.clip_return_value)
            catcher_returns = torch.clamp(catcher_returns, -self.clip_return_value, self.clip_return_value)

        # --- Calculate Policy Loss ---
        policy_runner_loss = []
        # Iterate using the stored log probabilities and calculated returns
        for log_prob, Gt in zip(self.log_runner_probs, runner_returns):
             # Ensure log_prob requires grad if it somehow got detached
             # (usually not needed if sampled correctly from network output)
             # log_prob.requires_grad_()
            policy_runner_loss.append(-log_prob * Gt) # REINFORCE loss: -log_prob * Gt

        policy_catcher_loss = []
        for log_prob, Gt in zip(self.log_catcher_probs, catcher_returns):
            policy_catcher_loss.append(-log_prob * Gt)

        # --- Perform Backpropagation and Optimizer Step ---
        self.runner_optimizer.zero_grad()
        # Sum losses over the episode; stack creates a tensor from the list of tensors
        runner_total_loss = torch.stack(policy_runner_loss).sum()
        runner_total_loss.backward() # Compute gradients for runner network
        self.runner_optimizer.step()  # Update runner network parameters

        self.catcher_optimizer.zero_grad()
        catcher_total_loss = torch.stack(policy_catcher_loss).sum()
        catcher_total_loss.backward() # Compute gradients for catcher network
        self.catcher_optimizer.step() # Update catcher network parameters

        # --- Clear Episode Data ---
        self._clear_episode_data()

    def _clear_episode_data(self):
        """Clears lists used to store data during an episode."""
        self.runner_rewards.clear()
        self.catcher_rewards.clear()
        self.log_runner_probs.clear()
        self.log_catcher_probs.clear()


    def train(self, num_episodes=2500000):
        """
        Main training loop.

        Args:
            num_episodes (int): The total number of episodes to train for.
        """
        print(f"Starting training for {num_episodes} episodes...")
        log_dir = f"runs/RunCatchTrain"
        writer = SummaryWriter(log_dir=log_dir) # Initialize TensorBoard writer
        for episode in range(num_episodes):

            state = self.env.reset()
            done = False # Use a single 'done' flag if episode ends for both simultaneously
            steps = 0
            total_runner_reward_episode = 0
            total_catcher_reward_episode = 0

            localstart = self.env.agent_pos
            localgoal = self.env.goal

            # Clear episode data at the start of each episode
            # (Although finish_episode also clears, good practice to ensure clean start)
            self._clear_episode_data()
            prev_distance = self.env.distance_to_goal(localstart,localgoal)

            while not done:
                steps += 1

                # --- Runner's Turn ---
                runner_state = state # State observed by runner
                runner_action = self.select_action(runner_state, "runner")
                # Assuming env.step takes agent type now for clarity
                next_state, runner_reward, runner_done = self.env.runner_step(runner_action)
                if steps%10 == 9:
                    runner_reward = runner_reward+5
                
                if steps == 256:
                    runner_reward +=100
                    runner_done = True
                
                self.runner_rewards.append(runner_reward) # Store runner's log prob
                total_runner_reward_episode += runner_reward

                # Update state for the catcher's turn
                state = next_state

                # Check if runner's action ended the episode
                if runner_done:
                    # Catcher doesn't get a turn if runner ended it
                    # Need to decide if catcher gets a final reward/penalty here
                    # Assuming no final reward/action for catcher if runner ends episode
                    done = True
                else: # Exit the while loop

                    # --- Catcher's Turn ---
                    catcher_state = state # State observed by catcher (after runner moved)
                    catcher_action = self.select_action(catcher_state, "catcher")
                    # Environment steps based on catcher's action
                    next_state, catcher_reward, catcher_done = self.env.step(catcher_action)
                    if steps%10 == 9:
                        catcher_reward = catcher_reward-5

                    self.catcher_rewards.append(catcher_reward) # Store catcher's log prob
                    total_catcher_reward_episode += catcher_reward

                    # Update state for the next iteration (runner's turn)
                    state = next_state

                    # Check if catcher's action ended the episode
                    if catcher_done:
                        done = True
                    # Episode ends after catcher's move

                # Optional: Add a step limit to prevent infinite episodes
                # if steps >= max_steps_per_episode:
                #     done = True

            # --- End of Episode ---
            # print(len(self.runner_rewards), len(self.catcher_rewards))
            # print(len(self.log_runner_probs), len(self.log_catcher_probs))
            self.finish_episode() # Perform learning update

            startdistance = ((self.env.distance_to_goal(localstart,localgoal)-1)*0.4) +10

            rewarddiff = total_catcher_reward_episode - startdistance

            writer.add_scalar('Catcher/Reward', total_catcher_reward_episode, episode)
            writer.add_scalar('Catcher/Steps', steps, episode)
            writer.add_scalar('Catcher/diff', rewarddiff, episode)
            # --- Logging and Saving ---
            if episode % 1000 == 0:
                print(f"Episode {episode}, Steps: {steps}, Runner Reward: {total_runner_reward_episode:.2f}, Catcher Reward: {total_catcher_reward_episode:.2f}")
               
                try:
                    # Use standard torch.save for state dictionaries
                    torch.save(self.runner_network.state_dict(), f"Runner_GridMapStep2.5M.pth")
                    torch.save(self.catcher_network.state_dict(), f"Catcher_GridMapStep2.5M.pth")
                except Exception as e:
                    print(f"Error saving models at episode {episode}: {e}")

        print("Training finished.")


if __name__ == "__main__":
    trainer = RunCatchTrain(gamma=0.99, runner_lr=0.0001, catcher_lr=0.0001) # Pass hyperparameters
    trainer.train(num_episodes=2500000) # Specify number of episodes