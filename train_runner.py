import torch
import torch.optim as optim
import numpy as np
from environment import TownEnvironment  # Make sure this points to your MODIFIED environment.py file
from policy_network import PolicyNetwork # Make sure this points to your policy_network.py file
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque
from queue import PriorityQueue, Queue
from util import get_shortest_path_action
# Hyperparameters (same as before)
HIDDEN_SIZE = 256
LEARNING_RATE = 0.000001
NUM_EPISODES = 10000000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99995
MEMORY_SIZE = 10000
BATCH_SIZE = 32
REWARD_SCALING = 0.1
MAX_STEPS = 100
SURVIVAL_REWARD = 1.0
def load_models(runner_network):
    """Load saved models if they exist"""
    runner_path = f'models/runner_network_solo.pth'

    if os.path.exists(runner_path):
        print(f"Loading existing models")
        runner_network.load_state_dict(torch.load(runner_path))
        return True
    return False



class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def update_network(batch, network, optimizer, writer, step_count): # Added writer and step_count
    states = torch.FloatTensor(np.array([x[0] for x in batch]))
    actions = torch.LongTensor(np.array([x[1] for x in batch]))
    rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
    next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
    dones = torch.FloatTensor(np.array([x[4] for x in batch]))

    # Get current action probabilities
    probs = network(states)
    action_probs = probs[range(len(actions)), actions]

    # Calculate loss
    loss = -torch.mean(torch.log(action_probs) * rewards)

    # Update network
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
    optimizer.step()

    # --- ADDED METRICS FOR TENSORBOARD ---
    avg_action_probs = torch.mean(probs, dim=0) # Average probabilities across the batch for each action
    for i in range(avg_action_probs.size(0)): # Assuming 4 actions (0, 1, 2, 3)
        writer.add_scalar(f'Action_Probabilities/Action_{i}', avg_action_probs[i].item(), step_count)
    # -------------------------------------

    return loss.item()

def save_model(runner_network, episode):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(runner_network.state_dict(), f'models/runner_network_solo.pth')

def train():
    # --- CLEAR RUNS DIRECTORY AT START ---
    log_dir = 'runs/runner_training'
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    writer = SummaryWriter(log_dir)
    # ------------------------------------

    env = TownEnvironment()

    # Initialize runner network and optimizer
    runner_network = PolicyNetwork(100, HIDDEN_SIZE, 4)
    if load_models(runner_network):
        print("Loaded existing models")
    else:
        print("No existing models found")
    runner_optimizer = optim.Adam(runner_network.parameters(), lr=LEARNING_RATE)

    # Initialize experience buffer
    runner_memory = ExperienceBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    total_steps = 0

    for episode in range(NUM_EPISODES):
        state = env.set(episode)
        episode_reward = 0
        step = 0

        # Store episode experiences
        episode_memory = []

        while step < MAX_STEPS:
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)

            # --- NEW ACTION SELECTION WITH VALID MOVES ---
            valid_actions = []
            for action_idx in range(4):
                if env.is_valid_action(env.runner_pos, action_idx):
                    valid_actions.append(action_idx)


            if not valid_actions:
                print(f"Episode {episode}, Step {step}: No valid actions for runner!")
                runner_reward = 0 # Assign default reward here if no valid actions - IMPORTANT
                done = True # End episode as no valid action is available
                next_state = state # next state is same as current state.
                break


            with torch.no_grad():
                runner_probs_all_actions = runner_network(state_tensor)[0]
                runner_probs_valid_actions = runner_probs_all_actions[valid_actions]

            if random.random() < epsilon:
                runner_action = random.choice(valid_actions)
            else:
                runner_action = valid_actions[torch.argmax(runner_probs_valid_actions).item()]


            # Get catcher action using pathfinding (now BFS)
            catcher_action = get_shortest_path_action(
                state,
                env.catcher_pos,
                env.runner_pos
            )

            # Take step
            next_state, (runner_reward, _), done = env.step(runner_action, catcher_action) # runner_reward is now guaranteed to be assigned

            # --- DEBUGGING LOGGING AFTER STEP ---
            # print(f"Episode: {episode}, Step: {step}, Runner Action: {runner_action}, Catcher Action: {catcher_action}, Runner Reward: {runner_reward}, Done: {done}, Valid Actions: {valid_actions}")


            # Scale runner reward
            runner_reward *= REWARD_SCALING

            # Store experience
            episode_memory.append((state, runner_action, runner_reward, next_state, done))

            episode_reward += runner_reward

            if done:
                break

            state = next_state
            step += 1
            total_steps += 1

        if step == MAX_STEPS: # Episode ended because runner survived MAX_STEPS
            episode_reward += SURVIVAL_REWARD  # Add survival bonus reward
            print(f"Episode {episode}: Runner survived and gets survival bonus reward: {SURVIVAL_REWARD}")

        # If runner survived (reached max steps or game ended due to other reasons), update memory
        if step >= 0: # Changed condition to always update memory if episode ended (either done or max steps)
            for exp in episode_memory:
                runner_memory.push(*exp)

        # Train network if enough samples
        if len(runner_memory) > BATCH_SIZE and step == MAX_STEPS - 1:
            runner_batch = runner_memory.sample(BATCH_SIZE)
            runner_loss = update_network(runner_batch, runner_network, runner_optimizer, writer, total_steps)
            writer.add_scalar('Loss/runner', runner_loss, total_steps)

        # Decay epsilon
        if episode%1000 == 0: #Decay every 1000 episodes
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Log episode stats
        writer.add_scalar('Reward/runner', episode_reward, episode)
        writer.add_scalar('Steps/episode', step, episode)
        writer.add_scalar('Epsilon', epsilon, episode)
        writer.add_scalar('Memory/size', len(runner_memory), episode)
        writer.flush()

        # Save model periodically
        if (episode + 1) % 1000 == 0:
            save_model(runner_network, episode)
            print(f"Episode {episode+1}: Runner Reward: {episode_reward:.2f}, Steps: {step}")
            print(f"Memory size: {len(runner_memory)}")


    writer.close()

if __name__ == "__main__":
    train()