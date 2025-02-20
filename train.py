import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
from policy_network import CRNN
from environment import RunnerCatcherEnv
import os
from tensorboardX import SummaryWriter  # Import SummaryWriter

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.2
EPS_DECAY = 10000000
MEMORY_SIZE = 10000
LEARNING_RATE = 0.000001
NUM_EPISODES = 1000000
TARGET_UPDATE = 10000
SAVE_INTERVAL = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return None  # Return None when not enough samples in memory

    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0]).to(device)
    action_batch = torch.tensor(batch[1], dtype=torch.long).to(device)
    reward_batch = torch.tensor(batch[2], dtype=torch.float).to(device)
    next_state_batch = torch.cat(batch[3]).to(device)
    done_batch = torch.tensor(batch[4], dtype=torch.bool).to(device)

    q_values, _ = policy_net(state_batch)
    state_action_values = q_values.gather(1, action_batch.unsqueeze(1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q_values, _ = target_net(next_state_batch)
        next_state_values[~done_batch] = next_q_values[~done_batch].max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()

def select_action(state, policy_net, epsilon, device):
    if random.random() > epsilon:
        with torch.no_grad():
            q_values, _ = policy_net(state.to(device))
            return q_values.max(1)[1].item()
    else:
        return random.randrange(5)

def save_models(runner_policy, catcher_policy, episode):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(runner_policy.state_dict(), f'models/runner_network_episode_{episode}.pth')
    torch.save(catcher_policy.state_dict(), f'models/catcher_network_episode_{episode}.pth')
    print(f"Models saved at episode {episode}")

def train_agent(policy_net, target_net, optimizer, memory, state, epsilon, device):
    """Selects action using epsilon-greedy policy."""
    return select_action(state, policy_net, epsilon, device)


def train():
    env = RunnerCatcherEnv()
    writer = SummaryWriter()  # Initialize SummaryWriter

    agents = {
        'runner': {
            'policy': CRNN().to(DEVICE),
            'target': CRNN().to(DEVICE),
            'optimizer': optim.Adam(CRNN().parameters(), lr=LEARNING_RATE),
            'memory': ReplayMemory(MEMORY_SIZE),
            'hidden': None
        },
        'catcher': {
            'policy': CRNN().to(DEVICE),
            'target': CRNN().to(DEVICE),
            'optimizer': optim.Adam(CRNN().parameters(), lr=LEARNING_RATE),
            'memory': ReplayMemory(MEMORY_SIZE),
            'hidden': None
        },
    }
    agents['runner']['target'].load_state_dict(agents['runner']['policy'].state_dict())
    agents['catcher']['target'].load_state_dict(agents['catcher']['policy'].state_dict())
    agents['runner']['target'].eval()
    agents['catcher']['target'].eval()

    steps_done = 0

    for episode in range(NUM_EPISODES):
        state = env.reset(episode)
        # state = torch.from_numpy(state).float().unsqueeze(0) # Convert state to tensor

        total_runner_reward = 0
        total_catcher_reward = 0

        runner_loss=0
        catcher_loss=0

        while True:
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)

            runner_action = train_agent(agents['runner']['policy'], agents['runner']['target'], agents['runner']['optimizer'], agents['runner']['memory'], state, eps, DEVICE)
            catcher_action = train_agent(agents['catcher']['policy'], agents['catcher']['target'], agents['catcher']['optimizer'], agents['catcher']['memory'], state, eps, DEVICE)

            next_state, runner_reward, catcher_reward, done = env.step(runner_action, catcher_action)
            # next_state = torch.from_numpy(next_state).float().unsqueeze(0) # Convert next_state to tensor

            agents['runner']['memory'].push(state, runner_action, runner_reward, next_state, done)
            agents['catcher']['memory'].push(state, catcher_action, catcher_reward, next_state, done)

            runner_loss = optimize_model(agents['runner']['policy'], agents['runner']['target'], agents['runner']['optimizer'], agents['runner']['memory'], DEVICE)
            catcher_loss = optimize_model(agents['catcher']['policy'], agents['catcher']['target'], agents['catcher']['optimizer'], agents['catcher']['memory'], DEVICE)

            total_runner_reward += runner_reward
            total_catcher_reward += catcher_reward

            state = next_state
            steps_done += 1

            if done:
                break

        # Log to TensorBoard
        writer.add_scalar('Runner/reward', total_runner_reward, episode)
        writer.add_scalar('Catcher/reward', total_catcher_reward, episode)
        writer.add_scalar('Runner/loss', runner_loss, episode)
        writer.add_scalar('Catcher/loss', catcher_loss, episode)
        writer.add_scalar('Steps', steps_done, episode)


        if episode % TARGET_UPDATE == 0:
            agents['runner']['target'].load_state_dict(agents['runner']['policy'].state_dict())
            agents['catcher']['target'].load_state_dict(agents['catcher']['policy'].state_dict())

        if episode % SAVE_INTERVAL == 0:
            save_models(agents['runner']['policy'], agents['catcher']['policy'], episode)

        if episode % TARGET_UPDATE == 10000-1:
            print(f'Episode {episode}/{NUM_EPISODES}: Runner Reward: {total_runner_reward:.2f}, Catcher Reward: {total_catcher_reward:.2f}, Epsilon: {eps:.4f}')

    writer.close()  # Close the SummaryWriter


if __name__ == "__main__":
    train()