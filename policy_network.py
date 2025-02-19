import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class CRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) for the runner-catcher game.  Includes CNN, LSTM, and Attention.
    Rewritten for 10x10 grid and 5 output actions.
    """

    def __init__(self, grid_size=10, num_actions=5): # Default to 10x10 grid and 5 actions
        super(CRNN, self).__init__()
        self.grid_size = grid_size
        self.num_actions = num_actions

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input channels: 3 (empty, runner, catcher)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16 x (grid_size/2) x (grid_size/2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x (grid_size/4) x (grid_size/4)


        # --- Calculate the flattened size after convolutions ---
        conv_output_size = self._calculate_conv_output_size()

        # --- LSTM Layer ---
        self.lstm = nn.LSTM(conv_output_size, 128, batch_first=True)  # 128 LSTM units

        # --- Attention Mechanism ---
        self.attention_weights = nn.Linear(128, 1)  # Learnable weights for attention

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(128, 64)  # Hidden layer
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_actions)  # Output layer (Q-values for each action), now 5 outputs


    def _calculate_conv_output_size(self):
        """Calculates the flattened size of the convolutional output for 10x10 grid."""
        # Dummy input to get the output size after convolutional layers
        dummy_input = torch.zeros(1, 3, self.grid_size, self.grid_size) # Use self.grid_size, which is now 10
        x = self.pool1(self.relu1(self.conv1(dummy_input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x.view(1, -1).size(1)


    def forward(self, x, hidden_state=None):
        """
        Forward pass of the CRNN.  No changes needed in forward pass for grid size or output dim.

        Args:
            x (torch.Tensor): Input tensor (grid state).  Shape: (batch_size, channels, height, width), height and width now 10
            hidden_state (tuple, optional):  Previous hidden state for the LSTM. (h_0, c_0).

        Returns:
           tuple: (Q-values, new hidden state), Q-values now length 5
        """
        # --- Convolutional part ---
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, flattened_features)

        # --- Reshape for LSTM ---
        x = x.unsqueeze(1)  # Add a time step dimension: (batch_size, time_steps=1, flattened_features)


        # --- LSTM part ---
        lstm_out, hidden_state = self.lstm(x, hidden_state)  # lstm_out: (batch_size, 1, 128)

        # --- Attention Mechanism ---
        attention_scores = self.attention_weights(lstm_out)  # (batch_size, 1, 1)
        attention_weights = torch.softmax(attention_scores, dim=1) # Apply softmax to get probabilities
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, 128)  Weighted sum

        # --- Fully connected part ---
        x = self.relu3(self.fc1(context_vector))
        q_values = self.fc2(x) # Output q_values will now have 5 dimensions

        return q_values, hidden_state


# --- Replay Memory ---
class ReplayMemory:
    # ReplayMemory class remains unchanged as it's independent of grid size/action space
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
