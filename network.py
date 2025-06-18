import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """
    Neural network for a policy gradient agent in an 8x8 maze.

    Takes the flattened state as input and outputs a probability
    distribution over the possible actions.
    """
    def __init__(self, input_size=64, hidden_size1=64, hidden_size2=32, output_size=4, use_leaky_relu=False):
        """
        Initializes the Policy Network.

        Args:
            input_size (int): The size of the flattened input state (8*8=64).
            hidden_size1 (int): The number of neurons in the first hidden layer.
            hidden_size2 (int): The number of neurons in the second hidden layer.
            output_size (int): The number of possible actions (e.g., 4 for Up, Down, Left, Right).
            use_leaky_relu (bool): If True, use Leaky ReLU; otherwise, use standard ReLU.
        """
        super(Network, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size1) # Input layer to first hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(hidden_size2, output_size) # Second hidden layer to output layer (logits)

        # Choose activation function for hidden layers
        if use_leaky_relu:
            self.hidden_activation = nn.LeakyReLU()
        else:
            self.hidden_activation = nn.ReLU()

        # Note: Softmax is applied in the forward pass to the output of fc3
    

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input state tensor, expected shape (batch_size, input_size).

        Returns:
            torch.Tensor: A tensor of action probabilities, shape (batch_size, output_size).
        """
        # Pass input through the first layer and apply activation
        x = self.hidden_activation(self.fc1(x))

        # Pass through the second layer and apply activation
        x = self.hidden_activation(self.fc2(x))

        # Pass through the output layer to get logits
        # No activation here yet, Softmax will be applied next
        logits = self.fc3(x)

        # Apply Softmax to convert logits into a probability distribution
        # dim=1 specifies that the softmax is applied across the action dimension
        action_probabilities = F.softmax(logits, dim=1)

        return action_probabilities
