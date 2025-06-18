import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GridMapNetwork(nn.Module):
    def __init__(self,size=None):
        super().__init__()
        if size is None:
            size = 8
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Simplified architecture (removed third conv layer)
        self.fc1 = nn.Linear(32 * size * size, 128)
        self.ln1 = nn.LayerNorm(128)  # Layer normalization
        self.fc2 = nn.Linear(128, 4)
        
        # Improved initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                # Small initial weights for final layer
                if m is self.fc2:
                    init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: (batch_size, 1, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(1)  # Flatten except batch dimension
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x