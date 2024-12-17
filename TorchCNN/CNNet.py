import torch
import torch.nn as nn


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=1, padding=0)  # Output: [batch_size, 32, 21, 89]

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)  # Output: [batch_size, 64, 9, 43]

        self.flatten = nn.Flatten()

        # Adjusted input size based on pooling and conv layers
        self.linear1 = nn.Linear(86 * 10 * 64, 32)  # Adjust input size here
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 36)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        x = self.linear4(x)

        return x
