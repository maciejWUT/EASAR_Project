import torch
import torch.nn as nn


class CNNet(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=1, padding=0)  # Output: [batch_size, 32, 13, 89]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [batch_size, 32, 7, 45]

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)  # Output: [batch_size, 64, 4, 42]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [batch_size, 64, 2, 21]

        self.flatten = nn.Flatten()

        # Input size based on pooling and conv layers
        self.linear1 = nn.Linear(64 * 1 * 20, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        x = self.linear4(x)

        return x
