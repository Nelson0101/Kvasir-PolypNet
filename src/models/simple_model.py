import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # Output shape = (batch_size, channels, 1, 1)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial size
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial size
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.ff_layers = nn.Sequential(
            nn.Linear(128, 64), nn.Linear(64, 32), nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)  # Reduce to (batch, channels, 1, 1)
        x = torch.flatten(
            x, start_dim=1
        )  # Flatten (batch, channels) instead of full spatial size
        x = self.ff_layers(x)
        return x
