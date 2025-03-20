import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes=2, num_filters=32):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.ff_layers = nn.Sequential(
            nn.Linear(num_filters * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ff_layers(x)
        return x
