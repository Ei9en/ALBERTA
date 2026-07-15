# ResNet.py

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(channels),
        )


    def forward(self, x):

        return torch.relu(
            x + self.block(x)
        )



class ChessResNet(nn.Module):

    def __init__(
        self,
        in_channels=19,
        channels=64,
        blocks=4,
        num_actions=20160,
    ):

        super().__init__()


        self.input = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )


        self.residuals = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(blocks)
            ]
        )


        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                channels * 8 * 8,
                num_actions
            )
        )


    def forward(self, x):

        x = self.input(x)

        x = self.residuals(x)

        x = self.policy(x)

        return x