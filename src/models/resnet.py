from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Residual block
# ============================================================

class ResidualBlock(nn.Module):
    """
    Residual convolutional block used by ChessResNet.

    The block applies two 3x3 convolutions with batch
    normalization and a ReLU activation after the residual
    connection.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                channels
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                channels
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.relu(
            x + self.block(x)
        )


# ============================================================
# Behavioral-cloning policy network
# ============================================================

class ChessResNet(nn.Module):
    """
    Residual policy network used for behavioral cloning.

    Input:
        Encoded chess board of shape
        (batch_size, in_channels, 8, 8).

    Output:
        Policy logits over ALBERTA's fixed action space.

    The network does not apply a softmax. Training losses and
    agents operate directly on the returned logits.
    """

    def __init__(
        self,
        in_channels: int = 19,
        channels: int = 32,
        blocks: int = 4,
        num_actions: int = 20160,
    ) -> None:

        super().__init__()

        # Stored explicitly because downstream actor-critic
        # construction uses the trunk channel width.
        self.channels = channels

        self.input = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(
                channels
            ),
            nn.ReLU(),
        )

        self.residuals = nn.Sequential(
            *[
                ResidualBlock(
                    channels
                )
                for _ in range(blocks)
            ]
        )

        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                channels * 8 * 8,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                num_actions,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.input(x)
        x = self.residuals(x)
        x = self.policy(x)

        return x