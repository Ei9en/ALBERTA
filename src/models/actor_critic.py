from __future__ import annotations

import torch
from torch import nn

from src.models.resnet import ChessResNet


class ActorCritic(nn.Module):
    """
    Actor-critic network initialized from a pretrained BC model.

    The convolutional backbone and policy head are reused directly
    from the behavioral-cloning model. A new value head is added on
    top of the shared spatial features.

    Notes
    -----
    The BC modules are reused by reference, not deep-copied.
    Training this ActorCritic therefore updates the transferred BC
    backbone and policy parameters.

    The value head is newly initialized. Its initialization is
    stochastic and must therefore be controlled by the training
    script's global random seed before constructing this model.
    """

    def __init__(
        self,
        bc_model: ChessResNet,
    ) -> None:

        super().__init__()

        # ----------------------------------------------------
        # Shared BC feature extractor
        # ----------------------------------------------------

        self.backbone = nn.Sequential(
            bc_model.input,
            bc_model.residuals,
        )

        # ----------------------------------------------------
        # Pretrained BC policy head
        # ----------------------------------------------------

        self.policy = (
            bc_model.policy
        )

        # ----------------------------------------------------
        # Newly initialized value head
        # ----------------------------------------------------

        channels = (
            bc_model.channels
        )

        self.value = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                channels * 8 * 8,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute policy logits and state-value estimates.

        Parameters
        ----------
        x
            Encoded board tensor with shape
            [batch_size, 19, 8, 8].

        Returns
        -------
        policy
            Raw action logits with shape
            [batch_size, num_actions].

        value
            State-value estimates with shape
            [batch_size, 1].
        """

        features = self.backbone(
            x
        )

        policy = self.policy(
            features
        )

        value = self.value(
            features
        )

        return (
            policy,
            value,
        )