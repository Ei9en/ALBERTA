# ============================================================
# criticality.py
#
# Modèle indépendant dédié à la prédiction de la criticité
# d'une position Atomic Chess.
#
# Classes :
#   0 = outcome_independent
#   1 = non_critical
#   2 = critical
#
# Ce modèle est totalement indépendant du ActorCritic RL.
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

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


    def forward(self, x):

        return torch.relu(
            x + self.block(x)
        )


# ============================================================
# Criticality Network
# ============================================================

class CriticalityNet(nn.Module):
    """
    Réseau indépendant dédié à la criticité.

    Input:
        Tensor [B, 19, 8, 8]

    Output:
        logits [B, 3]

    Classes:
        0 = outcome_independent
        1 = non_critical
        2 = critical
    """

    NUM_CLASSES = 3


    def __init__(
        self,
        in_channels=19,
        channels=32,
        blocks=4,
        num_classes=3,
    ):

        super().__init__()

        if num_classes != 3:

            raise ValueError(
                "CriticalityNet expects exactly 3 classes."
            )


        self.channels = channels

        self.num_classes = num_classes


        # ====================================================
        # Backbone
        # ====================================================

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


        # ====================================================
        # Criticality head
        # ====================================================

        self.criticality = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                channels * 8 * 8,
                512,
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                num_classes,
            ),
        )


    def forward(self, x):

        x = self.input(x)

        x = self.residuals(x)

        logits = self.criticality(x)

        return logits


    def probabilities(self, x):

        logits = self.forward(x)

        return torch.softmax(
            logits,
            dim=1,
        )


    def criticality_score(self, x):

        """
        Retourne P(critical).

        Indice :
            colonne 2 = critical
        """

        probabilities = self.probabilities(
            x
        )

        return probabilities[:, 2]


    def predict(self, x):

        """
        Retourne la classe prédite.
        """

        logits = self.forward(x)

        return torch.argmax(
            logits,
            dim=1,
        )