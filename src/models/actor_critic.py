# actor_critic.py

import torch.nn as nn

class ActorCritic(nn.Module):

    def __init__(self, bc_model):

        super().__init__()

        self.backbone = nn.Sequential(
            bc_model.input,
            bc_model.residuals
        )

        self.policy = bc_model.policy

        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


    def forward(self, x):

        features = self.backbone(x)

        policy = self.policy(features)

        value = self.value(features)

        return policy, value