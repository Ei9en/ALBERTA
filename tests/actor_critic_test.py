import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic


model_bc = ChessResNet()

model_rl = ActorCritic(model_bc)

x = torch.randn(2, 19, 8, 8)

policy, value = model_rl(x)

print(policy.shape)
print(value.shape)