import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic


# Charger BC
bc_model = ChessResNet()

checkpoint = torch.load(
    "checkpoints/bc_v2_5_epoch_3.pt",
    map_location="cpu"
)

bc_model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("BC chargé")
print("Loss BC :", checkpoint["loss"])


# Créer Actor-Critic
model = ActorCritic(bc_model)

print("Actor-Critic créé")


# Test forward
x = torch.randn(2, 19, 8, 8)

policy, value = model(x)

print(policy.shape)
print(value.shape)