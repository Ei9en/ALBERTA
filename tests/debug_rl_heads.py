import torch

ckpt = torch.load(
    "checkpoints/rl_best.pt",
    map_location="cpu"
)

state = ckpt["model_state_dict"]


for key in [
    "policy.1.weight",
    "policy.1.bias",
    "value.1.weight",
    "value.1.bias",
    "value.3.weight",
    "value.3.bias",
]:

    t = state[key]

    print("\n", key)
    print("shape:", t.shape)
    print("mean:", t.mean().item())
    print("std :", t.std().item())
    print("min :", t.min().item())
    print("max :", t.max().item())