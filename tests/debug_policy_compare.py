from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.encoding import encode_fen
from src.actions_space import ACTIONS, INDEX_TO_ACTION


DEVICE = "cpu"

BC_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bc_v2_5_epoch_3.pt"
RL_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "rl_best.pt"


def load_bc():

    ckpt = torch.load(
        BC_CHECKPOINT,
        map_location=DEVICE
    )

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return model


def load_rl():

    ckpt = torch.load(
        RL_CHECKPOINT,
        map_location=DEVICE
    )

    backbone = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    model = ActorCritic(
        backbone
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return model


bc = load_bc()
rl = load_rl()


# position de départ
x = encode_fen(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
)

x = x.unsqueeze(0)


with torch.no_grad():

    bc_logits = bc(x)

    rl_policy, value = rl(x)


print("BC shape:", bc_logits.shape)
print("RL shape:", rl_policy.shape)


bc_logits = bc_logits[0]
rl_policy = rl_policy[0]


print("\n=== BC TOP ===")

bc_probs = torch.softmax(
    bc_logits,
    dim=0
)

values, indices = torch.topk(
    bc_probs,
    5
)

for v, idx in zip(values, indices):
    print(
        INDEX_TO_ACTION[idx.item()],
        v.item()
    )


print("\n=== RL TOP ===")

rl_probs = torch.softmax(
    rl_policy,
    dim=0
)

values, indices = torch.topk(
    rl_probs,
    5
)

for v, idx in zip(values, indices):
    print(
        INDEX_TO_ACTION[idx.item()],
        v.item()
    )