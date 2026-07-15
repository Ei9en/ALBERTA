from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame
from src.actions_space import ACTIONS, INDEX_TO_ACTION


BC_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_v2_5_epoch_3.pt"
)

RL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_best.pt"
)

DEVICE = "cpu"

class BCWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        policy = self.model(x)

        value = torch.zeros(
            (x.shape[0], 1),
            device=x.device,
        )

        return policy, value
    

def load_model(checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    state_dict = checkpoint["model_state_dict"]


    #
    # RL checkpoint
    #
    if any(
        key.startswith("backbone")
        for key in state_dict.keys()
    ):

        bc_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=64,
            blocks=4,
        )

        model = ActorCritic(
            bc_model
        )

        model.load_state_dict(
            state_dict
        )


    #
    # BC checkpoint
    #
    else:

        bc_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=64,
            blocks=4,
        )

        bc_model.load_state_dict(
            state_dict
        )

        model = BCWrapper(
            bc_model
        )


    model.to(DEVICE)
    model.eval()

    return model


def main():

    print("=== RL vs BC ===")

    rl_model = load_model(
        RL_CHECKPOINT
    )

    bc_model = load_model(
        BC_CHECKPOINT
    )


    rl_agent = ActorCriticAgent(
        rl_model,
        deterministic=False,
        temperature=0.5,
    )


    bc_agent = ActorCriticAgent(
        bc_model,
        deterministic=True,
        temperature=0.0,
    )


    rl_wins = 0
    bc_wins = 0
    draws = 0

    total_moves = 0


    N_GAMES = 20


    for i in range(N_GAMES):

        if i % 2 == 0:

            white = rl_agent
            black = bc_agent
            white_name = "RL"

        else:

            white = bc_agent
            black = rl_agent
            white_name = "BC"


        game = SelfPlayGame(
            white,
            black,
        )


        trajectory, result = game.play()


        moves = len(trajectory)

        total_moves += moves


        print(
            f"Game {i+1}/{N_GAMES} | "
            f"White={white_name} | "
            f"Result={result} | "
            f"Moves={moves}"
        )


        if result == "1-0":

            if white_name == "RL":
                rl_wins += 1
            else:
                bc_wins += 1


        elif result == "0-1":

            if white_name == "RL":
                bc_wins += 1
            else:
                rl_wins += 1


        else:

            draws += 1



    print("\n=== RESULTS ===")

    print(
        f"RL wins : {rl_wins}"
    )

    print(
        f"BC wins : {bc_wins}"
    )

    print(
        f"Draws   : {draws}"
    )

    print(
        f"Average length : "
        f"{total_moves/N_GAMES:.2f}"
    )


if __name__ == "__main__":
    main()