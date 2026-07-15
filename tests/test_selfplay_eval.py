from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame
from src.actions_space import ACTIONS


CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_v2_5_epoch_3.pt"
)

DEVICE = "cpu"

N_GAMES = 150


def load_model():

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    bc_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = ActorCritic(bc_model)

    model.to(DEVICE)

    model.eval()

    return model



def update_result(counter, result):

    if result == "1-0":
        counter["win"] += 1

    elif result == "0-1":
        counter["loss"] += 1

    else:
        counter["draw"] += 1



def main():

    print("=== SELF PLAY EVALUATION ===")

    model = load_model()

    print("Model loaded.")


    deterministic_white = {
        "win": 0,
        "loss": 0,
        "draw": 0,
    }

    stochastic_white = {
        "win": 0,
        "loss": 0,
        "draw": 0,
    }


    total_lengths = []


    for i in range(N_GAMES):


        #
        # Alternance des couleurs
        #
        if i % 2 == 0:

            white_agent = ActorCriticAgent(
                model,
                deterministic=True,
            )

            black_agent = ActorCriticAgent(
                model,
                deterministic=False,
                temperature=0.5,
            )

            matchup = "deterministic white"


        else:

            white_agent = ActorCriticAgent(
                model,
                deterministic=False,
                temperature=0.5,
            )

            black_agent = ActorCriticAgent(
                model,
                deterministic=True,
            )

            matchup = "stochastic white"



        game = SelfPlayGame(
            white_agent,
            black_agent,
        )


        trajectory, result = game.play()


        total_lengths.append(
            len(trajectory)
        )


        if matchup == "deterministic white":

            update_result(
                deterministic_white,
                result,
            )

        else:

            update_result(
                stochastic_white,
                result,
            )


        print(
            f"Game {i+1}/{N_GAMES} "
            f"| {matchup} "
            f"| result={result} "
            f"| moves={len(trajectory)}"
        )


    print("\n=== RESULTS ===")


    print(
        "Deterministic white:"
    )

    print(
        deterministic_white
    )


    print(
        "\nStochastic white:"
    )

    print(
        stochastic_white
    )


    print(
        "\nAverage game length:",
        sum(total_lengths)
        /
        len(total_lengths)
    )


if __name__ == "__main__":

    main()