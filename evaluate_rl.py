from pathlib import Path

import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame
from src.actions_space import ACTIONS


# ============================================================
# CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEVICE = "cpu"

RL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_20.pt"
)

RLAL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "al_epoch"
    / "al_epoch_20.pt"
)

GAMES = 1000
TEMPERATURE = 1.5


# ============================================================
# MODEL
# ============================================================

def build_model():

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        bc_model
    )

    model.to(DEVICE)

    return model


def load_model(path):

    model = build_model()

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# MATCH
# ============================================================

def play_game(
    white_model,
    black_model,
):

    white_agent = ActorCriticAgent(
        white_model,
        deterministic=False,
        temperature=TEMPERATURE,
    )

    black_agent = ActorCriticAgent(
        black_model,
        deterministic=False,
        temperature=TEMPERATURE,
    )

    game = SelfPlayGame(
        white_agent,
        black_agent,
    )

    _, result = game.play()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ALBERTA - RL10 vs RLAL10")
    print("=" * 70)

    print(
        f"RL10  : {RL_CHECKPOINT}"
    )

    print(
        f"RLAL10: {RLAL_CHECKPOINT}"
    )

    print(
        f"Games : {GAMES}"
    )

    print(
        f"Temperature : {TEMPERATURE}"
    )

    print(
        f"Device : {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("Loading RL10...")

    rl10 = load_model(
        RL_CHECKPOINT
    )

    print("Loading RLAL10...")

    rlal10 = load_model(
        RLAL_CHECKPOINT
    )

    print()
    print("Starting match...")
    print()

    rl10_wins = 0
    rlal10_wins = 0
    draws = 0

    # ========================================================
    # RL10 WHITE
    # ========================================================

    for i in range(
        GAMES // 2
    ):

        result = play_game(
            rl10,
            rlal10,
        )

        if result == "1-0":

            rl10_wins += 1

        elif result == "0-1":

            rlal10_wins += 1

        else:

            draws += 1

        print(
            f"Game {i + 1:2d}/{GAMES}: "
            f"RL10 White vs RLAL10 Black -> "
            f"{result}"
        )

    # ========================================================
    # RLAL10 WHITE
    # ========================================================

    for i in range(
        GAMES // 2
    ):

        result = play_game(
            rlal10,
            rl10,
        )

        if result == "1-0":

            rlal10_wins += 1

        elif result == "0-1":

            rl10_wins += 1

        else:

            draws += 1

        print(
            f"Game {GAMES // 2 + i + 1:2d}/{GAMES}: "
            f"RLAL10 White vs RL10 Black -> "
            f"{result}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    rl10_score = (
        rl10_wins
        + 0.5 * draws
    )

    rlal10_score = (
        rlal10_wins
        + 0.5 * draws
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"RL10   wins : {rl10_wins}"
    )

    print(
        f"RLAL10 wins : {rlal10_wins}"
    )

    print(
        f"Draws       : {draws}"
    )

    print()

    print(
        f"RL10 score   : "
        f"{rl10_score / GAMES * 100:.1f}%"
    )

    print(
        f"RLAL10 score : "
        f"{rlal10_score / GAMES * 100:.1f}%"
    )

    print()

    if rl10_score > rlal10_score:

        print("Winner: RL10")

    elif rlal10_score > rl10_score:

        print("Winner: RLAL10")

    else:

        print("Result: DRAW")

    print("=" * 70)


if __name__ == "__main__":
    main()