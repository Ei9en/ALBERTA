from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from lichess_bot.atomic_engine.bc_bot_stochastic import BCBotStochastic
from lichess_bot.atomic_engine.rl_bot import RLBot
from src.selfplay.game import SelfPlayGame


PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

BC_CHECKPOINT = (
    CHECKPOINT_DIR
    / "bc_epoch"
    / "bc_v2_5_epoch_5.pt"
)

RL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "rl_epoch"
    / "rl_epoch_39.pt"
)

TEMPERATURE = 0.5
GAMES = 100

# ============================================================
# Evaluation
# ============================================================

def play_match(
    bc_bot,
    rl_bot,
    games,
):

    bc_wins = 0
    rl_wins = 0
    draws = 0

    half = games // 2

    # --------------------------------------------------------
    # BC = White
    # --------------------------------------------------------

    print("\nBC5 = White")

    for _ in tqdm(
        range(half),
        desc="BC5 White",
    ):

        game = SelfPlayGame(
            bc_bot,
            rl_bot,
        )

        _, result = game.play()

        if result == "1-0":
            bc_wins += 1

        elif result == "0-1":
            rl_wins += 1

        else:
            draws += 1


    # --------------------------------------------------------
    # RL = White
    # --------------------------------------------------------

    print("\nRL20 = White")

    for _ in tqdm(
        range(half),
        desc="BC5 White",
    ):

        game = SelfPlayGame(
            rl_bot,
            bc_bot,
        )

        _, result = game.play()

        if result == "1-0":
            rl_wins += 1

        elif result == "0-1":
            bc_wins += 1

        else:
            draws += 1


    return (
        bc_wins,
        rl_wins,
        draws,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n==============================")
    print(" BC5 vs RL20")
    print("==============================")

    print("\nBC checkpoint:")
    print(BC_CHECKPOINT)

    print("\nRL checkpoint:")
    print(RL_CHECKPOINT)


    # --------------------------------------------------------
    # Load bots
    # --------------------------------------------------------

    bc_bot = BCBotStochastic(
        checkpoint=BC_CHECKPOINT,
        temperature=TEMPERATURE,
    )

    rl_bot = RLBot(
        checkpoint=RL_CHECKPOINT,
        temperature=TEMPERATURE,
    )


    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print(
        f"\nStarting {GAMES} games..."
    )

    bc_wins, rl_wins, draws = play_match(
        bc_bot,
        rl_bot,
        GAMES,
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    bc_score = (
        bc_wins
        + 0.5 * draws
    ) / GAMES

    rl_score = (
        rl_wins
        + 0.5 * draws
    ) / GAMES


    print("\n==============================")
    print(" RESULTS")
    print("==============================")

    print(
        f"BC5  : {bc_wins} W"
    )

    print(
        f"RL20 : {rl_wins} W"
    )

    print(
        f"Draw : {draws}"
    )

    print(
        f"\nBC5 score  : {100 * bc_score:.2f}%"
    )

    print(
        f"RL20 score : {100 * rl_score:.2f}%"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output = {

        "bc_checkpoint":
            str(BC_CHECKPOINT),

        "rl_checkpoint":
            str(RL_CHECKPOINT),

        "temperature":
            TEMPERATURE,

        "games":
            GAMES,

        "bc_wins":
            bc_wins,

        "rl_wins":
            rl_wins,

        "draws":
            draws,

        "bc_score":
            bc_score,

        "rl_score":
            rl_score,

    }


    output_path = (
        PROJECT_ROOT
        / f"rl_vs_bc5_{timestamp}.json"
    )


    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )


    print(
        "\nSaved:",
        output_path,
    )


# ============================================================

if __name__ == "__main__":
    main()