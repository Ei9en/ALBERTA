from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import json
from datetime import datetime


from lichess_bot.atomic_engine.bc_bot_stochastic import BCBotStochastic
from lichess_bot.atomic_engine.rl_bot import RLBot
from src.selfplay.game import SelfPlayGame


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)


BC_CHECKPOINT = (
    CHECKPOINT_DIR
    / "bc_epoch"
    / "bc_v3_epoch_5.pt"
)


RL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "rl_epoch"
    / "rl_epoch_10.pt"
)


TEMPERATURE = 2.0

GAMES = 500

WORKERS = 3


# ============================================================
# WORKER STATE
# ============================================================

_WORKER_BC = None
_WORKER_RL = None


# ============================================================
# Worker initialization
# ============================================================

def init_worker(
    bc_checkpoint,
    rl_checkpoint,
    temperature,
):

    global _WORKER_BC
    global _WORKER_RL


    # --------------------------------------------------------
    # Load BC
    # --------------------------------------------------------

    _WORKER_BC = BCBotStochastic(
        checkpoint=bc_checkpoint,
        temperature=temperature,
    )


    # --------------------------------------------------------
    # Load RL
    # --------------------------------------------------------

    _WORKER_RL = RLBot(
        checkpoint=rl_checkpoint,
        temperature=temperature,
    )


# ============================================================
# Play one game
# ============================================================

def play_one_game(
    bc_is_white,
):

    global _WORKER_BC
    global _WORKER_RL


    # --------------------------------------------------------
    # BC = White
    # --------------------------------------------------------

    if bc_is_white:

        game = SelfPlayGame(
            _WORKER_BC,
            _WORKER_RL,
        )


    # --------------------------------------------------------
    # RL = White
    # --------------------------------------------------------

    else:

        game = SelfPlayGame(
            _WORKER_RL,
            _WORKER_BC,
        )


    _, result = game.play()


    # --------------------------------------------------------
    # Convert chess result to model winner
    # --------------------------------------------------------

    if result == "1-0":

        if bc_is_white:
            return "bc"

        else:
            return "rl"


    elif result == "0-1":

        if bc_is_white:
            return "rl"

        else:
            return "bc"


    else:

        return "draw"


# ============================================================
# Evaluation
# ============================================================

def play_match(
    bc_checkpoint,
    rl_checkpoint,
    games,
    workers,
):

    # --------------------------------------------------------
    # Color balance
    # --------------------------------------------------------

    half = games // 2


    tasks = (
        [True] * half
        +
        [False] * half
    )


    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print()

    print(
        "Evaluation configuration:"
    )

    print(
        f"Games   : {games}"
    )

    print(
        f"Workers : {workers}"
    )

    print(
        f"BC White: {half}"
    )

    print(
        f"RL White: {half}"
    )


    # --------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------

    print()

    print(
        "Loading workers..."
    )


    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(
            bc_checkpoint,
            rl_checkpoint,
            TEMPERATURE,
        ),
    ) as pool:


        results = list(
            tqdm(
                pool.imap_unordered(
                    play_one_game,
                    tasks,
                    chunksize=1,
                ),
                total=games,
                desc="Evaluation",
            )
        )


    # --------------------------------------------------------
    # Count results
    # --------------------------------------------------------

    bc_wins = results.count(
        "bc"
    )

    rl_wins = results.count(
        "rl"
    )

    draws = results.count(
        "draw"
    )


    return (
        bc_wins,
        rl_wins,
        draws,
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=============================="
    )
    print(
        " BC5 vs RL10"
    )
    print(
        "=============================="
    )


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    print()

    print(
        "BC checkpoint:"
    )

    print(
        BC_CHECKPOINT
    )


    print()

    print(
        "RL checkpoint:"
    )

    print(
        RL_CHECKPOINT
    )


    print()

    print(
        f"Temperature : {TEMPERATURE}"
    )

    print(
        f"Games       : {GAMES}"
    )

    print(
        f"Workers     : {WORKERS}"
    )


    # --------------------------------------------------------
    # Check checkpoint existence
    # --------------------------------------------------------

    if not BC_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"BC checkpoint not found:\n"
            f"{BC_CHECKPOINT}"
        )


    if not RL_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"RL checkpoint not found:\n"
            f"{RL_CHECKPOINT}"
        )


    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print()

    print(
        "=============================="
    )

    print(
        f"Starting {GAMES} games..."
    )

    print(
        "=============================="
    )


    bc_wins, rl_wins, draws = play_match(
        BC_CHECKPOINT,
        RL_CHECKPOINT,
        GAMES,
        WORKERS,
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


    decisive_games = (
        bc_wins
        + rl_wins
    )


    if decisive_games > 0:

        rl_winrate_decisive = (
            rl_wins
            / decisive_games
        )

    else:

        rl_winrate_decisive = 0.5


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()

    print(
        "=============================="
    )

    print(
        " RESULTS"
    )

    print(
        "=============================="
    )


    print()

    print(
        f"BC5  : {bc_wins} W"
    )

    print(
        f"RL10 : {rl_wins} W"
    )

    print(
        f"Draw : {draws}"
    )


    print()

    print(
        f"BC5 score  : "
        f"{100 * bc_score:.2f}%"
    )

    print(
        f"RL10 score : "
        f"{100 * rl_score:.2f}%"
    )


    print()

    print(
        f"RL10 decisive-game winrate : "
        f"{100 * rl_winrate_decisive:.2f}%"
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    output = {

        "evaluation": "BC5_vs_RL10",

        "bc_checkpoint":
            str(BC_CHECKPOINT),

        "rl_checkpoint":
            str(RL_CHECKPOINT),

        "temperature":
            TEMPERATURE,

        "games":
            GAMES,

        "workers":
            WORKERS,

        "color_balance": {
            "bc_white": GAMES // 2,
            "rl_white": GAMES // 2,
        },

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

        "rl_decisive_game_winrate":
            rl_winrate_decisive,

        "timestamp":
            timestamp,
    }


    output_path = (
        PROJECT_ROOT
        / f"rl10_vs_bc5_{timestamp}.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )


    print()

    print(
        "Saved:"
    )

    print(
        output_path
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()