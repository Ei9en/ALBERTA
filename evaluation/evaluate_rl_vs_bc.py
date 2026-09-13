from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lichess_bot.atomic_engine.bc_bot_stochastic import BCBotStochastic
from lichess_bot.atomic_engine.rl_bot import RLBot
from src.selfplay.game import SelfPlayGame


# ============================================================
# Default configuration
# ============================================================

DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_epoch_7.pt"
)

DEFAULT_RL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

DEFAULT_TEMPERATURE = 2.0
DEFAULT_GAMES = 500
DEFAULT_WORKERS = 3

DEFAULT_SEED = 42


# ============================================================
# Worker state
# ============================================================

_WORKER_BC = None
_WORKER_RL = None


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:
    """
    Seed all RNGs potentially involved in stochastic move
    selection.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_game_seed(
    master_seed: int,
    game_index: int,
) -> int:
    """
    Deterministically derive one RNG seed per evaluation game.

    This makes each game independent of multiprocessing worker
    scheduling.
    """

    modulus = 2_147_483_647

    seed = (
        master_seed
        + game_index * 7_919
        + 104_729
    ) % modulus

    if seed == 0:
        seed = 1

    return seed


# ============================================================
# Worker initialization
# ============================================================

def init_worker(
    bc_checkpoint: Path,
    rl_checkpoint: Path,
    temperature: float,
) -> None:

    global _WORKER_BC
    global _WORKER_RL

    torch.set_num_threads(1)

    _WORKER_BC = BCBotStochastic(
        checkpoint=bc_checkpoint,
        temperature=temperature,
    )

    _WORKER_RL = RLBot(
        checkpoint=rl_checkpoint,
        temperature=temperature,
    )


# ============================================================
# Single game
# ============================================================

def play_one_game(
    task: tuple[int, bool, int],
) -> tuple[int, str]:
    """
    Play one game.

    Parameters
    ----------
    task
        (
            game_index,
            bc_is_white,
            game_seed,
        )

    Returns
    -------
    tuple
        (
            game_index,
            "bc" | "rl" | "draw",
        )
    """

    global _WORKER_BC
    global _WORKER_RL

    (
        game_index,
        bc_is_white,
        game_seed,
    ) = task

    # --------------------------------------------------------
    # Per-game deterministic RNG state
    # --------------------------------------------------------

    seed_everything(
        game_seed
    )

    # --------------------------------------------------------
    # Color assignment
    # --------------------------------------------------------

    if bc_is_white:

        game = SelfPlayGame(
            _WORKER_BC,
            _WORKER_RL,
        )

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

        winner = (
            "bc"
            if bc_is_white
            else "rl"
        )

    elif result == "0-1":

        winner = (
            "rl"
            if bc_is_white
            else "bc"
        )

    else:

        winner = "draw"

    return (
        game_index,
        winner,
    )


# ============================================================
# Evaluation
# ============================================================

def play_match(
    bc_checkpoint: Path,
    rl_checkpoint: Path,
    games: int,
    workers: int,
    temperature: float,
    seed: int,
) -> tuple[
    int,
    int,
    int,
    list[dict],
]:
    """
    Run a color-balanced stochastic BC-vs-RL evaluation.
    """

    half = (
        games
        // 2
    )

    # --------------------------------------------------------
    # Deterministic game definitions
    #
    # First half:
    #     BC White
    #
    # Second half:
    #     RL White
    # --------------------------------------------------------

    tasks = []

    for game_index in range(
        games
    ):

        bc_is_white = (
            game_index
            < half
        )

        game_seed = derive_game_seed(
            master_seed=seed,
            game_index=game_index,
        )

        tasks.append(
            (
                game_index,
                bc_is_white,
                game_seed,
            )
        )

    print()
    print("Evaluation configuration:")

    print(
        f"Games        : {games}"
    )

    print(
        f"Workers      : {workers}"
    )

    print(
        f"Temperature  : {temperature}"
    )

    print(
        f"Master seed  : {seed}"
    )

    print(
        f"BC White     : {half}"
    )

    print(
        f"RL White     : {half}"
    )

    # --------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------

    ctx = mp.get_context(
        "spawn"
    )

    indexed_results = []

    print()
    print(
        "Loading workers..."
    )

    with ctx.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(
            bc_checkpoint,
            rl_checkpoint,
            temperature,
        ),
    ) as pool:

        for result in tqdm(
            pool.imap_unordered(
                play_one_game,
                tasks,
                chunksize=1,
            ),
            total=games,
            desc="Evaluation",
        ):

            indexed_results.append(
                result
            )

    # --------------------------------------------------------
    # Restore deterministic game order.
    # --------------------------------------------------------

    indexed_results.sort(
        key=lambda item: item[0]
    )

    winners = [
        winner
        for _, winner
        in indexed_results
    ]

    bc_wins = winners.count(
        "bc"
    )

    rl_wins = winners.count(
        "rl"
    )

    draws = winners.count(
        "draw"
    )

    game_details = []

    for game_index, winner in indexed_results:

        bc_is_white = (
            game_index
            < half
        )

        game_details.append(
            {
                "game":
                    game_index,

                "seed":
                    derive_game_seed(
                        master_seed=seed,
                        game_index=game_index,
                    ),

                "white":
                    (
                        "BC"
                        if bc_is_white
                        else "RL"
                    ),

                "black":
                    (
                        "RL"
                        if bc_is_white
                        else "BC"
                    ),

                "winner":
                    winner,
            }
        )

    return (
        bc_wins,
        rl_wins,
        draws,
        game_details,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an ALBERTA behavioral-cloning checkpoint "
            "against an ActorCritic RL checkpoint."
        )
    )

    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=DEFAULT_BC_CHECKPOINT,
    )

    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        default=DEFAULT_RL_CHECKPOINT,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if not args.bc_checkpoint.exists():

        raise FileNotFoundError(
            "BC checkpoint not found: "
            f"{args.bc_checkpoint}"
        )

    if not args.rl_checkpoint.exists():

        raise FileNotFoundError(
            "RL checkpoint not found: "
            f"{args.rl_checkpoint}"
        )

    if args.games <= 0:

        raise ValueError(
            "--games must be greater than zero."
        )

    if args.games % 2 != 0:

        raise ValueError(
            "--games must be even so that colors "
            "are exactly balanced."
        )

    if args.temperature < 0.0:

        raise ValueError(
            "--temperature cannot be negative."
        )

    if args.workers <= 0:

        raise ValueError(
            "--workers must be greater than zero."
        )

    if args.seed < 0:

        raise ValueError(
            "--seed must be non-negative."
        )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 70)
    print("ALBERTA - BC VS RL EVALUATION")
    print("=" * 70)

    print()
    print(
        f"BC checkpoint : {args.bc_checkpoint}"
    )

    print(
        f"RL checkpoint : {args.rl_checkpoint}"
    )

    print(
        f"Temperature   : {args.temperature}"
    )

    print(
        f"Games         : {args.games}"
    )

    print(
        f"Workers       : {args.workers}"
    )

    print(
        f"Seed          : {args.seed}"
    )

    # ========================================================
    # Evaluation
    # ========================================================

    (
        bc_wins,
        rl_wins,
        draws,
        game_details,
    ) = play_match(
        bc_checkpoint=args.bc_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        games=args.games,
        workers=args.workers,
        temperature=args.temperature,
        seed=args.seed,
    )

    # ========================================================
    # Statistics
    # ========================================================

    bc_score = (
        bc_wins
        + 0.5 * draws
    ) / args.games

    rl_score = (
        rl_wins
        + 0.5 * draws
    ) / args.games

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

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print()
    print(
        f"BC wins : {bc_wins}"
    )

    print(
        f"RL wins : {rl_wins}"
    )

    print(
        f"Draws   : {draws}"
    )

    print()
    print(
        f"BC score : "
        f"{bc_score:.2%}"
    )

    print(
        f"RL score : "
        f"{rl_score:.2%}"
    )

    print()
    print(
        f"RL decisive-game win rate: "
        f"{rl_winrate_decisive:.2%}"
    )

    # ========================================================
    # Save
    # ========================================================

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    if args.output is None:

        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"bc_vs_rl_{timestamp}.json"
        )

    else:

        output_path = (
            args.output
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "seed":
            args.seed,

        "bc_checkpoint":
            str(
                args.bc_checkpoint
            ),

        "rl_checkpoint":
            str(
                args.rl_checkpoint
            ),

        "temperature":
            args.temperature,

        "games":
            args.games,

        "workers":
            args.workers,

        "color_balance":
            {
                "bc_white":
                    args.games // 2,

                "rl_white":
                    args.games // 2,
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

        "games_detail":
            game_details,

        "timestamp_utc":
            timestamp,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        f"Results saved to: {output_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()