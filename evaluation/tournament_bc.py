from __future__ import annotations

import argparse
import json
import random
import sys

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import chess
import chess.variant
import numpy as np
import torch

from tqdm import tqdm


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions_space import ACTIONS, INDEX_TO_ACTION
from src.encoding import encode_fen
from src.models.resnet import ChessResNet


# ============================================================
# Default configuration
# ============================================================

DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
)

DEFAULT_CHECKPOINT_PATTERN = (
    "bc_epoch_*.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

DEFAULT_TEMPERATURE = 1.0

DEFAULT_GAMES_PER_MATCHUP = 10

DEFAULT_CHANNELS = 32
DEFAULT_BLOCKS = 4

DEFAULT_MAX_PLIES = 1000

DEFAULT_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:
    """
    Seed all random-number generators potentially involved
    in tournament evaluation.

    BC move sampling itself uses torch.multinomial().
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


# ============================================================
# Checkpoint ordering
# ============================================================

def checkpoint_epoch(
    path: Path,
) -> int:
    """
    Extract the numeric epoch from checkpoints named:

        bc_epoch_<N>.pt
    """

    try:
        return int(
            path.stem.split("_")[-1]
        )

    except ValueError as exc:
        raise ValueError(
            f"Cannot extract epoch number from {path.name}"
        ) from exc


# ============================================================
# Model loading
# ============================================================

def load_model(
    path: Path,
    device: torch.device,
    channels: int,
    blocks: int,
) -> ChessResNet:

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        checkpoint.get("actions")
        != len(ACTIONS)
    ):
        raise ValueError(
            f"Action-space mismatch in {path}"
        )

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=channels,
        blocks=blocks,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# ============================================================
# Move selection
# ============================================================

def select_move(
    model: ChessResNet,
    board: chess.variant.AtomicBoard,
    device: torch.device,
    temperature: float,
) -> chess.Move:
    """
    Select one legal move from the BC policy.

    temperature == 0:
        deterministic greedy action.

    temperature > 0:
        sample from the legal-action distribution obtained
        after temperature scaling.
    """

    x = encode_fen(
        board.fen()
    ).unsqueeze(0).to(
        device
    )

    with torch.no_grad():
        logits = model(x)[0]

    # --------------------------------------------------------
    # Legal-action mask
    # --------------------------------------------------------

    legal_moves = {
        move.uci()
        for move in board.legal_moves
    }

    masked_logits = torch.full_like(
        logits,
        -float("inf"),
    )

    for index, uci in INDEX_TO_ACTION.items():

        if uci in legal_moves:
            masked_logits[index] = logits[index]

    # --------------------------------------------------------
    # Greedy evaluation
    # --------------------------------------------------------

    if temperature == 0.0:

        action_index = (
            masked_logits.argmax()
        ).item()

    # --------------------------------------------------------
    # Stochastic evaluation
    # --------------------------------------------------------

    else:

        probabilities = torch.softmax(
            masked_logits / temperature,
            dim=0,
        )

        action_index = torch.multinomial(
            probabilities,
            num_samples=1,
        ).item()

    return chess.Move.from_uci(
        INDEX_TO_ACTION[
            action_index
        ]
    )


# ============================================================
# Single game
# ============================================================

def play_game(
    white_model: ChessResNet,
    black_model: ChessResNet,
    device: torch.device,
    temperature: float,
    max_plies: int,
) -> tuple[str, int]:
    """
    Play one Atomic Chess game.

    Returns:
        result,
        number of plies played.

    If max_plies is reached before a terminal state, "*"
    is returned. Tournament scoring treats this as a draw,
    matching the historical ALBERTA evaluation protocol.
    """

    board = chess.variant.AtomicBoard()

    for ply in range(
        1,
        max_plies + 1,
    ):

        model = (
            white_model
            if board.turn == chess.WHITE
            else black_model
        )

        move = select_move(
            model=model,
            board=board,
            device=device,
            temperature=temperature,
        )

        board.push(move)

        if board.is_game_over():

            return (
                board.result(),
                ply,
            )

    return (
        "*",
        max_plies,
    )


# ============================================================
# Match
# ============================================================

def play_match(
    model_a: ChessResNet,
    model_b: ChessResNet,
    device: torch.device,
    temperature: float,
    games_per_matchup: int,
    max_plies: int,
) -> list[dict]:
    """
    Play one balanced matchup.

    Model A has White on even-numbered games and model B has
    White on odd-numbered games.
    """

    games = []

    for game_index in tqdm(
        range(games_per_matchup),
        leave=False,
        desc="Games",
    ):

        if game_index % 2 == 0:

            white = "a"
            black = "b"

            result, plies = play_game(
                white_model=model_a,
                black_model=model_b,
                device=device,
                temperature=temperature,
                max_plies=max_plies,
            )

        else:

            white = "b"
            black = "a"

            result, plies = play_game(
                white_model=model_b,
                black_model=model_a,
                device=device,
                temperature=temperature,
                max_plies=max_plies,
            )

        games.append(
            {
                "game":
                    game_index,

                "white":
                    white,

                "black":
                    black,

                "result":
                    result,

                "plies":
                    plies,
            }
        )

    return games


# ============================================================
# Process matchup
# ============================================================

def summarize_matchup(
    games: list[dict],
) -> dict:
    """
    Convert game results into model-A/model-B statistics.
    """

    a_wins = 0
    b_wins = 0
    draws = 0
    unfinished = 0

    for game in games:

        result = game["result"]
        white = game["white"]
        black = game["black"]

        if result == "1-0":

            if white == "a":
                a_wins += 1
            else:
                b_wins += 1

        elif result == "0-1":

            if black == "a":
                a_wins += 1
            else:
                b_wins += 1

        else:

            draws += 1

            if result == "*":
                unfinished += 1

    total = (
        a_wins
        + b_wins
        + draws
    )

    a_points = (
        a_wins
        + 0.5 * draws
    )

    b_points = (
        b_wins
        + 0.5 * draws
    )

    return {
        "a_wins":
            a_wins,

        "b_wins":
            b_wins,

        "draws":
            draws,

        "unfinished":
            unfinished,

        "a_points":
            a_points,

        "b_points":
            b_points,

        "a_score":
            a_points / total,

        "b_score":
            b_points / total,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run a round-robin tournament between ALBERTA "
            "behavioral-cloning checkpoints."
        )
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Directory containing BC checkpoints "
            f"(default: {DEFAULT_CHECKPOINT_DIR})."
        ),
    )

    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default=DEFAULT_CHECKPOINT_PATTERN,
        help=(
            "Checkpoint glob pattern "
            f"(default: {DEFAULT_CHECKPOINT_PATTERN})."
        ),
    )

    # --------------------------------------------------------
    # Tournament
    # --------------------------------------------------------

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Sampling temperature. Use 0 for greedy "
            f"evaluation (default: {DEFAULT_TEMPERATURE})."
        ),
    )

    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=DEFAULT_GAMES_PER_MATCHUP,
        help=(
            "Number of games per checkpoint pair "
            f"(default: {DEFAULT_GAMES_PER_MATCHUP})."
        ),
    )

    parser.add_argument(
        "--max-plies",
        type=int,
        default=DEFAULT_MAX_PLIES,
        help=(
            "Maximum number of plies before a game is "
            "recorded as unfinished."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Master random seed "
            f"(default: {DEFAULT_SEED})."
        ),
    )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=DEFAULT_BLOCKS,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON file. When omitted, a timestamped "
            "file is written under evaluation/results."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Configuration validation
    # ========================================================

    if not args.checkpoint_dir.exists():

        raise FileNotFoundError(
            "Checkpoint directory does not exist: "
            f"{args.checkpoint_dir}"
        )

    if args.temperature < 0.0:

        raise ValueError(
            "--temperature cannot be negative."
        )

    if args.games_per_matchup <= 0:

        raise ValueError(
            "--games-per-matchup must be greater than zero."
        )

    if args.games_per_matchup % 2 != 0:

        raise ValueError(
            "--games-per-matchup must be even so that "
            "colors are exactly balanced."
        )

    if args.max_plies <= 0:

        raise ValueError(
            "--max-plies must be greater than zero."
        )

    # ========================================================
    # Reproducibility
    # ========================================================

    seed_everything(
        args.seed
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("ALBERTA - BC ROUND-ROBIN TOURNAMENT")
    print("=" * 70)

    print(
        f"Device             : {device}"
    )

    print(
        f"Temperature        : {args.temperature}"
    )

    print(
        f"Games / matchup    : {args.games_per_matchup}"
    )

    print(
        f"Maximum plies      : {args.max_plies}"
    )

    print(
        f"Seed               : {args.seed}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU                : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Checkpoints
    # ========================================================

    checkpoints = sorted(
        args.checkpoint_dir.glob(
            args.checkpoint_pattern
        ),
        key=checkpoint_epoch,
    )

    if len(checkpoints) < 2:

        raise RuntimeError(
            "At least two BC checkpoints are required. "
            f"Found {len(checkpoints)} matching "
            f"{args.checkpoint_pattern!r} in "
            f"{args.checkpoint_dir}."
        )

    print()
    print(
        f"Found {len(checkpoints)} checkpoints:"
    )

    for checkpoint in checkpoints:
        print(
            f"  - {checkpoint.name}"
        )

    # ========================================================
    # Models
    # ========================================================

    models: dict[
        str,
        ChessResNet,
    ] = {}

    for path in checkpoints:

        name = path.stem

        print(
            f"Loading {name}..."
        )

        models[name] = load_model(
            path=path,
            device=device,
            channels=args.channels,
            blocks=args.blocks,
        )

    # ========================================================
    # Scores
    # ========================================================

    scores = {
        name: {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0.0,
        }
        for name in models
    }

    matchup_results = []

    total_games = 0

    # ========================================================
    # Round robin
    # ========================================================

    print()
    print("=" * 70)
    print("TOURNAMENT")
    print("=" * 70)

    for matchup_index, (
        name_a,
        name_b,
    ) in enumerate(
        combinations(
            models.keys(),
            2,
        )
    ):

        model_a = models[name_a]
        model_b = models[name_b]

        # ----------------------------------------------------
        # Each matchup receives a deterministic seed derived
        # from the global master seed.
        #
        # This prevents changes in an earlier matchup from
        # shifting the stochastic trajectories of every later
        # matchup.
        # ----------------------------------------------------

        matchup_seed = (
            args.seed
            + matchup_index
        )

        seed_everything(
            matchup_seed
        )

        print()
        print(
            f"{name_a} vs {name_b} "
            f"(seed={matchup_seed})"
        )

        games = play_match(
            model_a=model_a,
            model_b=model_b,
            device=device,
            temperature=args.temperature,
            games_per_matchup=args.games_per_matchup,
            max_plies=args.max_plies,
        )

        summary = summarize_matchup(
            games
        )

        # ----------------------------------------------------
        # Global tournament score
        # ----------------------------------------------------

        scores[name_a]["wins"] += (
            summary["a_wins"]
        )

        scores[name_a]["losses"] += (
            summary["b_wins"]
        )

        scores[name_a]["draws"] += (
            summary["draws"]
        )

        scores[name_a]["points"] += (
            summary["a_points"]
        )

        scores[name_b]["wins"] += (
            summary["b_wins"]
        )

        scores[name_b]["losses"] += (
            summary["a_wins"]
        )

        scores[name_b]["draws"] += (
            summary["draws"]
        )

        scores[name_b]["points"] += (
            summary["b_points"]
        )

        total_games += (
            len(games)
        )

        matchup_results.append(
            {
                "model_a":
                    name_a,

                "model_b":
                    name_b,

                "seed":
                    matchup_seed,

                **summary,

                "games":
                    games,
            }
        )

        print(
            f"  {name_a}: "
            f"{summary['a_wins']}W "
            f"{summary['draws']}D "
            f"{summary['b_wins']}L "
            f"({100 * summary['a_score']:.1f}%)"
        )

        print(
            f"  {name_b}: "
            f"{summary['b_wins']}W "
            f"{summary['draws']}D "
            f"{summary['a_wins']}L "
            f"({100 * summary['b_score']:.1f}%)"
        )

        if summary["unfinished"] > 0:

            print(
                f"  Unfinished: "
                f"{summary['unfinished']}"
            )

    # ========================================================
    # Ranking
    # ========================================================

    ranking = sorted(
        scores.items(),
        key=lambda item: (
            -item[1]["points"],
            -item[1]["wins"],
            item[1]["losses"],
            item[0],
        ),
    )

    # ========================================================
    # Print final ranking
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RANKING")
    print("=" * 70)

    print()

    print(
        f"{'Rank':<6}"
        f"{'Model':<22}"
        f"{'W':>6}"
        f"{'D':>6}"
        f"{'L':>6}"
        f"{'Points':>10}"
        f"{'Score':>10}"
    )

    print("-" * 70)

    ranking_output = []

    for rank, (
        name,
        data,
    ) in enumerate(
        ranking,
        start=1,
    ):

        games_played = (
            data["wins"]
            + data["draws"]
            + data["losses"]
        )

        score = (
            data["points"]
            / games_played
        )

        print(
            f"{rank:<6}"
            f"{name:<22}"
            f"{data['wins']:>6}"
            f"{data['draws']:>6}"
            f"{data['losses']:>6}"
            f"{data['points']:>10.1f}"
            f"{100 * score:>9.1f}%"
        )

        ranking_output.append(
            {
                "rank":
                    rank,

                "model":
                    name,

                **data,

                "games":
                    games_played,

                "score":
                    score,
            }
        )

    number_of_matchups = (
        len(checkpoints)
        * (
            len(checkpoints) - 1
        )
        // 2
    )

    print()
    print(
        f"Total games       : {total_games}"
    )

    print(
        f"Games / matchup   : "
        f"{args.games_per_matchup}"
    )

    print(
        f"Matchups          : "
        f"{number_of_matchups}"
    )

    print("=" * 70)

    # ========================================================
    # Save
    # ========================================================

    if args.output is None:

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d_%H%M%S"
        )

        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"bc_tournament_{timestamp}.json"
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

        "temperature":
            args.temperature,

        "games_per_matchup":
            args.games_per_matchup,

        "max_plies":
            args.max_plies,

        "channels":
            args.channels,

        "blocks":
            args.blocks,

        "checkpoint_dir":
            str(args.checkpoint_dir),

        "checkpoint_pattern":
            args.checkpoint_pattern,

        "checkpoints":
            [
                str(path)
                for path in checkpoints
            ],

        "total_games":
            total_games,

        "matchups":
            matchup_results,

        "ranking":
            ranking_output,
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
    main()