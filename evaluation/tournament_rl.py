from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions_space import ACTIONS
from src.agents.actor_critic_agent import ActorCriticAgent
from src.models.actor_critic import ActorCritic
from src.models.resnet import ChessResNet
from src.selfplay.game import SelfPlayGame


# ============================================================
# Default configuration
# ============================================================

DEFAULT_RL_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
)

DEFAULT_ORACLE_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "oracle_epoch_rndm"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

DEFAULT_GAMES_PER_MATCH = 100
DEFAULT_TEMPERATURE = 2.0

DEFAULT_DEVICE = "cpu"

DEFAULT_CHANNELS = 32
DEFAULT_BLOCKS = 4

DEFAULT_SEED = 42

DEFAULT_INITIAL_ELO = 1500.0
DEFAULT_ELO_SCALE = 400.0

DEFAULT_ELO_ITERATIONS = 5000
DEFAULT_ELO_LEARNING_RATE = 1.0


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_game_seed(
    master_seed: int,
    match_index: int,
    game_index: int,
) -> int:
    """
    Derive one deterministic RNG seed per tournament game.

    The master experimental seed remains 42 by default.
    """

    modulus = 2_147_483_647

    seed = (
        master_seed
        + match_index * 1_000_003
        + game_index * 7_919
        + 104_729
    ) % modulus

    if seed == 0:
        seed = 1

    return seed


# ============================================================
# Device
# ============================================================

def resolve_device(
    requested: str,
) -> torch.device:

    if requested == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return device


# ============================================================
# Model
# ============================================================

def build_model(
    device: torch.device,
    channels: int,
    blocks: int,
) -> ActorCritic:

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=channels,
        blocks=blocks,
    )

    model = ActorCritic(
        bc_model
    ).to(
        device
    )

    return model


def load_model(
    path: Path,
    device: torch.device,
    channels: int,
    blocks: int,
) -> ActorCritic:

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    model = build_model(
        device=device,
        channels=channels,
        blocks=blocks,
    )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    checkpoint_actions = checkpoint.get(
        "actions"
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(ACTIONS)
    ):

        raise ValueError(
            f"Action-space mismatch in {path}: "
            f"{checkpoint_actions} != {len(ACTIONS)}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# ============================================================
# Checkpoint discovery
# ============================================================

def extract_epoch(
    path: Path,
) -> int:

    try:

        return int(
            path.stem.split("_")[-1]
        )

    except ValueError as exc:

        raise ValueError(
            f"Cannot extract epoch from {path.name}"
        ) from exc


def get_checkpoints(
    rl_dir: Path,
    oracle_dir: Path,
    include_oracle: bool,
) -> list[dict]:

    checkpoints = []

    # --------------------------------------------------------
    # RL
    # --------------------------------------------------------

    if rl_dir.exists():

        paths = sorted(
            rl_dir.glob(
                "rl_epoch_*.pt"
            ),
            key=extract_epoch,
        )

        for path in paths:

            epoch = extract_epoch(
                path
            )

            checkpoints.append(
                {
                    "name":
                        f"RL{epoch}",

                    "path":
                        path,

                    "type":
                        "RL",

                    "epoch":
                        epoch,
                }
            )

    # --------------------------------------------------------
    # Oracle / AL
    # --------------------------------------------------------

    if (
        include_oracle
        and oracle_dir.exists()
    ):

        paths = sorted(
            oracle_dir.glob(
                "al_epoch_*.pt"
            ),
            key=extract_epoch,
        )

        for path in paths:

            epoch = extract_epoch(
                path
            )

            checkpoints.append(
                {
                    "name":
                        f"ORACLE{epoch}",

                    "path":
                        path,

                    "type":
                        "ORACLE",

                    "epoch":
                        epoch,
                }
            )

    # --------------------------------------------------------
    # Deterministic global ordering
    # --------------------------------------------------------

    checkpoints.sort(
        key=lambda checkpoint: (
            checkpoint[
                "epoch"
            ],
            0
            if checkpoint[
                "type"
            ] == "RL"
            else 1,
        )
    )

    return checkpoints


# ============================================================
# Single game
# ============================================================

def play_game(
    white_model: ActorCritic,
    black_model: ActorCritic,
    device: torch.device,
    temperature: float,
) -> str:

    white_agent = ActorCriticAgent(
        white_model,
        device=device,
        deterministic=False,
        temperature=temperature,
    )

    black_agent = ActorCriticAgent(
        black_model,
        device=device,
        deterministic=False,
        temperature=temperature,
    )

    game = SelfPlayGame(
        white_agent,
        black_agent,
    )

    _, result = game.play()

    return result


# ============================================================
# Global Elo fit
# ============================================================

def expected_score(
    rating_a: float,
    rating_b: float,
    elo_scale: float,
) -> float:

    return 1.0 / (
        1.0
        + 10.0 ** (
            (
                rating_b
                - rating_a
            )
            / elo_scale
        )
    )


def fit_global_elo(
    agents: list[str],
    match_results: list[dict],
    initial_elo: float,
    elo_scale: float,
    iterations: int,
    learning_rate: float,
) -> dict[str, float]:
    """
    Fit one set of Elo ratings jointly from the complete
    round-robin results.

    Unlike sequential Elo updates, the result does not depend
    on tournament match order.

    Draws count as 0.5 points.

    The final ratings are centered so that their mean equals
    initial_elo.
    """

    ratings = {
        agent:
            initial_elo
        for agent in agents
    }

    # Precompute number of games per agent.
    games_per_agent = {
        agent: 0
        for agent in agents
    }

    for match in match_results:

        games = (
            match[
                "a_wins"
            ]
            + match[
                "b_wins"
            ]
            + match[
                "draws"
            ]
        )

        games_per_agent[
            match[
                "agent_a"
            ]
        ] += games

        games_per_agent[
            match[
                "agent_b"
            ]
        ] += games

    # --------------------------------------------------------
    # Iterative global fit
    # --------------------------------------------------------

    for _ in range(
        iterations
    ):

        gradients = {
            agent: 0.0
            for agent in agents
        }

        for match in match_results:

            agent_a = match[
                "agent_a"
            ]

            agent_b = match[
                "agent_b"
            ]

            a_wins = match[
                "a_wins"
            ]

            b_wins = match[
                "b_wins"
            ]

            draws = match[
                "draws"
            ]

            total = (
                a_wins
                + b_wins
                + draws
            )

            if total == 0:
                continue

            observed_a = (
                a_wins
                + 0.5 * draws
            ) / total

            predicted_a = expected_score(
                ratings[
                    agent_a
                ],
                ratings[
                    agent_b
                ],
                elo_scale,
            )

            error = (
                observed_a
                - predicted_a
            )

            gradients[
                agent_a
            ] += (
                error
                * total
            )

            gradients[
                agent_b
            ] -= (
                error
                * total
            )

        max_change = 0.0

        for agent in agents:

            games = (
                games_per_agent[
                    agent
                ]
            )

            if games == 0:
                continue

            change = (
                learning_rate
                * gradients[
                    agent
                ]
                / games
            )

            ratings[
                agent
            ] += change

            max_change = max(
                max_change,
                abs(
                    change
                ),
            )

        if max_change < 1e-7:
            break

    # --------------------------------------------------------
    # Elo is identifiable only up to an additive constant.
    # Center ratings around the requested baseline.
    # --------------------------------------------------------

    mean_rating = (
        sum(
            ratings.values()
        )
        / len(
            ratings
        )
    )

    shift = (
        initial_elo
        - mean_rating
    )

    for agent in ratings:

        ratings[
            agent
        ] += shift

    return ratings


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run a round-robin tournament between ALBERTA "
            "ActorCritic checkpoints."
        )
    )

    parser.add_argument(
        "--rl-dir",
        type=Path,
        default=DEFAULT_RL_CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=DEFAULT_ORACLE_CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--include-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include Oracle/AL checkpoints in addition "
            "to RL checkpoints."
        ),
    )

    parser.add_argument(
        "--games-per-match",
        type=int,
        default=DEFAULT_GAMES_PER_MATCH,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

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

    parser.add_argument(
        "--initial-elo",
        type=float,
        default=DEFAULT_INITIAL_ELO,
    )

    parser.add_argument(
        "--elo-scale",
        type=float,
        default=DEFAULT_ELO_SCALE,
    )

    parser.add_argument(
        "--elo-iterations",
        type=int,
        default=DEFAULT_ELO_ITERATIONS,
    )

    parser.add_argument(
        "--elo-learning-rate",
        type=float,
        default=DEFAULT_ELO_LEARNING_RATE,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-json",
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

    if args.games_per_match <= 0:

        raise ValueError(
            "--games-per-match must be greater than zero."
        )

    if args.games_per_match % 2 != 0:

        raise ValueError(
            "--games-per-match must be even so colors "
            "are exactly balanced."
        )

    if args.temperature < 0.0:

        raise ValueError(
            "--temperature cannot be negative."
        )

    if args.seed < 0:

        raise ValueError(
            "--seed must be non-negative."
        )

    if args.elo_scale <= 0.0:

        raise ValueError(
            "--elo-scale must be greater than zero."
        )

    if args.elo_iterations <= 0:

        raise ValueError(
            "--elo-iterations must be greater than zero."
        )

    if args.elo_learning_rate <= 0.0:

        raise ValueError(
            "--elo-learning-rate must be greater than zero."
        )

    device = resolve_device(
        args.device
    )

    # ========================================================
    # Reproducibility
    # ========================================================

    seed_everything(
        args.seed
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 80)
    print("ALBERTA - ACTOR-CRITIC ROUND ROBIN")
    print("=" * 80)

    print(
        f"RL directory       : {args.rl_dir}"
    )

    print(
        f"Oracle directory   : {args.oracle_dir}"
    )

    print(
        f"Include Oracle     : {args.include_oracle}"
    )

    print(
        f"Games / match      : {args.games_per_match}"
    )

    print(
        f"Temperature        : {args.temperature}"
    )

    print(
        f"Device             : {device}"
    )

    print(
        f"Master seed        : {args.seed}"
    )

    # ========================================================
    # Discover checkpoints
    # ========================================================

    checkpoints = get_checkpoints(
        rl_dir=args.rl_dir,
        oracle_dir=args.oracle_dir,
        include_oracle=args.include_oracle,
    )

    if len(checkpoints) < 2:

        raise RuntimeError(
            "At least two ActorCritic checkpoints are required."
        )

    print()
    print("=" * 80)
    print("CHECKPOINTS")
    print("=" * 80)

    for checkpoint in checkpoints:

        print(
            f"{checkpoint['name']:<12} "
            f"{checkpoint['path']}"
        )

    num_agents = len(
        checkpoints
    )

    total_matches = (
        num_agents
        * (
            num_agents
            - 1
        )
        // 2
    )

    total_games = (
        total_matches
        * args.games_per_match
    )

    print()
    print(
        f"Agents          : {num_agents}"
    )

    print(
        f"Matches         : {total_matches}"
    )

    print(
        f"Games / match   : {args.games_per_match}"
    )

    print(
        f"Total games     : {total_games}"
    )

    # ========================================================
    # Load models
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    models = {}

    for checkpoint in checkpoints:

        name = checkpoint[
            "name"
        ]

        print(
            f"Loading {name}..."
        )

        models[
            name
        ] = load_model(
            path=checkpoint[
                "path"
            ],
            device=device,
            channels=args.channels,
            blocks=args.blocks,
        )

    # ========================================================
    # Statistics
    # ========================================================

    stats = {
        checkpoint[
            "name"
        ]: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "score": 0.0,
        }
        for checkpoint in checkpoints
    }

    match_results = []

    # ========================================================
    # Tournament
    # ========================================================

    print()
    print("=" * 80)
    print("TOURNAMENT")
    print("=" * 80)

    matchups = list(
        itertools.combinations(
            checkpoints,
            2,
        )
    )

    for zero_based_match_index, (
        checkpoint_a,
        checkpoint_b,
    ) in enumerate(
        matchups
    ):

        match_number = (
            zero_based_match_index
            + 1
        )

        name_a = checkpoint_a[
            "name"
        ]

        name_b = checkpoint_b[
            "name"
        ]

        model_a = models[
            name_a
        ]

        model_b = models[
            name_b
        ]

        a_wins = 0
        b_wins = 0
        draws = 0

        half = (
            args.games_per_match
            // 2
        )

        print()
        print(
            f"[{match_number}/{total_matches}] "
            f"{name_a} vs {name_b}"
        )

        # ----------------------------------------------------
        # First half: A White
        # ----------------------------------------------------

        for local_game_index in range(
            half
        ):

            game_seed = derive_game_seed(
                master_seed=args.seed,
                match_index=zero_based_match_index,
                game_index=local_game_index,
            )

            seed_everything(
                game_seed
            )

            result = play_game(
                white_model=model_a,
                black_model=model_b,
                device=device,
                temperature=args.temperature,
            )

            if result == "1-0":
                a_wins += 1

            elif result == "0-1":
                b_wins += 1

            else:
                draws += 1

        # ----------------------------------------------------
        # Second half: B White
        # ----------------------------------------------------

        for local_game_index in range(
            half,
            args.games_per_match,
        ):

            game_seed = derive_game_seed(
                master_seed=args.seed,
                match_index=zero_based_match_index,
                game_index=local_game_index,
            )

            seed_everything(
                game_seed
            )

            result = play_game(
                white_model=model_b,
                black_model=model_a,
                device=device,
                temperature=args.temperature,
            )

            if result == "1-0":
                b_wins += 1

            elif result == "0-1":
                a_wins += 1

            else:
                draws += 1

        # ====================================================
        # Match score
        # ====================================================

        a_score = (
            a_wins
            + 0.5 * draws
        )

        b_score = (
            b_wins
            + 0.5 * draws
        )

        a_percentage = (
            a_score
            / args.games_per_match
            * 100.0
        )

        b_percentage = (
            b_score
            / args.games_per_match
            * 100.0
        )

        print(
            f"    {name_a:<12} "
            f"{a_wins:>3}W / "
            f"{b_wins:>3}L / "
            f"{draws:>3}D "
            f"-> {a_percentage:5.1f}%"
        )

        print(
            f"    {name_b:<12} "
            f"{b_wins:>3}W / "
            f"{a_wins:>3}L / "
            f"{draws:>3}D "
            f"-> {b_percentage:5.1f}%"
        )

        # ====================================================
        # Global statistics
        # ====================================================

        stats[
            name_a
        ][
            "games"
        ] += args.games_per_match

        stats[
            name_a
        ][
            "wins"
        ] += a_wins

        stats[
            name_a
        ][
            "losses"
        ] += b_wins

        stats[
            name_a
        ][
            "draws"
        ] += draws

        stats[
            name_a
        ][
            "score"
        ] += a_score

        stats[
            name_b
        ][
            "games"
        ] += args.games_per_match

        stats[
            name_b
        ][
            "wins"
        ] += b_wins

        stats[
            name_b
        ][
            "losses"
        ] += a_wins

        stats[
            name_b
        ][
            "draws"
        ] += draws

        stats[
            name_b
        ][
            "score"
        ] += b_score

        match_results.append(
            {
                "agent_a":
                    name_a,

                "agent_b":
                    name_b,

                "a_wins":
                    a_wins,

                "b_wins":
                    b_wins,

                "draws":
                    draws,

                "a_score_pct":
                    a_percentage,

                "b_score_pct":
                    b_percentage,
            }
        )

    # ========================================================
    # Global Elo
    # ========================================================

    print()
    print("=" * 80)
    print("ESTIMATING GLOBAL ELO")
    print("=" * 80)

    agent_names = [
        checkpoint[
            "name"
        ]
        for checkpoint in checkpoints
    ]

    elo = fit_global_elo(
        agents=agent_names,
        match_results=match_results,
        initial_elo=args.initial_elo,
        elo_scale=args.elo_scale,
        iterations=args.elo_iterations,
        learning_rate=args.elo_learning_rate,
    )

    # ========================================================
    # Final ranking
    # ========================================================

    ranking = sorted(
        agent_names,
        key=lambda name: (
            -elo[
                name
            ],
            name,
        ),
    )

    print()
    print("=" * 80)
    print("FINAL RANKING")
    print("=" * 80)

    print(
        f"{'Rank':<6}"
        f"{'Agent':<12}"
        f"{'Elo':<10}"
        f"{'Score':<10}"
        f"{'Wins':<8}"
        f"{'Losses':<8}"
        f"{'Draws':<8}"
    )

    print("-" * 80)

    ranking_output = []

    for rank, name in enumerate(
        ranking,
        start=1,
    ):

        agent_stats = stats[
            name
        ]

        score_fraction = (
            agent_stats[
                "score"
            ]
            / agent_stats[
                "games"
            ]
        )

        print(
            f"{rank:<6}"
            f"{name:<12}"
            f"{elo[name]:>7.0f}   "
            f"{100 * score_fraction:>6.1f}%   "
            f"{agent_stats['wins']:<8}"
            f"{agent_stats['losses']:<8}"
            f"{agent_stats['draws']:<8}"
        )

        ranking_output.append(
            {
                "rank":
                    rank,

                "agent":
                    name,

                "elo":
                    elo[
                        name
                    ],

                **agent_stats,

                "score_fraction":
                    score_fraction,
            }
        )

    # ========================================================
    # Output paths
    # ========================================================

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    if args.output_csv is None:

        output_csv = (
            DEFAULT_OUTPUT_DIR
            / f"round_robin_{timestamp}.csv"
        )

    else:

        output_csv = (
            args.output_csv
        )

    if args.output_json is None:

        output_json = (
            DEFAULT_OUTPUT_DIR
            / f"round_robin_{timestamp}.json"
        )

    else:

        output_json = (
            args.output_json
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Save CSV
    # ========================================================

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "agent_a",
                "agent_b",
                "a_wins",
                "b_wins",
                "draws",
                "a_score_pct",
                "b_score_pct",
            ]
        )

        for result in match_results:

            writer.writerow(
                [
                    result[
                        "agent_a"
                    ],
                    result[
                        "agent_b"
                    ],
                    result[
                        "a_wins"
                    ],
                    result[
                        "b_wins"
                    ],
                    result[
                        "draws"
                    ],
                    f"{result['a_score_pct']:.4f}",
                    f"{result['b_score_pct']:.4f}",
                ]
            )

    # ========================================================
    # Save JSON
    # ========================================================

    output = {
        "seed":
            args.seed,

        "temperature":
            args.temperature,

        "games_per_match":
            args.games_per_match,

        "include_oracle":
            args.include_oracle,

        "device":
            str(
                device
            ),

        "checkpoints":
            [
                {
                    **checkpoint,
                    "path":
                        str(
                            checkpoint[
                                "path"
                            ]
                        ),
                }
                for checkpoint
                in checkpoints
            ],

        "total_matches":
            total_matches,

        "total_games":
            total_games,

        "match_results":
            match_results,

        "elo":
            elo,

        "ranking":
            ranking_output,

        "timestamp_utc":
            timestamp,
    }

    with open(
        output_json,
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
        f"CSV saved to:  {output_csv}"
    )

    print(
        f"JSON saved to: {output_json}"
    )

    print()
    print("=" * 80)
    print("TOURNAMENT FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()