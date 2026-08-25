from pathlib import Path
import sys
import json
import random
import math
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.variant
import numpy as np
import torch

from lichess_bot.atomic_engine.rl_bot import RLBot


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_RL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_AL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "al_epoch"
    / "al_epoch_10.pt"
)

DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "data"
    / "oracle_queue_1-10.jsonl"
)

DEFAULT_POSITIONS = 1000

DEFAULT_SEED = 42


# ============================================================
# NUMERICAL HELPERS
# ============================================================

EPS = 1e-12


def safe_normalize(probs):
    probs = np.asarray(
        probs,
        dtype=np.float64,
    )

    total = probs.sum()

    if total <= 0:
        raise ValueError(
            "Probability vector has non-positive sum."
        )

    return probs / total


def entropy(probs):

    probs = safe_normalize(probs)

    mask = probs > 0

    return -np.sum(
        probs[mask]
        * np.log(
            probs[mask]
        )
    )


def kl_divergence(p, q):

    p = safe_normalize(p)
    q = safe_normalize(q)

    return np.sum(
        p
        * (
            np.log(p + EPS)
            -
            np.log(q + EPS)
        )
    )


def l1_distance(p, q):

    p = safe_normalize(p)
    q = safe_normalize(q)

    return np.sum(
        np.abs(
            p - q
        )
    )


def percentile(values, p):

    if len(values) == 0:
        return float("nan")

    return float(
        np.percentile(
            values,
            p,
        )
    )


def mean(values):

    if len(values) == 0:
        return float("nan")

    return float(
        np.mean(values)
    )


def median(values):

    if len(values) == 0:
        return float("nan")

    return float(
        np.median(values)
    )


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

def load_annotations(path):

    samples = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)

            if (
                sample.get("status")
                != "answered"
            ):
                continue

            required = [
                "query_id",
                "fen",
                "oracle_move",
                "oracle_confidence",
                "oracle_situation",
            ]

            if not all(
                key in sample
                for key in required
            ):
                continue

            samples.append(
                sample
            )

    return samples


# ============================================================
# POLICY EXTRACTION
# ============================================================

def evaluate_position(
    bot,
    board,
):
    """
    Ask RLBot for the complete legal policy.

    This assumes RLBot.evaluate_policy(board)
    returns something containing:

        moves
        probs

    where moves are UCI strings.
    """

    result = bot.evaluate_policy(
        board
    )

    moves = result["moves"]

    probs = result["probs"]

    if hasattr(
        probs,
        "tolist",
    ):
        probs = probs.tolist()

    probs = np.asarray(
        probs,
        dtype=np.float64,
    )

    probs = safe_normalize(
        probs
    )

    return moves, probs


# ============================================================
# POSITION COMPARISON
# ============================================================

def compare_position(
    rl_bot,
    al_bot,
    fen,
    oracle_move=None,
):

    board = chess.variant.AtomicBoard(
        fen
    )

    rl_moves, rl_probs = evaluate_position(
        rl_bot,
        board,
    )

    al_moves, al_probs = evaluate_position(
        al_bot,
        board,
    )

    # --------------------------------------------------------
    # Ensure identical action ordering
    # --------------------------------------------------------

    rl_dict = {
        move: prob
        for move, prob
        in zip(
            rl_moves,
            rl_probs,
        )
    }

    al_dict = {
        move: prob
        for move, prob
        in zip(
            al_moves,
            al_probs,
        )
    }

    common_moves = list(
        dict.fromkeys(
            list(rl_dict.keys())
            +
            list(al_dict.keys())
        )
    )

    rl_vector = np.array(
        [
            rl_dict.get(
                move,
                0.0,
            )
            for move in common_moves
        ],
        dtype=np.float64,
    )

    al_vector = np.array(
        [
            al_dict.get(
                move,
                0.0,
            )
            for move in common_moves
        ],
        dtype=np.float64,
    )

    rl_vector = safe_normalize(
        rl_vector
    )

    al_vector = safe_normalize(
        al_vector
    )

    # --------------------------------------------------------
    # Policy metrics
    # --------------------------------------------------------

    kl = kl_divergence(
        rl_vector,
        al_vector,
    )

    l1 = l1_distance(
        rl_vector,
        al_vector,
    )

    h_rl = entropy(
        rl_vector
    )

    h_al = entropy(
        al_vector
    )

    # --------------------------------------------------------
    # Best move
    # --------------------------------------------------------

    rl_best_index = int(
        np.argmax(
            rl_vector
        )
    )

    al_best_index = int(
        np.argmax(
            al_vector
        )
    )

    rl_best = common_moves[
        rl_best_index
    ]

    al_best = common_moves[
        al_best_index
    ]

    best_changed = (
        rl_best
        !=
        al_best
    )

    result = {
        "kl": kl,
        "l1": l1,
        "entropy_rl": h_rl,
        "entropy_al": h_al,
        "entropy_delta":
            h_al - h_rl,
        "rl_best":
            rl_best,
        "al_best":
            al_best,
        "best_changed":
            best_changed,
    }

    # --------------------------------------------------------
    # Oracle metrics
    # --------------------------------------------------------

    if oracle_move is not None:

        if oracle_move not in rl_dict:

            raise ValueError(
                f"Oracle move {oracle_move} "
                f"is not legal in FEN:\n{fen}"
            )

        rl_oracle_prob = (
            rl_dict[
                oracle_move
            ]
        )

        al_oracle_prob = (
            al_dict[
                oracle_move
            ]
        )

        rl_rank = (
            1
            +
            sum(
                p > rl_oracle_prob
                for p in rl_vector
            )
        )

        al_rank = (
            1
            +
            sum(
                p > al_oracle_prob
                for p in al_vector
            )
        )

        result.update(
            {
                "oracle_prob_rl":
                    rl_oracle_prob,

                "oracle_prob_al":
                    al_oracle_prob,

                "oracle_prob_delta":
                    al_oracle_prob
                    -
                    rl_oracle_prob,

                "oracle_rank_rl":
                    rl_rank,

                "oracle_rank_al":
                    al_rank,

                "oracle_rank_delta":
                    al_rank
                    -
                    rl_rank,

                "oracle_best_rl":
                    rl_best
                    ==
                    oracle_move,

                "oracle_best_al":
                    al_best
                    ==
                    oracle_move,
            }
        )

    return result


# ============================================================
# STATISTICS
# ============================================================

def print_distribution_stats(
    title,
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    print()
    print(title)
    print("-" * 70)

    print(
        f"mean     : {np.mean(values):.6f}"
    )

    print(
        f"median   : {np.median(values):.6f}"
    )

    print(
        f"std      : {np.std(values):.6f}"
    )

    print(
        f"p05      : {np.percentile(values, 5):.6f}"
    )

    print(
        f"p95      : {np.percentile(values, 95):.6f}"
    )

    print(
        f"p99      : {np.percentile(values, 99):.6f}"
    )

    print(
        f"min      : {np.min(values):.6f}"
    )

    print(
        f"max      : {np.max(values):.6f}"
    )


def print_random_summary(
    results,
):

    print()
    print("=" * 70)
    print("RANDOM POSITIONS")
    print("=" * 70)

    print(
        f"Positions : {len(results):,}"
    )

    print_distribution_stats(
        "KL divergence RL || RLAL",
        [
            r["kl"]
            for r in results
        ],
    )

    print_distribution_stats(
        "L1 policy distance",
        [
            r["l1"]
            for r in results
        ],
    )

    print_distribution_stats(
        "Entropy delta (RLAL - RL)",
        [
            r["entropy_delta"]
            for r in results
        ],
    )

    changed = sum(
        r["best_changed"]
        for r in results
    )

    print()
    print(
        f"Best move changed : "
        f"{changed / len(results) * 100:.2f}%"
    )


# ============================================================
# ANNOTATED SUMMARY
# ============================================================

def print_annotated_summary(
    results,
    annotations,
):

    print()
    print("=" * 70)
    print("ANNOTATED POSITIONS")
    print("=" * 70)

    print(
        f"Positions : {len(results):,}"
    )

    # --------------------------------------------------------
    # Policy change
    # --------------------------------------------------------

    print_distribution_stats(
        "KL divergence RL || RLAL",
        [
            r["kl"]
            for r in results
        ],
    )

    print_distribution_stats(
        "L1 policy distance",
        [
            r["l1"]
            for r in results
        ],
    )

    # --------------------------------------------------------
    # Oracle probability
    # --------------------------------------------------------

    print_distribution_stats(
        "Oracle probability - RL10",
        [
            r["oracle_prob_rl"]
            for r in results
        ],
    )

    print_distribution_stats(
        "Oracle probability - RLAL10",
        [
            r["oracle_prob_al"]
            for r in results
        ],
    )

    print_distribution_stats(
        "Oracle probability delta",
        [
            r["oracle_prob_delta"]
            for r in results
        ],
    )

    # --------------------------------------------------------
    # Oracle rank
    # --------------------------------------------------------

    print_distribution_stats(
        "Oracle rank - RL10",
        [
            r["oracle_rank_rl"]
            for r in results
        ],
    )

    print_distribution_stats(
        "Oracle rank - RLAL10",
        [
            r["oracle_rank_al"]
            for r in results
        ],
    )

    # --------------------------------------------------------
    # Best move
    # --------------------------------------------------------

    rl_best = sum(
        r["oracle_best_rl"]
        for r in results
    )

    al_best = sum(
        r["oracle_best_al"]
        for r in results
    )

    print()
    print(
        f"Oracle is best move"
    )

    print(
        f"  RL10   : "
        f"{rl_best}/{len(results)} "
        f"({100 * rl_best / len(results):.2f}%)"
    )

    print(
        f"  RLAL10 : "
        f"{al_best}/{len(results)} "
        f"({100 * al_best / len(results):.2f}%)"
    )

    # --------------------------------------------------------
    # Improvement / degradation
    # --------------------------------------------------------

    prob_improved = sum(
        r["oracle_prob_delta"] > 0
        for r in results
    )

    prob_worsened = sum(
        r["oracle_prob_delta"] < 0
        for r in results
    )

    prob_same = len(results) - (
        prob_improved
        +
        prob_worsened
    )

    rank_improved = sum(
        r["oracle_rank_delta"] < 0
        for r in results
    )

    rank_worsened = sum(
        r["oracle_rank_delta"] > 0
        for r in results
    )

    rank_same = len(results) - (
        rank_improved
        +
        rank_worsened
    )

    print()
    print(
        "Oracle probability:"
    )

    print(
        f"  improved : "
        f"{prob_improved} "
        f"({100 * prob_improved / len(results):.2f}%)"
    )

    print(
        f"  worsened : "
        f"{prob_worsened} "
        f"({100 * prob_worsened / len(results):.2f}%)"
    )

    print(
        f"  unchanged: "
        f"{prob_same} "
        f"({100 * prob_same / len(results):.2f}%)"
    )

    print()
    print(
        "Oracle rank:"
    )

    print(
        f"  improved : "
        f"{rank_improved} "
        f"({100 * rank_improved / len(results):.2f}%)"
    )

    print(
        f"  worsened : "
        f"{rank_worsened} "
        f"({100 * rank_worsened / len(results):.2f}%)"
    )

    print(
        f"  unchanged: "
        f"{rank_same} "
        f"({100 * rank_same / len(results):.2f}%)"
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    print_distribution_stats(
        "Entropy delta (RLAL - RL)",
        [
            r["entropy_delta"]
            for r in results
        ],
    )

    # --------------------------------------------------------
    # Breakdown by annotation
    # --------------------------------------------------------

    print()
    print(
        "BREAKDOWN BY CONFIDENCE"
    )
    print("-" * 70)

    grouped = {}

    for result, annotation in zip(
        results,
        annotations,
    ):

        key = annotation[
            "oracle_confidence"
        ]

        grouped.setdefault(
            key,
            [],
        ).append(
            result
        )

    for key in [
        "high",
        "medium",
        "low",
    ]:

        group = grouped.get(
            key,
            [],
        )

        if not group:
            continue

        improved = sum(
            r["oracle_prob_delta"] > 0
            for r in group
        )

        print(
            f"{key:8s} : "
            f"n={len(group):3d} | "
            f"ΔP={mean([r['oracle_prob_delta'] for r in group]):+.6f} | "
            f"rank Δ={mean([r['oracle_rank_delta'] for r in group]):+.3f} | "
            f"improved={100 * improved / len(group):.1f}%"
        )

    print()
    print(
        "BREAKDOWN BY SITUATION"
    )
    print("-" * 70)

    grouped = {}

    for result, annotation in zip(
        results,
        annotations,
    ):

        key = annotation[
            "oracle_situation"
        ]

        grouped.setdefault(
            key,
            [],
        ).append(
            result
        )

    for key in [
        "unique_move",
        "multiple_good",
        "everything_wins",
    ]:

        group = grouped.get(
            key,
            [],
        )

        if not group:
            continue

        improved = sum(
            r["oracle_prob_delta"] > 0
            for r in group
        )

        print(
            f"{key:16s} : "
            f"n={len(group):3d} | "
            f"ΔP={mean([r['oracle_prob_delta'] for r in group]):+.6f} | "
            f"rank Δ={mean([r['oracle_rank_delta'] for r in group]):+.3f} | "
            f"improved={100 * improved / len(group):.1f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        default=DEFAULT_RL_CHECKPOINT,
    )

    parser.add_argument(
        "--al-checkpoint",
        type=Path,
        default=DEFAULT_AL_CHECKPOINT,
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
    )

    parser.add_argument(
        "--positions",
        type=int,
        default=DEFAULT_POSITIONS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    print("=" * 70)
    print("ALBERTA - RL10 vs RLAL10")
    print("=" * 70)

    print(
        f"RL checkpoint  : "
        f"{args.rl_checkpoint}"
    )

    print(
        f"AL checkpoint  : "
        f"{args.al_checkpoint}"
    )

    # ========================================================
    # Load bots
    # ========================================================

    print()
    print("Loading RL10...")

    rl_bot = RLBot(
        checkpoint=args.rl_checkpoint,
        deterministic=True,
    )

    print()
    print("Loading RLAL10...")

    al_bot = RLBot(
        checkpoint=args.al_checkpoint,
        deterministic=True,
    )

    # ========================================================
    # Random positions
    # ========================================================

    print()
    print(
        "Generating random positions..."
    )

    # Use positions already available through the
    # annotation queue as a fallback-independent source.
    #
    # For a true random evaluation, generate random legal
    # Atomic positions from random playouts.

    random_positions = []

    while len(
        random_positions
    ) < args.positions:

        board = chess.variant.AtomicBoard()

        n_moves = random.randint(
            5,
            80,
        )

        for _ in range(
            n_moves
        ):

            legal = list(
                board.legal_moves
            )

            if not legal:
                break

            move = random.choice(
                legal
            )

            board.push(
                move
            )

            if board.is_game_over():
                break

        if not board.is_game_over():

            random_positions.append(
                board.fen()
            )

    print(
        f"Loaded random positions : "
        f"{len(random_positions):,}"
    )

    # ========================================================
    # Compare random positions
    # ========================================================

    random_results = []

    for i, fen in enumerate(
        random_positions
    ):

        result = compare_position(
            rl_bot,
            al_bot,
            fen,
        )

        random_results.append(
            result
        )

        if (
            (i + 1) % 100
            == 0
        ):

            print(
                f"Compared "
                f"{i + 1:,}/"
                f"{len(random_positions):,}"
            )

    print_random_summary(
        random_results
    )

    # ========================================================
    # Annotated positions
    # ========================================================

    annotations = load_annotations(
        args.queue
    )

    print()
    print("=" * 70)
    print("ANNOTATED POSITIONS")
    print("=" * 70)

    print(
        f"Answered annotations : "
        f"{len(annotations):,}"
    )

    annotated_results = []

    for i, sample in enumerate(
        annotations
    ):

        result = compare_position(
            rl_bot,
            al_bot,
            sample["fen"],
            oracle_move=sample[
                "oracle_move"
            ],
        )

        annotated_results.append(
            result
        )

        if (
            (i + 1) % 50
            == 0
        ):

            print(
                f"Compared "
                f"{i + 1:,}/"
                f"{len(annotations):,}"
            )

    if annotated_results:

        print_annotated_summary(
            annotated_results,
            annotations,
        )

    else:

        print(
            "No answered annotations."
        )

    # ========================================================
    # Global interpretation
    # ========================================================

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    random_kl = mean(
        [
            r["kl"]
            for r in random_results
        ]
    )

    random_changed = (
        100
        *
        mean(
            [
                float(
                    r["best_changed"]
                )
                for r in random_results
            ]
        )
    )

    print(
        f"Random mean KL        : "
        f"{random_kl:.6f}"
    )

    print(
        f"Random best-move Δ    : "
        f"{random_changed:.2f}%"
    )

    if annotated_results:

        oracle_delta = mean(
            [
                r["oracle_prob_delta"]
                for r in annotated_results
            ]
        )

        oracle_rank_delta = mean(
            [
                r["oracle_rank_delta"]
                for r in annotated_results
            ]
        )

        print(
            f"Annotated mean ΔP    : "
            f"{oracle_delta:+.6f}"
        )

        print(
            f"Annotated mean rank Δ: "
            f"{oracle_rank_delta:+.3f}"
        )

    print()
    print(
        "Done."
    )


if __name__ == "__main__":
    main()