from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# Imports from project
# ============================================================

from oracle_queue import OracleQueue

from active_learning import (
    ActiveLearningConfig,
    calibrate_config,
    compute_score,
    compute_threshold,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats.json"
)

DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "data"
    / "oracle_queue.jsonl"
)


# ============================================================
# Load data
# ============================================================

def load_observations(
    path: Path,
) -> list[dict]:

    print()
    print("=" * 70)
    print("ALBERTA - SEED ORACLE QUEUE")
    print("=" * 70)

    print()
    print("Loading uncertainty statistics")
    print("-" * 70)
    print(f"File: {path}")

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    print(
        f"Raw records: {len(data):,}"
    )

    return data


# ============================================================
# Compute scores
# ============================================================

def compute_scored_observations(
    observations: list[dict],
    config: ActiveLearningConfig,
) -> list[dict]:
    """
    Compute I for every valid observation.
    """

    scored = []

    skipped = 0

    for observation in observations:

        try:

            fen = observation["fen"]

            H = float(
                observation["H"]
            )

            U = float(
                observation["U"]
            )

            HU = float(
                observation["HU"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            skipped += 1
            continue

        score = compute_score(
            H,
            U,
            HU,
            config,
        )

        scored.append(
            {
                "fen": fen,
                "H": H,
                "U": U,
                "HU": HU,
                "score": score,
            }
        )

    print()
    print(
        f"Valid observations: "
        f"{len(scored):,}"
    )

    if skipped:

        print(
            f"Skipped observations: "
            f"{skipped:,}"
        )

    return scored


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Seed the oracle queue from "
            "uncertainty_stats.json."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "Maximum number of positions "
            "inserted into the queue."
        ),
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help=(
            "Override query budget. "
            "Example: 0.005 for 0.5%%."
        ),
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation.",
    )

    args = parser.parse_args()

    if args.limit <= 0:

        raise ValueError(
            "--limit must be greater than 0."
        )

    # ========================================================
    # Load observations
    # ========================================================

    observations = load_observations(
        args.input
    )

    if not observations:

        raise RuntimeError(
            "No observations found."
        )

    # ========================================================
    # Calibration
    # ========================================================

    print()
    print("CALIBRATING ACTIVE LEARNING")
    print("-" * 70)

    #
    # Compute the normalization ranges from
    # the current historical dataset.
    #
    config = calibrate_config(
        observations
    )

    #
    # Optional budget override.
    #
    if args.budget is not None:

        if not 0.0 < args.budget <= 1.0:

            raise ValueError(
                "--budget must be in (0, 1]."
            )

        config.query_budget = (
            args.budget
        )

    print(
        f"H   : "
        f"[{config.h_low:.6f}, "
        f"{config.h_high:.6f}]"
    )

    print(
        f"U   : "
        f"[{config.u_low:.6f}, "
        f"{config.u_high:.6f}]"
    )

    print(
        f"HU  : "
        f"[{config.hu_low:.6f}, "
        f"{config.hu_high:.6f}]"
    )

    print()
    print(
        f"Query budget : "
        f"{config.query_budget * 100:.3f}%"
    )

    # ========================================================
    # Compute scores
    # ========================================================

    scored = compute_scored_observations(
        observations,
        config,
    )

    if not scored:

        raise RuntimeError(
            "No valid observations found."
        )

    # ========================================================
    # Compute threshold
    # ========================================================

    scores = [
        observation["score"]
        for observation in scored
    ]

    threshold = compute_threshold(
        scores,
        config,
    )

    # ========================================================
    # Sort
    # ========================================================

    scored.sort(
        key=lambda observation: (
            observation["score"]
        ),
        reverse=True,
    )

    # ========================================================
    # Budget statistics
    # ========================================================

    budget_selected = sum(
        observation["score"] >= threshold
        for observation in scored
    )

    actual_budget = (
        budget_selected
        / len(scored)
    )

    # ========================================================
    # Display calibration
    # ========================================================

    print()
    print("ACTIVE LEARNING CALIBRATION")
    print("-" * 70)

    print(
        f"Budget    : "
        f"{config.query_budget * 100:.3f}%"
    )

    print(
        f"Threshold : "
        f"{threshold:.6f}"
    )

    print(
        f"Budget-selected positions: "
        f"{budget_selected:,}"
    )

    print(
        f"Actual fraction           : "
        f"{actual_budget * 100:.3f}%"
    )

    # ========================================================
    # Preview
    # ========================================================

    print()
    print("TOP POSITIONS")
    print("-" * 70)

    preview_count = min(
        10,
        len(scored),
    )

    for i in range(preview_count):

        observation = scored[i]

        print(
            f"{i + 1:3d} | "
            f"I={observation['score']:.6f} | "
            f"H={observation['H']:.6f} | "
            f"U={observation['U']:.6f} | "
            f"HU={observation['HU']:.6f}"
        )

    # ========================================================
    # Select positions above threshold
    # ========================================================

    selected = [
        observation
        for observation in scored
        if observation["score"] >= threshold
    ]

    if not selected:

        raise RuntimeError(
            "No observations reached the threshold."
        )

    #
    # For testing, --limit restricts the number
    # inserted into the queue.
    #
    selected = selected[
        :args.limit
    ]

    print()
    print(
        f"Positions to insert: "
        f"{len(selected):,}"
    )

    # ========================================================
    # Queue
    # ========================================================

    queue = OracleQueue(
        args.queue
    )

    current_stats = queue.stats()

    print()
    print("CURRENT QUEUE")
    print("-" * 70)

    print(
        f"Total     : "
        f"{current_stats['total']}"
    )

    print(
        f"Pending   : "
        f"{current_stats['pending']}"
    )

    print(
        f"Answered  : "
        f"{current_stats['answered']}"
    )

    print(
        f"Discarded : "
        f"{current_stats['discarded']}"
    )

    # ========================================================
    # Confirmation
    # ========================================================

    if not args.yes:

        print()
        print(
            f"About to add "
            f"{len(selected)} "
            "historical positions."
        )

        confirmation = input(
            "Continue? [y/n] > "
        ).strip().lower()

        if confirmation != "y":

            print()
            print(
                "Nothing added."
            )

            return

    # ========================================================
    # Insert
    # ========================================================

    added = 0

    for observation in selected:

        queue.add(
            fen=observation["fen"],

            H=observation["H"],
            U=observation["U"],
            HU=observation["HU"],

            score=observation["score"],
            threshold=threshold,

            model="historical_json",

            #
            # Historical data are not associated
            # with a specific RL epoch/game/ply.
            #
            epoch=-1,
            game_id=-1,
            ply=-1,
        )

        added += 1

    # ========================================================
    # Final statistics
    # ========================================================

    final_stats = queue.stats()

    print()
    print("=" * 70)
    print("QUEUE SEEDED")
    print("=" * 70)

    print(
        f"Added     : "
        f"{added:,}"
    )

    print(
        f"Total     : "
        f"{final_stats['total']:,}"
    )

    print(
        f"Pending   : "
        f"{final_stats['pending']:,}"
    )

    print(
        f"Answered  : "
        f"{final_stats['answered']:,}"
    )

    print(
        f"Discarded : "
        f"{final_stats['discarded']:,}"
    )

    print()
    print(
        f"Queue file: "
        f"{args.queue}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()