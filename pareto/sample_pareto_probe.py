#!/usr/bin/env python3

"""
ALBERTA - Pareto probe queue seeding
====================================

Build a stratified experimental probe over the percentile distribution
of the Active Learning score I and write it directly in the Oracle HMI
queue format.

This is NOT the final Pareto selection.

Purpose
-------

The queue is used to manually annotate positions spread across the
whole I spectrum, in order to later estimate:

    I -> (Delta KL, Delta V)

Sampling
--------

Default:
    40 percentile bins
    5 positions per bin
    200 annotations total

Each bin spans 2.5 percentile points.

Output
------

    checkpoints/queue/oracle_queue_1-10_pareto_probe.jsonl

Usage
-----

    python pareto/sample_pareto_probe.py
"""

from __future__ import annotations

import sys
import json
import uuid
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import chess.variant


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Active Learning parameters
# ============================================================

from data.uncertainty_analysis.active_learning_weights import (
    TAU,
    RAW_W_H,
    RAW_W_U,
    RAW_W_HU,
)


# ============================================================
# Paths
# ============================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

QUEUE_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)


# ============================================================
# Configuration
# ============================================================

N_BINS = 40

SAMPLES_PER_BIN = 5

RANDOM_SEED = 42


# ============================================================
# Percentile rank
# ============================================================

def percentile_rank(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n < 2:
        raise ValueError(
            "Not enough values for percentile rank."
        )

    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[
        order
    ]

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and
            sorted_values[end]
            ==
            sorted_values[start]
        ):

            end += 1

        rank = (
            start + end - 1
        ) / 2.0

        ranks[
            order[start:end]
        ] = (
            rank
            /
            (n - 1)
        )

        start = end

    return ranks


# ============================================================
# Side to move
# ============================================================

def extract_side(fens):

    sides = []

    for fen in fens:

        parts = fen.split()

        if len(parts) < 2:

            raise ValueError(
                f"Invalid FEN: {fen}"
            )

        side = parts[1]

        if side not in (
            "w",
            "b",
        ):

            raise ValueError(
                f"Invalid FEN side: {fen}"
            )

        sides.append(
            side
        )

    return np.asarray(
        sides,
        dtype="<U1",
    )


# ============================================================
# Atomic legal move count
# ============================================================

def count_legal_moves(fen):

    try:

        board = chess.variant.AtomicBoard(
            fen
        )

        return board.legal_moves.count()

    except Exception as e:

        raise ValueError(
            f"Could not parse Atomic FEN:\n"
            f"{fen}\n"
            f"Error: {e}"
        )


# ============================================================
# Side-aware normalization
# ============================================================

def normalize_side_aware(
    values,
    sides,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    normalized = np.zeros_like(
        values,
        dtype=np.float64,
    )

    for side in (
        "w",
        "b",
    ):

        mask = (
            sides == side
        )

        if np.sum(mask) < 2:

            raise ValueError(
                f"Not enough positions for side "
                f"{side} normalization."
            )

        normalized[
            mask
        ] = percentile_rank(
            values[
                mask
            ]
        )

    return normalized


# ============================================================
# Min-max normalization
# ============================================================

def minmax_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    minimum = np.min(
        values
    )

    maximum = np.max(
        values
    )

    if maximum <= minimum:

        raise ValueError(
            "Cannot min-max normalize a constant score."
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# Load existing queue IDs
# ============================================================

def load_existing_ids():

    if not QUEUE_FILE.exists():

        return set()

    existing_ids = set()

    with open(
        QUEUE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():

                continue

            item = json.loads(
                line
            )

            existing_ids.add(
                item["query_id"]
            )

    return existing_ids


# ============================================================
# Build I
# ============================================================

def build_score(data):

    fens = np.array(
        [
            record["fen"]
            for record in data
        ],
        dtype=object,
    )

    H = np.array(
        [
            record["H"]
            for record in data
        ],
        dtype=np.float64,
    )

    U = np.array(
        [
            record["U"]
            for record in data
        ],
        dtype=np.float64,
    )

    HU = np.array(
        [
            record["HU"]
            for record in data
        ],
        dtype=np.float64,
    )

    sides = extract_side(
        fens
    )

    # --------------------------------------------------------
    # U logarithmic transform
    # --------------------------------------------------------

    U_log = np.log1p(
        U / TAU
    )

    # --------------------------------------------------------
    # Side-aware normalization
    # --------------------------------------------------------

    H_norm = normalize_side_aware(
        H,
        sides,
    )

    U_log_norm = normalize_side_aware(
        U_log,
        sides,
    )

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    HU_log_norm = (
        H_norm
        *
        U_log_norm
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    H_std = H_norm.std()
    U_std = U_log_norm.std()
    HU_std = HU_log_norm.std()

    if (
        H_std <= 0
        or U_std <= 0
        or HU_std <= 0
    ):

        raise ValueError(
            "Cannot standardize predictor."
        )

    H_star = (
        H_norm
        -
        H_norm.mean()
    ) / H_std

    U_star = (
        U_log_norm
        -
        U_log_norm.mean()
    ) / U_std

    HU_star = (
        HU_log_norm
        -
        HU_log_norm.mean()
    ) / HU_std

    # --------------------------------------------------------
    # Raw I
    # --------------------------------------------------------

    I = (
        RAW_W_H
        * H_star
        +
        RAW_W_U
        * U_star
        +
        RAW_W_HU
        * HU_star
    )

    # --------------------------------------------------------
    # Min-max representation
    # --------------------------------------------------------

    I_norm = minmax_normalize(
        I
    )

    # --------------------------------------------------------
    # Global percentile coordinate
    # --------------------------------------------------------

    I_percentile = (
        percentile_rank(
            I
        )
        *
        100.0
    )

    return (
        H,
        U,
        HU,
        I,
        I_norm,
        I_percentile,
        sides,
    )


# ============================================================
# Stratified selection
# ============================================================

def select_probe(
    data,
    I_percentile,
    existing_ids,
):

    rng = random.Random(
        RANDOM_SEED
    )

    bin_width = (
        100.0
        /
        N_BINS
    )

    selected = []

    total_checked = 0
    rejected_moves = 0
    rejected_duplicates = 0

    for bin_id in range(
        N_BINS
    ):

        lower = (
            bin_id
            *
            bin_width
        )

        upper = (
            (bin_id + 1)
            *
            bin_width
        )

        # percentile_rank lies in [0, 100]
        # Use:
        #   [lower, upper)
        # except final bin which includes 100.

        if bin_id < (
            N_BINS - 1
        ):

            candidate_indices = np.where(
                (
                    I_percentile >= lower
                )
                &
                (
                    I_percentile < upper
                )
            )[0]

        else:

            candidate_indices = np.where(
                (
                    I_percentile >= lower
                )
                &
                (
                    I_percentile <= upper
                )
            )[0]

        candidate_indices = (
            candidate_indices.tolist()
        )

        rng.shuffle(
            candidate_indices
        )

        bin_selected = []

        for idx in candidate_indices:

            total_checked += 1

            record = data[idx]

            query_id = uuid.uuid5(
                uuid.NAMESPACE_DNS,
                record["fen"],
            ).hex

            if query_id in existing_ids:

                rejected_duplicates += 1
                continue

            if (
                count_legal_moves(
                    record["fen"]
                )
                <= 1
            ):

                rejected_moves += 1
                continue

            bin_selected.append(
                idx
            )

            existing_ids.add(
                query_id
            )

            if (
                len(bin_selected)
                >=
                SAMPLES_PER_BIN
            ):

                break

        if (
            len(bin_selected)
            <
            SAMPLES_PER_BIN
        ):

            raise RuntimeError(
                f"Could not obtain enough valid "
                f"positions in percentile bin "
                f"{bin_id}: "
                f"[{lower:.2f}, {upper:.2f})."
            )

        selected.extend(
            bin_selected
        )

        print(
            f"Bin {bin_id:02d} | "
            f"[{lower:6.2f}, {upper:6.2f}) | "
            f"candidates={len(candidate_indices):7,d} | "
            f"selected={len(bin_selected)}"
        )

    return (
        np.asarray(
            selected,
            dtype=np.int64,
        ),
        total_checked,
        rejected_moves,
        rejected_duplicates,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "ALBERTA - PARETO PROBE QUEUE"
    )
    print("=" * 70)

    print()

    print(
        f"Bins            : {N_BINS}"
    )

    print(
        f"Samples / bin   : {SAMPLES_PER_BIN}"
    )

    print(
        f"Target total    : "
        f"{N_BINS * SAMPLES_PER_BIN}"
    )

    print(
        f"Random seed     : {RANDOM_SEED}"
    )

    print()

    print(
        f"Queue file      : {QUEUE_FILE}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print(
        "Loading uncertainty statistics..."
    )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    if not data:

        raise RuntimeError(
            "No positions found."
        )

    print(
        f"Positions loaded : {len(data):,}"
    )

    # --------------------------------------------------------
    # Build score
    # --------------------------------------------------------

    print()
    print(
        "Computing score I..."
    )

    (
        H,
        U,
        HU,
        I,
        I_norm,
        I_percentile,
        sides,
    ) = build_score(
        data
    )

    print(
        f"I min        : {I.min():+.9f}"
    )

    print(
        f"I max        : {I.max():+.9f}"
    )

    print(
        f"I_norm min   : {I_norm.min():.9f}"
    )

    print(
        f"I_norm max   : {I_norm.max():.9f}"
    )

    print(
        f"Percentiles  : "
        f"{I_percentile.min():.6f}% "
        f"-> "
        f"{I_percentile.max():.6f}%"
    )

    # --------------------------------------------------------
    # Existing queue
    # --------------------------------------------------------

    existing_ids = load_existing_ids()

    print()
    print(
        f"Existing queue entries : "
        f"{len(existing_ids):,}"
    )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    print()
    print(
        "Sampling I spectrum..."
    )

    (
        selected_indices,
        checked,
        rejected_moves,
        rejected_duplicates,
    ) = select_probe(
        data,
        I_percentile,
        existing_ids,
    )

    # ============================================================
    # Write HMI-compatible queue
    # ============================================================

    QUEUE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    added = 0

    with open(
        QUEUE_FILE,
        "a",
        encoding="utf-8",
    ) as f:

        for idx in selected_indices:

            record = data[idx]

            query_id = uuid.uuid5(
                uuid.NAMESPACE_DNS,
                record["fen"],
            ).hex

            item = {

                "query_id":
                    query_id,

                "fen":
                    record["fen"],

                "H":
                    float(
                        H[idx]
                    ),

                "U":
                    float(
                        U[idx]
                    ),

                "HU":
                    float(
                        HU[idx]
                    ),

                "score":
                    float(
                        I[idx]
                    ),

                "I_norm":
                    float(
                        I_norm[idx]
                    ),

                "threshold":
                    None,

                "model":
                    "historical_json",

                "epoch":
                    -1,

                "game_id":
                    -1,

                "ply":
                    -1,

                "created_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "status":
                    "pending",

                "oracle_move":
                    None,

                "oracle_confidence":
                    None,

                "oracle_situation":
                    None,

                "reward":
                    None,

                "answered_at":
                    None,
            }

            f.write(
                json.dumps(
                    item
                )
                +
                "\n"
            )

            added += 1

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    selected_sides = sides[
        selected_indices
    ]

    print()
    print("=" * 70)
    print(
        "PARETO PROBE QUEUE CREATED"
    )
    print("=" * 70)

    print()

    print(
        f"Selected             : "
        f"{len(selected_indices):,}"
    )

    print(
        f"Added                : "
        f"{added:,}"
    )

    print(
        f"Candidates checked   : "
        f"{checked:,}"
    )

    print(
        f"Rejected <=1 move    : "
        f"{rejected_moves:,}"
    )

    print(
        f"Rejected duplicates  : "
        f"{rejected_duplicates:,}"
    )

    print()

    print(
        f"White-to-move        : "
        f"{np.sum(selected_sides == 'w'):,}"
    )

    print(
        f"Black-to-move        : "
        f"{np.sum(selected_sides == 'b'):,}"
    )

    print()

    print(
        f"I percentile min     : "
        f"{I_percentile[selected_indices].min():.6f}%"
    )

    print(
        f"I percentile max     : "
        f"{I_percentile[selected_indices].max():.6f}%"
    )

    print()

    print(
        f"File                 : "
        f"{QUEUE_FILE}"
    )


if __name__ == "__main__":
    main()