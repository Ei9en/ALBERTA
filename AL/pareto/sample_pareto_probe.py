#!/usr/bin/env python3

"""
ALBERTA - Pareto Probe Queue Seeding
====================================

Construct a small, manually annotated calibration probe used by the
dynamic Pareto acquisition pipeline.

Pipeline
--------

    corrected RL self-play pool
            |
            v
    compute canonical acquisition score I
            |
            v
    stratify the full I-percentile spectrum
            |
            v
    sample a fixed number of FENs per percentile bin
            |
            v
    Oracle HMI probe queue
            |
            v
    manual annotations
            |
            v
    local-response measurement
            |
            v
    Pareto response surrogate

This script does NOT perform Pareto acquisition itself.

Purpose
-------

The goal is to obtain a small calibration set spanning the acquisition
score spectrum, rather than selecting only high-I positions.

The resulting annotations are later reused to estimate local learner
response as a function of H/U/HU.

Default design
--------------

    40 percentile bins
    5 positions per bin
    200 probe positions total

The score I follows the same acquisition coordinate system as the
canonical Active Learning pipeline:

    U_log = log1p(U / tau)

    H_norm     = side-aware percentile(H)
    U_norm     = side-aware percentile(U_log)
    HU_norm    = H_norm * U_norm

    H*  = standardized(H_norm)
    U*  = standardized(U_norm)
    HU* = standardized(HU_norm)

    I =
        w_H  H*
        + w_U  U*
        + w_HU HU*

Important
---------

The input self-play statistics MUST have been generated with the
corrected league uncertainty estimator.

In particular, U must exclude untrained BC value heads.

The stored raw HU field is recomputed as:

    HU = H * U

rather than trusted from the input JSON.

Output
------

    checkpoints/queue/oracle_queue_1-10_pareto_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path

import chess.variant
import numpy as np


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Canonical Active Learning weights
# ============================================================

from AL.AL_weights import (
    TAU,
    RAW_W_H,
    RAW_W_U,
    RAW_W_HU,
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

DEFAULT_QUEUE_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)

DEFAULT_N_BINS = 40
DEFAULT_SAMPLES_PER_BIN = 5
DEFAULT_SEED = 42


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build the stratified Oracle calibration probe used "
            "by the ALBERTA dynamic Pareto acquisition pipeline."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help=(
            "Corrected self-play uncertainty JSON containing "
            "FEN, H and U."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_QUEUE_FILE,
        help="Output Oracle HMI queue.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_N_BINS,
    )

    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=DEFAULT_SAMPLES_PER_BIN,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty probe queue.",
    )

    return parser.parse_args()


# ============================================================
# Generic utilities
# ============================================================

def safe_float(
    value,
) -> float:

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return np.nan

    if not np.isfinite(
        value
    ):

        return np.nan

    return value


def query_id_from_fen(
    fen: str,
) -> str:

    return uuid.uuid5(
        uuid.NAMESPACE_DNS,
        fen,
    ).hex


# ============================================================
# Input loading
# ============================================================

def load_records(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )

    if isinstance(
        payload,
        list,
    ):

        return payload

    if isinstance(
        payload,
        dict,
    ):

        for key in (
            "data",
            "records",
            "positions",
            "stats",
            "uncertainty_stats",
        ):

            value = payload.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                return value

    raise ValueError(
        f"Unsupported JSON structure:\n{path}"
    )


# ============================================================
# Percentile ranks
# ============================================================

def percentile_rank(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(
        values
    )

    if n < 2:

        raise ValueError(
            "At least two values are required."
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

        end = (
            start
            + 1
        )

        while (
            end < n
            and sorted_values[
                end
            ] == sorted_values[
                start
            ]
        ):

            end += 1

        average_rank = (
            start
            + end
            - 1
        ) / 2.0

        ranks[
            order[
                start:end
            ]
        ] = (
            average_rank
            / (
                n
                - 1
            )
        )

        start = end

    return ranks


# ============================================================
# Side to move
# ============================================================

def extract_side(
    fens: list[str],
) -> np.ndarray:

    sides = []

    for fen in fens:

        parts = fen.split()

        if len(
            parts
        ) < 2:

            raise ValueError(
                f"Invalid FEN:\n{fen}"
            )

        side = parts[
            1
        ]

        if side not in {
            "w",
            "b",
        }:

            raise ValueError(
                f"Invalid side-to-move in FEN:\n{fen}"
            )

        sides.append(
            side
        )

    return np.asarray(
        sides,
        dtype="<U1",
    )


# ============================================================
# Side-aware normalization
# ============================================================

def normalize_side_aware(
    values: np.ndarray,
    sides: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    normalized = np.empty_like(
        values,
        dtype=np.float64,
    )

    for side in (
        "w",
        "b",
    ):

        mask = (
            sides
            == side
        )

        if np.sum(
            mask
        ) < 2:

            raise ValueError(
                f"Not enough positions for side '{side}'."
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
# Min-max representation
# ============================================================

def minmax_normalize(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    minimum = float(
        np.min(
            values
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    if maximum <= minimum:

        raise ValueError(
            "Cannot min-max normalize a constant score."
        )

    return (
        values
        - minimum
    ) / (
        maximum
        - minimum
    )


# ============================================================
# Build canonical acquisition coordinates
# ============================================================

def build_score(
    records: list[dict],
) -> dict:
    """
    Build the canonical AL score I and its percentile coordinate.

    Raw HU is stored for provenance as H * U.

    The interaction entering I is instead:

        HU_norm = H_norm * U_log_norm
    """

    cleaned_records = []

    H_values = []
    U_values = []
    fens = []

    skipped_invalid = 0

    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            skipped_invalid += 1
            continue

        fen = record.get(
            "fen"
        )

        H = safe_float(
            record.get(
                "H"
            )
        )

        U = safe_float(
            record.get(
                "U"
            )
        )

        if (
            not isinstance(
                fen,
                str,
            )
            or not fen
            or not np.isfinite(
                H
            )
            or not np.isfinite(
                U
            )
            or U < 0.0
        ):

            skipped_invalid += 1
            continue

        # Validate FEN now.
        chess.variant.AtomicBoard(
            fen
        )

        cleaned_records.append(
            record
        )

        fens.append(
            fen
        )

        H_values.append(
            H
        )

        U_values.append(
            U
        )

    if len(
        cleaned_records
    ) < 2:

        raise RuntimeError(
            "Too few valid H/U records."
        )

    H = np.asarray(
        H_values,
        dtype=np.float64,
    )

    U = np.asarray(
        U_values,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Raw interaction stored with the queue
    # --------------------------------------------------------

    HU = (
        H
        * U
    )

    # --------------------------------------------------------
    # Side-to-move
    # --------------------------------------------------------

    sides = extract_side(
        fens
    )

    # --------------------------------------------------------
    # U transformation used by acquisition score
    # --------------------------------------------------------

    U_log = np.log1p(
        U
        / TAU
    )

    # --------------------------------------------------------
    # Side-aware percentile coordinates
    # --------------------------------------------------------

    H_norm = normalize_side_aware(
        H,
        sides,
    )

    U_norm = normalize_side_aware(
        U_log,
        sides,
    )

    # --------------------------------------------------------
    # Interaction in normalized coordinate system
    # --------------------------------------------------------

    HU_norm = (
        H_norm
        * U_norm
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    H_std = float(
        H_norm.std(
            ddof=0
        )
    )

    U_std = float(
        U_norm.std(
            ddof=0
        )
    )

    HU_std = float(
        HU_norm.std(
            ddof=0
        )
    )

    if (
        H_std <= 0.0
        or U_std <= 0.0
        or HU_std <= 0.0
    ):

        raise ValueError(
            "Cannot standardize constant acquisition predictor."
        )

    H_star = (
        H_norm
        - H_norm.mean()
    ) / H_std

    U_star = (
        U_norm
        - U_norm.mean()
    ) / U_std

    HU_star = (
        HU_norm
        - HU_norm.mean()
    ) / HU_std

    # --------------------------------------------------------
    # Canonical raw acquisition score
    # --------------------------------------------------------

    I = (
        RAW_W_H
        * H_star
        + RAW_W_U
        * U_star
        + RAW_W_HU
        * HU_star
    )

    I_norm = minmax_normalize(
        I
    )

    I_percentile = (
        percentile_rank(
            I
        )
        * 100.0
    )

    return {
        "records":
            cleaned_records,

        "H":
            H,

        "U":
            U,

        "HU":
            HU,

        "I":
            I,

        "I_norm":
            I_norm,

        "I_percentile":
            I_percentile,

        "sides":
            sides,

        "skipped_invalid":
            skipped_invalid,
    }


# ============================================================
# Atomic legal move count
# ============================================================

def count_legal_moves(
    fen: str,
) -> int:

    board = chess.variant.AtomicBoard(
        fen
    )

    return board.legal_moves.count()


# ============================================================
# Stratified probe selection
# ============================================================

def select_probe(
    *,
    records: list[dict],
    I_percentile: np.ndarray,
    n_bins: int,
    samples_per_bin: int,
    seed: int,
) -> tuple[
    np.ndarray,
    int,
    int,
    int,
]:

    rng = random.Random(
        seed
    )

    bin_width = (
        100.0
        / n_bins
    )

    selected = []

    selected_ids = set()

    total_checked = 0
    rejected_moves = 0
    rejected_duplicates = 0

    for bin_id in range(
        n_bins
    ):

        lower = (
            bin_id
            * bin_width
        )

        upper = (
            (
                bin_id
                + 1
            )
            * bin_width
        )

        if bin_id < (
            n_bins
            - 1
        ):

            candidate_indices = np.flatnonzero(
                (
                    I_percentile
                    >= lower
                )
                &
                (
                    I_percentile
                    < upper
                )
            )

        else:

            candidate_indices = np.flatnonzero(
                (
                    I_percentile
                    >= lower
                )
                &
                (
                    I_percentile
                    <= upper
                )
            )

        candidate_indices = (
            candidate_indices.tolist()
        )

        rng.shuffle(
            candidate_indices
        )

        bin_selected = []

        for index in candidate_indices:

            total_checked += 1

            fen = records[
                index
            ][
                "fen"
            ]

            query_id = query_id_from_fen(
                fen
            )

            if query_id in selected_ids:

                rejected_duplicates += 1
                continue

            try:

                legal_moves = count_legal_moves(
                    fen
                )

            except Exception:

                rejected_moves += 1
                continue

            if legal_moves <= 1:

                rejected_moves += 1
                continue

            bin_selected.append(
                index
            )

            selected_ids.add(
                query_id
            )

            if len(
                bin_selected
            ) >= samples_per_bin:

                break

        if len(
            bin_selected
        ) < samples_per_bin:

            raise RuntimeError(
                "Could not obtain enough eligible positions "
                f"from percentile bin {bin_id}: "
                f"[{lower:.2f}, {upper:.2f}). "
                f"Needed {samples_per_bin}, "
                f"found {len(bin_selected)}."
            )

        selected.extend(
            bin_selected
        )

        closing_bracket = (
            "]"
            if bin_id == n_bins - 1
            else ")"
        )

        print(
            f"Bin {bin_id:02d} | "
            f"[{lower:6.2f}, {upper:6.2f}"
            f"{closing_bracket} | "
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
# Queue writer
# ============================================================

def write_probe_queue(
    *,
    output_path: Path,
    records: list[dict],
    selected_indices: np.ndarray,
    H: np.ndarray,
    U: np.ndarray,
    HU: np.ndarray,
    I: np.ndarray,
    I_norm: np.ndarray,
    I_percentile: np.ndarray,
    force: bool,
) -> None:

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not force
    ):

        raise FileExistsError(
            f"Probe queue already exists:\n"
            f"{output_path}\n\n"
            f"Use --force to replace it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for index in selected_indices:

            index = int(
                index
            )

            record = records[
                index
            ]

            fen = record[
                "fen"
            ]

            item = {
                "query_id":
                    query_id_from_fen(
                        fen
                    ),

                "fen":
                    fen,

                "H":
                    float(
                        H[
                            index
                        ]
                    ),

                "U":
                    float(
                        U[
                            index
                        ]
                    ),

                "HU":
                    float(
                        HU[
                            index
                        ]
                    ),

                "score":
                    float(
                        I[
                            index
                        ]
                    ),

                "I_norm":
                    float(
                        I_norm[
                            index
                        ]
                    ),

                "I_percentile":
                    float(
                        I_percentile[
                            index
                        ]
                    ),

                "threshold":
                    None,

                "model":
                    record.get(
                        "model",
                        "historical_json",
                    ),

                "epoch":
                    record.get(
                        "epoch",
                        -1,
                    ),

                "game_id":
                    record.get(
                        "game_id",
                        -1,
                    ),

                "ply":
                    record.get(
                        "ply",
                        -1,
                    ),

                "created_at":
                    created_at,

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

            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.bins <= 0:

        raise ValueError(
            "--bins must be strictly positive."
        )

    if args.samples_per_bin <= 0:

        raise ValueError(
            "--samples-per-bin must be strictly positive."
        )

    print()
    print("=" * 72)
    print("ALBERTA - PARETO PROBE QUEUE")
    print("=" * 72)

    print(
        f"Input:           {args.input}"
    )

    print(
        f"Output:          {args.output}"
    )

    print(
        f"Bins:            {args.bins}"
    )

    print(
        f"Samples / bin:   {args.samples_per_bin}"
    )

    print(
        f"Target total:    "
        f"{args.bins * args.samples_per_bin}"
    )

    print(
        f"Seed:            {args.seed}"
    )

    print()

    print(
        "IMPORTANT: input U must come from the corrected "
        "league uncertainty estimator."
    )

    # ========================================================
    # Load
    # ========================================================

    raw_records = load_records(
        args.input
    )

    print()
    print(
        f"Raw pool positions: {len(raw_records):,}"
    )

    # ========================================================
    # Score
    # ========================================================

    score_data = build_score(
        raw_records
    )

    records = score_data[
        "records"
    ]

    H = score_data[
        "H"
    ]

    U = score_data[
        "U"
    ]

    HU = score_data[
        "HU"
    ]

    I = score_data[
        "I"
    ]

    I_norm = score_data[
        "I_norm"
    ]

    I_percentile = score_data[
        "I_percentile"
    ]

    sides = score_data[
        "sides"
    ]

    print(
        f"Valid H/U positions: "
        f"{len(records):,}"
    )

    print(
        f"Invalid skipped:     "
        f"{score_data['skipped_invalid']:,}"
    )

    print()

    print(
        f"I range: "
        f"{I.min():+.9f} -> {I.max():+.9f}"
    )

    print(
        f"I percentile range: "
        f"{I_percentile.min():.6f}% -> "
        f"{I_percentile.max():.6f}%"
    )

    # ========================================================
    # Selection
    # ========================================================

    (
        selected_indices,
        checked,
        rejected_moves,
        rejected_duplicates,
    ) = select_probe(
        records=records,
        I_percentile=I_percentile,
        n_bins=args.bins,
        samples_per_bin=args.samples_per_bin,
        seed=args.seed,
    )

    # ========================================================
    # Write
    # ========================================================

    write_probe_queue(
        output_path=args.output,
        records=records,
        selected_indices=selected_indices,
        H=H,
        U=U,
        HU=HU,
        I=I,
        I_norm=I_norm,
        I_percentile=I_percentile,
        force=args.force,
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    selected_sides = sides[
        selected_indices
    ]

    print()
    print("=" * 72)
    print("PARETO PROBE QUEUE CREATED")
    print("=" * 72)

    print(
        f"Selected:            "
        f"{len(selected_indices):,}"
    )

    print(
        f"Candidates checked:  "
        f"{checked:,}"
    )

    print(
        f"Rejected <=1 move:   "
        f"{rejected_moves:,}"
    )

    print(
        f"Rejected duplicates: "
        f"{rejected_duplicates:,}"
    )

    print()

    print(
        f"White to move: "
        f"{np.sum(selected_sides == 'w'):,}"
    )

    print(
        f"Black to move: "
        f"{np.sum(selected_sides == 'b'):,}"
    )

    print()

    print(
        f"Selected I percentile range: "
        f"{I_percentile[selected_indices].min():.6f}% -> "
        f"{I_percentile[selected_indices].max():.6f}%"
    )

    print()

    print(
        f"Queue: {args.output}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()