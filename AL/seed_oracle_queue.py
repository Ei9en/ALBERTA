import sys
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone
import math

import numpy as np
import chess
import chess.variant


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Active Learning weights
# ============================================================

from data.uncertainty_analysis.active_learning_weights import (
    TAU,
    RAW_W_H,
    RAW_W_U,
    RAW_W_HU,
)


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

QUEUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
)


# ============================================================
# Configuration
# ============================================================

AL_BUDGET = 0.0002


# ============================================================
# Percentile rank normalization
# ============================================================

def percentile_rank(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    n = len(values)

    if n < 2:
        raise ValueError(
            "Not enough values for percentile rank."
        )

    order = np.argsort(
        values,
        kind="stable"
    )

    sorted_values = values[order]

    ranks = np.empty(
        n,
        dtype=np.float64
    )

    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and sorted_values[end]
            ==
            sorted_values[start]
        ):
            end += 1

        rank = (
            (start + end - 1)
            /
            2.0
        )

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
# Extract side to move
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

        if side not in ("w", "b"):

            raise ValueError(
                f"Invalid FEN side: {fen}"
            )

        sides.append(side)

    return np.array(
        sides,
        dtype="<U1"
    )


# ============================================================
# Count legal Atomic moves
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
    sides
):

    normalized = np.zeros_like(
        values,
        dtype=np.float64
    )

    for side in (
        "w",
        "b"
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
        dtype=np.float64
    )

    minimum = np.min(values)
    maximum = np.max(values)

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
# Build candidate order
# ============================================================

def build_candidate_order(
    I,
    mode
):

    n = len(I)

    if n == 0:

        raise ValueError(
            "No positions available."
        )

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if mode == "high":

        order = np.argsort(
            I,
            kind="stable"
        )[::-1]

        description = (
            "Highest I"
        )

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    elif mode == "low":

        order = np.argsort(
            I,
            kind="stable"
        )

        description = (
            "Lowest I"
        )

    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    elif mode == "middle":

        sorted_indices = np.argsort(
            I,
            kind="stable"
        )

        center = n // 2

        order_list = []

        left = center - 1
        right = center

        while (
            left >= 0
            or right < n
        ):

            if right < n:

                order_list.append(
                    sorted_indices[right]
                )

                right += 1

            if left >= 0:

                order_list.append(
                    sorted_indices[left]
                )

                left -= 1

        order = np.array(
            order_list,
            dtype=np.int64
        )

        description = (
            "Middle I (around median)"
        )

    # --------------------------------------------------------
    # RANDOM
    #
    # I is completely ignored.
    # --------------------------------------------------------

    elif mode == "random":

        order = np.random.permutation(
            n
        )

        description = (
            "Uniform random selection (I ignored)"
        )

    else:

        raise ValueError(
            f"Unknown selection mode: {mode}"
        )

    return order, description


# ============================================================
# Queue path
# ============================================================

def get_queue_file(mode):

    if mode == "random":

        filename = (
            "oracle_queue_1-10_random.jsonl"
        )

    elif mode == "high":

        filename = (
            "oracle_queue_1-10_AL.jsonl"
        )

    elif mode == "low":

        filename = (
            "oracle_queue_1-10_low.jsonl"
        )

    elif mode == "middle":

        filename = (
            "oracle_queue_1-10_middle.jsonl"
        )

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return (
        QUEUE_DIR
        / filename
    )


# ============================================================
# Load existing queue IDs
# ============================================================

def load_existing_ids(
    queue_file
):

    if not queue_file.exists():

        return set()

    existing_ids = set()

    with open(
        queue_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():

                continue

            item = json.loads(line)

            existing_ids.add(
                item["query_id"]
            )

    return existing_ids


# ============================================================
# Select positions
# ============================================================

def select_positions(
    data,
    I,
    mode,
    existing_ids
):

    n = len(I)

    if n == 0:

        raise ValueError(
            "No positions available."
        )

    # --------------------------------------------------------
    # Budget is 0.02% of the original dataset.
    # --------------------------------------------------------

    target_count = max(
        1,
        math.ceil(
            n * AL_BUDGET
        )
    )

    candidate_order, description = (
        build_candidate_order(
            I,
            mode
        )
    )

    selected = []

    checked = 0
    rejected_moves = 0
    rejected_duplicates = 0

    # --------------------------------------------------------
    # Examine candidates in priority order.
    # --------------------------------------------------------

    for idx in candidate_order:

        checked += 1

        record = data[idx]

        query_id = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            record["fen"]
        ).hex

        # ----------------------------------------------------
        # Duplicate check
        # ----------------------------------------------------

        if query_id in existing_ids:

            rejected_duplicates += 1

            continue

        # ----------------------------------------------------
        # Legal move check
        # ----------------------------------------------------

        legal_count = count_legal_moves(
            record["fen"]
        )

        if legal_count <= 1:

            rejected_moves += 1

            continue

        # ----------------------------------------------------
        # Valid candidate
        # ----------------------------------------------------

        selected.append(
            idx
        )

        existing_ids.add(
            query_id
        )

        if len(selected) >= target_count:

            break

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(selected) < target_count:

        raise RuntimeError(
            "Could not find enough new eligible positions "
            "to reach the requested annotation budget."
        )

    return (
        np.array(
            selected,
            dtype=np.int64
        ),
        description,
        checked,
        rejected_moves,
        rejected_duplicates
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Seed ALBERTA oracle queue using the frozen "
        "active-learning score or uniform random sampling."
    )

    parser.add_argument(
        "--mode",
        choices=[
            "high",
            "low",
            "middle",
            "random"
        ],
        default="high",
        help=
        "Selection mode: "
        "high = highest I, "
        "low = lowest I, "
        "middle = around median, "
        "random = uniform random selection."
    )

    args = parser.parse_args()

    queue_file = get_queue_file(
        args.mode
    )

    print("=" * 70)
    print(
        "ALBERTA - SEED ORACLE QUEUE"
    )
    print("=" * 70)

    print()
    print(
        f"Selection mode : {args.mode}"
    )

    print(
        f"Queue file     : {queue_file}"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print(
        "CONFIGURATION"
    )
    print("-" * 70)

    print(
        f"TAU            : {TAU:.6f}"
    )

    print(
        f"AL budget      : "
        f"{100 * AL_BUDGET:.5f}%"
    )

    print()
    print(
        "RAW OLS COEFFICIENTS"
    )

    print(
        f"RAW_W_H        : {RAW_W_H:+.9f}"
    )

    print(
        f"RAW_W_U        : {RAW_W_U:+.9f}"
    )

    print(
        f"RAW_W_HU       : {RAW_W_HU:+.9f}"
    )

    print()
    print(
        "Coefficients are used at their original "
        "OLS scale; no coefficient normalization."
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
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        f"Positions loaded : {len(data):,}"
    )

    if len(data) == 0:

        raise RuntimeError(
            "No positions found."
        )

    # --------------------------------------------------------
    # Extract signals
    # --------------------------------------------------------

    fens = np.array(
        [
            x["fen"]
            for x in data
        ],
        dtype=object
    )

    H = np.array(
        [
            x["H"]
            for x in data
        ],
        dtype=np.float64
    )

    U = np.array(
        [
            x["U"]
            for x in data
        ],
        dtype=np.float64
    )

    HU = np.array(
        [
            x["HU"]
            for x in data
        ],
        dtype=np.float64
    )

    sides = extract_side(
        fens
    )

    # --------------------------------------------------------
    # Log transform of U
    # --------------------------------------------------------

    print()
    print(
        "Applying log transform to U..."
    )

    U_log = np.log1p(
        U / TAU
    )

    # --------------------------------------------------------
    # Side-aware normalization
    # --------------------------------------------------------

    print()
    print(
        "Computing side-aware normalization..."
    )

    H_norm = normalize_side_aware(
        H,
        sides
    )

    U_log_norm = normalize_side_aware(
        U_log,
        sides
    )

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    HU_log_norm = (
        H_norm
        * U_log_norm
    )

    # --------------------------------------------------------
    # Standardize predictors
    #
    # IMPORTANT:
    #
    # The OLS coefficients were estimated on standardized
    # predictors, therefore we must reproduce exactly the same
    # standardization here.
    # --------------------------------------------------------

    print()
    print(
        "Standardizing predictors..."
    )

    H_mean = H_norm.mean()
    H_std = H_norm.std()

    U_log_mean = U_log_norm.mean()
    U_log_std = U_log_norm.std()

    HU_log_mean = HU_log_norm.mean()
    HU_log_std = HU_log_norm.std()

    if H_std <= 0:
        raise ValueError(
            "H normalized standard deviation is zero."
        )

    if U_log_std <= 0:
        raise ValueError(
            "U_log normalized standard deviation is zero."
        )

    if HU_log_std <= 0:
        raise ValueError(
            "H*U_log normalized standard deviation is zero."
        )

    H_star = (
        H_norm - H_mean
    ) / H_std

    U_log_star = (
        U_log_norm - U_log_mean
    ) / U_log_std

    HU_log_star = (
        HU_log_norm - HU_log_mean
    ) / HU_log_std

    # --------------------------------------------------------
    # Compute raw I
    # --------------------------------------------------------

    I = (
        RAW_W_H * H_star
        +
        RAW_W_U * U_log_star
        +
        RAW_W_HU * HU_log_star
    )

    # --------------------------------------------------------
    # Final min-max normalization
    #
    # ONLY for the final score representation.
    #
    # This does not affect ranking.
    # --------------------------------------------------------

    I_norm = minmax_normalize(
        I
    )

    print()
    print(
        "Active learning score computed."
    )

    print(
        f"I min       : {I.min():+.9f}"
    )

    print(
        f"I max       : {I.max():+.9f}"
    )

    print(
        f"I_norm min  : {I_norm.min():.9f}"
    )

    print(
        f"I_norm max  : {I_norm.max():.9f}"
    )

    # --------------------------------------------------------
    # Existing queue
    # --------------------------------------------------------

    existing_ids = load_existing_ids(
        queue_file
    )

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
        "Searching candidates..."
    )

    (
        selected_indices,
        description,
        checked,
        rejected_moves,
        rejected_duplicates
    ) = select_positions(
        data,
        I,
        args.mode,
        existing_ids
    )

    selected_I = I[
        selected_indices
    ]

    selected_I_norm = I_norm[
        selected_indices
    ]

    # --------------------------------------------------------
    # Selection statistics
    # --------------------------------------------------------

    print()
    print(
        "ACTIVE LEARNING SELECTION"
    )
    print("-" * 70)

    print(
        f"Mode                : "
        f"{description}"
    )

    print(
        f"Target annotations  : "
        f"{len(selected_indices):,}"
    )

    print(
        f"Candidates checked  : "
        f"{checked:,}"
    )

    print(
        f"Rejected (<=1 move) : "
        f"{rejected_moves:,}"
    )

    print(
        f"Rejected (duplicate): "
        f"{rejected_duplicates:,}"
    )

    print(
        f"Budget              : "
        f"{100 * len(selected_indices) / len(data):.5f}%"
    )

    print(
        f"I min selected      : "
        f"{selected_I.min():+.9f}"
    )

    print(
        f"I max selected      : "
        f"{selected_I.max():+.9f}"
    )

    print(
        f"I mean selected     : "
        f"{selected_I.mean():+.9f}"
    )

    print(
        f"I median selected   : "
        f"{np.median(selected_I):+.9f}"
    )

    print(
        f"I_norm min selected : "
        f"{selected_I_norm.min():.9f}"
    )

    print(
        f"I_norm max selected : "
        f"{selected_I_norm.max():.9f}"
    )

    # --------------------------------------------------------
    # Side-to-move statistics
    # --------------------------------------------------------

    selected_sides = sides[
        selected_indices
    ]

    global_white = np.sum(
        sides == "w"
    )

    global_black = np.sum(
        sides == "b"
    )

    selected_white = np.sum(
        selected_sides == "w"
    )

    selected_black = np.sum(
        selected_sides == "b"
    )

    print()
    print(
        "SIDE-TO-MOVE"
    )
    print("-" * 70)

    print(
        f"Global White      : "
        f"{global_white:,} "
        f"({100 * global_white / len(data):.3f}%)"
    )

    print(
        f"Global Black      : "
        f"{global_black:,} "
        f"({100 * global_black / len(data):.3f}%)"
    )

    print(
        f"Selected White    : "
        f"{selected_white:,} "
        f"({100 * selected_white / len(selected_indices):.3f}%)"
    )

    print(
        f"Selected Black    : "
        f"{selected_black:,} "
        f"({100 * selected_black / len(selected_indices):.3f}%)"
    )

    white_enrichment = (
        (
            selected_white
            /
            len(selected_indices)
        )
        /
        (
            global_white
            /
            len(data)
        )
    )

    black_enrichment = (
        (
            selected_black
            /
            len(selected_indices)
        )
        /
        (
            global_black
            /
            len(data)
        )
    )

    print(
        f"White enrichment : "
        f"{white_enrichment:.3f}x"
    )

    print(
        f"Black enrichment : "
        f"{black_enrichment:.3f}x"
    )

    # --------------------------------------------------------
    # Append to queue
    # --------------------------------------------------------

    QUEUE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    added = 0

    with open(
        queue_file,
        "a",
        encoding="utf-8"
    ) as f:

        for idx in selected_indices:

            record = data[idx]

            query_id = uuid.uuid5(
                uuid.NAMESPACE_DNS,
                record["fen"]
            ).hex

            item = {

                "query_id":
                    query_id,

                "fen":
                    record["fen"],

                "H":
                    float(H[idx]),

                "U":
                    float(U[idx]),

                "HU":
                    float(HU[idx]),

                "I":
                    float(I[idx]),

                "I_norm":
                    float(I_norm[idx]),

                "status":
                    "pending",

                "oracle_move":
                    None,

                "oracle_confidence":
                    None,

                "oracle_situation":
                    None,

                "created_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "reward":
                    None,

                "answered_at":
                    None
            }

            f.write(
                json.dumps(item)
                +
                "\n"
            )

            added += 1

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "QUEUE UPDATED"
    )
    print("=" * 70)

    print(
        f"Selected : {len(selected_indices):,}"
    )

    print(
        f"Added    : {added:,}"
    )

    print(
        f"Budget   : "
        f"{100 * added / len(data):.5f}%"
    )

    print(
        f"File     : {queue_file}"
    )


if __name__ == "__main__":

    main()