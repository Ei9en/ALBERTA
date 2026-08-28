import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone
import math

import numpy as np
import chess
import chess.variant

from distribution_I import W_H, W_U, W_HU


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats_oracle_21-30.json"
)

QUEUE_FILE = (
    PROJECT_ROOT
    / "data"
    / "oracle_queue_21-30.jsonl"
)


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
            and
            sorted_values[end]
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

        side = fen.split()[1]

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
    # Highest I first
    # --------------------------------------------------------

    if mode == "high":

        order = np.argsort(
            I,
            kind="stable"
        )[::-1]

        description = (
            "Highest I (top 0.02%)"
        )

    # --------------------------------------------------------
    # LOW
    # Lowest I first
    # --------------------------------------------------------

    elif mode == "low":

        order = np.argsort(
            I,
            kind="stable"
        )

        description = (
            "Lowest I (bottom 0.02%)"
        )

    # --------------------------------------------------------
    # MIDDLE
    # Start at median and expand outward
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
            "Middle I (around median, ±0.01%)"
        )

    else:

        raise ValueError(
            f"Unknown selection mode: {mode}"
        )

    return order, description


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
    # IMPORTANT:
    #
    # Budget is 0.02% of the ORIGINAL dataset.
    #
    # ceil() guarantees that we reach the requested budget.
    # --------------------------------------------------------

    target_count = max(
        1,
        math.ceil(
            n * 0.0002
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
    # Examine candidates in I priority order.
    #
    # Legal move count is calculated ONLY when the position
    # is reached as a candidate.
    # --------------------------------------------------------

    for idx in candidate_order:

        checked += 1

        record = data[idx]

        # ----------------------------------------------------
        # Check duplicate BEFORE expensive chess processing
        # ----------------------------------------------------

        query_id = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            record["fen"]
        ).hex

        if query_id in existing_ids:

            rejected_duplicates += 1

            continue

        # ----------------------------------------------------
        # Check legal move count
        # ----------------------------------------------------

        legal_count = count_legal_moves(
            record["fen"]
        )

        if legal_count <= 1:

            rejected_moves += 1

            continue

        # ----------------------------------------------------
        # Valid new annotation
        # ----------------------------------------------------

        selected.append(
            idx
        )

        # IMPORTANT:
        # Add immediately so the same FEN cannot be selected
        # twice during this run.
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
            "to reach the requested 0.02% annotation budget."
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
        "Seed ALBERTA oracle queue using active learning score."
    )

    parser.add_argument(
        "--mode",
        choices=[
            "high",
            "low",
            "middle"
        ],
        default="high",
        help=
        "Selection mode: "
        "high = highest I, "
        "low = lowest I, "
        "middle = around median."
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "ALBERTA - SEED ORACLE QUEUE"
    )
    print("=" * 70)

    print()
    print(
        f"Selection mode : {args.mode}"
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

    U_norm = normalize_side_aware(
        U,
        sides
    )

    HU_norm = normalize_side_aware(
        HU,
        sides
    )

    # --------------------------------------------------------
    # Compute I
    # --------------------------------------------------------

    I = (
        W_H * H_norm
        +
        W_U * U_norm
        +
        W_HU * HU_norm
    )

    print(
        "Active learning score computed."
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
        f"Budget               : "
        f"{100 * len(selected_indices) / len(data):.5f}%"
    )

    print(
        f"I min selected       : "
        f"{selected_I.min():.9f}"
    )

    print(
        f"I max selected       : "
        f"{selected_I.max():.9f}"
    )

    print(
        f"I mean selected      : "
        f"{selected_I.mean():.9f}"
    )

    print(
        f"I median selected    : "
        f"{np.median(selected_I):.9f}"
    )

    # --------------------------------------------------------
    # Append to queue
    # --------------------------------------------------------

    added = 0

    with open(
        QUEUE_FILE,
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
        f"File     : {QUEUE_FILE}"
    )


if __name__ == "__main__":

    main()