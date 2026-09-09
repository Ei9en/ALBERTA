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

# Fixed Active Learning budget:
# 0.02% of the dataset.
AL_BUDGET = 0.0002


# ------------------------------------------------------------
# Non-linear acquisition function
#
# I(H,U) = H^alpha / (1 + beta * U)
#          + gamma * H * U
#
# Last-stand configuration:
#
#     alpha = 1
#     beta  = 1
#     gamma = 0.5
#
# H and U are used on their RAW scale.
# ------------------------------------------------------------

ALPHA = 0.75
BETA = 2000
GAMMA = 2


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
# Non-linear acquisition function
# ============================================================

def compute_acquisition_score(
    H,
    U,
):

    H = np.asarray(
        H,
        dtype=np.float64
    )

    U = np.asarray(
        U,
        dtype=np.float64
    )

    if len(H) != len(U):

        raise ValueError(
            "H and U must have the same length."
        )

    if np.any(~np.isfinite(H)):

        raise ValueError(
            "H contains non-finite values."
        )

    if np.any(~np.isfinite(U)):

        raise ValueError(
            "U contains non-finite values."
        )

    denominator = (
        1.0
        +
        BETA * U
    )

    if np.any(denominator <= 0):

        raise ValueError(
            "Invalid denominator in acquisition function."
        )

    # --------------------------------------------------------
    # First term
    #
    # H^alpha / (1 + beta * U)
    # --------------------------------------------------------

    first_term = (
        np.power(
            H,
            ALPHA
        )
        /
        denominator
    )

    # --------------------------------------------------------
    # Second term
    #
    # gamma * H * U
    # --------------------------------------------------------

    interaction_term = (
        GAMMA
        * np.sqrt(H*U)
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    I = (
        first_term
        +
        interaction_term
    )

    return (
        I,
        first_term,
        interaction_term
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
            "Highest non-linear I"
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
            "Lowest non-linear I"
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
            "Middle non-linear I (around median)"
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

    return (
        order,
        description
    )


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
            "oracle_queue_1-10_AL_nonlinear.jsonl"
        )

    elif mode == "low":

        filename = (
            "oracle_queue_1-10_low_nonlinear.jsonl"
        )

    elif mode == "middle":

        filename = (
            "oracle_queue_1-10_middle_nonlinear.jsonl"
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
        "non-linear active-learning score or uniform "
        "random sampling."
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
        "NON-LINEAR ACQUISITION FUNCTION"
    )

    print(
        "I(H,U) = H^alpha / (1 + beta*U) "
        "+ gamma*H*U"
    )

    print(
        f"alpha          : {ALPHA:.6f}"
    )

    print(
        f"beta           : {BETA:.6f}"
    )

    print(
        f"gamma          : {GAMMA:.6f}"
    )

    print(
        f"AL budget      : "
        f"{100 * AL_BUDGET:.5f}%"
    )

    print()
    print(
        "H and U are used on their RAW scale."
    )

    print(
        "No log transform."
    )

    print(
        "No OLS coefficients."
    )

    print(
        "No predictor standardization."
    )

    print(
        "No side-aware percentile normalization."
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

    sides = extract_side(
        fens
    )

    # --------------------------------------------------------
    # Compute non-linear acquisition score
    # --------------------------------------------------------

    print()
    print(
        "Computing non-linear acquisition score..."
    )

    (
        I,
        first_term,
        interaction_term
    ) = compute_acquisition_score(
        H,
        U
    )

    # --------------------------------------------------------
    # Final min-max normalization
    #
    # ONLY for representation.
    #
    # Ranking is performed on I directly.
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
        f"I mean      : {I.mean():+.9f}"
    )

    print(
        f"I median    : {np.median(I):+.9f}"
    )

    print(
        f"I_norm min  : {I_norm.min():.9f}"
    )

    print(
        f"I_norm max  : {I_norm.max():.9f}"
    )

    # --------------------------------------------------------
    # Score decomposition
    # --------------------------------------------------------

    print()
    print(
        "SCORE DECOMPOSITION"
    )
    print("-" * 70)

    print(
        f"First term mean       : "
        f"{first_term.mean():+.9f}"
    )

    print(
        f"First term median     : "
        f"{np.median(first_term):+.9f}"
    )

    print(
        f"Interaction mean      : "
        f"{interaction_term.mean():+.9f}"
    )

    print(
        f"Interaction median    : "
        f"{np.median(interaction_term):+.9f}"
    )

    print(
        f"Interaction / total   : "
        f"{interaction_term.sum() / I.sum():.2%}"
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

    selected_first_term = (
        first_term[
            selected_indices
        ]
    )

    selected_interaction = (
        interaction_term[
            selected_indices
        ]
    )

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
        f"First term mean     : "
        f"{selected_first_term.mean():+.9f}"
    )

    print(
        f"Interaction mean    : "
        f"{selected_interaction.mean():+.9f}"
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
                    float(H[idx] * U[idx]),

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