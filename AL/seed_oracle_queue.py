from __future__ import annotations

import argparse
import json
import math
import uuid

from datetime import datetime, timezone
from pathlib import Path

import chess.variant
import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

DEFAULT_QUEUE_DIR = (
    PROJECT_ROOT
    / "data"
    / "queue"
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_AL_BUDGET = 0.0001
DEFAULT_SEED = 42


# ============================================================
# Query ID
# ============================================================

def fen_to_query_id(
    fen: str,
) -> str:
    """
    Return the deterministic ALBERTA query identifier
    associated with a FEN.
    """

    return uuid.uuid5(
        uuid.NAMESPACE_DNS,
        fen,
    ).hex


# ============================================================
# Percentile-rank normalization
# ============================================================

def percentile_rank(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n < 2:
        raise ValueError(
            "Not enough values for percentile-rank normalization."
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
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        rank = (
            start
            + end
            - 1
        ) / 2.0

        ranks[
            order[
                start:end
            ]
        ] = (
            rank
            / (n - 1)
        )

        start = end

    return ranks


# ============================================================
# Side to move
# ============================================================

def extract_side(
    fens: np.ndarray,
) -> np.ndarray:

    sides = []

    for fen in fens:

        parts = fen.split()

        if len(parts) < 2:
            raise ValueError(
                f"Invalid FEN: {fen}"
            )

        side = parts[1]

        if side not in {
            "w",
            "b",
        }:
            raise ValueError(
                f"Invalid side-to-move field in FEN: {fen}"
            )

        sides.append(
            side
        )

    return np.asarray(
        sides,
        dtype="<U1",
    )


# ============================================================
# Atomic legal moves
# ============================================================

def count_legal_moves(
    fen: str,
) -> int:

    try:

        board = (
            chess.variant.AtomicBoard(
                fen
            )
        )

        return (
            board.legal_moves.count()
        )

    except Exception as exc:

        raise ValueError(
            "Could not parse Atomic Chess FEN:\n"
            f"{fen}\n"
            f"Error: {exc}"
        ) from exc


# ============================================================
# Side-aware normalization
# ============================================================

def normalize_temporal_side_aware(
    values: np.ndarray,
    sides: np.ndarray,
    n_windows: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Percentile-rank normalization conditional on:

        - chronological window
        - side to move

    Records are assumed to preserve chronological append order.

    The full dataset is divided into n_windows contiguous,
    approximately equal-size windows.

    Within each (window, side) stratum, values are independently
    converted to percentile ranks in [0, 1].
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sides = np.asarray(
        sides
    )

    n = len(
        values
    )

    if len(
        sides
    ) != n:

        raise ValueError(
            "values and sides must have identical lengths."
        )

    if n_windows < 1:

        raise ValueError(
            "n_windows must be >= 1."
        )

    if n < n_windows:

        raise ValueError(
            "Not enough observations for requested temporal windows."
        )

    normalized = np.empty_like(
        values,
        dtype=np.float64,
    )

    window_ids = np.empty(
        n,
        dtype=np.int64,
    )

    edges = np.linspace(
        0,
        n,
        n_windows + 1,
        dtype=np.int64,
    )

    for window in range(
        n_windows
    ):

        start = int(
            edges[
                window
            ]
        )

        end = int(
            edges[
                window + 1
            ]
        )

        window_ids[
            start:end
        ] = window

        for side in (
            "w",
            "b",
        ):

            side_mask = (
                sides[
                    start:end
                ]
                == side
            )

            count = int(
                np.sum(
                    side_mask
                )
            )

            if count < 2:

                raise ValueError(
                    f"Not enough observations in "
                    f"window {window + 1}, side {side}: "
                    f"{count}"
                )

            local_values = values[
                start:end
            ][
                side_mask
            ]

            local_ranks = percentile_rank(
                local_values
            )

            local_indices = np.flatnonzero(
                side_mask
            )

            normalized[
                start
                + local_indices
            ] = local_ranks

    return (
        normalized,
        window_ids,
    )


# ============================================================
# Min-max normalization
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
# Candidate order
# ============================================================

def build_candidate_order(
    scores: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    str,
]:

    n = len(
        scores
    )

    if n == 0:
        raise ValueError(
            "No positions available."
        )

    # --------------------------------------------------------
    # Highest score
    # --------------------------------------------------------

    if mode == "high":

        order = (
            np.argsort(
                scores,
                kind="stable",
            )[::-1]
        )

        description = (
            "Highest I"
        )

    # --------------------------------------------------------
    # Lowest score
    # --------------------------------------------------------

    elif mode == "low":

        order = np.argsort(
            scores,
            kind="stable",
        )

        description = (
            "Lowest I"
        )

    # --------------------------------------------------------
    # Around median
    # --------------------------------------------------------

    elif mode == "middle":

        sorted_indices = np.argsort(
            scores,
            kind="stable",
        )

        center = (
            n // 2
        )

        order_list = []

        left = (
            center - 1
        )

        right = center

        while (
            left >= 0
            or right < n
        ):

            if right < n:

                order_list.append(
                    sorted_indices[
                        right
                    ]
                )

                right += 1

            if left >= 0:

                order_list.append(
                    sorted_indices[
                        left
                    ]
                )

                left -= 1

        order = np.asarray(
            order_list,
            dtype=np.int64,
        )

        description = (
            "Middle I (around median)"
        )

    # --------------------------------------------------------
    # Uniform random
    #
    # Acquisition score is deliberately ignored.
    # --------------------------------------------------------

    elif mode == "random":

        order = rng.permutation(
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
        description,
    )


# ============================================================
# Default queue path
# ============================================================

def get_queue_file(
    queue_dir: Path,
    mode: str,
) -> Path:

    filenames = {
        "random":
            "oracle_queue_1-10_random.jsonl",

        "high":
            "oracle_queue_1-10_AL.jsonl",

        "low":
            "oracle_queue_1-10_low.jsonl",

        "middle":
            "oracle_queue_1-10_middle.jsonl",
    }

    try:

        filename = filenames[
            mode
        ]

    except KeyError as exc:

        raise ValueError(
            f"Unknown selection mode: {mode}"
        ) from exc

    return (
        queue_dir
        / filename
    )


# ============================================================
# Existing queue IDs
# ============================================================

def load_existing_ids(
    queue_file: Path,
) -> set[str]:

    if not queue_file.exists():
        return set()

    existing_ids = set()

    with queue_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSONL in {queue_file} "
                    f"at line {line_number}."
                ) from exc

            query_id = item.get(
                "query_id"
            )

            if query_id is None:
                raise RuntimeError(
                    f"Missing query_id in {queue_file} "
                    f"at line {line_number}."
                )

            existing_ids.add(
                query_id
            )

    return existing_ids


# ============================================================
# Position selection
# ============================================================

def select_positions(
    data: list[dict],
    scores: np.ndarray,
    mode: str,
    existing_ids: set[str],
    budget: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    str,
    int,
    int,
    int,
]:

    n = len(
        scores
    )

    if n == 0:
        raise ValueError(
            "No positions available."
        )

    if not (
        0.0
        < budget
        <= 1.0
    ):
        raise ValueError(
            "budget must satisfy 0 < budget <= 1."
        )

    # --------------------------------------------------------
    # Annotation budget
    # --------------------------------------------------------

    target_count = max(
        1,
        math.ceil(
            n
            * budget
        ),
    )

    candidate_order, description = (
        build_candidate_order(
            scores=scores,
            mode=mode,
            rng=rng,
        )
    )

    selected = []

    checked = 0
    rejected_moves = 0
    rejected_duplicates = 0

    # --------------------------------------------------------
    # Examine candidates in priority order
    # --------------------------------------------------------

    for idx in candidate_order:

        checked += 1

        record = data[
            int(
                idx
            )
        ]

        fen = record[
            "fen"
        ]

        query_id = fen_to_query_id(
            fen
        )

        # ----------------------------------------------------
        # Existing position
        # ----------------------------------------------------

        if query_id in existing_ids:

            rejected_duplicates += 1

            continue

        # ----------------------------------------------------
        # Trivial / forced position
        # ----------------------------------------------------

        legal_count = count_legal_moves(
            fen
        )

        if legal_count <= 1:

            rejected_moves += 1

            continue

        # ----------------------------------------------------
        # Valid candidate
        # ----------------------------------------------------

        selected.append(
            int(
                idx
            )
        )

        existing_ids.add(
            query_id
        )

        if len(
            selected
        ) >= target_count:
            break

    if len(
        selected
    ) < target_count:

        raise RuntimeError(
            "Could not find enough new eligible positions "
            "to reach the requested annotation budget."
        )

    return (
        np.asarray(
            selected,
            dtype=np.int64,
        ),
        description,
        checked,
        rejected_moves,
        rejected_duplicates,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Seed an ALBERTA Oracle queue using the frozen "
            "active-learning score or uniform random sampling."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "high",
            "low",
            "middle",
            "random",
        ],
        default="high",
        help=(
            "Selection mode: high = highest I, "
            "low = lowest I, middle = around median, "
            "random = uniform random."
        ),
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Input uncertainty-statistics JSON file.",
    )

    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=DEFAULT_QUEUE_DIR,
        help="Directory containing Oracle queues.",
    )

    parser.add_argument(
        "--queue-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit output queue path. "
            "Overrides the mode-specific default."
        ),
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_AL_BUDGET,
        help=(
            "Fraction of input observations to annotate. "
            "Default: 0.0001 = 0.01%%."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Master seed used for random acquisition. "
            "Default: 42."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.seed < 0:
        raise ValueError(
            "--seed must be non-negative."
        )

    if not (
        0.0
        < args.budget
        <= 1.0
    ):
        raise ValueError(
            "--budget must satisfy 0 < budget <= 1."
        )

    rng = np.random.default_rng(
        args.seed
    )

    queue_file = (
        args.queue_file
        if args.queue_file is not None
        else get_queue_file(
            queue_dir=args.queue_dir,
            mode=args.mode,
        )
    )

    # ========================================================
    # Load AL weights only when acquisition uses I
    #
    # Random acquisition must remain completely independent
    # from the fitted active-learning score.
    # ========================================================

    if args.mode != "random":

        from AL_weights import (
            RAW_W_H,
            RAW_W_HU,
            RAW_W_U,
            TEMPORAL_WINDOWS,
        )

    else:

        RAW_W_H = None
        RAW_W_U = None
        RAW_W_HU = None
        TEMPORAL_WINDOWS = None

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 70)
    print("ALBERTA - SEED ORACLE QUEUE")
    print("=" * 70)

    print(
        f"Selection mode : {args.mode}"
    )

    print(
        f"Input file     : {args.data_file}"
    )

    print(
        f"Queue file     : {queue_file}"
    )

    print(
        f"Master seed    : {args.seed}"
    )

    print(
        f"AL budget      : {100 * args.budget:.5f}%"
    )

    # ========================================================
    # Acquisition-score configuration
    # ========================================================

    print()
    print("ACQUISITION SCORE")
    print("-" * 70)

    if args.mode == "random":

        print(
            "Random acquisition: I is not computed."
        )

        print(
            "AL_weights.py is not imported or used."
        )

    else:

        print(
            f"TEMPORAL WINDOWS            : {TEMPORAL_WINDOWS:.6f}"
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
            "Coefficients are used at their original OLS scale; "
            "no coefficient normalization."
        )

    # ========================================================
    # Load uncertainty data
    # ========================================================

    print()
    print("Loading uncertainty statistics...")

    if not args.data_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.data_file}"
        )

    with args.data_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Input uncertainty file must contain a JSON list."
        )

    if len(
        data
    ) == 0:
        raise RuntimeError(
            "No positions found."
        )

    print(
        f"Positions loaded : {len(data):,}"
    )

    # ========================================================
    # Raw signals
    # ========================================================

    fens = np.asarray(
        [
            record[
                "fen"
            ]
            for record in data
        ],
        dtype=object,
    )

    H = np.asarray(
        [
            record[
                "H"
            ]
            for record in data
        ],
        dtype=np.float64,
    )

    U = np.asarray(
        [
            record[
                "U"
            ]
            for record in data
        ],
        dtype=np.float64,
    )

    HU = np.asarray(
        [
            record[
                "HU"
            ]
            for record in data
        ],
        dtype=np.float64,
    )

    sides = extract_side(
        fens
    )

    # ========================================================
    # Acquisition score
    # ========================================================

    if args.mode == "random":

        # ----------------------------------------------------
        # Random selection does not use I.
        #
        # build_candidate_order() only needs an array whose
        # length equals the number of records.
        # ----------------------------------------------------

        I = np.zeros(
            len(data),
            dtype=np.float64,
        )

        I_norm = np.zeros(
            len(data),
            dtype=np.float64,
        )

    else:

        # ====================================================
        # Temporal-window x side-to-move percentile ranks
        #
        # IMPORTANT:
        # This must remain identical to the normalization used
        # when fitting RAW_W_*.
        # ====================================================

        (
            H_norm,
            window_ids,
        ) = normalize_temporal_side_aware(
            values=H,
            sides=sides,
            n_windows=TEMPORAL_WINDOWS,
        )

        (
            U_norm,
            window_ids_u,
        ) = normalize_temporal_side_aware(
            values=U,
            sides=sides,
            n_windows=TEMPORAL_WINDOWS,
        )

        if not np.array_equal(
            window_ids,
            window_ids_u,
        ):
            raise RuntimeError(
                "Temporal-window assignments are inconsistent."
            )

        # ====================================================
        # Factorial interaction
        # ====================================================

        HU_interaction = (
            H_norm
            * U_norm
        )

        # ====================================================
        # Standardization
        #
        # This is performed on the same complete RL1-10
        # reference population used for coefficient fitting.
        # ====================================================

        H_mean = float(
            H_norm.mean()
        )

        H_std = float(
            H_norm.std()
        )

        U_mean = float(
            U_norm.mean()
        )

        U_std = float(
            U_norm.std()
        )

        HU_mean = float(
            HU_interaction.mean()
        )

        HU_std = float(
            HU_interaction.std()
        )

        if H_std <= 0.0:
            raise ValueError(
                "H normalized standard deviation is zero."
            )

        if U_std <= 0.0:
            raise ValueError(
                "U normalized standard deviation is zero."
            )

        if HU_std <= 0.0:
            raise ValueError(
                "H*U normalized standard deviation is zero."
            )

        H_star = (
            H_norm
            - H_mean
        ) / H_std

        U_star = (
            U_norm
            - U_mean
        ) / U_std

        HU_star = (
            HU_interaction
            - HU_mean
        ) / HU_std

        # ====================================================
        # Canonical I score
        # ====================================================

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

        print()
        print("Active-learning score computed.")

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

    # ========================================================
    # Existing queue
    # ========================================================

    existing_ids = load_existing_ids(
        queue_file
    )

    print()
    print(
        f"Existing queue entries : {len(existing_ids):,}"
    )

    # ========================================================
    # Selection
    # ========================================================

    (
        selected_indices,
        description,
        checked,
        rejected_moves,
        rejected_duplicates,
    ) = select_positions(
        data=data,
        scores=I,
        mode=args.mode,
        existing_ids=existing_ids,
        budget=args.budget,
        rng=rng,
    )

    if args.mode != "random":

        selected_I = I[
            selected_indices
        ]

        selected_I_norm = I_norm[
            selected_indices
        ]

    else:

        selected_I = None
        selected_I_norm = None

    # ========================================================
    # Selection report
    # ========================================================

    print()
    print("ACTIVE LEARNING SELECTION")
    print("-" * 70)

    print(
        f"Mode                 : {description}"
    )

    print(
        f"Target annotations   : {len(selected_indices):,}"
    )

    print(
        f"Candidates checked   : {checked:,}"
    )

    print(
        f"Rejected (<=1 move)  : {rejected_moves:,}"
    )

    print(
        f"Rejected (duplicate) : {rejected_duplicates:,}"
    )

    print(
        f"Budget               : "
        f"{100 * len(selected_indices) / len(data):.5f}%"
    )

    if args.mode != "random":

        print(
            f"I min selected       : {selected_I.min():+.9f}"
        )

        print(
            f"I max selected       : {selected_I.max():+.9f}"
        )

        print(
            f"I mean selected      : {selected_I.mean():+.9f}"
        )

        print(
            f"I median selected    : {np.median(selected_I):+.9f}"
        )

        print(
            f"I_norm min selected  : {selected_I_norm.min():.9f}"
        )

        print(
            f"I_norm max selected  : {selected_I_norm.max():.9f}"
        )

    else:

        print(
            "I statistics         : N/A (random acquisition)"
        )

    # ========================================================
    # Side-to-move report
    # ========================================================

    selected_sides = sides[
        selected_indices
    ]

    global_white = int(
        np.sum(
            sides == "w"
        )
    )

    global_black = int(
        np.sum(
            sides == "b"
        )
    )

    selected_white = int(
        np.sum(
            selected_sides == "w"
        )
    )

    selected_black = int(
        np.sum(
            selected_sides == "b"
        )
    )

    print()
    print("SIDE-TO-MOVE")
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

    if global_white > 0:

        white_enrichment = (
            selected_white
            / len(
                selected_indices
            )
        ) / (
            global_white
            / len(
                data
            )
        )

        print(
            f"White enrichment  : {white_enrichment:.3f}x"
        )

    if global_black > 0:

        black_enrichment = (
            selected_black
            / len(
                selected_indices
            )
        ) / (
            global_black
            / len(
                data
            )
        )

        print(
            f"Black enrichment  : {black_enrichment:.3f}x"
        )

    # ========================================================
    # Append selected positions
    # ========================================================

    queue_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    added = 0

    with queue_file.open(
        "a",
        encoding="utf-8",
    ) as f:

        for idx in selected_indices:

            idx = int(
                idx
            )

            record = data[
                idx
            ]

            fen = record[
                "fen"
            ]

            query_id = (
                fen_to_query_id(
                    fen
                )
            )

            item = {
                "query_id":
                    query_id,

                "fen":
                    fen,

                "H":
                    float(
                        H[
                            idx
                        ]
                    ),

                "U":
                    float(
                        U[
                            idx
                        ]
                    ),

                "HU":
                    float(
                        HU[
                            idx
                        ]
                    ),

                # Random acquisition has no meaningful I score.
                "score":
                    (
                        None
                        if args.mode == "random"
                        else float(
                            I[
                                idx
                            ]
                        )
                    ),

                "I_norm":
                    (
                        None
                        if args.mode == "random"
                        else float(
                            I_norm[
                                idx
                            ]
                        )
                    ),

                # No single threshold is meaningful for all
                # acquisition modes.
                "threshold":
                    None,

                "model":
                    str(
                        record.get(
                            "model",
                            "historical_json",
                        )
                    ),

                "epoch":
                    int(
                        record.get(
                            "epoch",
                            -1,
                        )
                    ),

                "game_id":
                    int(
                        record.get(
                            "game_id",
                            -1,
                        )
                    ),

                "ply":
                    int(
                        record.get(
                            "ply",
                            -1,
                        )
                    ),

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
                    item,
                    ensure_ascii=False,
                )
            )

            f.write(
                "\n"
            )

            added += 1

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 70)
    print("QUEUE UPDATED")
    print("=" * 70)

    print(
        f"Selected : {len(selected_indices):,}"
    )

    print(
        f"Added    : {added:,}"
    )

    print(
        f"Budget   : {100 * added / len(data):.5f}%"
    )

    print(
        f"File     : {queue_file}"
    )


if __name__ == "__main__":
    main()