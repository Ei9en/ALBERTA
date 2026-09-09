#!/usr/bin/env python3

"""
ALBERTA - Pareto Selection
==========================

Pipeline
--------

1. Load the 200 measured Pareto probes:

       (H, U, HU, score)
            ->
       (Delta_KL, Delta_V)

2. Cross-validate two surrogates:

       score only
       versus
       H + U + HU + score

3. Fit the rich surrogate on all probe observations.

4. Recompute the exact ALBERTA score I on the complete
   uncertainty pool.

5. Remove:
       - invalid / <= 1 legal move positions
       - duplicate FENs
       - the 200 probe positions

6. Predict:

       Delta_KL_hat
       Delta_V_hat

7. Compute exact 2D Pareto layers.

8. Select exactly BUDGET positions using:

       non-dominated sorting
       +
       crowding distance on the last front

9. Write an HMI-compatible Oracle queue with EXACTLY the
   standard schema. No Pareto-specific fields are added.

Output
------

    checkpoints/queue/oracle_queue_1-10_pareto.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path

import chess.variant
import numpy as np

from scipy.stats import spearmanr

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    cross_val_predict,
)


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# ALBERTA score parameters
# ============================================================

from data.uncertainty_analysis.active_learning_weights import (
    RAW_W_H,
    RAW_W_U,
    RAW_W_HU,
    TAU,
)


# ============================================================
# Paths
# ============================================================

PROBE_RESPONSE_PATH = (
    PROJECT_ROOT
    / "pareto"
    / "pareto_local_responses.jsonl"
)

PROBE_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)

POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

OUTPUT_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto.jsonl"
)


# ============================================================
# Experiment configuration
# ============================================================

BUDGET = 246

RANDOM_STATE = 42

CV_FOLDS = 5

N_TREES = 300

PREDICT_CHUNK_SIZE = 100000


# ============================================================
# Features
# ============================================================

SCORE_ONLY_FEATURES = [
    "score",
]

RICH_FEATURES = [
    "H",
    "U",
    "HU",
    "score",
]


# ============================================================
# JSON loading
# ============================================================

def load_json_records(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(
            f
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
        ):

            if (
                key in payload
                and isinstance(
                    payload[key],
                    list,
                )
            ):

                return payload[
                    key
                ]

    raise ValueError(
        f"Unsupported JSON structure in:\n"
        f"{path}"
    )


# ============================================================
# JSONL loading
# ============================================================

def load_jsonl(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    rows = []

    with open(
        path,
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

                rows.append(
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSONL at line "
                    f"{line_number} in {path}"
                ) from exc

    return rows


# ============================================================
# Exact percentile rank
#
# Matches seed_oracle_queue.py:
#
#   average rank for ties
#   divided by n - 1
#
# Output approximately / exactly in [0, 1].
# ============================================================

def percentile_rank(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(
        values
    )

    if n == 0:

        return np.array(
            [],
            dtype=np.float64,
        )

    if n == 1:

        return np.zeros(
            1,
            dtype=np.float64,
        )

    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[
        order
    ]

    # --------------------------------------------------------
    # Find equal-value groups
    # --------------------------------------------------------

    change = np.empty(
        n,
        dtype=bool,
    )

    change[0] = True

    change[1:] = (
        sorted_values[1:]
        !=
        sorted_values[:-1]
    )

    starts = np.flatnonzero(
        change
    )

    ends = np.concatenate(
        (
            starts[1:],
            np.array(
                [n],
                dtype=np.int64,
            ),
        )
    )

    ranked_sorted = np.empty(
        n,
        dtype=np.float64,
    )

    for start, end in zip(
        starts,
        ends,
    ):

        average_rank = (
            start
            +
            end
            -
            1
        ) / 2.0

        ranked_sorted[
            start:end
        ] = (
            average_rank
            /
            (n - 1)
        )

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    ranks[
        order
    ] = ranked_sorted

    return ranks


# ============================================================
# Side to move
# ============================================================

def extract_side(
    fen,
):

    fields = fen.split()

    if len(fields) < 2:

        raise ValueError(
            f"Invalid FEN:\n{fen}"
        )

    side = fields[1]

    if side not in (
        "w",
        "b",
    ):

        raise ValueError(
            f"Invalid side to move in FEN:\n"
            f"{fen}"
        )

    return side


# ============================================================
# Side-aware percentile normalization
# ============================================================

def side_aware_percentile(
    values,
    sides,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sides = np.asarray(
        sides
    )

    result = np.empty(
        len(values),
        dtype=np.float64,
    )

    for side in (
        "w",
        "b",
    ):

        mask = (
            sides
            ==
            side
        )

        if not np.any(
            mask
        ):

            continue

        result[
            mask
        ] = percentile_rank(
            values[
                mask
            ]
        )

    return result


# ============================================================
# Min-max normalize
# ============================================================

def minmax_normalize(
    values,
):

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

        return np.zeros_like(
            values
        )

    return (
        values
        -
        minimum
    ) / (
        maximum
        -
        minimum
    )


# ============================================================
# Compute exact ALBERTA score
# ============================================================

def compute_scores(
    records,
):

    print(
        "\nComputing exact ALBERTA I score...",
        flush=True,
    )

    n = len(
        records
    )

    H = np.empty(
        n,
        dtype=np.float64,
    )

    U = np.empty(
        n,
        dtype=np.float64,
    )

    HU = np.empty(
        n,
        dtype=np.float64,
    )

    sides = np.empty(
        n,
        dtype="<U1",
    )

    for i, record in enumerate(
        records
    ):

        try:

            H[i] = float(
                record["H"]
            )

            U[i] = float(
                record["U"]
            )

            HU[i] = float(
                record["HU"]
            )

            sides[i] = extract_side(
                record["fen"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"Invalid uncertainty record "
                f"at index {i}"
            ) from exc

    # ========================================================
    # H
    # ========================================================

    H_norm = side_aware_percentile(
        H,
        sides,
    )

    # ========================================================
    # U
    # ========================================================

    U_log = np.log1p(
        U
        /
        TAU
    )

    U_log_norm = (
        side_aware_percentile(
            U_log,
            sides,
        )
    )

    # ========================================================
    # Interaction
    # ========================================================

    HU_log_norm = (
        H_norm
        *
        U_log_norm
    )

    # ========================================================
    # Global z-scores
    #
    # np.std default:
    # ddof = 0
    # ========================================================

    H_std = np.std(
        H_norm
    )

    U_std = np.std(
        U_log_norm
    )

    HU_std = np.std(
        HU_log_norm
    )

    if (
        H_std == 0
        or U_std == 0
        or HU_std == 0
    ):

        raise RuntimeError(
            "Zero variance encountered "
            "while computing I."
        )

    H_star = (
        H_norm
        -
        np.mean(
            H_norm
        )
    ) / H_std

    U_log_star = (
        U_log_norm
        -
        np.mean(
            U_log_norm
        )
    ) / U_std

    HU_log_star = (
        HU_log_norm
        -
        np.mean(
            HU_log_norm
        )
    ) / HU_std

    # ========================================================
    # Raw score
    # ========================================================

    score = (
        RAW_W_H
        *
        H_star

        +

        RAW_W_U
        *
        U_log_star

        +

        RAW_W_HU
        *
        HU_log_star
    )

    I_norm = minmax_normalize(
        score
    )

    return {
        "H":
            H,

        "U":
            U,

        "HU":
            HU,

        "score":
            score,

        "I_norm":
            I_norm,
    }


# ============================================================
# Probe dataset
# ============================================================

def load_probe_dataset():

    rows = load_jsonl(
        PROBE_RESPONSE_PATH
    )

    required = {
        "query_id",
        "fen",
        "H",
        "U",
        "HU",
        "score",
        "delta_kl",
        "delta_v",
    }

    cleaned = []

    for i, row in enumerate(
        rows
    ):

        missing = (
            required
            -
            set(
                row.keys()
            )
        )

        if missing:

            raise ValueError(
                f"Probe response {i} "
                f"is missing fields: "
                f"{sorted(missing)}"
            )

        delta_kl = float(
            row["delta_kl"]
        )

        delta_v = float(
            row["delta_v"]
        )

        if (
            not np.isfinite(
                delta_kl
            )
            or
            not np.isfinite(
                delta_v
            )
        ):

            raise ValueError(
                f"Non-finite response "
                f"at probe index {i}"
            )

        if (
            delta_kl < 0
            or delta_v < 0
        ):

            raise ValueError(
                f"Negative response "
                f"at probe index {i}"
            )

        cleaned.append(
            row
        )

    if len(
        cleaned
    ) < 20:

        raise RuntimeError(
            "Too few Pareto probe responses."
        )

    return cleaned


# ============================================================
# Feature matrix
# ============================================================

def make_feature_matrix(
    rows,
    feature_names,
):

    return np.asarray(
        [
            [
                float(
                    row[name]
                )
                for name
                in feature_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


# ============================================================
# Reconstruct original I probe groups
#
# The Pareto probe was sampled as:
#
#     40 I bins x 5 random positions
#
# The HMI queue intentionally contains no Pareto-specific
# metadata, so we reconstruct these groups by sorting the
# probes by the raw score I and grouping consecutive probes.
#
# This is only used for cross-validation.
# ============================================================

def make_i_groups(
    rows,
    n_groups=40,
):

    n = len(
        rows
    )

    if n < n_groups:

        raise ValueError(
            f"Cannot create {n_groups} I groups "
            f"from only {n} probes."
        )

    scores = np.asarray(
        [
            float(
                row["score"]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    order = np.argsort(
        scores,
        kind="stable",
    )

    groups = np.empty(
        n,
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # Split sorted probes into contiguous regions of I.
    #
    # With 200 probes / 40 groups this gives exactly
    # 5 observations per group.
    # --------------------------------------------------------

    chunks = np.array_split(
        order,
        n_groups,
    )

    for group_id, indices in enumerate(
        chunks
    ):

        groups[
            indices
        ] = group_id

    return groups

# ============================================================
# Surrogate
# ============================================================

def create_surrogate():

    return ExtraTreesRegressor(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=1.0,
    )


# ============================================================
# Metrics
# ============================================================

def safe_spearman(
    y_true,
    y_pred,
):

    result = spearmanr(
        y_true,
        y_pred,
    )

    rho = result.statistic

    if not np.isfinite(
        rho
    ):

        return float(
            "nan"
        )

    return float(
        rho
    )


def print_target_metrics(
    name,
    y_true,
    y_pred,
):

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rho = safe_spearman(
        y_true,
        y_pred,
    )

    print(
        f"  {name:<10} "
        f"R2={r2:+.4f} "
        f"| MAE={mae:.6e} "
        f"| Spearman={rho:+.4f}"
    )

    return {
        "r2":
            float(
                r2
            ),

        "mae":
            float(
                mae
            ),

        "spearman":
            rho,
    }


# ============================================================
# Cross-validation
#
# mode:
#
#   "random"
#       ordinary shuffled K-fold
#
#   "i_grouped"
#       entire local regions of the I spectrum are held out
#
# ============================================================

def cross_validate_model(
    rows,
    feature_names,
    label,
    mode="random",
):

    X = make_feature_matrix(
        rows,
        feature_names,
    )

    delta_kl = np.asarray(
        [
            float(
                row["delta_kl"]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    delta_v = np.asarray(
        [
            float(
                row["delta_v"]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Log targets
    # --------------------------------------------------------

    Y_log = np.column_stack(
        (
            np.log1p(
                delta_kl
            ),

            np.log1p(
                delta_v
            ),
        )
    )

    # --------------------------------------------------------
    # CV strategy
    # --------------------------------------------------------

    if mode == "random":

        cv = KFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        predicted_log = cross_val_predict(
            create_surrogate(),
            X,
            Y_log,
            cv=cv,
            n_jobs=1,
        )

        cv_description = (
            "random shuffled K-fold"
        )

    elif mode == "i_grouped":

        groups = make_i_groups(
            rows,
            n_groups=40,
        )

        cv = GroupKFold(
            n_splits=CV_FOLDS,
        )

        predicted_log = cross_val_predict(
            create_surrogate(),
            X,
            Y_log,
            cv=cv,
            groups=groups,
            n_jobs=1,
        )

        cv_description = (
            "grouped CV over local I regions"
        )

    else:

        raise ValueError(
            f"Unknown CV mode: {mode}"
        )

    # --------------------------------------------------------
    # Back to original response space
    # --------------------------------------------------------

    predicted = np.expm1(
        predicted_log
    )

    predicted = np.maximum(
        predicted,
        0.0,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print(
        "-" * 70
    )

    print(
        f"CROSS-VALIDATION: {label}"
    )

    print(
        f"Strategy: {cv_description}"
    )

    print(
        f"Features: {feature_names}"
    )

    print(
        "-" * 70
    )

    kl_metrics = (
        print_target_metrics(
            "Delta KL",
            delta_kl,
            predicted[:, 0],
        )
    )

    v_metrics = (
        print_target_metrics(
            "Delta V",
            delta_v,
            predicted[:, 1],
        )
    )

    return {
        "delta_kl":
            kl_metrics,

        "delta_v":
            v_metrics,
    }


# ============================================================
# Fit final surrogate
# ============================================================

def fit_final_surrogate(
    rows,
):

    X = make_feature_matrix(
        rows,
        RICH_FEATURES,
    )

    Y = np.asarray(
        [
            [
                math.log1p(
                    float(
                        row["delta_kl"]
                    )
                ),
                math.log1p(
                    float(
                        row["delta_v"]
                    )
                ),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    model = create_surrogate()

    model.fit(
        X,
        Y,
    )

    return model


# ============================================================
# Probe IDs to exclude from final selection
# ============================================================

def load_probe_query_ids():

    rows = load_jsonl(
        PROBE_QUEUE_PATH
    )

    ids = set()

    for row in rows:

        query_id = row.get(
            "query_id"
        )

        if query_id:

            ids.add(
                query_id
            )

    return ids


# ============================================================
# Eligibility
# ============================================================

def is_eligible_position(
    fen,
):

    try:

        board = (
            chess.variant.AtomicBoard(
                fen
            )
        )

        return (
            board.legal_moves.count()
            >
            1
        )

    except Exception:

        return False


# ============================================================
# Build final candidate pool
# ============================================================

def build_candidate_pool(
    records,
    scores,
    excluded_ids,
):

    print()
    print(
        "Filtering final decision pool...",
        flush=True,
    )

    candidates = []

    seen_ids = set()

    rejected_probe = 0
    rejected_duplicate = 0
    rejected_illegal = 0

    matched_probe_ids = set()

    for i, record in enumerate(
        records
    ):

        if (
            i > 0
            and i % 100000 == 0
        ):

            print(
                f"  checked "
                f"{i:,} / {len(records):,}",
                flush=True,
            )

        fen = record[
            "fen"
        ]

        query_id = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            fen,
        ).hex

        # ----------------------------------------------------
        # Probe leakage exclusion
        # ----------------------------------------------------

        if query_id in excluded_ids:

            rejected_probe += 1

            matched_probe_ids.add(
                query_id
            )

            continue

        # ----------------------------------------------------
        # Duplicate FEN
        # ----------------------------------------------------

        if query_id in seen_ids:

            rejected_duplicate += 1
            continue

        seen_ids.add(
            query_id
        )

        # ----------------------------------------------------
        # Final Oracle eligibility
        # ----------------------------------------------------

        if not is_eligible_position(
            fen
        ):

            rejected_illegal += 1
            continue

        candidates.append(
            {
                "source_index":
                    i,

                "query_id":
                    query_id,

                "fen":
                    fen,

                "H":
                    float(
                        scores["H"][i]
                    ),

                "U":
                    float(
                        scores["U"][i]
                    ),

                "HU":
                    float(
                        scores["HU"][i]
                    ),

                "score":
                    float(
                        scores["score"][i]
                    ),

                "I_norm":
                    float(
                        scores["I_norm"][i]
                    ),
            }
        )

    print()
    print(
        f"Eligible candidates : "
        f"{len(candidates):,}"
    )

    print(
        f"Excluded probe      : "
        f"{rejected_probe:,}"
    )

    print(
        f"Duplicate FEN       : "
        f"{rejected_duplicate:,}"
    )

    print(
        f"<=1 legal move      : "
        f"{rejected_illegal:,}"
    )

    print(
        f"Matched probe IDs   : "
        f"{len(matched_probe_ids):,} "
        f"/ {len(excluded_ids):,}"
    )

    missing_probe_ids = (
        excluded_ids
        -
        matched_probe_ids
    )

    print(
        f"Missing probe IDs   : "
        f"{len(missing_probe_ids):,}"
    )

    if matched_probe_ids:

        print(
            f"Probe row multiplicity: "
            f"{rejected_probe / len(matched_probe_ids):.2f}x"
        )

    return candidates


# ============================================================
# Chunked surrogate prediction
# ============================================================

def predict_candidates(
    model,
    candidates,
):

    n = len(
        candidates
    )

    predicted_kl = np.empty(
        n,
        dtype=np.float64,
    )

    predicted_v = np.empty(
        n,
        dtype=np.float64,
    )

    print()
    print(
        "Projecting candidate pool "
        "into response space...",
        flush=True,
    )

    for start in range(
        0,
        n,
        PREDICT_CHUNK_SIZE,
    ):

        end = min(
            start
            +
            PREDICT_CHUNK_SIZE,
            n,
        )

        chunk = candidates[
            start:end
        ]

        X = make_feature_matrix(
            chunk,
            RICH_FEATURES,
        )

        prediction_log = (
            model.predict(
                X
            )
        )

        prediction = np.expm1(
            prediction_log
        )

        prediction = np.maximum(
            prediction,
            0.0,
        )

        predicted_kl[
            start:end
        ] = prediction[
            :,
            0
        ]

        predicted_v[
            start:end
        ] = prediction[
            :,
            1
        ]

        print(
            f"  predicted "
            f"{end:,} / {n:,}",
            flush=True,
        )

    return (
        predicted_kl,
        predicted_v,
    )


# ============================================================
# Fenwick tree for maximum Pareto depth
# ============================================================

class FenwickMax:

    def __init__(
        self,
        size,
    ):

        self.size = size

        self.tree = np.zeros(
            size + 1,
            dtype=np.int32,
        )

    def query(
        self,
        index,
    ):

        result = 0

        while index > 0:

            value = int(
                self.tree[
                    index
                ]
            )

            if value > result:

                result = value

            index -= (
                index
                &
                -index
            )

        return result

    def update(
        self,
        index,
        value,
    ):

        while index <= self.size:

            if (
                value
                >
                self.tree[
                    index
                ]
            ):

                self.tree[
                    index
                ] = value

            index += (
                index
                &
                -index
            )


# ============================================================
# Exact 2D Pareto ranks
#
# Maximize BOTH objectives.
#
# Rank 1 = non-dominated front.
# Rank 2 = next front.
# ...
#
# O(N log N)
# ============================================================

def pareto_ranks_2d(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    n = len(
        x
    )

    if n != len(
        y
    ):

        raise ValueError(
            "Objective arrays have "
            "different lengths."
        )

    print()
    print(
        "Computing exact 2D Pareto layers...",
        flush=True,
    )

    # --------------------------------------------------------
    # y rank:
    #
    # index 1 = highest y
    # --------------------------------------------------------

    unique_y = np.unique(
        y
    )

    y_index = (
        len(unique_y)
        -
        np.searchsorted(
            unique_y,
            y,
            side="left",
        )
    )

    # --------------------------------------------------------
    # Sort:
    #
    # x descending
    # y descending
    # --------------------------------------------------------

    order = np.lexsort(
        (
            -y,
            -x,
        )
    )

    ranks = np.empty(
        n,
        dtype=np.int32,
    )

    fenwick = FenwickMax(
        len(
            unique_y
        )
    )

    position = 0

    while position < n:

        idx = order[
            position
        ]

        current_x = x[
            idx
        ]

        current_y = y[
            idx
        ]

        end = (
            position
            +
            1
        )

        # ----------------------------------------------------
        # Identical objective vectors do not dominate
        # one another.
        # ----------------------------------------------------

        while end < n:

            idx2 = order[
                end
            ]

            if (
                x[idx2] != current_x
                or
                y[idx2] != current_y
            ):

                break

            end += 1

        yi = int(
            y_index[
                idx
            ]
        )

        layer = (
            fenwick.query(
                yi
            )
            +
            1
        )

        group = order[
            position:end
        ]

        ranks[
            group
        ] = layer

        # ----------------------------------------------------
        # Update only after all identical vectors received
        # the same rank.
        # ----------------------------------------------------

        fenwick.update(
            yi,
            layer,
        )

        position = end

        if (
            position % 100000
            <
            len(group)
        ):

            print(
                f"  ranked "
                f"{position:,} / {n:,}",
                flush=True,
            )

    return ranks


# ============================================================
# Crowding distance
#
# Standard NSGA-II-style distance inside ONE front.
# ============================================================

def crowding_distance(
    indices,
    objective_1,
    objective_2,
):

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    n = len(
        indices
    )

    if n == 0:

        return np.array(
            [],
            dtype=np.float64,
        )

    if n <= 2:

        return np.full(
            n,
            np.inf,
            dtype=np.float64,
        )

    distance = np.zeros(
        n,
        dtype=np.float64,
    )

    objectives = (
        objective_1[
            indices
        ],
        objective_2[
            indices
        ],
    )

    for values in objectives:

        order = np.argsort(
            values,
            kind="stable",
        )

        minimum = values[
            order[0]
        ]

        maximum = values[
            order[-1]
        ]

        distance[
            order[0]
        ] = np.inf

        distance[
            order[-1]
        ] = np.inf

        span = (
            maximum
            -
            minimum
        )

        if span <= 0:

            continue

        interior = order[
            1:-1
        ]

        previous_values = values[
            order[:-2]
        ]

        next_values = values[
            order[2:]
        ]

        distance[
            interior
        ] += (
            next_values
            -
            previous_values
        ) / span

    return distance


# ============================================================
# Budgeted Pareto selection
# ============================================================

def select_budget(
    pareto_ranks,
    predicted_kl,
    predicted_v,
    budget,
):

    selected = []

    max_rank = int(
        np.max(
            pareto_ranks
        )
    )

    print()
    print(
        "Pareto front occupancy:",
        flush=True,
    )

    for rank in range(
        1,
        max_rank + 1,
    ):

        front = np.flatnonzero(
            pareto_ranks
            ==
            rank
        )

        if len(
            front
        ) == 0:

            continue

        remaining = (
            budget
            -
            len(
                selected
            )
        )

        print(
            f"  Front {rank:3d}: "
            f"{len(front):,} positions",
            flush=True,
        )

        if remaining <= 0:

            break

        # ----------------------------------------------------
        # Entire front fits
        # ----------------------------------------------------

        if len(
            front
        ) <= remaining:

            selected.extend(
                front.tolist()
            )

            continue

        # ----------------------------------------------------
        # Last partially included front:
        # preserve objective-space diversity
        # ----------------------------------------------------

        distances = crowding_distance(
            front,
            predicted_kl,
            predicted_v,
        )

        # Deterministic tie-break:
        #
        # 1. larger crowding distance
        # 2. larger predicted KL
        # 3. larger predicted V
        # 4. smaller candidate index
        #
        local_order = sorted(
            range(
                len(front)
            ),
            key=lambda j: (
                -distances[j],
                -predicted_kl[
                    front[j]
                ],
                -predicted_v[
                    front[j]
                ],
                int(
                    front[j]
                ),
            ),
        )

        chosen_local = (
            local_order[
                :remaining
            ]
        )

        selected.extend(
            [
                int(
                    front[j]
                )
                for j in chosen_local
            ]
        )

        break

    if len(
        selected
    ) != budget:

        raise RuntimeError(
            f"Expected exactly {budget} "
            f"selected positions, "
            f"got {len(selected)}."
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


# ============================================================
# Write strict HMI queue
# ============================================================

def write_hmi_queue(
    candidates,
    selected_indices,
    output_path,
    force=False,
):

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not force
    ):

        raise FileExistsError(
            f"Output queue already exists:\n"
            f"{output_path}\n\n"
            f"Use --force to overwrite it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    now = datetime.now(
        timezone.utc
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for offset, idx in enumerate(
            selected_indices
        ):

            candidate = candidates[
                int(
                    idx
                )
            ]

            created_at = (
                now.isoformat()
            )

            item = {
                "query_id":
                    candidate[
                        "query_id"
                    ],

                "fen":
                    candidate[
                        "fen"
                    ],

                "H":
                    float(
                        candidate[
                            "H"
                        ]
                    ),

                "U":
                    float(
                        candidate[
                            "U"
                        ]
                    ),

                "HU":
                    float(
                        candidate[
                            "HU"
                        ]
                    ),

                "score":
                    float(
                        candidate[
                            "score"
                        ]
                    ),

                "I_norm":
                    float(
                        candidate[
                            "I_norm"
                        ]
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

            f.write(
                json.dumps(
                    item
                )
                +
                "\n"
            )


# ============================================================
# Diagnostics
# ============================================================

# ============================================================
# Distribution diagnostics
# ============================================================

def print_quantiles(
    name,
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    percentiles = (
        0,
        10,
        25,
        50,
        75,
        90,
        100,
    )

    quantiles = np.percentile(
        values,
        percentiles,
    )

    print()
    print(
        name
    )

    for percentile, value in zip(
        percentiles,
        quantiles,
    ):

        print(
            f"  p{percentile:03d}: "
            f"{value:.6e}"
        )


def print_probe_correlations(
    rows,
):

    print()
    print(
        "=" * 70
    )

    print(
        "RAW PROBE CORRELATIONS"
    )

    print(
        "=" * 70
    )

    delta_kl = np.asarray(
        [
            float(
                row["delta_kl"]
            )
            for row in rows
        ]
    )

    delta_v = np.asarray(
        [
            float(
                row["delta_v"]
            )
            for row in rows
        ]
    )

    for feature in (
        "H",
        "U",
        "HU",
        "score",
        "I_norm",
    ):

        values = np.asarray(
            [
                float(
                    row[feature]
                )
                for row in rows
            ]
        )

        rho_kl = safe_spearman(
            values,
            delta_kl,
        )

        rho_v = safe_spearman(
            values,
            delta_v,
        )

        print(
            f"{feature:<8} "
            f"| rho(KL)={rho_kl:+.4f} "
            f"| rho(V)={rho_v:+.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ALBERTA Pareto acquisition selection"
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing final Pareto queue."
        ),
    )

    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Run probe diagnostics and CV only. "
            "Do not project the full pool."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "ALBERTA - PARETO SELECTION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Probe responses : "
        f"{PROBE_RESPONSE_PATH}"
    )

    print(
        f"Candidate pool  : "
        f"{POOL_PATH}"
    )

    print(
        f"Budget          : "
        f"{BUDGET}"
    )

    print(
        f"Trees           : "
        f"{N_TREES}"
    )

    print(
        f"CV folds        : "
        f"{CV_FOLDS}"
    )

    # ========================================================
    # Load probe responses
    # ========================================================

    probes = load_probe_dataset()

    print()

    print(
        f"Probe responses loaded: "
        f"{len(probes)}"
    )

    # ========================================================
    # Raw correlations
    # ========================================================

    print_probe_correlations(
        probes
    )

    # ========================================================
    # Random CV
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "RANDOM CROSS-VALIDATION"
    )

    print(
        "=" * 70
    )

    score_metrics_random = (
        cross_validate_model(
            probes,
            SCORE_ONLY_FEATURES,
            "I / score only",
            mode="random",
        )
    )

    rich_metrics_random = (
        cross_validate_model(
            probes,
            RICH_FEATURES,
            "H + U + HU + score",
            mode="random",
        )
    )

    # ========================================================
    # Grouped-I CV
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "GROUPED-I CROSS-VALIDATION"
    )

    print(
        "=" * 70
    )

    score_metrics_grouped = (
        cross_validate_model(
            probes,
            SCORE_ONLY_FEATURES,
            "I / score only",
            mode="i_grouped",
        )
    )

    rich_metrics_grouped = (
        cross_validate_model(
            probes,
            RICH_FEATURES,
            "H + U + HU + score",
            mode="i_grouped",
        )
    )

    # ========================================================
    # Surrogate comparison
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "SURROGATE COMPARISON"
    )

    print(
        "=" * 70
    )

    for target in (
        "delta_kl",
        "delta_v",
    ):

        random_i = (
            score_metrics_random[
                target
            ][
                "spearman"
            ]
        )

        random_rich = (
            rich_metrics_random[
                target
            ][
                "spearman"
            ]
        )

        grouped_i = (
            score_metrics_grouped[
                target
            ][
                "spearman"
            ]
        )

        grouped_rich = (
            rich_metrics_grouped[
                target
            ][
                "spearman"
            ]
        )

        print()

        print(
            target
        )

        print(
            f"  Random CV "
            f"| I={random_i:+.4f} "
            f"| rich={random_rich:+.4f} "
            f"| gain="
            f"{random_rich - random_i:+.4f}"
        )

        print(
            f"  Grouped-I "
            f"| I={grouped_i:+.4f} "
            f"| rich={grouped_rich:+.4f} "
            f"| gain="
            f"{grouped_rich - grouped_i:+.4f}"
        )

    # ========================================================
    # Interpretation warning
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "DIAGNOSTIC SUMMARY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "The random CV estimates ordinary "
        "out-of-sample interpolation performance."
    )

    print(
        "The Grouped-I CV is stricter: "
        "local regions of the I spectrum are held out "
        "together."
    )

    print()

    print(
        "The final Pareto queue should only be interpreted "
        "if the rich surrogate retains meaningful ranking "
        "ability, especially under Grouped-I CV."
    )

    # ========================================================
    # Diagnostic-only mode
    # ========================================================

    if args.diagnostic_only:

        print()

        print(
            "Diagnostic-only mode: stopping here."
        )

        return

    # ========================================================
    # Fit final rich surrogate
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL SURROGATE FIT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Fitting rich surrogate "
        "on all probe responses...",
        flush=True,
    )

    surrogate = fit_final_surrogate(
        probes
    )

    # ========================================================
    # Load full candidate pool
    # ========================================================

    records = load_json_records(
        POOL_PATH
    )

    print()

    print(
        f"Raw pool positions: "
        f"{len(records):,}"
    )

    # ========================================================
    # Recompute exact I
    # ========================================================

    scores = compute_scores(
        records
    )

    print()

    print(
        f"I raw range : "
        f"{scores['score'].min():.6f} "
        f"-> "
        f"{scores['score'].max():.6f}"
    )

    print(
        f"I norm range: "
        f"{scores['I_norm'].min():.6f} "
        f"-> "
        f"{scores['I_norm'].max():.6f}"
    )

    # ========================================================
    # Exclude calibration probes
    # ========================================================

    excluded_ids = load_probe_query_ids()

    print()

    print(
        f"Probe IDs excluded from final pool: "
        f"{len(excluded_ids)}"
    )

    # ========================================================
    # Build eligible candidate pool
    # ========================================================

    candidates = build_candidate_pool(
        records,
        scores,
        excluded_ids,
    )

    if len(
        candidates
    ) < BUDGET:

        raise RuntimeError(
            "Eligible pool is smaller than "
            "the requested budget."
        )

    # ========================================================
    # Predict local response
    # ========================================================

    (
        predicted_kl,
        predicted_v,
    ) = predict_candidates(
        surrogate,
        candidates,
    )

    print()

    print(
        "Predicted response ranges:"
    )

    print(
        f"  Delta KL: "
        f"{predicted_kl.min():.6e} "
        f"-> "
        f"{predicted_kl.max():.6e}"
    )

    print(
        f"  Delta V : "
        f"{predicted_v.min():.6e} "
        f"-> "
        f"{predicted_v.max():.6e}"
    )

    # ========================================================
    # Exact Pareto layers
    # ========================================================

    ranks = pareto_ranks_2d(
        predicted_kl,
        predicted_v,
    )

    # ========================================================
    # Budgeted selection
    # ========================================================

    selected = select_budget(
        ranks,
        predicted_kl,
        predicted_v,
        BUDGET,
    )

    # ========================================================
    # Selected diagnostics
    # ========================================================

    selected_kl = (
        predicted_kl[
            selected
        ]
    )

    selected_v = (
        predicted_v[
            selected
        ]
    )

    selected_ranks = (
        ranks[
            selected
        ]
    )

    selected_i = np.asarray(
        [
            candidates[
                int(idx)
            ][
                "I_norm"
            ]
            for idx in selected
        ],
        dtype=np.float64,
    )

    selected_h = np.asarray(
        [
            candidates[
                int(idx)
            ][
                "H"
            ]
            for idx in selected
        ],
        dtype=np.float64,
    )

    selected_u = np.asarray(
        [
            candidates[
                int(idx)
            ][
                "U"
            ]
            for idx in selected
        ],
        dtype=np.float64,
    )


    selected_hu = np.asarray(
        [
            candidates[
                int(idx)
            ][
                "HU"
            ]
            for idx in selected
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Final selection report
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL PARETO SELECTION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Selected positions : "
        f"{len(selected)}"
    )

    print(
        f"Pareto rank range  : "
        f"{selected_ranks.min()} "
        f"-> "
        f"{selected_ranks.max()}"
    )

    print()

    print(
        "Predicted Delta KL:"
    )

    print(
        f"  mean   : "
        f"{selected_kl.mean():.6e}"
    )

    print(
        f"  median : "
        f"{np.median(selected_kl):.6e}"
    )

    print(
        f"  min    : "
        f"{selected_kl.min():.6e}"
    )

    print(
        f"  max    : "
        f"{selected_kl.max():.6e}"
    )

    print()

    print(
        "Predicted Delta V:"
    )

    print(
        f"  mean   : "
        f"{selected_v.mean():.6e}"
    )

    print(
        f"  median : "
        f"{np.median(selected_v):.6e}"
    )

    print(
        f"  min    : "
        f"{selected_v.min():.6e}"
    )

    print(
        f"  max    : "
        f"{selected_v.max():.6e}"
    )

    print()

    print(
        "Selected I_norm:"
    )

    print(
        f"  mean   : "
        f"{selected_i.mean():.6f}"
    )

    print(
        f"  median : "
        f"{np.median(selected_i):.6f}"
    )

    print(
        f"  min    : "
        f"{selected_i.min():.6f}"
    )

    print(
        f"  max    : "
        f"{selected_i.max():.6f}"
    )

    print()

    print(
        "Selected raw H:"
    )

    print(
        f"  mean   : "
        f"{selected_h.mean():.6f}"
    )

    print(
        f"  median : "
        f"{np.median(selected_h):.6f}"
    )

    print()

    print(
        "Selected raw U:"
    )

    print(
        f"  mean   : "
        f"{selected_u.mean():.6f}"
    )

    print(
        f"  median : "
        f"{np.median(selected_u):.6f}"
    )

        # ========================================================
    # Final sanity checks
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL SANITY CHECKS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Selected-set quantiles
    # --------------------------------------------------------

    print_quantiles(
        "Selected I_norm",
        selected_i,
    )

    print_quantiles(
        "Selected raw H",
        selected_h,
    )

    print_quantiles(
        "Selected raw U",
        selected_u,
    )

    print_quantiles(
        "Selected raw HU",
        selected_hu,
    )

    print_quantiles(
        "Selected predicted Delta KL",
        selected_kl,
    )

    print_quantiles(
        "Selected predicted Delta V",
        selected_v,
    )

    # --------------------------------------------------------
    # Actor / critic response relationship
    # --------------------------------------------------------

    selected_response_rho = safe_spearman(
        selected_kl,
        selected_v,
    )

    pool_response_rho = safe_spearman(
        predicted_kl,
        predicted_v,
    )

    print()
    print(
        "Predicted response correlation:"
    )

    print(
        f"  Full eligible pool "
        f"| rho(KL, V)="
        f"{pool_response_rho:+.4f}"
    )

    print(
        f"  Selected Pareto set "
        f"| rho(KL, V)="
        f"{selected_response_rho:+.4f}"
    )

    # --------------------------------------------------------
    # Compare selected feature distribution to decision pool
    # --------------------------------------------------------

    pool_i = np.asarray(
        [
            candidate["I_norm"]
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    pool_h = np.asarray(
        [
            candidate["H"]
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    pool_u = np.asarray(
        [
            candidate["U"]
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    print_quantiles(
        "Eligible pool I_norm",
        pool_i,
    )

    print_quantiles(
        "Eligible pool raw H",
        pool_h,
    )

    print_quantiles(
        "Eligible pool raw U",
        pool_u,
    )

    # ========================================================
    # Write strict HMI queue
    # ========================================================

    write_hmi_queue(
        candidates,
        selected,
        OUTPUT_QUEUE_PATH,
        force=args.force,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "PARETO QUEUE CREATED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Final HMI queue saved to:\n"
        f"  {OUTPUT_QUEUE_PATH}"
    )

    print()

    print(
        f"The queue contains exactly "
        f"{BUDGET} pending Oracle annotations."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()