#!/usr/bin/env python3

"""
ALBERTA - Pareto Selection
==========================

Dynamic learner-response acquisition.

Pipeline
--------

1. Load the measured probe responses:

       (H, U, HU)
            ->
       (Delta_KL, Delta_V)

2. Cross-validate the response surrogate.

3. Fit the surrogate on all probe observations.

4. Load the current self-play uncertainty pool.

5. Remove:
       - invalid / <= 1 legal move positions
       - duplicate FENs
       - calibration probe positions

6. Predict:

       Delta_KL_hat
       Delta_V_hat

7. Compute exact 2D Pareto layers.

8. Select:

       round(0.02% * raw_pool_size)

   positions using:

       non-dominated sorting
       +
       crowding distance on the last front

9. Write an HMI-compatible Oracle queue.

Method
------

    (H, U, HU)
        ->
    (Delta_KL_hat, Delta_V_hat)
        ->
    Pareto acquisition

The historical scalar score I is NOT used.

No OLS or uncertainty scalarization is involved in the
Pareto acquisition stage.
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
# Default paths
#
# These can be overridden from the command line so the same
# script can be reused at every dynamic AL generation.
# ============================================================

DEFAULT_PROBE_RESPONSE_PATH = (
    PROJECT_ROOT
    / "pareto"
    / "pareto_local_responses.jsonl"
)

DEFAULT_PROBE_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)

DEFAULT_POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

DEFAULT_OUTPUT_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_dynamic_pareto.jsonl"
)


# ============================================================
# Experiment configuration
# ============================================================

# 0.02 %
BUDGET_FRACTION = 0.0002

RANDOM_STATE = 42

CV_FOLDS = 5

N_TREES = 300

PREDICT_CHUNK_SIZE = 100000


# ============================================================
# Features
#
# No I.
# No score.
# No OLS-derived feature.
# ============================================================

FEATURES = [
    "H",
    "U",
    "HU",
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
            "stats",
            "uncertainty_stats",
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
# Probe dataset
# ============================================================

def load_probe_dataset(
    path,
):

    rows = load_jsonl(
        path
    )

    required = {
        "query_id",
        "fen",
        "H",
        "U",
        "HU",
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
            or
            delta_v < 0
        ):

            raise ValueError(
                f"Negative response "
                f"at probe index {i}"
            )

        for feature in FEATURES:

            value = float(
                row[feature]
            )

            if not np.isfinite(
                value
            ):

                raise ValueError(
                    f"Non-finite {feature} "
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
):

    return np.asarray(
        [
            [
                float(
                    row[name]
                )
                for name
                in FEATURES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


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
# Random cross-validation
#
# We mainly care about ranking ability because Pareto
# acquisition depends on relative response predictions.
# ============================================================

def cross_validate_surrogate(
    rows,
):

    X = make_feature_matrix(
        rows
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

    predicted = np.expm1(
        predicted_log
    )

    predicted = np.maximum(
        predicted,
        0.0,
    )

    print()
    print(
        "-" * 70
    )

    print(
        "SURROGATE CROSS-VALIDATION"
    )

    print(
        f"Features: {FEATURES}"
    )

    print(
        "-" * 70
    )

    kl_metrics = print_target_metrics(
        "Delta KL",
        delta_kl,
        predicted[:, 0],
    )

    v_metrics = print_target_metrics(
        "Delta V",
        delta_v,
        predicted[:, 1],
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
        rows
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
# Probe IDs to exclude
# ============================================================

def load_probe_query_ids(
    path,
):

    rows = load_jsonl(
        path
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

        board = chess.variant.AtomicBoard(
            fen
        )

        return (
            board.legal_moves.count()
            >
            1
        )

    except Exception:

        return False


# ============================================================
# Build candidate pool
#
# H/U/HU are taken DIRECTLY from the current self-play JSON.
# ============================================================

def build_candidate_pool(
    records,
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
    rejected_invalid = 0

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

        try:

            fen = record[
                "fen"
            ]

            H = float(
                record["H"]
            )

            U = float(
                record["U"]
            )

            HU = float(
                record["HU"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            rejected_invalid += 1
            continue

        if not (
            np.isfinite(H)
            and
            np.isfinite(U)
            and
            np.isfinite(HU)
        ):

            rejected_invalid += 1
            continue

        query_id = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            fen,
        ).hex


        # ----------------------------------------------------
        # Calibration leakage exclusion
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
        # Oracle eligibility
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
                    H,

                "U":
                    U,

                "HU":
                    HU,
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
        f"Invalid records     : "
        f"{rejected_invalid:,}"
    )

    print(
        f"Matched probe IDs   : "
        f"{len(matched_probe_ids):,} "
        f"/ {len(excluded_ids):,}"
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
            chunk
        )

        prediction_log = model.predict(
            X
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
# Maximize both objectives.
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

        fenwick.update(
            yi,
            layer,
        )

        position = end

    return ranks


# ============================================================
# Crowding distance
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

        if len(
            front
        ) <= remaining:

            selected.extend(
                front.tolist()
            )

            continue


        # ----------------------------------------------------
        # Last partially included front
        # ----------------------------------------------------

        distances = crowding_distance(
            front,
            predicted_kl,
            predicted_v,
        )

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

        selected.extend(
            [
                int(
                    front[j]
                )
                for j in local_order[
                    :remaining
                ]
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
# Strict HMI queue
#
# NOTE:
#
# score / I_norm are kept as None ONLY if your HMI schema
# expects these keys to exist.
#
# They have no role in the acquisition method.
# ============================================================

def write_hmi_queue(
    candidates,
    selected_indices,
    output_path,
    force=False,
):

    if (
        output_path.exists()
        and
        output_path.stat().st_size > 0
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

        for idx in selected_indices:

            candidate = candidates[
                int(
                    idx
                )
            ]

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

                # --------------------------------------------
                # Legacy HMI fields.
                #
                # Kept only for schema compatibility.
                # Not used by dynamic Pareto.
                # --------------------------------------------

                "score":
                    None,

                "I_norm":
                    None,

                "threshold":
                    None,

                "model":
                    "dynamic_pareto",

                "epoch":
                    -1,

                "game_id":
                    -1,

                "ply":
                    -1,

                "created_at":
                    now.isoformat(),

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


# ============================================================
# Probe correlations
# ============================================================

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

    for feature in FEATURES:

        values = np.asarray(
            [
                float(
                    row[feature]
                )
                for row in rows
            ],
            dtype=np.float64,
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
            "ALBERTA dynamic Pareto acquisition"
        )
    )

    parser.add_argument(
        "--probe-responses",
        type=Path,
        default=DEFAULT_PROBE_RESPONSE_PATH,
        help=(
            "JSONL containing current local probe responses."
        ),
    )

    parser.add_argument(
        "--probe-queue",
        type=Path,
        default=DEFAULT_PROBE_QUEUE_PATH,
        help=(
            "Original calibration queue used for leakage exclusion."
        ),
    )

    parser.add_argument(
        "--pool",
        type=Path,
        default=DEFAULT_POOL_PATH,
        help=(
            "Current self-play uncertainty JSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_QUEUE_PATH,
        help=(
            "Output HMI queue."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing output queue."
        ),
    )

    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Run probe diagnostics and CV only."
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
        "ALBERTA - DYNAMIC PARETO SELECTION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Probe responses : "
        f"{args.probe_responses}"
    )

    print(
        f"Candidate pool  : "
        f"{args.pool}"
    )

    print(
        f"Budget fraction : "
        f"{BUDGET_FRACTION:.6f} "
        f"({BUDGET_FRACTION * 100:.4f}%)"
    )

    print(
        f"Features        : "
        f"{FEATURES}"
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

    probes = load_probe_dataset(
        args.probe_responses
    )

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
    # Surrogate CV
    # ========================================================

    metrics = cross_validate_surrogate(
        probes
    )


    print()
    print(
        "=" * 70
    )

    print(
        "CV SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Pareto acquisition primarily requires "
        "useful response ranking."
    )

    print(
        f"Delta KL Spearman : "
        f"{metrics['delta_kl']['spearman']:+.4f}"
    )

    print(
        f"Delta V  Spearman : "
        f"{metrics['delta_v']['spearman']:+.4f}"
    )


    if args.diagnostic_only:

        print()

        print(
            "Diagnostic-only mode: stopping here."
        )

        return


    # ========================================================
    # Fit final surrogate
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

    surrogate = fit_final_surrogate(
        probes
    )


    # ========================================================
    # Load current candidate pool
    # ========================================================

    records = load_json_records(
        args.pool
    )

    raw_pool_size = len(
        records
    )


    # ========================================================
    # Dynamic annotation budget
    #
    # 0.02% of the JSON produced by the previous generation.
    #
    # Example:
    #
    # 1,229,546 * 0.0002 = 245.9092 -> 246
    # ========================================================

    budget = max(
        1,
        int(
            round(
                raw_pool_size
                *
                BUDGET_FRACTION
            )
        ),
    )


    print()
    print(
        f"Raw pool positions : "
        f"{raw_pool_size:,}"
    )

    print(
        f"Dynamic budget     : "
        f"{budget:,}"
    )


    # ========================================================
    # Probe leakage exclusion
    # ========================================================

    excluded_ids = load_probe_query_ids(
        args.probe_queue
    )

    print()

    print(
        f"Calibration probe IDs excluded: "
        f"{len(excluded_ids)}"
    )


    # ========================================================
    # Candidate pool
    # ========================================================

    candidates = build_candidate_pool(
        records,
        excluded_ids,
    )


    if len(
        candidates
    ) < budget:

        raise RuntimeError(
            "Eligible candidate pool is smaller "
            "than the dynamic budget."
        )


    # ========================================================
    # Predict response
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
    # Pareto layers
    # ========================================================

    ranks = pareto_ranks_2d(
        predicted_kl,
        predicted_v,
    )


    # ========================================================
    # Budget selection
    # ========================================================

    selected = select_budget(
        ranks,
        predicted_kl,
        predicted_v,
        budget,
    )


    selected_kl = predicted_kl[
        selected
    ]

    selected_v = predicted_v[
        selected
    ]

    selected_ranks = ranks[
        selected
    ]


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
    # Final report
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


    print_quantiles(
        "Selected H",
        selected_h,
    )

    print_quantiles(
        "Selected U",
        selected_u,
    )

    print_quantiles(
        "Selected HU",
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


    # ========================================================
    # Response relationship
    # ========================================================

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
        f"| rho(KL,V)="
        f"{pool_response_rho:+.4f}"
    )

    print(
        f"  Selected Pareto set "
        f"| rho(KL,V)="
        f"{selected_response_rho:+.4f}"
    )


    # ========================================================
    # Compare feature distributions
    # ========================================================

    pool_h = np.asarray(
        [
            candidate[
                "H"
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    pool_u = np.asarray(
        [
            candidate[
                "U"
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    pool_hu = np.asarray(
        [
            candidate[
                "HU"
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )


    print_quantiles(
        "Eligible pool H",
        pool_h,
    )

    print_quantiles(
        "Eligible pool U",
        pool_u,
    )

    print_quantiles(
        "Eligible pool HU",
        pool_hu,
    )


    # ========================================================
    # Write queue
    # ========================================================

    write_hmi_queue(
        candidates,
        selected,
        args.output,
        force=args.force,
    )


    print()
    print(
        "=" * 70
    )

    print(
        "DYNAMIC PARETO QUEUE CREATED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Raw pool size : "
        f"{raw_pool_size:,}"
    )

    print(
        f"Budget        : "
        f"{budget:,} "
        f"({BUDGET_FRACTION * 100:.4f}%)"
    )

    print()

    print(
        f"Saved to:\n"
        f"  {args.output}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()