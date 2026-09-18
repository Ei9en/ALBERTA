#!/usr/bin/env python3

"""
ALBERTA - Active Learning Score Distribution
============================================

Analyze the distribution of the ALBERTA acquisition score I.

Pipeline
--------

    H raw
        |
        v
    temporal-window x side-aware percentile normalization
        |
        v
    H_norm
        |
        v
    standardization
        |
        v
    H*

    U raw
        |
        v
    F(U) = log1p(U / TAU)
        |
        v
    temporal-window x side-aware percentile normalization
        |
        v
    F(U)_norm
        |
        v
    standardization
        |
        v
    F(U)*

    H_norm * F(U)_norm
        |
        v
    standardization
        |
        v
    (H * F(U))*

    H*, F(U)*, (H * F(U))*
        |
        v
    raw OLS coefficients
        |
        v
    I
        |
        v
    min-max normalization
        |
        v
    I_norm in [0, 1]

The acquisition score is:

    I =
        RAW_W_H  * H*
        + RAW_W_U  * F(U)*
        + RAW_W_HU * (H * F(U))*

The OLS coefficients retain their original sign and magnitude.

The chronological windows are contiguous equal-size partitions
of the append-ordered uncertainty dataset. They are coarse temporal
strata, not exact RL epochs.

The normalization used here must exactly match the normalization
used by estimate_active_learning_weights.py.

The final min-max normalization is only a representation of the
score and does not affect ranking.

The annotation budget remains defined as a fraction of observations,
not as a fraction of the numerical score range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from AL_weights import (
    RAW_W_H,
    RAW_W_HU,
    RAW_W_U,
    TAU,
    TEMPORAL_WINDOWS,
)


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

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "I_distribution.png"
)

# 0.01 %
DEFAULT_AL_BUDGET = 0.0001


# ============================================================
# Percentile-rank normalization
# ============================================================

def percentile_rank(
    values: np.ndarray,
) -> np.ndarray:
    """
    Percentile rank in [0, 1].

    Ties receive their average zero-based rank.

    This implementation must remain identical to the one used by
    estimate_active_learning_weights.py and seed_oracle_queue.py.
    """

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

        average_rank = (
            start
            + end
            - 1
        ) / 2.0

        ranks[
            order[start:end]
        ] = (
            average_rank
            / (n - 1)
        )

        start = end

    return ranks


# ============================================================
# Side to move
# ============================================================

def extract_side_to_move(
    fens: np.ndarray,
) -> np.ndarray:

    sides = []

    for fen in fens:

        parts = str(
            fen
        ).split()

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
                f"Invalid side-to-move in FEN: {fen}"
            )

        sides.append(
            side
        )

    return np.asarray(
        sides,
        dtype="<U1",
    )


# ============================================================
# Temporal + side-aware normalization
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

    The dataset is split into contiguous approximately equal-size
    windows. Within each window x side stratum, values are converted
    independently to percentile ranks in [0, 1].
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sides = np.asarray(
        sides
    )

    n = len(values)

    if len(sides) != n:

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
            edges[window]
        )

        end = int(
            edges[window + 1]
        )

        window_ids[
            start:end
        ] = window

        local_sides = sides[
            start:end
        ]

        for side in (
            "w",
            "b",
        ):

            side_mask = (
                local_sides
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
                    f"W{window + 1}, side={side}: "
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
# Standardization
# ============================================================

def z_score(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        np.mean(
            values
        )
    )

    std = float(
        np.std(
            values,
            ddof=0,
        )
    )

    if std <= 0.0:

        raise ValueError(
            "Cannot standardize a constant array."
        )

    return (
        values
        - mean
    ) / std


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
# Data loading
# ============================================================

def load_data(
    path: Path,
) -> dict[str, np.ndarray]:

    print()
    print("=" * 70)
    print("LOADING UNCERTAINTY STATISTICS")
    print("=" * 70)

    print()
    print(
        f"File: {path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        raw = json.load(
            f
        )

    if not isinstance(
        raw,
        list,
    ):

        raise ValueError(
            "Input uncertainty file must contain a JSON list."
        )

    print(
        f"Raw records: {len(raw):,}"
    )

    fens = []
    H = []
    U = []
    HU = []

    rejected = 0

    for record in raw:

        try:

            fen = record[
                "fen"
            ]

            h = float(
                record[
                    "H"
                ]
            )

            u = float(
                record[
                    "U"
                ]
            )

            hu = float(
                record[
                    "HU"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            rejected += 1
            continue

        if not isinstance(
            fen,
            str,
        ):

            rejected += 1
            continue

        if not (
            np.isfinite(h)
            and np.isfinite(u)
            and np.isfinite(hu)
        ):

            rejected += 1
            continue

        if u < 0.0:

            rejected += 1
            continue

        fens.append(
            fen
        )

        H.append(
            h
        )

        U.append(
            u
        )

        HU.append(
            hu
        )

    if not fens:

        raise RuntimeError(
            "No valid observations."
        )

    print(
        f"Valid records: {len(fens):,}"
    )

    print(
        f"Rejected     : {rejected:,}"
    )

    return {
        "fen":
            np.asarray(
                fens,
                dtype=object,
            ),

        "H":
            np.asarray(
                H,
                dtype=np.float64,
            ),

        "U":
            np.asarray(
                U,
                dtype=np.float64,
            ),

        # Kept for compatibility / diagnostics.
        # The acquisition interaction is recomputed from the
        # normalized H and U features.
        "HU":
            np.asarray(
                HU,
                dtype=np.float64,
            ),
    }


# ============================================================
# Configuration
# ============================================================

def validate_configuration(
    budget: float,
) -> None:

    print()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print()
    print(
        f"TAU              : {TAU:.6f}"
    )

    print(
        f"Temporal windows : {TEMPORAL_WINDOWS}"
    )

    print(
        f"AL budget        : {budget:.5%}"
    )

    print()
    print("RAW OLS COEFFICIENTS")
    print("-" * 70)

    print(
        f"RAW_W_H   : {RAW_W_H:+.9f}"
    )

    print(
        f"RAW_W_U   : {RAW_W_U:+.9f}"
    )

    print(
        f"RAW_W_HU  : {RAW_W_HU:+.9f}"
    )

    print()
    print(
        "Coefficients retain their original OLS scale; "
        "no coefficient normalization is applied."
    )

    if TAU <= 0.0:

        raise ValueError(
            "TAU must be strictly positive."
        )

    if TEMPORAL_WINDOWS < 1:

        raise ValueError(
            "TEMPORAL_WINDOWS must be >= 1."
        )

    if not (
        0.0
        < budget
        <= 1.0
    ):

        raise ValueError(
            "budget must satisfy 0 < budget <= 1."
        )


# ============================================================
# Score
# ============================================================

def build_score(
    data: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:

    print()
    print("=" * 70)
    print("BUILDING ACTIVE LEARNING SCORE")
    print("=" * 70)

    fens = data[
        "fen"
    ]

    H = data[
        "H"
    ]

    U = data[
        "U"
    ]

    sides = extract_side_to_move(
        fens
    )

    # --------------------------------------------------------
    # 1. H temporal + side-aware normalization
    # --------------------------------------------------------

    (
        H_norm,
        window_ids,
    ) = normalize_temporal_side_aware(
        values=H,
        sides=sides,
        n_windows=TEMPORAL_WINDOWS,
    )

    # --------------------------------------------------------
    # 2. U logarithmic transform
    #
    # Retained because the current calibration was fitted with
    # exactly this pipeline.
    # --------------------------------------------------------

    F_U_raw = np.log1p(
        U / TAU
    )

    # --------------------------------------------------------
    # 3. U temporal + side-aware normalization
    # --------------------------------------------------------

    (
        F_U_norm,
        window_ids_u,
    ) = normalize_temporal_side_aware(
        values=F_U_raw,
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

    # --------------------------------------------------------
    # 4. Interaction
    # --------------------------------------------------------

    H_F_U = (
        H_norm
        * F_U_norm
    )

    # --------------------------------------------------------
    # 5. Standardization
    # --------------------------------------------------------

    H_star = z_score(
        H_norm
    )

    F_U_star = z_score(
        F_U_norm
    )

    H_F_U_star = z_score(
        H_F_U
    )

    # --------------------------------------------------------
    # 6. Raw acquisition score
    # --------------------------------------------------------

    I = (
        RAW_W_H
        * H_star
        + RAW_W_U
        * F_U_star
        + RAW_W_HU
        * H_F_U_star
    )

    # --------------------------------------------------------
    # 7. Display normalization
    # --------------------------------------------------------

    I_norm = minmax_normalize(
        I
    )

    components = {
        "H_norm":
            H_norm,

        "F_U_raw":
            F_U_raw,

        "F_U_norm":
            F_U_norm,

        "H_F_U":
            H_F_U,

        "H_star":
            H_star,

        "F_U_star":
            F_U_star,

        "H_F_U_star":
            H_F_U_star,
    }

    return (
        I,
        I_norm,
        sides,
        window_ids,
        components,
    )


# ============================================================
# Budget threshold
# ============================================================

def compute_budget_threshold(
    I: np.ndarray,
    I_norm: np.ndarray,
    budget: float,
) -> tuple[
    int,
    float,
    float,
    np.ndarray,
    float,
]:

    n = len(
        I
    )

    target = max(
        1,
        int(
            np.ceil(
                n
                * budget
            )
        ),
    )

    order = np.argsort(
        -I,
        kind="stable",
    )

    selected_indices = order[
        :target
    ]

    threshold = float(
        I[
            selected_indices[-1]
        ]
    )

    threshold_norm = float(
        I_norm[
            selected_indices[-1]
        ]
    )

    selected_fraction = (
        len(
            selected_indices
        )
        / n
    )

    return (
        target,
        threshold,
        threshold_norm,
        selected_indices,
        selected_fraction,
    )


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostics(
    I: np.ndarray,
    I_norm: np.ndarray,
    sides: np.ndarray,
    window_ids: np.ndarray,
    components: dict[str, np.ndarray],
    budget: float,
) -> np.ndarray:

    H_norm = components[
        "H_norm"
    ]

    F_U_raw = components[
        "F_U_raw"
    ]

    F_U_norm = components[
        "F_U_norm"
    ]

    H_F_U = components[
        "H_F_U"
    ]

    H_star = components[
        "H_star"
    ]

    F_U_star = components[
        "F_U_star"
    ]

    H_F_U_star = components[
        "H_F_U_star"
    ]

    print()
    print("=" * 70)
    print("ACTIVE LEARNING SCORE I")
    print("=" * 70)

    # --------------------------------------------------------
    # Temporal normalization sanity check
    # --------------------------------------------------------

    print()
    print("TEMPORAL NORMALIZATION")
    print("-" * 70)

    for window in range(
        TEMPORAL_WINDOWS
    ):

        window_mask = (
            window_ids
            == window
        )

        print()
        print(
            f"W{window + 1}: "
            f"{np.sum(window_mask):,} observations"
        )

        for side in (
            "w",
            "b",
        ):

            mask = (
                window_mask
                & (sides == side)
            )

            print(
                f"  {side}: "
                f"n={np.sum(mask):,}, "
                f"H_norm_mean={np.mean(H_norm[mask]):.6f}, "
                f"U_norm_mean={np.mean(F_U_norm[mask]):.6f}"
            )

    # --------------------------------------------------------
    # Distribution by side
    # --------------------------------------------------------

    print()
    print("I DISTRIBUTION BY SIDE-TO-MOVE")
    print("-" * 70)

    for side, name in (
        ("w", "White"),
        ("b", "Black"),
    ):

        values = I[
            sides == side
        ]

        print()
        print(name)

        for q in (
            0.50,
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
            1.00,
        ):

            print(
                f"P{q * 100:g} : "
                f"{np.quantile(values, q):+.9f}"
            )

    # --------------------------------------------------------
    # Predictors
    # --------------------------------------------------------

    print()
    print("PREDICTORS")
    print("-" * 70)

    for label, values in (
        (
            "H*",
            H_star,
        ),
        (
            "F(U)*",
            F_U_star,
        ),
        (
            "(H*F(U))*",
            H_F_U_star,
        ),
    ):

        print(
            f"{label:<12}: "
            f"mean={np.mean(values):+.6f} "
            f"std={np.std(values):.6f}"
        )

    # --------------------------------------------------------
    # Transform diagnostics
    # --------------------------------------------------------

    print()
    print("TRANSFORMATION")
    print("-" * 70)

    print(
        f"F(U) raw       : "
        f"[{np.min(F_U_raw):.6f}, "
        f"{np.max(F_U_raw):.6f}]"
    )

    print(
        f"F(U) norm      : "
        f"[{np.min(F_U_norm):.6f}, "
        f"{np.max(F_U_norm):.6f}]"
    )

    print(
        f"H norm mean/std: "
        f"{np.mean(H_norm):.9f} / "
        f"{np.std(H_norm):.9f}"
    )

    print(
        f"U norm mean/std: "
        f"{np.mean(F_U_norm):.9f} / "
        f"{np.std(F_U_norm):.9f}"
    )

    print(
        f"HU mean/std    : "
        f"{np.mean(H_F_U):.9f} / "
        f"{np.std(H_F_U):.9f}"
    )

    # --------------------------------------------------------
    # Score quantiles
    # --------------------------------------------------------

    print()
    print("SCORE DISTRIBUTION")
    print("-" * 70)

    quantiles = (
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.975,
        0.99,
        0.995,
        0.999,
        0.9999,
        1.00,
    )

    for q in quantiles:

        print(
            f"P{q * 100:<6g}: "
            f"{np.quantile(I, q):+.9f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("SUMMARY")
    print("-" * 70)

    print(
        f"N      : {len(I):,}"
    )

    print(
        f"Mean   : {np.mean(I):+.9f}"
    )

    print(
        f"Std    : {np.std(I):.9f}"
    )

    print(
        f"Median : {np.median(I):+.9f}"
    )

    print(
        f"Min    : {np.min(I):+.9f}"
    )

    print(
        f"Max    : {np.max(I):+.9f}"
    )

    # --------------------------------------------------------
    # Budget
    # --------------------------------------------------------

    (
        target,
        threshold,
        threshold_norm,
        selected_indices,
        selected_fraction,
    ) = compute_budget_threshold(
        I=I,
        I_norm=I_norm,
        budget=budget,
    )

    print()
    print("ACTIVE LEARNING SELECTION")
    print("-" * 70)

    print(
        f"Budget             : {budget:.5%}"
    )

    print(
        f"Target positions   : {target:,}"
    )

    print(
        f"Threshold I        : {threshold:+.9f}"
    )

    print(
        f"Threshold I_norm   : {threshold_norm:.9f}"
    )

    print(
        f"Threshold range    : "
        f"{100 * threshold_norm:.4f}% of [I_min, I_max]"
    )

    print(
        f"Actual fraction    : {selected_fraction:.6%}"
    )

    # --------------------------------------------------------
    # Side composition
    # --------------------------------------------------------

    selected_sides = sides[
        selected_indices
    ]

    total_white = int(
        np.sum(
            sides == "w"
        )
    )

    total_black = int(
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

    white_fraction = (
        total_white
        / len(sides)
    )

    black_fraction = (
        total_black
        / len(sides)
    )

    selected_white_fraction = (
        selected_white
        / len(selected_indices)
    )

    selected_black_fraction = (
        selected_black
        / len(selected_indices)
    )

    print()
    print("SIDE-TO-MOVE")
    print("-" * 70)

    print(
        f"Global White   : "
        f"{total_white:,} "
        f"({white_fraction:.3%})"
    )

    print(
        f"Global Black   : "
        f"{total_black:,} "
        f"({black_fraction:.3%})"
    )

    print(
        f"Selected White : "
        f"{selected_white:,} "
        f"({selected_white_fraction:.3%})"
    )

    print(
        f"Selected Black : "
        f"{selected_black:,} "
        f"({selected_black_fraction:.3%})"
    )

    if white_fraction > 0.0:

        print(
            f"White enrichment: "
            f"{selected_white_fraction / white_fraction:.3f}x"
        )

    if black_fraction > 0.0:

        print(
            f"Black enrichment: "
            f"{selected_black_fraction / black_fraction:.3f}x"
        )

    # --------------------------------------------------------
    # Temporal composition of selected positions
    # --------------------------------------------------------

    print()
    print("TEMPORAL COMPOSITION OF I SELECTION")
    print("-" * 70)

    selected_window_ids = window_ids[
        selected_indices
    ]

    for window in range(
        TEMPORAL_WINDOWS
    ):

        global_count = int(
            np.sum(
                window_ids == window
            )
        )

        selected_count = int(
            np.sum(
                selected_window_ids == window
            )
        )

        global_fraction = (
            global_count
            / len(window_ids)
        )

        selected_window_fraction = (
            selected_count
            / len(selected_indices)
        )

        enrichment = (
            selected_window_fraction
            / global_fraction
            if global_fraction > 0.0
            else float("nan")
        )

        print(
            f"W{window + 1}: "
            f"{selected_count:4d} selected "
            f"({selected_window_fraction:7.2%}), "
            f"expected={global_fraction:7.2%}, "
            f"enrichment={enrichment:.3f}x"
        )

    return selected_indices


# ============================================================
# Smooth histogram
# ============================================================

def smooth_histogram(
    values: np.ndarray,
    bins: int = 160,
    smoothing_window: int = 9,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    counts, edges = np.histogram(
        values,
        bins=bins,
        range=(
            0.0,
            1.0,
        ),
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    window = np.ones(
        smoothing_window,
        dtype=np.float64,
    )

    window /= np.sum(
        window
    )

    smooth_counts = np.convolve(
        counts,
        window,
        mode="same",
    )

    return (
        counts,
        centers,
        smooth_counts,
    )


# ============================================================
# Plot
# ============================================================

def plot_distribution(
    I_norm: np.ndarray,
    threshold_norm: float,
    output_file: Path,
) -> None:

    print()
    print("=" * 70)
    print("GENERATING DISTRIBUTION PLOT")
    print("=" * 70)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, centers, smooth_counts = smooth_histogram(
        I_norm
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.hist(
        I_norm,
        bins=160,
        range=(
            0.0,
            1.0,
        ),
        alpha=0.35,
        label="Histogram",
    )

    axis.plot(
        centers,
        smooth_counts,
        linewidth=2.0,
        label="Smoothed distribution",
    )

    axis.axvline(
        threshold_norm,
        linestyle="--",
        linewidth=1.5,
        label=(
            "AL budget threshold "
            f"({threshold_norm:.3f})"
        ),
    )

    axis.set_xlabel(
        "Normalized acquisition score $I_{norm}$"
    )

    axis.set_ylabel(
        "Number of positions"
    )

    axis.set_title(
        "Distribution of ALBERTA Active-Learning Acquisition Score"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.grid(
        alpha=0.2,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Plot saved to: {output_file}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze and plot the distribution of the "
            "ALBERTA active-learning acquisition score."
        )
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Input uncertainty-statistics JSON file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output PNG file.",
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_AL_BUDGET,
        help=(
            "Active-learning annotation budget. "
            "Default: 0.0001 = 0.01%%."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    validate_configuration(
        args.budget
    )

    data = load_data(
        args.data_file
    )

    (
        I,
        I_norm,
        sides,
        window_ids,
        components,
    ) = build_score(
        data
    )

    selected_indices = print_diagnostics(
        I=I,
        I_norm=I_norm,
        sides=sides,
        window_ids=window_ids,
        components=components,
        budget=args.budget,
    )

    threshold_norm = float(
        np.min(
            I_norm[
                selected_indices
            ]
        )
    )

    plot_distribution(
        I_norm=I_norm,
        threshold_norm=threshold_norm,
        output_file=args.output,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()