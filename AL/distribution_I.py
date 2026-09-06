#!/usr/bin/env python3

"""
ALBERTA - Active Learning Score I
=================================

Analyse et distribution du score d'Active Learning I.

Pipeline
--------

    H raw
        |
        v
    side-aware percentile normalization
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
    side-aware percentile normalization
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
    (H*F(U))*

    H*, F(U)*, (H*F(U))*
        |
        v
    OLS coefficients
        |
        v
    I
        |
        v
    min-max normalization
        |
        v
    I_norm in [0, 1]

Les coefficients RAW_W_H, RAW_W_U et RAW_W_HU sont les
coefficients OLS bruts estimés dans AL_weights.py.

Ils NE sont PAS normalisés entre eux.

Le score est donc :

    I =
        RAW_W_H  * H*
        + RAW_W_U  * F(U)*
        + RAW_W_HU * (H*F(U))*

Les coefficients conservent leur signe et leur amplitude
statistique originale.

Une seconde normalisation min-max est appliquée uniquement
au score final :

    I_norm = (I - I_min) / (I_max - I_min)

Ainsi :

    I_norm = 0 -> score minimal
    I_norm = 1 -> score maximal

Le seuil correspondant au budget AL est également affiché
en pourcentage de l'étendue [I_min, I_max].

IMPORTANT
---------

Le budget reste défini en nombre de positions / fraction
du dataset.

Le pourcentage de l'étendue de I est une information
supplémentaire et ne remplace PAS le budget.

Output
------

    data/I_distribution.png

Usage
-----

    python AL/distribution_I.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
# Raw OLS coefficients
# ============================================================

from data.uncertainty_analysis.active_learning_weights import (
    RAW_W_H,
    RAW_W_U,
    RAW_W_HU,
    TAU,
)


# ============================================================
# Configuration
# ============================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "I_distribution.png"
)

AL_BUDGET = 0.0002


# ============================================================
# Utilities
# ============================================================

def percentile_rank(values):
    """
    Percentile rank in [0, 1].

    Ties receive their average rank.
    """

    series = pd.Series(
        values
    )

    return (
        series.rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=float
        )
    )


def extract_side_to_move(fens):
    """
    Extract side-to-move from full FEN.

    Returns:
        'w' for White
        'b' for Black
    """

    sides = []

    for fen in fens:

        try:

            side = fen.split()[1]

            if side not in {
                "w",
                "b",
            }:

                raise ValueError

        except Exception:

            raise ValueError(
                f"Invalid FEN side-to-move: {fen}"
            )

        sides.append(
            side
        )

    return np.asarray(
        sides
    )


def normalize_side_aware(
    values,
    sides,
):
    """
    Percentile-rank normalization performed
    independently for White-to-move and Black-to-move.

    Output range:
        [0, 1]
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    sides = np.asarray(
        sides
    )

    result = np.empty(
        len(values),
        dtype=float,
    )

    for side in [
        "w",
        "b",
    ]:

        mask = (
            sides == side
        )

        if not np.any(mask):
            continue

        result[mask] = (
            percentile_rank(
                values[mask]
            )
        )

    return result


def z_score(values):
    """
    Standard score.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    mean = np.mean(
        values
    )

    std = np.std(
        values
    )

    if std == 0:

        raise ValueError(
            "Cannot standardize a constant array."
        )

    return (
        values - mean
    ) / std


def minmax_normalize(values):
    """
    Min-max normalization to [0, 1].

        x_norm = (x - min) / (max - min)
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = np.min(
        values
    )

    maximum = np.max(
        values
    )

    span = (
        maximum - minimum
    )

    if span <= 0:

        raise ValueError(
            "Cannot min-max normalize a constant score."
        )

    return (
        values - minimum
    ) / span


# ============================================================
# Load data
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING UNCERTAINTY STATISTICS")
    print("=" * 70)

    print()
    print(
        f"File: {DATA_FILE}"
    )

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        raw = pd.read_json(
            f
        )

    print(
        f"Raw records: "
        f"{len(raw):,}"
    )

    required = [
        "fen",
        "H",
        "U",
        "HU",
    ]

    missing = [
        column
        for column in required
        if column not in raw.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = raw[
        required
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in [
        "H",
        "U",
        "HU",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Validity
    # --------------------------------------------------------

    valid = (
        df["fen"].notna()
        &
        df["H"].notna()
        &
        df["U"].notna()
        &
        df["HU"].notna()
        &
        np.isfinite(
            df["H"]
        )
        &
        np.isfinite(
            df["U"]
        )
        &
        np.isfinite(
            df["HU"]
        )
    )

    df = df[
        valid
    ].reset_index(
        drop=True
    )

    print(
        f"Valid records: "
        f"{len(df):,}"
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid observations."
        )

    return df


# ============================================================
# Configuration validation
# ============================================================

def validate_configuration():

    print()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print()
    print(
        f"TAU      : "
        f"{TAU:.6f}"
    )

    print(
        f"AL budget: "
        f"{AL_BUDGET:.5%}"
    )

    print()

    print(
        "RAW OLS COEFFICIENTS"
    )
    print("-" * 70)

    print(
        f"RAW_W_H  : "
        f"{RAW_W_H:+.9f}"
    )

    print(
        f"RAW_W_U  : "
        f"{RAW_W_U:+.9f}"
    )

    print(
        f"RAW_W_HU : "
        f"{RAW_W_HU:+.9f}"
    )

    print()

    print(
        "Coefficients are used at their original "
        "OLS scale; no coefficient normalization is applied."
    )

    if TAU <= 0:

        raise ValueError(
            "TAU must be strictly positive."
        )

    if not (
        0 < AL_BUDGET <= 1
    ):

        raise ValueError(
            "AL_BUDGET must be in (0, 1]."
        )


# ============================================================
# Build score
# ============================================================

def build_score(df):

    print()
    print("=" * 70)
    print("BUILDING ACTIVE LEARNING SCORE")
    print("=" * 70)

    H = df[
        "H"
    ].to_numpy(
        dtype=float
    )

    U = df[
        "U"
    ].to_numpy(
        dtype=float
    )

    sides = (
        extract_side_to_move(
            df["fen"]
        )
    )

    # --------------------------------------------------------
    # 1. Raw H -> side-aware normalization
    # --------------------------------------------------------

    H_norm = (
        normalize_side_aware(
            H,
            sides,
        )
    )

    # --------------------------------------------------------
    # 2. Raw U -> logarithmic transform
    # --------------------------------------------------------

    F_U_raw = np.log1p(
        U / TAU
    )

    # --------------------------------------------------------
    # 3. F(U) -> side-aware normalization
    # --------------------------------------------------------

    F_U_norm = (
        normalize_side_aware(
            F_U_raw,
            sides,
        )
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
    # 6. Active Learning score
    # --------------------------------------------------------

    I = (
        RAW_W_H
        * H_star
        +
        RAW_W_U
        * F_U_star
        +
        RAW_W_HU
        * H_F_U_star
    )

    # --------------------------------------------------------
    # 7. Final min-max normalization
    # --------------------------------------------------------

    I_norm = minmax_normalize(
        I
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()
    print(
        "SIDE-AWARE NORMALIZATION"
    )
    print("-" * 70)

    print(
        "H and F(U) normalized independently "
        "for White-to-move and Black-to-move."
    )

    # --------------------------------------------------------
    # I distribution by side
    # --------------------------------------------------------

    print()
    print(
        "I DISTRIBUTION BY SIDE-TO-MOVE"
    )
    print("-" * 70)

    for side, name in [
        ("w", "White"),
        ("b", "Black"),
    ]:

        values = I[
            sides == side
        ]

        print()
        print(name)

        for q in [
            0.50,
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
            1.00,
        ]:

            print(
                f"P{q * 100:g} : "
                f"{np.quantile(values, q):+.9f}"
            )

    # --------------------------------------------------------
    # Normalized I distribution by side
    # --------------------------------------------------------

    print()
    print(
        "I_NORM DISTRIBUTION BY SIDE-TO-MOVE"
    )
    print("-" * 70)

    for side, name in [
        ("w", "White"),
        ("b", "Black"),
    ]:

        values = I_norm[
            sides == side
        ]

        print()
        print(name)

        for q in [
            0.50,
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
            1.00,
        ]:

            print(
                f"P{q * 100:g} : "
                f"{np.quantile(values, q):.9f}"
            )

    # --------------------------------------------------------
    # Predictor components
    # --------------------------------------------------------

    print()
    print(
        "PREDICTOR COMPONENTS BY SIDE-TO-MOVE"
    )
    print("-" * 70)

    components = [
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
    ]

    for label, values in components:

        print()
        print(label)

        for side, name in [
            ("w", "White"),
            ("b", "Black"),
        ]:

            subset = values[
                sides == side
            ]

            print(
                f"  {name:<6}: "
                f"mean={np.mean(subset):+.9f} | "
                f"std={np.std(subset):.9f} | "
                f"P50={np.quantile(subset, .50):+.9f} | "
                f"P90={np.quantile(subset, .90):+.9f} | "
                f"P99={np.quantile(subset, .99):+.9f} | "
                f"P99.9={np.quantile(subset, .999):+.9f} | "
                f"Max={np.max(subset):+.9f}"
            )

    # --------------------------------------------------------
    # Predictor diagnostics
    # --------------------------------------------------------

    print()
    print(
        "PREDICTORS"
    )
    print("-" * 70)

    print(
        f"H*       : "
        f"mean={np.mean(H_star):+.6f} "
        f"std={np.std(H_star):.6f}"
    )

    print(
        f"F(U)*    : "
        f"mean={np.mean(F_U_star):+.6f} "
        f"std={np.std(F_U_star):.6f}"
    )

    print(
        f"(HF(U))* : "
        f"mean={np.mean(H_F_U_star):+.6f} "
        f"std={np.std(H_F_U_star):.6f}"
    )

    # --------------------------------------------------------
    # Transformation diagnostics
    # --------------------------------------------------------

    print()
    print(
        "TRANSFORMATION"
    )
    print("-" * 70)

    print(
        f"TAU      : "
        f"{TAU:.6f}"
    )

    print(
        f"U raw    : "
        f"[{np.min(U):.6f}, "
        f"{np.max(U):.6f}]"
    )

    print(
        f"F(U) raw : "
        f"[{np.min(F_U_raw):.6f}, "
        f"{np.max(F_U_raw):.6f}]"
    )

    print(
        f"F(U) norm: "
        f"[{np.min(F_U_norm):.6f}, "
        f"{np.max(F_U_norm):.6f}]"
    )

    # --------------------------------------------------------
    # Standardization diagnostics
    # --------------------------------------------------------

    print()
    print(
        "STANDARDIZATION PARAMETERS"
    )
    print("-" * 70)

    print(
        f"H_norm mean/std       : "
        f"{np.mean(H_norm):.9f} / "
        f"{np.std(H_norm):.9f}"
    )

    print(
        f"F(U) norm mean/std    : "
        f"{np.mean(F_U_norm):.9f} / "
        f"{np.std(F_U_norm):.9f}"
    )

    print(
        f"H*F(U) mean/std      : "
        f"{np.mean(H_F_U):.9f} / "
        f"{np.std(H_F_U):.9f}"
    )

    return (
        I,
        I_norm,
        sides,
        H_norm,
        F_U_norm,
        F_U_raw,
        H_star,
        F_U_star,
        H_F_U_star,
    )


# ============================================================
# Budget / threshold
# ============================================================

def compute_budget_threshold(
    I,
    I_norm,
):

    n = len(I)

    target = max(
        1,
        int(
            np.ceil(
                n * AL_BUDGET
            )
        ),
    )

    # --------------------------------------------------------
    # Sort descending
    # --------------------------------------------------------

    order = np.argsort(
        I
    )[::-1]

    selected_indices = (
        order[:target]
    )

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
        len(selected_indices)
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
    I,
    I_norm,
    sides,
):

    print()
    print("=" * 70)
    print("ACTIVE LEARNING SCORE I")
    print("=" * 70)

    # --------------------------------------------------------
    # Raw OLS coefficients
    # --------------------------------------------------------

    print()
    print(
        "RAW OLS COEFFICIENTS"
    )
    print("-" * 70)

    print(
        f"RAW_W_H  : "
        f"{RAW_W_H:+.9f}"
    )

    print(
        f"RAW_W_U  : "
        f"{RAW_W_U:+.9f}"
    )

    print(
        f"RAW_W_HU : "
        f"{RAW_W_HU:+.9f}"
    )

    print()
    print(
        "No normalization is applied to the coefficients."
    )

    # --------------------------------------------------------
    # Raw score
    # --------------------------------------------------------

    print()
    print(
        "SCORE DISTRIBUTION"
    )
    print("-" * 70)

    for q in [
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
    ]:

        print(
            f"P{q * 100:<6g}: "
            f"{np.quantile(I, q):+.9f}"
        )

    # --------------------------------------------------------
    # Normalized score
    # --------------------------------------------------------

    print()
    print(
        "NORMALIZED SCORE DISTRIBUTION"
    )
    print("-" * 70)

    for q in [
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
    ]:

        print(
            f"P{q * 100:<6g}: "
            f"{np.quantile(I_norm, q):.9f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    I_min = float(
        np.min(I)
    )

    I_max = float(
        np.max(I)
    )

    print()
    print(
        "SUMMARY"
    )
    print("-" * 70)

    print(
        f"N      : "
        f"{len(I):,}"
    )

    print(
        f"Mean   : "
        f"{np.mean(I):+.9f}"
    )

    print(
        f"Std    : "
        f"{np.std(I):.9f}"
    )

    print(
        f"Median : "
        f"{np.median(I):+.9f}"
    )

    print(
        f"Min    : "
        f"{I_min:+.9f}"
    )

    print(
        f"Max    : "
        f"{I_max:+.9f}"
    )

    # --------------------------------------------------------
    # Min-max normalization
    # --------------------------------------------------------

    print()
    print(
        "MIN-MAX NORMALIZATION"
    )
    print("-" * 70)

    print(
        "I_norm = (I - I_min) / (I_max - I_min)"
    )

    print(
        f"I_norm min : "
        f"{np.min(I_norm):.9f}"
    )

    print(
        f"I_norm max : "
        f"{np.max(I_norm):.9f}"
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
        I,
        I_norm,
    )

    threshold_percentage = (
        threshold_norm
        * 100.0
    )

    print()
    print(
        "ACTIVE LEARNING SELECTION"
    )
    print("-" * 70)

    print(
        f"Budget             : "
        f"{AL_BUDGET:.5%}"
    )

    print(
        f"Target positions   : "
        f"{target:,}"
    )

    print(
        f"Threshold I        : "
        f"{threshold:+.9f}"
    )

    print(
        f"Threshold I_norm   : "
        f"{threshold_norm:.9f}"
    )

    print(
        f"Threshold range    : "
        f"{threshold_percentage:.4f}% "
        f"of [I_min, I_max]"
    )

    print(
        f"Selected positions : "
        f"{len(selected_indices):,}"
    )

    print(
        f"Actual fraction    : "
        f"{selected_fraction:.6%}"
    )

    # --------------------------------------------------------
    # Side composition
    # --------------------------------------------------------

    selected_sides = (
        sides[
            selected_indices
        ]
    )

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
    print(
        "SIDE-TO-MOVE"
    )
    print("-" * 70)

    print(
        f"Global White : "
        f"{total_white:,} "
        f"({white_fraction:.3%})"
    )

    print(
        f"Global Black : "
        f"{total_black:,} "
        f"({black_fraction:.3%})"
    )

    print()

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

    print()

    print(
        f"White enrichment : "
        f"{selected_white_fraction / white_fraction:.3f}x"
    )

    print(
        f"Black enrichment : "
        f"{selected_black_fraction / black_fraction:.3f}x"
    )

    return selected_indices


# ============================================================
# Plot
# ============================================================

def plot_distribution(
    I_norm,
):

    print()
    print("=" * 70)
    print("GENERATING DISTRIBUTION PLOT")
    print("=" * 70)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        I_norm,
        bins=100,
        alpha=0.75,
    )

    plt.xlabel(
        "Normalized active learning score I_norm"
    )

    plt.ylabel(
        "Number of positions"
    )

    plt.title(
        "Distribution of normalized Active Learning score"
    )

    plt.xlim(
        0,
        1,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
    )

    plt.close()

    print(
        f"Plot saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# Main
# ============================================================

def main():

    validate_configuration()

    df = load_data()

    (
        I,
        I_norm,
        sides,
        H_norm,
        F_U_norm,
        F_U_raw,
        H_star,
        F_U_star,
        H_F_U_star,
    ) = build_score(
        df
    )

    print_diagnostics(
        I,
        I_norm,
        sides,
    )

    plot_distribution(
        I_norm,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()