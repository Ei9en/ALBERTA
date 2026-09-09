#!/usr/bin/env python3

"""
ALBERTA - Active Learning Score I
=================================

Analyse et distribution du score d'Active Learning I.

Score
-----

Le score utilisé est la forme non linéaire retenue pour ALBERTA :

    I(H,U) = H / (1 + U) + 0.5 * sqrt(HU)

où :

    H  = incertitude / entropie individuelle
    U  = incertitude / désaccord collectif
    HU = interaction H * U fournie par le dataset

La formulation utilise directement les valeurs brutes H et U.

Aucune :
    - normalisation percentile side-aware
    - transformation logarithmique de U
    - standardisation
    - régression OLS
    - normalisation des coefficients

n'est appliquée avant le calcul de I.

Les paramètres sont fixes :

    alpha = 1
    beta  = 1
    gamma = 0.5

soit :

    I(H,U) = H / (1 + U) + 0.5 * sqrt(HU)

Normalisation finale
--------------------

Une normalisation min-max est appliquée uniquement au score final :

    I_norm = (I - I_min) / (I_max - I_min)

Elle ne modifie PAS le classement des positions.

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
    / "I_distribution_NL.png"
)

AL_BUDGET = 0.0002

# Fixed parameters of the nonlinear score.
ALPHA = 0.75
BETA = 2000
GAMMA = 2


# ============================================================
# Utilities
# ============================================================

def minmax_normalize(values):
    """
    Min-max normalization to [0, 1].

        x_norm = (x - min) / (max - min)

    This normalization is only representational.
    It does not change the ranking.
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

    # --------------------------------------------------------
    # Non-negativity
    # --------------------------------------------------------

    if (
        df["H"] < 0
    ).any():

        raise ValueError(
            "H contains negative values."
        )

    if (
        df["U"] < 0
    ).any():

        raise ValueError(
            "U contains negative values."
        )

    if (
        df["HU"] < 0
    ).any():

        raise ValueError(
            "HU contains negative values."
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
        "NONLINEAR ACTIVE LEARNING SCORE"
    )

    print("-" * 70)

    print(
        f"ALPHA    : "
        f"{ALPHA:.6f}"
    )

    print(
        f"BETA     : "
        f"{BETA:.6f}"
    )

    print(
        f"GAMMA    : "
        f"{GAMMA:.6f}"
    )

    print(
        f"AL budget: "
        f"{AL_BUDGET:.5%}"
    )

    print()

    print(
        "Formula:"
    )

    print(
        "I(H,U) = H / (1 + U) + 0.5 * sqrt(HU)"
    )

    print()

    if ALPHA <= 0:

        raise ValueError(
            "ALPHA must be strictly positive."
        )

    if BETA < 0:

        raise ValueError(
            "BETA must be non-negative."
        )

    if GAMMA < 0:

        raise ValueError(
            "GAMMA must be non-negative."
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

    HU = df[
        "HU"
    ].to_numpy(
        dtype=float
    )

    sides = (
        extract_side_to_move(
            df["fen"]
        )
    )

    # --------------------------------------------------------
    # 1. Main H term
    # --------------------------------------------------------

    H_term = (
        np.power(
            H,
            ALPHA,
        )
        /
        (
            1.0
            +
            BETA * U
        )
    )

    # --------------------------------------------------------
    # 2. Nonlinear interaction term
    # --------------------------------------------------------

    interaction_term = (
        GAMMA
        *
        np.sqrt(
            HU
        )
    )

    # --------------------------------------------------------
    # 3. Final score
    # --------------------------------------------------------

    I = (
        H_term
        +
        interaction_term
    )

    # --------------------------------------------------------
    # 4. Final min-max normalization
    # --------------------------------------------------------

    I_norm = minmax_normalize(
        I
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()
    print(
        "SCORE COMPONENTS"
    )
    print("-" * 70)

    print(
        "Main term:"
    )

    print(
        "    H / (1 + U)"
    )

    print(
        "Interaction term:"
    )

    print(
        "    0.5 * sqrt(HU)"
    )

    print()

    # --------------------------------------------------------
    # Score components globally
    # --------------------------------------------------------

    print(
        "GLOBAL COMPONENT DISTRIBUTION"
    )
    print("-" * 70)

    for label, values in [
        (
            "H term",
            H_term,
        ),
        (
            "Interaction term",
            interaction_term,
        ),
        (
            "Final I",
            I,
        ),
    ]:

        print()
        print(label)

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
    # Score distribution by side
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
    # Normalized score distribution by side
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
    # Raw predictors by side
    # --------------------------------------------------------

    print()
    print(
        "RAW PREDICTORS BY SIDE-TO-MOVE"
    )
    print("-" * 70)

    predictors = [
        (
            "H",
            H,
        ),
        (
            "U",
            U,
        ),
        (
            "HU",
            HU,
        ),
    ]

    for label, values in predictors:

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
    # Component contribution
    # --------------------------------------------------------

    print()
    print(
        "COMPONENT CONTRIBUTION"
    )
    print("-" * 70)

    print(
        f"Mean H term           : "
        f"{np.mean(H_term):+.9f}"
    )

    print(
        f"Mean interaction term : "
        f"{np.mean(interaction_term):+.9f}"
    )

    print(
        f"Mean I                : "
        f"{np.mean(I):+.9f}"
    )

    print()

    print(
        f"Median H term           : "
        f"{np.median(H_term):+.9f}"
    )

    print(
        f"Median interaction term : "
        f"{np.median(interaction_term):+.9f}"
    )

    print(
        f"Median I                : "
        f"{np.median(I):+.9f}"
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
        "No logarithmic transformation is applied to U."
    )

    print(
        "No percentile normalization is applied."
    )

    print(
        "No standardization is applied."
    )

    print(
        "H, U and HU are used on their raw dataset scales."
    )

    print()

    print(
        f"H raw  : "
        f"[{np.min(H):.9f}, "
        f"{np.max(H):.9f}]"
    )

    print(
        f"U raw  : "
        f"[{np.min(U):.9f}, "
        f"{np.max(U):.9f}]"
    )

    print(
        f"HU raw : "
        f"[{np.min(HU):.9f}, "
        f"{np.max(HU):.9f}]"
    )

    # --------------------------------------------------------
    # Interaction consistency
    # --------------------------------------------------------

    expected_HU = (
        H * U
    )

    absolute_error = np.abs(
        HU - expected_HU
    )

    max_error = float(
        np.max(
            absolute_error
        )
    )

    mean_error = float(
        np.mean(
            absolute_error
        )
    )

    print()
    print(
        "HU CONSISTENCY CHECK"
    )
    print("-" * 70)

    print(
        "HU is retained from the dataset and used directly."
    )

    print(
        f"Mean |HU - H*U| : "
        f"{mean_error:.12e}"
    )

    print(
        f"Max  |HU - H*U| : "
        f"{max_error:.12e}"
    )

    if max_error > 1e-8:

        print()

        print(
            "WARNING: HU is not numerically identical "
            "to H * U at the 1e-8 tolerance."
        )

        print(
            "The score continues to use the dataset HU field."
        )

    return (
        I,
        I_norm,
        sides,
        H,
        U,
        HU,
        H_term,
        interaction_term,
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
    # Formula
    # --------------------------------------------------------

    print()
    print(
        "SCORE DEFINITION"
    )
    print("-" * 70)

    print(
        "I(H,U) = H / (1 + U) + 0.5 * sqrt(HU)"
    )

    print()

    print(
        f"ALPHA : "
        f"{ALPHA:.6f}"
    )

    print(
        f"BETA  : "
        f"{BETA:.6f}"
    )

    print(
        f"GAMMA : "
        f"{GAMMA:.6f}"
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
        "Distribution of nonlinear Active Learning score"
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
        H,
        U,
        HU,
        H_term,
        interaction_term,
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