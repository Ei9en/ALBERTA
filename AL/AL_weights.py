#!/usr/bin/env python3

"""
ALBERTA - Active Learning Weight Estimation
============================================

Estimate the coefficients of the active learning score:

    I = W_H * H
      + W_U * log(1 + U / TAU)
      + W_HU * H * log(1 + U / TAU)

The coefficients are estimated using multivariate ordinary
least squares (OLS).

Target
------

The target is the absolute terminal reward:

    Y = |R|

This measures how strongly the position is associated with
a decisive outcome, independently of its direction.

Preprocessing
-------------

1. Extract side-to-move from the FEN.
2. Transform U using:

       log(1 + U / TAU)

3. Apply side-aware percentile-rank normalization to H and
   transformed U independently for White and Black.
4. Build the interaction:

       H_norm * U_log_norm

5. Standardize the three predictors and the target.
6. Fit:

       Y* = W_H H*
          + W_U U*
          + W_HU (H U)* + epsilon

The fitted coefficients are then normalized so that:

       W_H + W_U + W_HU = 1

Outputs
-------

    data/uncertainty_analysis/
        active_learning_weights.py
        active_learning_weight_report.txt

The generated Python file can be imported directly by:

    seed_oracle_queue.py

Usage
-----

    python estimate_active_learning_weights.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_analysis"
)

WEIGHTS_FILE = (
    OUTPUT_DIR
    / "active_learning_weights.py"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "active_learning_weight_report.txt"
)

# ------------------------------------------------------------
# U transformation
# ------------------------------------------------------------

TAU = 0.05


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
            == sorted_values[start]
        ):
            end += 1

        # Mid-rank for ties.
        rank = (
            (start + end - 1)
            / 2.0
        )

        ranks[
            order[start:end]
        ] = (
            rank
            / (n - 1)
        )

        start = end

    return ranks


# ============================================================
# Side extraction
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
                f"Invalid side-to-move: {fen}"
            )

        sides.append(side)

    return np.asarray(
        sides,
        dtype="<U1"
    )


# ============================================================
# Side-aware normalization
# ============================================================

def normalize_side_aware(
    values,
    sides
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    normalized = np.empty_like(
        values
    )

    for side in (
        "w",
        "b"
    ):

        mask = (
            sides == side
        )

        count = int(
            np.sum(mask)
        )

        if count < 2:

            raise ValueError(
                f"Not enough positions for "
                f"side {side} normalization."
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
# Standardization
# ============================================================

def standardize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(
            values,
            ddof=0
        )
    )

    if std <= 0:

        raise ValueError(
            "Cannot standardize a constant variable."
        )

    return (
        (values - mean) / std,
        mean,
        std,
    )


# ============================================================
# Reward conversion
# ============================================================

def result_to_reward(
    result,
    side
):

    # Reward from side-to-move perspective.
    #
    # White result:
    #   1-0      -> +1 for White
    #   0-1      -> -1 for White
    #
    # Black result:
    #   1-0      -> -1 for Black
    #   0-1      -> +1 for Black
    #
    # Draw:
    #   0

    if result == "1/2-1/2":

        return 0.0

    if result == "1-0":

        return 1.0 if side == "w" else -1.0

    if result == "0-1":

        return -1.0 if side == "w" else 1.0

    raise ValueError(
        f"Invalid result: {result}"
    )


# ============================================================
# Load data
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print(
        "ALBERTA - ACTIVE LEARNING WEIGHT ESTIMATION"
    )
    print("=" * 70)

    print()
    print(
        "Loading dataset..."
    )

    print(
        f"File: {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        raw = json.load(f)

    print(
        f"Raw records: {len(raw):,}"
    )

    fens = []
    H = []
    U = []
    results = []

    invalid = 0

    for record in raw:

        if not isinstance(
            record,
            dict
        ):

            invalid += 1
            continue

        fen = record.get(
            "fen"
        )

        h = record.get(
            "H"
        )

        u = record.get(
            "U"
        )

        result = record.get(
            "result"
        )

        if not isinstance(
            fen,
            str
        ):

            invalid += 1
            continue

        if result not in {
            "1-0",
            "0-1",
            "1/2-1/2",
        }:

            invalid += 1
            continue

        try:

            h = float(h)
            u = float(u)

        except (
            TypeError,
            ValueError
        ):

            invalid += 1
            continue

        if not (
            np.isfinite(h)
            and np.isfinite(u)
        ):

            invalid += 1
            continue

        if u < 0:

            invalid += 1
            continue

        fens.append(fen)
        H.append(h)
        U.append(u)
        results.append(result)

    print(
        f"Valid records: {len(fens):,}"
    )

    print(
        f"Invalid records skipped: {invalid:,}"
    )

    return (
        np.asarray(
            fens,
            dtype=object
        ),
        np.asarray(
            H,
            dtype=np.float64
        ),
        np.asarray(
            U,
            dtype=np.float64
        ),
        results,
    )


# ============================================================
# Build regression dataset
# ============================================================

def build_regression_data(
    fens,
    H,
    U,
    results
):

    print()
    print(
        "Building regression variables..."
    )

    sides = extract_side(
        fens
    )

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    rewards = np.asarray(
        [
            result_to_reward(
                result,
                side
            )
            for result, side
            in zip(
                results,
                sides
            )
        ],
        dtype=np.float64
    )

    # Target is reward magnitude. test on both abs(R) and 1 - abs(R)
    
    Y = np.abs(
        rewards
    )

    # --------------------------------------------------------
    # U transformation
    # --------------------------------------------------------

    U_log = np.log1p(
        U / TAU
    )

    # --------------------------------------------------------
    # Side-aware normalization
    # --------------------------------------------------------

    print(
        "Normalizing H by side-to-move..."
    )

    H_norm = normalize_side_aware(
        H,
        sides
    )

    print(
        "Normalizing log-transformed U "
        "by side-to-move..."
    )

    U_log_norm = normalize_side_aware(
        U_log,
        sides
    )

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    HU_interaction = (
        H_norm
        * U_log_norm
    )

    return (
        H_norm,
        U_log_norm,
        HU_interaction,
        Y,
        rewards,
        sides,
    )


# ============================================================
# Pearson correlation
# ============================================================

def correlation(
    x,
    y
):

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


# ============================================================
# OLS
# ============================================================

def fit_ols(
    X,
    y
):

    # No intercept because all variables are standardized.
    beta, residuals, rank, singular_values = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None
        )
    )

    predictions = (
        X @ beta
    )

    residual = (
        y - predictions
    )

    ss_res = np.sum(
        residual ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return (
        beta,
        predictions,
        residual,
        r_squared,
        rank,
        singular_values,
    )


# ============================================================
# VIF
# ============================================================

def calculate_vif(X):

    n_features = X.shape[1]

    vif = []

    for i in range(
        n_features
    ):

        target = X[:, i]

        others = np.delete(
            X,
            i,
            axis=1
        )

        beta, _, _, _ = np.linalg.lstsq(
            others,
            target,
            rcond=None
        )

        prediction = (
            others @ beta
        )

        residual = (
            target - prediction
        )

        ss_res = np.sum(
            residual ** 2
        )

        ss_tot = np.sum(
            (target - target.mean()) ** 2
        )

        r2 = (
            1.0 - ss_res / ss_tot
            if ss_tot > 0
            else 0.0
        )

        denominator = (
            1.0 - r2
        )

        if denominator <= 1e-12:

            vif_value = np.inf

        else:

            vif_value = (
                1.0 / denominator
            )

        vif.append(
            vif_value
        )

    return np.asarray(
        vif,
        dtype=np.float64
    )


# ============================================================
# Normalize weights
# ============================================================

def normalize_weights(
    coefficients
):

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64
    )

    total = np.sum(
        coefficients
    )

    if abs(total) < 1e-12:

        raise ValueError(
            "Coefficient sum is too close to zero; "
            "cannot normalize weights."
        )

    return (
        coefficients / total
    )


# ============================================================
# Report
# ============================================================

def build_report(
    n,
    coefficients,
    normalized_weights,
    r_squared,
    correlations,
    predictor_correlations,
    vif,
    sides,
    rewards,
):

    names = [
        "H",
        "log(U)",
        "H x log(U)",
    ]

    lines = []

    def add(text=""):
        lines.append(
            str(text)
        )

    add("=" * 70)
    add(
        "ALBERTA - ACTIVE LEARNING WEIGHT ESTIMATION"
    )
    add("=" * 70)
    add()

    add(
        f"Dataset size: {n:,}"
    )

    add(
        f"TAU: {TAU}"
    )

    add()

    # --------------------------------------------------------
    # Reward distribution
    # --------------------------------------------------------

    add(
        "TARGET"
    )
    add("-" * 70)

    add(
        "Target: |R|"
    )

    add(
        f"Mean |R|: "
        f"{np.mean(np.abs(rewards)):.6f}"
    )

    add(
        f"Mean R: "
        f"{np.mean(rewards):+.6f}"
    )

    add(
        f"White-to-move: "
        f"{np.sum(sides == 'w'):,}"
    )

    add(
        f"Black-to-move: "
        f"{np.sum(sides == 'b'):,}"
    )

    add()

    # --------------------------------------------------------
    # Simple correlations
    # --------------------------------------------------------

    add(
        "SIMPLE PEARSON CORRELATIONS WITH |R|"
    )
    add("-" * 70)

    for name in names:

        add(
            f"{name:<15} : "
            f"{correlations[name]:+.6f}"
        )

    add()

    # --------------------------------------------------------
    # Predictor correlations
    # --------------------------------------------------------

    add(
        "PREDICTOR CORRELATIONS"
    )
    add("-" * 70)

    for i in range(
        len(names)
    ):

        for j in range(
            i + 1,
            len(names)
        ):

            add(
                f"{names[i]:<15} vs "
                f"{names[j]:<15} : "
                f"{predictor_correlations[i, j]:+.6f}"
            )

    add()

    # --------------------------------------------------------
    # VIF
    # --------------------------------------------------------

    add(
        "VARIANCE INFLATION FACTORS"
    )
    add("-" * 70)

    for name, value in zip(
        names,
        vif
    ):

        add(
            f"{name:<15} : "
            f"{value:.4f}"
        )

    add()

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    add(
        "OLS COEFFICIENTS"
    )
    add("-" * 70)

    for name, value in zip(
        names,
        coefficients
    ):

        add(
            f"{name:<15} : "
            f"{value:+.8f}"
        )

    add()

    add(
        f"R²: {r_squared:.8f}"
    )

    add()

    # --------------------------------------------------------
    # Normalized weights
    # --------------------------------------------------------

    add(
        "NORMALIZED ACTIVE LEARNING WEIGHTS"
    )
    add("-" * 70)

    for name, value in zip(
        names,
        normalized_weights
    ):

        add(
            f"W_{name:<13} : "
            f"{value:+.8f}"
        )

    add()

    add(
        "These normalized weights sum to 1."
    )

    add(
        "They are intended to be frozen after estimation."
    )

    add()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    add(
        "INTERPRETATION"
    )
    add("-" * 70)

    add(
        "The coefficients estimate the marginal "
        "association of each standardized predictor "
        "with reward magnitude |R| while controlling "
        "for the other predictors."
    )

    add()

    add(
        "This is an observational statistical "
        "association and does not establish causality."
    )

    add()

    add(
        "The coefficients should be estimated on a "
        "reference dataset and then frozen before "
        "the active-learning selection experiment."
    )

    return "\n".join(lines)


# ============================================================
# Save weights
# ============================================================

def save_weights(
    coefficients,
    normalized_weights
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    names = [
        "W_H",
        "W_U",
        "W_HU",
    ]

    with open(
        WEIGHTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Auto-generated by "
            "estimate_active_learning_weights.py\n"
        )

        f.write(
            "# Do not modify manually.\n\n"
        )

        f.write(
            f"TAU = {TAU!r}\n\n"
        )

        for name, value in zip(
            names,
            normalized_weights
        ):

            f.write(
                f"{name} = "
                f"{float(value)!r}\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "# Raw standardized OLS coefficients\n"
        )

        for name, value in zip(
            names,
            coefficients
        ):

            f.write(
                f"RAW_{name} = "
                f"{float(value)!r}\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        fens,
        H,
        U,
        results,
    ) = load_data()

    if len(fens) == 0:

        raise RuntimeError(
            "No valid observations."
        )

    # --------------------------------------------------------
    # Build regression variables
    # --------------------------------------------------------

    (
        H_norm,
        U_log_norm,
        HU_interaction,
        Y,
        rewards,
        sides,
    ) = build_regression_data(
        fens,
        H,
        U,
        results
    )

    # --------------------------------------------------------
    # Standardize predictors and target
    # --------------------------------------------------------

    print()
    print(
        "Standardizing variables..."
    )

    H_star, H_mean, H_std = (
        standardize(H_norm)
    )

    U_star, U_mean, U_std = (
        standardize(U_log_norm)
    )

    HU_star, HU_mean, HU_std = (
        standardize(HU_interaction)
    )

    Y_star, Y_mean, Y_std = (
        standardize(Y)
    )

    X = np.column_stack(
        [
            H_star,
            U_star,
            HU_star,
        ]
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlations = {
        "H":
            correlation(
                H_star,
                Y_star
            ),

        "log(U)":
            correlation(
                U_star,
                Y_star
            ),

        "H x log(U)":
            correlation(
                HU_star,
                Y_star
            ),
    }

    predictor_correlations = (
        np.corrcoef(
            X,
            rowvar=False
        )
    )

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    print()
    print(
        "Fitting OLS..."
    )

    (
        coefficients,
        predictions,
        residual,
        r_squared,
        rank,
        singular_values,
    ) = fit_ols(
        X,
        Y_star
    )

    # --------------------------------------------------------
    # VIF
    # --------------------------------------------------------

    vif = calculate_vif(
        X
    )

    # --------------------------------------------------------
    # Normalize coefficients
    # --------------------------------------------------------

    normalized_weights = (
        normalize_weights(
            coefficients
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "OLS RESULTS"
    )
    print("=" * 70)

    print()

    for name, value in zip(
        [
            "W_H",
            "W_U",
            "W_HU",
        ],
        coefficients
    ):

        print(
            f"{name:<8}: "
            f"{value:+.8f}"
        )

    print()

    print(
        f"R²      : "
        f"{r_squared:.8f}"
    )

    print()

    print(
        "NORMALIZED WEIGHTS"
    )

    print("-" * 70)

    for name, value in zip(
        [
            "W_H",
            "W_U",
            "W_HU",
        ],
        normalized_weights
    ):

        print(
            f"{name:<8}: "
            f"{value:+.8f}"
        )

    print()

    print(
        f"Sum     : "
        f"{np.sum(normalized_weights):.8f}"
    )

    print()

    print(
        "SIMPLE CORRELATIONS WITH |R|"
    )

    print("-" * 70)

    for name, value in correlations.items():

        print(
            f"{name:<15}: "
            f"{value:+.6f}"
        )

    print()

    print(
        "VIF"
    )

    print("-" * 70)

    for name, value in zip(
        [
            "H",
            "log(U)",
            "H x log(U)",
        ],
        vif
    ):

        print(
            f"{name:<15}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Save generated weights
    # --------------------------------------------------------

    save_weights(
        coefficients,
        normalized_weights
    )

    report = build_report(
        len(fens),
        coefficients,
        normalized_weights,
        r_squared,
        correlations,
        predictor_correlations,
        vif,
        sides,
        rewards,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "OUTPUT"
    )
    print("=" * 70)

    print(
        f"Weights : {WEIGHTS_FILE}"
    )

    print(
        f"Report  : {REPORT_FILE}"
    )

    print()
    print(
        "The generated weights can now be imported "
        "by seed_oracle_queue.py."
    )


if __name__ == "__main__":

    main()