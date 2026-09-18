#!/usr/bin/env python3

"""
ALBERTA - Active Learning Weight Estimation
============================================

Estimate the coefficients of the ALBERTA active-learning
acquisition score from a reference uncertainty dataset.

Target
------

The regression target is the magnitude of the terminal reward:

    Y = |R|

where R is expressed from the side-to-move perspective.

Therefore:

    |R| = 1  -> decisive game
    |R| = 0  -> draw

This target measures association with decisiveness, not learning
value and not the direction of the outcome.

Predictors
----------

Given raw policy entropy H and value-disagreement uncertainty U,
H and U are percentile-rank normalized independently within each:

    chronological-window x side-to-move

stratum.

The chronological windows are contiguous equal-size partitions of
the append-ordered uncertainty dataset. They are coarse temporal
strata and must not be interpreted as exact RL epochs.

This temporal normalization is used because both H and historical
league disagreement U evolve systematically during RL training.
Without temporal stratification, the global percentile rank of U can
partly encode when an observation was collected rather than only its
relative disagreement within the current learning regime.

The interaction is then defined as:

    H_norm * U_norm

All three predictors and the target are standardized before OLS.

The regression is:

    Y* =
        RAW_W_H  * H*
        + RAW_W_U  * U*
        + RAW_W_HU * (H_norm * U_norm)*
        + epsilon

The RAW_W_* coefficients are the canonical coefficients used by
the ALBERTA acquisition score.

For diagnostics only, an additional set of coefficients normalized
to sum to one is also reported as W_H, W_U and W_HU.

Outputs
-------

    AL/AL_weights.py

    data/uncertainty_analysis/
        active_learning_weight_report.txt

The generated AL/AL_weights.py is imported directly by:

    AL/seed_oracle_queue.py
    AL/distribution_I.py

IMPORTANT
---------

The same chronological-window x side-to-move normalization used here
must also be used by seed_oracle_queue.py and distribution_I.py when
the resulting RAW_W_* coefficients are applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

DEFAULT_WEIGHTS_FILE = (
    PROJECT_ROOT
    / "AL"
    / "AL_weights.py"
)

DEFAULT_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_analysis"
    / "active_learning_weight_report.txt"
)

DEFAULT_TEMPORAL_WINDOWS = 6


# ============================================================
# Percentile-rank normalization
# ============================================================

def percentile_rank(
    values: np.ndarray,
) -> np.ndarray:
    """
    Percentile rank in [0, 1].

    Ties receive their average zero-based rank.

    This implementation must remain identical to the one used
    by seed_oracle_queue.py and distribution_I.py.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(
        values
    )

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

        end = (
            start + 1
        )

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
            order[
                start:end
            ]
        ] = (
            average_rank
            / (n - 1)
        )

        start = end

    return ranks


# ============================================================
# Side extraction
# ============================================================

def extract_side(
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

        side = parts[
            1
        ]

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

    The full dataset is divided into n_windows contiguous,
    approximately equal-size windows.

    Within each (window, side) stratum, values are independently
    converted to percentile ranks in [0, 1].

    This prevents systematic temporal drift in H or U from directly
    dominating the acquisition ranking.
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
# Standardization
# ============================================================

def standardize(
    values: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
    float,
]:

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
            "Cannot standardize a constant variable."
        )

    standardized = (
        values
        - mean
    ) / std

    return (
        standardized,
        mean,
        std,
    )


# ============================================================
# Reward conversion
# ============================================================

def result_to_reward(
    result: str,
    side: str,
) -> float:
    """
    Convert the terminal game result into reward from the
    side-to-move perspective.
    """

    if result == "1/2-1/2":

        return 0.0

    if result == "1-0":

        return (
            1.0
            if side == "w"
            else -1.0
        )

    if result == "0-1":

        return (
            -1.0
            if side == "w"
            else 1.0
        )

    raise ValueError(
        f"Invalid result: {result}"
    )


# ============================================================
# Data loading
# ============================================================

def load_data(
    input_file: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:

    print()
    print("=" * 70)
    print("ALBERTA - ACTIVE LEARNING WEIGHT ESTIMATION")
    print("=" * 70)

    print()
    print(
        f"Input file: {input_file}"
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    with input_file.open(
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
    results = []

    invalid = 0

    for record in raw:

        if not isinstance(
            record,
            dict,
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
            str,
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

            h = float(
                h
            )

            u = float(
                u
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid += 1
            continue

        if not (
            np.isfinite(
                h
            )
            and np.isfinite(
                u
            )
        ):

            invalid += 1
            continue

        if u < 0.0:

            invalid += 1
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

        results.append(
            result
        )

    print(
        f"Valid records          : {len(fens):,}"
    )

    print(
        f"Invalid records skipped: {invalid:,}"
    )

    if not fens:

        raise RuntimeError(
            "No valid observations."
        )

    return (
        np.asarray(
            fens,
            dtype=object,
        ),
        np.asarray(
            H,
            dtype=np.float64,
        ),
        np.asarray(
            U,
            dtype=np.float64,
        ),
        results,
    )


# ============================================================
# Regression dataset
# ============================================================

def build_regression_data(
    fens: np.ndarray,
    H: np.ndarray,
    U: np.ndarray,
    results: list[str],
    n_windows: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    print()
    print(
        "Building regression variables..."
    )

    sides = extract_side(
        fens
    )

    # --------------------------------------------------------
    # Terminal reward from side-to-move perspective
    # --------------------------------------------------------

    rewards = np.asarray(
        [
            result_to_reward(
                result,
                side,
            )
            for result, side in zip(
                results,
                sides,
            )
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Regression target
    #
    #     Y = |R|
    #
    # This is decisiveness, not signed outcome.
    # --------------------------------------------------------

    Y = np.abs(
        rewards
    )

    # --------------------------------------------------------
    # Temporal + side-aware normalization of H
    # --------------------------------------------------------

    print(
        "Normalizing H by chronological window and side-to-move..."
    )

    (
        H_norm,
        window_ids,
    ) = normalize_temporal_side_aware(
        values=H,
        sides=sides,
        n_windows=n_windows,
    )

    # --------------------------------------------------------
    # Temporal + side-aware normalization of raw U
    # --------------------------------------------------------

    print(
        "Normalizing U by chronological window and side-to-move..."
    )

    (
        U_norm,
        window_ids_u,
    ) = normalize_temporal_side_aware(
        values=U,
        sides=sides,
        n_windows=n_windows,
    )

    if not np.array_equal(
        window_ids,
        window_ids_u,
    ):

        raise RuntimeError(
            "Temporal-window assignments are inconsistent."
        )

    # --------------------------------------------------------
    # Factorial interaction
    # --------------------------------------------------------

    HU_interaction = (
        H_norm
        * U_norm
    )

    return (
        H_norm,
        U_norm,
        HU_interaction,
        Y,
        rewards,
        sides,
        window_ids,
    )


# ============================================================
# Temporal normalization diagnostics
# ============================================================

def print_temporal_normalization_diagnostics(
    H_norm: np.ndarray,
    U_norm: np.ndarray,
    sides: np.ndarray,
    window_ids: np.ndarray,
    n_windows: int,
) -> None:

    print()
    print("=" * 70)
    print("TEMPORAL NORMALIZATION DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        "Expected percentile means are approximately 0.5 "
        "within each window x side stratum."
    )

    for window in range(
        n_windows
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

            count = int(
                np.sum(
                    mask
                )
            )

            if count == 0:

                raise RuntimeError(
                    f"Empty temporal normalization stratum: "
                    f"W{window + 1}, side={side}"
                )

            print(
                f"  {side}: "
                f"n={count:,}, "
                f"H_norm_mean={np.mean(H_norm[mask]):.6f}, "
                f"U_norm_mean={np.mean(U_norm[mask]):.6f}, "
                f"H_norm_std={np.std(H_norm[mask]):.6f}, "
                f"U_norm_std={np.std(U_norm[mask]):.6f}"
            )


# ============================================================
# Correlation
# ============================================================

def correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    return float(
        np.corrcoef(
            x,
            y,
        )[
            0,
            1,
        ]
    )


# ============================================================
# OLS
# ============================================================

def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    np.ndarray,
]:
    """
    Fit OLS without intercept.

    All predictors and the target are already centered and
    standardized, therefore an intercept is unnecessary.
    """

    (
        beta,
        _,
        rank,
        singular_values,
    ) = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    predictions = (
        X
        @ beta
    )

    residual = (
        y
        - predictions
    )

    ss_res = float(
        np.sum(
            residual ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                y
                - np.mean(
                    y
                )
            ) ** 2
        )
    )

    r_squared = (
        1.0
        - ss_res
        / ss_tot
        if ss_tot > 0.0
        else float(
            "nan"
        )
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
# Variance inflation factors
# ============================================================

def calculate_vif(
    X: np.ndarray,
) -> np.ndarray:

    n_features = (
        X.shape[
            1
        ]
    )

    vif = []

    for i in range(
        n_features
    ):

        target = (
            X[
                :,
                i,
            ]
        )

        others = np.delete(
            X,
            i,
            axis=1,
        )

        (
            beta,
            _,
            _,
            _,
        ) = np.linalg.lstsq(
            others,
            target,
            rcond=None,
        )

        prediction = (
            others
            @ beta
        )

        residual = (
            target
            - prediction
        )

        ss_res = float(
            np.sum(
                residual ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    target
                    - np.mean(
                        target
                    )
                ) ** 2
            )
        )

        r_squared = (
            1.0
            - ss_res
            / ss_tot
            if ss_tot > 0.0
            else 0.0
        )

        denominator = (
            1.0
            - r_squared
        )

        if denominator <= 1e-12:

            vif_value = np.inf

        else:

            vif_value = (
                1.0
                / denominator
            )

        vif.append(
            vif_value
        )

    return np.asarray(
        vif,
        dtype=np.float64,
    )


# ============================================================
# Diagnostic coefficient normalization
# ============================================================

def normalize_coefficients(
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Normalize coefficients so they sum to one.

    These values are diagnostic only.

    The canonical ALBERTA acquisition score uses RAW_W_*.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    total = float(
        np.sum(
            coefficients
        )
    )

    if abs(
        total
    ) < 1e-12:

        raise ValueError(
            "Coefficient sum is too close to zero; "
            "cannot normalize coefficients."
        )

    return (
        coefficients
        / total
    )


# ============================================================
# Report
# ============================================================

def build_report(
    n: int,
    n_windows: int,
    coefficients: np.ndarray,
    normalized_weights: np.ndarray,
    r_squared: float,
    correlations: dict[str, float],
    predictor_correlations: np.ndarray,
    vif: np.ndarray,
    sides: np.ndarray,
    rewards: np.ndarray,
    rank: int,
    singular_values: np.ndarray,
    H_norm: np.ndarray,
    U_norm: np.ndarray,
    window_ids: np.ndarray,
) -> str:

    names = [
        "H",
        "U",
        "H x U",
    ]

    lines = []

    def add(
        text: str = "",
    ) -> None:

        lines.append(
            str(
                text
            )
        )

    add(
        "=" * 70
    )

    add(
        "ALBERTA - ACTIVE LEARNING WEIGHT ESTIMATION"
    )

    add(
        "=" * 70
    )

    add()

    add(
        f"Dataset size: {n:,}"
    )

    add(
        f"Chronological windows: {n_windows}"
    )

    add()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    add(
        "TARGET"
    )

    add(
        "-" * 70
    )

    add(
        "Target: |R|"
    )

    add(
        "Interpretation: terminal-game decisiveness."
    )

    add(
        "This is not a direct measure of learning value."
    )

    add()

    add(
        f"Mean |R|: {np.mean(np.abs(rewards)):.6f}"
    )

    add(
        f"Mean R: {np.mean(rewards):+.6f}"
    )

    add(
        f"White-to-move: {np.sum(sides == 'w'):,}"
    )

    add(
        f"Black-to-move: {np.sum(sides == 'b'):,}"
    )

    add()

    # --------------------------------------------------------
    # Temporal normalization
    # --------------------------------------------------------

    add(
        "TEMPORAL NORMALIZATION"
    )

    add(
        "-" * 70
    )

    add(
        f"Chronological windows: {n_windows}"
    )

    add()

    add(
        "H and U are percentile-rank normalized independently "
        "within each chronological-window x side-to-move stratum."
    )

    add()

    add(
        "The chronological windows are equal-size contiguous "
        "partitions of the append-ordered uncertainty dataset."
    )

    add(
        "They are coarse temporal strata and must not be "
        "interpreted as exact RL epochs."
    )

    add()

    add(
        "This normalization is intended to prevent systematic "
        "temporal drift in H or historical-league disagreement U "
        "from dominating acquisition ranking."
    )

    add()

    for window in range(
        n_windows
    ):

        window_mask = (
            window_ids
            == window
        )

        add(
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

            add(
                f"  {side}: "
                f"n={np.sum(mask):,}, "
                f"H_norm_mean={np.mean(H_norm[mask]):.6f}, "
                f"U_norm_mean={np.mean(U_norm[mask]):.6f}"
            )

    add()

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    add(
        "SIMPLE PEARSON CORRELATIONS WITH |R|"
    )

    add(
        "-" * 70
    )

    for name in names:

        add(
            f"{name:<15}: "
            f"{correlations[name]:+.6f}"
        )

    add()

    add(
        "PREDICTOR CORRELATIONS"
    )

    add(
        "-" * 70
    )

    for i in range(
        len(
            names
        )
    ):

        for j in range(
            i + 1,
            len(
                names
            ),
        ):

            add(
                f"{names[i]:<15} vs "
                f"{names[j]:<15}: "
                f"{predictor_correlations[i, j]:+.6f}"
            )

    add()

    # --------------------------------------------------------
    # VIF
    # --------------------------------------------------------

    add(
        "VARIANCE INFLATION FACTORS"
    )

    add(
        "-" * 70
    )

    for name, value in zip(
        names,
        vif,
    ):

        add(
            f"{name:<15}: "
            f"{value:.4f}"
        )

    add()

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    add(
        "RAW STANDARDIZED OLS COEFFICIENTS"
    )

    add(
        "-" * 70
    )

    for name, value in zip(
        names,
        coefficients,
    ):

        add(
            f"{name:<15}: "
            f"{value:+.8f}"
        )

    add()

    add(
        f"R²: {r_squared:.8f}"
    )

    add(
        f"Design-matrix rank: {rank}"
    )

    add(
        "Singular values: "
        + ", ".join(
            f"{value:.8f}"
            for value in singular_values
        )
    )

    add()

    # --------------------------------------------------------
    # Diagnostic normalized coefficients
    # --------------------------------------------------------

    add(
        "SUM-NORMALIZED COEFFICIENTS (DIAGNOSTIC ONLY)"
    )

    add(
        "-" * 70
    )

    for name, value in zip(
        names,
        normalized_weights,
    ):

        add(
            f"{name:<15}: "
            f"{value:+.8f}"
        )

    add()

    add(
        f"Sum: {np.sum(normalized_weights):.8f}"
    )

    add()

    add(
        "The acquisition score itself uses the RAW_W_* "
        "coefficients, not these sum-normalized values."
    )

    add()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    add(
        "INTERPRETATION"
    )

    add(
        "-" * 70
    )

    add(
        "The coefficients estimate the marginal association "
        "of each standardized predictor with terminal reward "
        "magnitude |R| while controlling for the other predictors."
    )

    add()

    add(
        "The temporal stratification means that H and U are "
        "interpreted relative to the local learning regime in which "
        "an observation was collected, rather than relative to the "
        "entire non-stationary RL1-10 population."
    )

    add()

    add(
        "This is an observational association. It does not "
        "establish causal learning value or useful update direction."
    )

    add()

    add(
        "Weights should be estimated once on the reference "
        "uncertainty dataset and frozen before query selection."
    )

    add()

    add(
        "The same temporal-window x side-to-move normalization "
        "must be used when applying these coefficients during "
        "query selection."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Save generated weights
# ============================================================

def save_weights(
    weights_file: Path,
    n_windows: int,
    coefficients: np.ndarray,
    normalized_weights: np.ndarray,
) -> None:

    weights_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_names = [
        "RAW_W_H",
        "RAW_W_U",
        "RAW_W_HU",
    ]

    normalized_names = [
        "W_H",
        "W_U",
        "W_HU",
    ]

    with weights_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            '"""\n'
        )

        f.write(
            "Auto-generated ALBERTA active-learning calibration.\n\n"
        )

        f.write(
            "Generated by AL/estimate_active_learning_weights.py.\n"
        )

        f.write(
            "Do not modify manually.\n\n"
        )

        f.write(
            "RAW_W_* are the canonical OLS coefficients used by\n"
        )

        f.write(
            "the acquisition score. W_* are diagnostic coefficients\n"
        )

        f.write(
            "normalized to sum to one.\n\n"
        )

        f.write(
            "Calibration uses chronological-window x side-to-move\n"
        )

        f.write(
            "percentile normalization of raw H and U.\n"
        )

        f.write(
            '"""\n\n'
        )

        f.write(
            f"TEMPORAL_WINDOWS = {int(n_windows)!r}\n\n"
        )

        f.write(
            "# Canonical standardized OLS coefficients\n"
        )

        for name, value in zip(
            raw_names,
            coefficients,
        ):

            f.write(
                f"{name} = {float(value)!r}\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "# Sum-normalized coefficients for diagnostics only\n"
        )

        for name, value in zip(
            normalized_names,
            normalized_weights,
        ):

            f.write(
                f"{name} = {float(value)!r}\n"
            )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Estimate ALBERTA active-learning acquisition "
            "coefficients from uncertainty statistics."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Reference uncertainty-statistics JSON file.",
    )

    parser.add_argument(
        "--weights-file",
        type=Path,
        default=DEFAULT_WEIGHTS_FILE,
        help="Generated Python calibration module.",
    )

    parser.add_argument(
        "--report-file",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help="Text report output.",
    )

    parser.add_argument(
        "--temporal-windows",
        type=int,
        default=DEFAULT_TEMPORAL_WINDOWS,
        help=(
            "Number of contiguous chronological windows used "
            "for temporal + side-aware percentile normalization."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.temporal_windows < 1:

        raise ValueError(
            "--temporal-windows must be >= 1."
        )

    # ========================================================
    # Load reference data
    # ========================================================

    (
        fens,
        H,
        U,
        results,
    ) = load_data(
        args.input_file
    )

    # ========================================================
    # Regression variables
    # ========================================================

    (
        H_norm,
        U_norm,
        HU_interaction,
        Y,
        rewards,
        sides,
        window_ids,
    ) = build_regression_data(
        fens=fens,
        H=H,
        U=U,
        results=results,
        n_windows=args.temporal_windows,
    )

    # ========================================================
    # Temporal normalization diagnostics
    # ========================================================

    print_temporal_normalization_diagnostics(
        H_norm=H_norm,
        U_norm=U_norm,
        sides=sides,
        window_ids=window_ids,
        n_windows=args.temporal_windows,
    )

    # ========================================================
    # Standardization
    # ========================================================

    print()
    print(
        "Standardizing variables..."
    )

    (
        H_star,
        _,
        _,
    ) = standardize(
        H_norm
    )

    (
        U_star,
        _,
        _,
    ) = standardize(
        U_norm
    )

    (
        HU_star,
        _,
        _,
    ) = standardize(
        HU_interaction
    )

    (
        Y_star,
        _,
        _,
    ) = standardize(
        Y
    )

    X = np.column_stack(
        [
            H_star,
            U_star,
            HU_star,
        ]
    )

    # ========================================================
    # Correlations
    # ========================================================

    correlations = {
        "H":
            correlation(
                H_star,
                Y_star,
            ),

        "U":
            correlation(
                U_star,
                Y_star,
            ),

        "H x U":
            correlation(
                HU_star,
                Y_star,
            ),
    }

    predictor_correlations = np.corrcoef(
        X,
        rowvar=False,
    )

    # ========================================================
    # OLS
    # ========================================================

    print()
    print(
        "Fitting OLS..."
    )

    (
        coefficients,
        _,
        _,
        r_squared,
        rank,
        singular_values,
    ) = fit_ols(
        X,
        Y_star,
    )

    vif = calculate_vif(
        X
    )

    # ========================================================
    # Diagnostic normalized coefficients
    # ========================================================

    normalized_weights = normalize_coefficients(
        coefficients
    )

    # ========================================================
    # Console report
    # ========================================================

    print()
    print("=" * 70)
    print("OLS RESULTS")
    print("=" * 70)

    print()

    for name, value in zip(
        (
            "RAW_W_H",
            "RAW_W_U",
            "RAW_W_HU",
        ),
        coefficients,
    ):

        print(
            f"{name:<12}: "
            f"{value:+.8f}"
        )

    print()

    print(
        f"R²          : {r_squared:.8f}"
    )

    print(
        f"Matrix rank : {rank}"
    )

    print()

    print(
        "SUM-NORMALIZED COEFFICIENTS "
        "(DIAGNOSTIC ONLY)"
    )

    print(
        "-" * 70
    )

    for name, value in zip(
        (
            "W_H",
            "W_U",
            "W_HU",
        ),
        normalized_weights,
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

    print(
        "-" * 70
    )

    for name, value in correlations.items():

        print(
            f"{name:<15}: "
            f"{value:+.6f}"
        )

    print()

    print(
        "VIF"
    )

    print(
        "-" * 70
    )

    for name, value in zip(
        (
            "H",
            "U",
            "H x U",
        ),
        vif,
    ):

        print(
            f"{name:<15}: "
            f"{value:.4f}"
        )

    # ========================================================
    # Save generated calibration module
    # ========================================================

    save_weights(
        weights_file=args.weights_file,
        n_windows=args.temporal_windows,
        coefficients=coefficients,
        normalized_weights=normalized_weights,
    )

    # ========================================================
    # Save report
    # ========================================================

    report = build_report(
        n=len(
            fens
        ),
        n_windows=args.temporal_windows,
        coefficients=coefficients,
        normalized_weights=normalized_weights,
        r_squared=r_squared,
        correlations=correlations,
        predictor_correlations=predictor_correlations,
        vif=vif,
        sides=sides,
        rewards=rewards,
        rank=rank,
        singular_values=singular_values,
        H_norm=H_norm,
        U_norm=U_norm,
        window_ids=window_ids,
    )

    args.report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.report_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            report
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Weights : {args.weights_file}"
    )

    print(
        f"Report  : {args.report_file}"
    )

    print()

    print(
        "AL/AL_weights.py can now be imported by "
        "seed_oracle_queue.py and distribution_I.py."
    )

    print()

    print(
        "IMPORTANT: seed_oracle_queue.py and distribution_I.py "
        "must use the same chronological-window x side-to-move "
        "normalization before applying these coefficients."
    )


if __name__ == "__main__":
    main()