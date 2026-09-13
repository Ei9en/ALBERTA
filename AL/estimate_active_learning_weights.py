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

Given raw policy entropy H and value-disagreement uncertainty U:

    F(U) = log1p(U / TAU)

H and F(U) are percentile-rank normalized independently for
White-to-move and Black-to-move positions.

The interaction is then defined as:

    H_norm * F(U)_norm

All three predictors and the target are standardized before OLS.

The regression is:

    Y* =
        RAW_W_H  * H*
        + RAW_W_U  * F(U)*
        + RAW_W_HU * (H_norm * F(U)_norm)*
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

DEFAULT_TAU = 0.05


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
# Side-aware normalization
# ============================================================

def normalize_side_aware(
    values: np.ndarray,
    sides: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    sides = np.asarray(
        sides
    )

    normalized = np.empty_like(
        values,
        dtype=np.float64,
    )

    for side in (
        "w",
        "b",
    ):

        mask = (
            sides == side
        )

        count = int(
            np.sum(
                mask
            )
        )

        if count < 2:

            raise ValueError(
                f"Not enough positions for side {side} normalization."
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
    tau: float,
) -> tuple[
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

    if tau <= 0.0:

        raise ValueError(
            "TAU must be strictly positive."
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
    # Regression target:
    #
    #     Y = |R|
    #
    # This is decisiveness, not signed outcome.
    # --------------------------------------------------------

    Y = np.abs(
        rewards
    )

    # --------------------------------------------------------
    # U transform
    # --------------------------------------------------------

    U_log = np.log1p(
        U
        / tau
    )

    # --------------------------------------------------------
    # Side-aware percentile normalization
    # --------------------------------------------------------

    print(
        "Normalizing H by side-to-move..."
    )

    H_norm = normalize_side_aware(
        H,
        sides,
    )

    print(
        "Normalizing log-transformed U by side-to-move..."
    )

    U_log_norm = normalize_side_aware(
        U_log,
        sides,
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

            vif_value = (
                np.inf
            )

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
    tau: float,
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
) -> str:

    names = [
        "H",
        "log(U)",
        "H x log(U)",
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
        f"TAU: {tau}"
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
        "This is an observational association. It does not "
        "establish causal learning value or useful update direction."
    )

    add()

    add(
        "Weights should be estimated once on the reference "
        "uncertainty dataset and frozen before query selection."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Save generated weights
# ============================================================

def save_weights(
    weights_file: Path,
    tau: float,
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
            "normalized to sum to one.\n"
        )

        f.write(
            '"""\n\n'
        )

        f.write(
            f"TAU = {float(tau)!r}\n\n"
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
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help="Scale parameter in log1p(U / TAU).",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.tau <= 0.0:

        raise ValueError(
            "--tau must be strictly positive."
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
        U_log_norm,
        HU_interaction,
        Y,
        rewards,
        sides,
    ) = build_regression_data(
        fens=fens,
        H=H,
        U=U,
        results=results,
        tau=args.tau,
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
        U_log_norm
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

        "log(U)":
            correlation(
                U_star,
                Y_star,
            ),

        "H x log(U)":
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
            "log(U)",
            "H x log(U)",
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
        tau=args.tau,
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
        tau=args.tau,
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


if __name__ == "__main__":
    main()