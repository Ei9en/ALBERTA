#!/usr/bin/env python3

"""
ALBERTA - Reward Information Analysis
======================================

Analyse complète de data/uncertainty_stats.json.

Objectif scientifique
---------------------
Quantifier dans quelle mesure le reward terminal {-1, 0, +1}
est informatif sur une position donnée.

Les quatre expériences principales sont :

1. Reward noise
   ----------------
   Mesure de H(R | s), E[R | s] et Var(R | s).

2. Sample-size effect
   -------------------
   Analyse de H(R | s) et E[R | s] en fonction du nombre
   d'observations d'une même FEN.

3. Confidence intervals
   ---------------------
   IC 95% de E[R | s] pour les positions répétées.

4. Variance explained
   ------------------
   Mesure de la fraction de variance du reward expliquée
   par la position :

       eta = Var(E[R | s]) / Var(R)

En complément :
- analyse H / U / HU
- gestion explicite des NaN dans H / U / HU
- diagnostic de couverture des métriques
- corrélations Pearson et Spearman sans scipy
- starting position
- export CSV
- graphiques PNG
- rapport texte complet

IMPORTANT
---------
Les NaN de H/U/HU ne suppriment PAS l'observation.

Une observation avec :

    reward = 1
    H = NaN
    U = 0.02

reste une observation parfaitement valide pour toutes les
analyses basées sur le reward.

Elle est simplement ignorée dans les analyses nécessitant H.

Usage
-----
    python analyze_uncertainty.py

Input
-----
    data/uncertainty_stats.json

Output
------
    data/uncertainty_analysis/
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path(
    "data/uncertainty_stats.json"
)

OUTPUT_DIR = Path(
    "data/uncertainty_analysis"
)

RANDOM_SEED = 42

MIN_OBSERVATIONS_DEFAULT = 10

REPETITION_THRESHOLDS = [
    5,
    10,
    20,
    50,
    100,
    200,
]

BOOTSTRAP_SAMPLES = 2000

CONFIDENCE_LEVEL = 0.95


# ============================================================
# Utilities
# ============================================================

def entropy_from_probs(probs):
    """
    Shannon entropy in bits.
    """

    probs = np.asarray(
        probs,
        dtype=float,
    )

    probs = probs[
        probs > 0
    ]

    if len(probs) == 0:
        return np.nan

    return float(
        -np.sum(
            probs * np.log2(probs)
        )
    )


def result_entropy(
    wins,
    draws,
    losses,
):
    """
    H(R | s) for R in {-1, 0, +1}.
    """

    n = (
        wins
        + draws
        + losses
    )

    if n == 0:
        return np.nan

    probs = np.array([
        wins / n,
        draws / n,
        losses / n,
    ])

    return entropy_from_probs(
        probs
    )


def result_expected_value(
    wins,
    draws,
    losses,
):
    """
    E[R | s] with:

        win  = +1
        draw =  0
        loss = -1
    """

    n = (
        wins
        + draws
        + losses
    )

    if n == 0:
        return np.nan

    return (
        wins - losses
    ) / n


def result_variance(
    wins,
    draws,
    losses,
):
    """
    Var(R | s), R in {-1, 0, +1}.
    """

    n = (
        wins
        + draws
        + losses
    )

    if n == 0:
        return np.nan

    mean = (
        wins - losses
    ) / n

    second_moment = (
        wins + losses
    ) / n

    return (
        second_moment
        - mean ** 2
    )


def normal_ci_mean(
    mean,
    variance,
    n,
    z=1.959963984540054,
):
    """
    Approximate 95% CI for the empirical mean.
    """

    if n <= 1:
        return (
            np.nan,
            np.nan,
        )

    se = math.sqrt(
        max(variance, 0.0)
        / n
    )

    return (
        mean - z * se,
        mean + z * se,
    )


def percentile_bootstrap_mean(
    values,
    n_samples=2000,
    seed=42,
):
    """
    Bootstrap confidence interval for the mean.

    Returns:
        mean, lower, upper
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    if n == 1:
        value = float(
            values[0]
        )

        return (
            value,
            value,
            value,
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        n,
        size=(
            n_samples,
            n,
        ),
    )

    bootstrap_means = (
        values[
            indices
        ].mean(axis=1)
    )

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_means,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2,
    )

    return (
        float(values.mean()),
        float(lower),
        float(upper),
    )


def rankdata(values):
    """
    Average-rank implementation.

    Used for Spearman correlation without scipy.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values
    )

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    i = 0

    while i < len(values):

        j = i + 1

        while (
            j < len(values)
            and values[
                order[j]
            ]
            == values[
                order[i]
            ]
        ):
            j += 1

        rank = (
            i + 1 + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = rank

        i = j

    return ranks


def pearson_corr(
    x,
    y,
):
    """
    Pearson correlation.

    NaN / infinite values are automatically removed
    pairwise.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def spearman_corr(
    x,
    y,
):
    """
    Spearman correlation without scipy.

    NaN / infinite values are automatically removed
    pairwise.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    return pearson_corr(
        rankdata(x),
        rankdata(y),
    )


def safe_mean(series):
    """
    Mean ignoring NaN / infinite values.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def safe_median(series):
    """
    Median ignoring NaN / infinite values.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.median()
    )


def safe_quantile(
    series,
    q,
):
    """
    Quantile ignoring NaN / infinite values.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.quantile(q)
    )


# ============================================================
# Load
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("ALBERTA - REWARD INFORMATION ANALYSIS")
    print("=" * 70)

    print()
    print("Loading uncertainty statistics")
    print("-" * 70)

    print(
        f"File: {INPUT_PATH}"
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
    ) as f:

        raw = json.load(f)

    print(
        f"Raw records: {len(raw):,}"
    )

    rows = []

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

        result = record.get(
            "result"
        )

        if (
            fen is None
            or result not in {
                "1-0",
                "0-1",
                "1/2-1/2",
            }
        ):
            invalid += 1
            continue

        rows.append(
            {
                "fen": fen,

                "actions":
                    record.get(
                        "actions",
                        np.nan,
                    ),

                "H":
                    record.get(
                        "H",
                        np.nan,
                    ),

                "U":
                    record.get(
                        "U",
                        np.nan,
                    ),

                "HU":
                    record.get(
                        "HU",
                        np.nan,
                    ),

                "result":
                    result,
            }
        )

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Explicit numeric conversion
    # --------------------------------------------------------

    for column in [
        "actions",
        "H",
        "U",
        "HU",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        f"Valid records: {len(df):,}"
    )

    print(
        f"Invalid records skipped: "
        f"{invalid:,}"
    )

    # --------------------------------------------------------
    # Metric coverage
    # --------------------------------------------------------

    print()
    print(
        "METRIC COVERAGE"
    )
    print("-" * 70)

    n = len(df)

    for column in [
        "H",
        "U",
        "HU",
    ]:

        valid = np.isfinite(
            df[column]
        ).sum()

        missing = (
            n - valid
        )

        print(
            f"{column:>3}: "
            f"{valid:,} valid "
            f"({valid / n:.2%}) | "
            f"{missing:,} NaN/invalid "
            f"({missing / n:.2%})"
        )

    print()

    print(
        "IMPORTANT: NaN values in H/U/HU "
        "do not invalidate reward observations."
    )

    return df


# ============================================================
# Result distribution
# ============================================================

def analyze_global_results(
    df,
):

    print()
    print("=" * 70)
    print("GLOBAL RESULT DISTRIBUTION")
    print("=" * 70)

    counts = (
        df["result"]
        .value_counts()
    )

    wins = int(
        counts.get(
            "1-0",
            0,
        )
    )

    losses = int(
        counts.get(
            "0-1",
            0,
        )
    )

    draws = int(
        counts.get(
            "1/2-1/2",
            0,
        )
    )

    n = len(df)

    probs = np.array([
        wins / n,
        draws / n,
        losses / n,
    ])

    H = entropy_from_probs(
        probs
    )

    print(
        f"Loss: {losses:>8,} "
        f"({losses / n:.2%})"
    )

    print(
        f"Draw: {draws:>8,} "
        f"({draws / n:.2%})"
    )

    print(
        f"Win:  {wins:>8,} "
        f"({wins / n:.2%})"
    )

    print()

    print(
        f"Global result entropy: "
        f"{H:.4f} bits"
    )

    return {
        "n": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "entropy": H,
    }


# ============================================================
# Group positions
# ============================================================

def build_position_table(
    df,
):

    grouped = (
        df.groupby(
            "fen",
            sort=False,
        )
    )

    rows = []

    for fen, group in grouped:

        counts = (
            group["result"]
            .value_counts()
        )

        wins = int(
            counts.get(
                "1-0",
                0,
            )
        )

        losses = int(
            counts.get(
                "0-1",
                0,
            )
        )

        draws = int(
            counts.get(
                "1/2-1/2",
                0,
            )
        )

        n = (
            wins
            + draws
            + losses
        )

        pW = wins / n
        pD = draws / n
        pL = losses / n

        mean_reward = (
            pW - pL
        )

        variance = (
            pW
            + pL
            - mean_reward ** 2
        )

        entropy = entropy_from_probs(
            [
                pW,
                pD,
                pL,
            ]
        )

        # ----------------------------------------------------
        # H / U / HU
        #
        # NaN are ignored independently.
        #
        # This is important:
        #
        # if H is NaN for one observation but U is valid,
        # that observation still contributes to U_mean.
        # ----------------------------------------------------

        H_values = pd.to_numeric(
            group["H"],
            errors="coerce",
        )

        U_values = pd.to_numeric(
            group["U"],
            errors="coerce",
        )

        HU_values = pd.to_numeric(
            group["HU"],
            errors="coerce",
        )

        H_valid = H_values[
            np.isfinite(H_values)
        ]

        U_valid = U_values[
            np.isfinite(U_values)
        ]

        HU_valid = HU_values[
            np.isfinite(HU_values)
        ]

        H_mean = (
            float(H_valid.mean())
            if len(H_valid)
            else np.nan
        )

        U_mean = (
            float(U_valid.mean())
            if len(U_valid)
            else np.nan
        )

        HU_mean = (
            float(HU_valid.mean())
            if len(HU_valid)
            else np.nan
        )

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        H_count = len(H_valid)
        U_count = len(U_valid)
        HU_count = len(HU_valid)

        H_fraction = (
            H_count / n
        )

        U_fraction = (
            U_count / n
        )

        HU_fraction = (
            HU_count / n
        )

        # ----------------------------------------------------
        # Confidence interval
        # ----------------------------------------------------

        ci_low, ci_high = (
            normal_ci_mean(
                mean_reward,
                variance,
                n,
            )
        )

        rows.append(
            {
                "fen": fen,

                "n": n,

                "W": wins,
                "D": draws,
                "L": losses,

                "pW": pW,
                "pD": pD,
                "pL": pL,

                "mean_reward":
                    mean_reward,

                "reward_variance":
                    variance,

                "result_entropy":
                    entropy,

                "ci95_low":
                    ci_low,

                "ci95_high":
                    ci_high,

                "ci95_width":
                    ci_high - ci_low,

                "ci_contains_zero":
                    (
                        ci_low <= 0
                        <= ci_high
                    ),

                # --------------------------------------------
                # Uncertainty metrics
                # --------------------------------------------

                "H_mean":
                    H_mean,

                "U_mean":
                    U_mean,

                "HU_mean":
                    HU_mean,

                "H_count":
                    H_count,

                "U_count":
                    U_count,

                "HU_count":
                    HU_count,

                "H_fraction":
                    H_fraction,

                "U_fraction":
                    U_fraction,

                "HU_fraction":
                    HU_fraction,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Starting position
# ============================================================

START_FEN = (
    "rnbqkbnr/"
    "pppppppp/"
    "8/8/8/8/"
    "PPPPPPPP/"
    "RNBQKBNR "
    "w KQkq - 0 1"
)


def analyze_starting_position(
    df,
    position_df,
):

    print()
    print("=" * 70)
    print("SANITY CHECK: STARTING POSITION")
    print("=" * 70)

    subset = df[
        df["fen"]
        == START_FEN
    ]

    if len(subset) == 0:

        print(
            "Starting position not found."
        )

        return None

    counts = (
        subset["result"]
        .value_counts()
    )

    wins = int(
        counts.get(
            "1-0",
            0,
        )
    )

    losses = int(
        counts.get(
            "0-1",
            0,
        )
    )

    draws = int(
        counts.get(
            "1/2-1/2",
            0,
        )
    )

    H = result_entropy(
        wins,
        draws,
        losses,
    )

    mean = result_expected_value(
        wins,
        draws,
        losses,
    )

    variance = result_variance(
        wins,
        draws,
        losses,
    )

    ci_low, ci_high = (
        normal_ci_mean(
            mean,
            variance,
            len(subset),
        )
    )

    print(
        f"Observations: {len(subset):,}"
    )

    print(
        f"Win:  {wins:,} "
        f"({wins / len(subset):.2%})"
    )

    print(
        f"Draw: {draws:,} "
        f"({draws / len(subset):.2%})"
    )

    print(
        f"Loss: {losses:,} "
        f"({losses / len(subset):.2%})"
    )

    print(
        f"Entropy: {H:.4f} bits"
    )

    print(
        f"E[R | s]: {mean:+.4f}"
    )

    print(
        f"95% CI: "
        f"[{ci_low:+.4f}, "
        f"{ci_high:+.4f}]"
    )

    return {
        "n": len(subset),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "entropy": H,
        "mean_reward": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ============================================================
# Experiment 1 - Reward noise
# ============================================================

def experiment_1(
    position_df,
):

    print()
    print("=" * 70)
    print("EXPERIMENT 1 - REWARD NOISE")
    print("=" * 70)

    repeated = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ].copy()

    print(
        f"Positions with n >= "
        f"{MIN_OBSERVATIONS_DEFAULT}: "
        f"{len(repeated):,}"
    )

    print()
    print(
        "Result entropy H(R|s):"
    )

    print(
        f"Mean:   "
        f"{safe_mean(repeated['result_entropy']):.4f}"
    )

    print(
        f"Median: "
        f"{safe_median(repeated['result_entropy']):.4f}"
    )

    for q in [
        0.90,
        0.95,
        0.99,
    ]:

        print(
            f"{int(q*100)}th percentile: "
            f"{safe_quantile(repeated['result_entropy'], q):.4f}"
        )

    print()

    max_entropy = (
        math.log2(3)
    )

    median_entropy = safe_median(
        repeated["result_entropy"]
    )

    print(
        f"Maximum possible entropy: "
        f"{max_entropy:.4f} bits"
    )

    print(
        f"Median / maximum: "
        f"{median_entropy / max_entropy:.2%}"
    )

    print()

    print(
        "Fraction above entropy thresholds:"
    )

    thresholds = [
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
        1.25,
        1.50,
    ]

    for threshold in thresholds:

        fraction = (
            repeated[
                "result_entropy"
            ]
            >= threshold
        ).mean()

        print(
            f"H >= {threshold:.2f}: "
            f"{fraction:.2%}"
        )

    return repeated


# ============================================================
# Experiment 2 - Sample size
# ============================================================

def experiment_2(
    position_df,
):

    print()
    print("=" * 70)
    print("EXPERIMENT 2 - SAMPLE SIZE EFFECT")
    print("=" * 70)

    rows = []

    for threshold in (
        REPETITION_THRESHOLDS
    ):

        subset = position_df[
            position_df["n"]
            >= threshold
        ]

        if len(subset) == 0:
            continue

        rows.append(
            {
                "min_n":
                    threshold,

                "positions":
                    len(subset),

                "mean_n":
                    safe_mean(
                        subset["n"]
                    ),

                "median_n":
                    safe_median(
                        subset["n"]
                    ),

                "mean_entropy":
                    safe_mean(
                        subset[
                            "result_entropy"
                        ]
                    ),

                "median_entropy":
                    safe_median(
                        subset[
                            "result_entropy"
                        ]
                    ),

                "mean_abs_reward":
                    safe_mean(
                        subset[
                            "mean_reward"
                        ].abs()
                    ),

                "median_abs_reward":
                    safe_median(
                        subset[
                            "mean_reward"
                        ].abs()
                    ),

                "mean_ci_width":
                    safe_mean(
                        subset[
                            "ci95_width"
                        ]
                    ),

                "median_ci_width":
                    safe_median(
                        subset[
                            "ci95_width"
                        ]
                    ),

                "fraction_ci_contains_zero":
                    safe_mean(
                        subset[
                            "ci_contains_zero"
                        ]
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return result


# ============================================================
# Experiment 3 - Confidence intervals
# ============================================================

def experiment_3(
    position_df,
):

    print()
    print("=" * 70)
    print("EXPERIMENT 3 - CONFIDENCE INTERVALS")
    print("=" * 70)

    subset = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ].copy()

    subset[
        "distance_from_zero"
    ] = subset[
        "mean_reward"
    ].abs()

    subset[
        "significant_95"
    ] = ~subset[
        "ci_contains_zero"
    ]

    print(
        f"Positions analysed: "
        f"{len(subset):,}"
    )

    print(
        f"95% CI contains zero: "
        f"{subset['ci_contains_zero'].mean():.2%}"
    )

    print(
        f"95% CI excludes zero: "
        f"{subset['significant_95'].mean():.2%}"
    )

    print()

    print(
        "Mean CI width: "
        f"{safe_mean(subset['ci95_width']):.4f}"
    )

    print(
        "Median CI width: "
        f"{safe_median(subset['ci95_width']):.4f}"
    )

    print()

    print(
        "Most uncertain estimates:"
    )

    display_cols = [
        "n",
        "W",
        "D",
        "L",
        "mean_reward",
        "ci95_low",
        "ci95_high",
        "ci95_width",
        "result_entropy",
    ]

    print(
        subset.sort_values(
            "ci95_width",
            ascending=False,
        )
        .head(20)[display_cols]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return subset


# ============================================================
# Experiment 4 - Variance explained
# ============================================================

def experiment_4(
    df,
    position_df,
):

    print()
    print("=" * 70)
    print("EXPERIMENT 4 - VARIANCE EXPLAINED BY POSITION")
    print("=" * 70)

    # --------------------------------------------------------
    # Global reward variance
    # --------------------------------------------------------

    reward_map = {
        "1-0": 1.0,
        "1/2-1/2": 0.0,
        "0-1": -1.0,
    }

    rewards = (
        df["result"]
        .map(reward_map)
        .astype(float)
    )

    global_mean = rewards.mean()

    global_variance = (
        rewards.var(
            ddof=0
        )
    )

    print(
        f"Global E[R]: "
        f"{global_mean:+.6f}"
    )

    print(
        f"Global Var(R): "
        f"{global_variance:.6f}"
    )

    print()

    results = []

    for threshold in (
        REPETITION_THRESHOLDS
    ):

        subset = position_df[
            position_df["n"]
            >= threshold
        ].copy()

        if len(subset) == 0:
            continue

        # ----------------------------------------------------
        # Important:
        #
        # Var(E[R|s]) must be weighted by the number
        # of observations of each position.
        # ----------------------------------------------------

        weights = (
            subset["n"]
            / subset["n"].sum()
        )

        means = (
            subset[
                "mean_reward"
            ]
        )

        weighted_mean = (
            np.sum(
                weights * means
            )
        )

        between_variance = (
            np.sum(
                weights
                * (
                    means
                    - weighted_mean
                ) ** 2
            )
        )

        # ----------------------------------------------------
        # Within-position variance
        # ----------------------------------------------------

        within_variance = (
            np.sum(
                weights
                * subset[
                    "reward_variance"
                ]
            )
        )

        total_decomposition = (
            between_variance
            + within_variance
        )

        eta = (
            between_variance
            / total_decomposition
            if total_decomposition > 0
            else np.nan
        )

        results.append(
            {
                "min_n":
                    threshold,

                "positions":
                    len(subset),

                "weighted_mean_reward":
                    weighted_mean,

                "between_position_variance":
                    between_variance,

                "within_position_variance":
                    within_variance,

                "total_variance":
                    total_decomposition,

                "eta_position":
                    eta,

                "fraction_unexplained":
                    1 - eta
                    if np.isfinite(eta)
                    else np.nan,
            }
        )

    result = pd.DataFrame(
        results
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()

    print(
        "Interpretation:"
    )

    for _, row in result.iterrows():

        print(
            f"n >= {int(row['min_n']):>3}: "
            f"eta = {row['eta_position']:.2%} "
            f"| unexplained = "
            f"{row['fraction_unexplained']:.2%}"
        )

    return result


# ============================================================
# H / U / HU analysis
# ============================================================

def analyze_uncertainty_relationships(
    position_df,
):

    print()
    print("=" * 70)
    print("H / U / HU RELATIONSHIPS")
    print("=" * 70)

    subset = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ].copy()

    # --------------------------------------------------------
    # Absolute expected reward
    # --------------------------------------------------------

    subset[
        "abs_mean_reward"
    ] = subset[
        "mean_reward"
    ].abs()

    pairs = [
        (
            "result_entropy",
            "U_mean",
        ),
        (
            "result_entropy",
            "H_mean",
        ),
        (
            "result_entropy",
            "HU_mean",
        ),
        (
            "abs_mean_reward",
            "U_mean",
        ),
        (
            "abs_mean_reward",
            "H_mean",
        ),
        (
            "abs_mean_reward",
            "HU_mean",
        ),
    ]

    rows = []

    for x_name, y_name in pairs:

        x = subset[x_name]
        y = subset[y_name]

        mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        n_pairs = int(
            mask.sum()
        )

        rows.append(
            {
                "x":
                    x_name,

                "y":
                    y_name,

                "n_pairs":
                    n_pairs,

                "coverage":
                    (
                        n_pairs / len(subset)
                        if len(subset)
                        else np.nan
                    ),

                "pearson":
                    pearson_corr(
                        x,
                        y,
                    ),

                "spearman":
                    spearman_corr(
                        x,
                        y,
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return result


# ============================================================
# Metric coverage
# ============================================================

def analyze_metric_coverage(
    df,
    position_df,
):

    print()
    print("=" * 70)
    print("H / U / HU COVERAGE")
    print("=" * 70)

    n = len(df)

    print()
    print(
        "Observation-level coverage:"
    )

    rows = []

    for metric in [
        "H",
        "U",
        "HU",
    ]:

        valid = np.isfinite(
            df[metric]
        )

        count = int(
            valid.sum()
        )

        rows.append(
            {
                "metric": metric,
                "valid_observations": count,
                "invalid_observations":
                    n - count,
                "coverage":
                    count / n
                    if n
                    else np.nan,
            }
        )

        print(
            f"{metric:>3}: "
            f"{count:,}/{n:,} "
            f"({count / n:.2%})"
        )

    result = pd.DataFrame(
        rows
    )

    print()

    print(
        "Position-level coverage "
        f"(FENs with n >= {MIN_OBSERVATIONS_DEFAULT}):"
    )

    repeated = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ]

    position_rows = []

    for metric in [
        "H",
        "U",
        "HU",
    ]:

        fraction_column = (
            f"{metric}_fraction"
        )

        count_column = (
            f"{metric}_count"
        )

        valid_positions = (
            repeated[
                count_column
            ] > 0
        ).sum()

        complete_positions = (
            repeated[
                fraction_column
            ] == 1.0
        ).sum()

        position_rows.append(
            {
                "metric":
                    metric,

                "positions_with_data":
                    int(valid_positions),

                "positions_with_complete_data":
                    int(complete_positions),

                "mean_fraction_valid":
                    safe_mean(
                        repeated[
                            fraction_column
                        ]
                    ),
            }
        )

        print(
            f"{metric:>3}: "
            f"{valid_positions:,} positions "
            f"with data | "
            f"{complete_positions:,} fully observed | "
            f"mean coverage = "
            f"{safe_mean(repeated[fraction_column]):.2%}"
        )

    position_result = pd.DataFrame(
        position_rows
    )

    return (
        result,
        position_result,
    )


# ============================================================
# Most noisy / deterministic
# ============================================================

def print_extremes(
    position_df,
):

    subset = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ]

    print()
    print("=" * 70)
    print("MOST NOISY POSITIONS")
    print("=" * 70)

    columns = [
        "n",
        "W",
        "L",
        "D",
        "pW",
        "pD",
        "pL",
        "mean_reward",
        "result_entropy",
        "U_mean",
        "H_mean",
        "HU_mean",
        "H_fraction",
        "U_fraction",
        "HU_fraction",
        "fen",
    ]

    print(
        subset.sort_values(
            "result_entropy",
            ascending=False,
        )
        .head(20)[columns]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print()
    print("=" * 70)
    print("MOST DETERMINISTIC POSITIONS")
    print("=" * 70)

    print(
        subset.sort_values(
            "result_entropy",
            ascending=True,
        )
        .head(20)[columns]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )


# ============================================================
# Plots
# ============================================================

def make_plots(
    position_df,
    sample_size_df,
    variance_df,
):

    print()
    print("=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ].copy()

    # --------------------------------------------------------
    # 1. Entropy vs number of observations
    # --------------------------------------------------------

    plt.figure()

    plt.scatter(
        subset["n"],
        subset["result_entropy"],
        alpha=0.35,
    )

    plt.xscale("log")

    plt.xlabel(
        "Observations per FEN"
    )

    plt.ylabel(
        "Result entropy H(R|s) [bits]"
    )

    plt.title(
        "Reward entropy vs sample size"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "entropy_vs_sample_size.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # 2. Mean reward vs sample size
    # --------------------------------------------------------

    plt.figure()

    plt.scatter(
        subset["n"],
        subset["mean_reward"],
        alpha=0.35,
    )

    plt.xscale("log")

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Observations per FEN"
    )

    plt.ylabel(
        "E[R|s]"
    )

    plt.title(
        "Empirical reward vs sample size"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "mean_reward_vs_sample_size.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # 3. Entropy vs U
    # --------------------------------------------------------

    valid = subset[
        np.isfinite(
            subset["U_mean"]
        )
        & np.isfinite(
            subset["result_entropy"]
        )
    ]

    plt.figure()

    if len(valid):

        plt.scatter(
            valid["U_mean"],
            valid["result_entropy"],
            alpha=0.35,
        )

    plt.xlabel(
        "Mean uncertainty U"
    )

    plt.ylabel(
        "Result entropy H(R|s) [bits]"
    )

    plt.title(
        "Reward noise vs model uncertainty"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "entropy_vs_U.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # 4. |E[R|s]| vs U
    # --------------------------------------------------------

    plt.figure()

    valid = subset[
        np.isfinite(
            subset["U_mean"]
        )
        & np.isfinite(
            subset["mean_reward"]
        )
    ]

    if len(valid):

        plt.scatter(
            valid["U_mean"],
            valid["mean_reward"].abs(),
            alpha=0.35,
        )

    plt.xlabel(
        "Mean uncertainty U"
    )

    plt.ylabel(
        "|E[R|s]|"
    )

    plt.title(
        "Reward directionality vs model uncertainty"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "absolute_reward_vs_U.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # 5. Eta vs threshold
    # --------------------------------------------------------

    if len(variance_df):

        plt.figure()

        plt.plot(
            variance_df["min_n"],
            variance_df["eta_position"],
            marker="o",
        )

        plt.xlabel(
            "Minimum observations per FEN"
        )

        plt.ylabel(
            "Variance explained by position"
        )

        plt.title(
            "Fraction of reward variance explained by position"
        )

        plt.ylim(
            0,
            1,
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / "variance_explained.png",
            dpi=200,
        )

        plt.close()

    print(
        "Plots saved."
    )


# ============================================================
# Save CSVs
# ============================================================

def save_csvs(
    df,
    position_df,
    experiment_1_df,
    experiment_2_df,
    experiment_3_df,
    experiment_4_df,
    correlations_df,
    coverage_df,
    position_coverage_df,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_DIR
        / "all_observations.csv",
        index=False,
    )

    position_df.to_csv(
        OUTPUT_DIR
        / "positions_by_fen.csv",
        index=False,
    )

    position_df[
        position_df["n"] >= 10
    ].to_csv(
        OUTPUT_DIR
        / "repeated_positions.csv",
        index=False,
    )

    experiment_1_df.to_csv(
        OUTPUT_DIR
        / "experiment_1_reward_noise.csv",
        index=False,
    )

    experiment_2_df.to_csv(
        OUTPUT_DIR
        / "experiment_2_sample_size.csv",
        index=False,
    )

    experiment_3_df.to_csv(
        OUTPUT_DIR
        / "experiment_3_confidence_intervals.csv",
        index=False,
    )

    experiment_4_df.to_csv(
        OUTPUT_DIR
        / "experiment_4_variance_explained.csv",
        index=False,
    )

    correlations_df.to_csv(
        OUTPUT_DIR
        / "uncertainty_correlations.csv",
        index=False,
    )

    coverage_df.to_csv(
        OUTPUT_DIR
        / "metric_coverage_observations.csv",
        index=False,
    )

    position_coverage_df.to_csv(
        OUTPUT_DIR
        / "metric_coverage_positions.csv",
        index=False,
    )

    print()
    print(
        "CSV files saved to:"
    )

    for path in [
        "all_observations.csv",
        "positions_by_fen.csv",
        "repeated_positions.csv",
        "experiment_1_reward_noise.csv",
        "experiment_2_sample_size.csv",
        "experiment_3_confidence_intervals.csv",
        "experiment_4_variance_explained.csv",
        "uncertainty_correlations.csv",
        "metric_coverage_observations.csv",
        "metric_coverage_positions.csv",
    ]:

        print(
            f"  {OUTPUT_DIR / path}"
        )


# ============================================================
# Report
# ============================================================

def build_report(
    global_stats,
    start_stats,
    position_df,
    experiment_2_df,
    experiment_4_df,
    correlations_df,
    coverage_df,
    position_coverage_df,
):

    lines = []

    def add(text=""):
        lines.append(
            str(text)
        )

    add("=" * 70)
    add(
        "ALBERTA - REWARD INFORMATION ANALYSIS"
    )
    add("=" * 70)
    add()

    add(
        f"Input: {INPUT_PATH}"
    )

    add(
        f"Records: "
        f"{global_stats['n']:,}"
    )

    add()

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    add(
        "GLOBAL RESULT DISTRIBUTION"
    )
    add("-" * 70)

    add(
        f"Wins:   {global_stats['wins']:,} "
        f"({global_stats['wins']/global_stats['n']:.2%})"
    )

    add(
        f"Draws:  {global_stats['draws']:,} "
        f"({global_stats['draws']/global_stats['n']:.2%})"
    )

    add(
        f"Losses: {global_stats['losses']:,} "
        f"({global_stats['losses']/global_stats['n']:.2%})"
    )

    add(
        f"Entropy: "
        f"{global_stats['entropy']:.6f} bits"
    )

    add()

    # --------------------------------------------------------
    # Metric coverage
    # --------------------------------------------------------

    add(
        "METRIC COVERAGE"
    )
    add("-" * 70)

    add(
        coverage_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    add(
        "Position-level coverage:"
    )

    add(
        position_coverage_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Position count
    # --------------------------------------------------------

    add(
        "POSITION COUNTS"
    )
    add("-" * 70)

    add(
        f"Unique FENs: "
        f"{len(position_df):,}"
    )

    add(
        f"FENs >= 5 observations: "
        f"{(position_df['n'] >= 5).sum():,}"
    )

    add(
        f"FENs >= 10 observations: "
        f"{(position_df['n'] >= 10).sum():,}"
    )

    add(
        f"FENs >= 50 observations: "
        f"{(position_df['n'] >= 50).sum():,}"
    )

    add(
        f"FENs >= 100 observations: "
        f"{(position_df['n'] >= 100).sum():,}"
    )

    add(
        f"Maximum observations/FEN: "
        f"{position_df['n'].max():,}"
    )

    add()

    # --------------------------------------------------------
    # Starting position
    # --------------------------------------------------------

    if start_stats:

        add(
            "STARTING POSITION"
        )

        add("-" * 70)

        add(
            f"Observations: "
            f"{start_stats['n']:,}"
        )

        add(
            f"W/D/L: "
            f"{start_stats['wins']}/"
            f"{start_stats['draws']}/"
            f"{start_stats['losses']}"
        )

        add(
            f"H(R|s): "
            f"{start_stats['entropy']:.6f} bits"
        )

        add(
            f"E[R|s]: "
            f"{start_stats['mean_reward']:+.6f}"
        )

        add(
            f"95% CI: "
            f"[{start_stats['ci_low']:+.6f}, "
            f"{start_stats['ci_high']:+.6f}]"
        )

        add()

    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    subset = position_df[
        position_df["n"] >= 10
    ]

    add(
        "EXPERIMENT 1 - REWARD NOISE"
    )

    add("-" * 70)

    add(
        f"Positions analysed: "
        f"{len(subset):,}"
    )

    add(
        f"Mean H(R|s): "
        f"{safe_mean(subset['result_entropy']):.6f}"
    )

    add(
        f"Median H(R|s): "
        f"{safe_median(subset['result_entropy']):.6f}"
    )

    add(
        f"Maximum entropy: "
        f"{math.log2(3):.6f}"
    )

    add(
        f"Median / maximum: "
        f"{safe_median(subset['result_entropy']) / math.log2(3):.2%}"
    )

    add()

    for threshold in [
        0.75,
        0.90,
        1.00,
        1.25,
        1.50,
    ]:

        fraction = (
            subset[
                "result_entropy"
            ] >= threshold
        ).mean()

        add(
            f"H >= {threshold:.2f}: "
            f"{fraction:.2%}"
        )

    add()

    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    add(
        "EXPERIMENT 2 - SAMPLE SIZE"
    )

    add("-" * 70)

    add(
        experiment_2_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Experiment 3
    # --------------------------------------------------------

    add(
        "EXPERIMENT 3 - CONFIDENCE INTERVALS"
    )

    add("-" * 70)

    add(
        f"95% CI contains zero: "
        f"{subset['ci_contains_zero'].mean():.2%}"
    )

    add(
        f"95% CI excludes zero: "
        f"{(~subset['ci_contains_zero']).mean():.2%}"
    )

    add(
        f"Mean CI width: "
        f"{safe_mean(subset['ci95_width']):.6f}"
    )

    add(
        f"Median CI width: "
        f"{safe_median(subset['ci95_width']):.6f}"
    )

    add()

    # --------------------------------------------------------
    # Experiment 4
    # --------------------------------------------------------

    add(
        "EXPERIMENT 4 - VARIANCE EXPLAINED"
    )

    add("-" * 70)

    add(
        experiment_4_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    add(
        "H / U / HU CORRELATIONS"
    )

    add("-" * 70)

    add(
        correlations_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Scientific summary
    # --------------------------------------------------------

    add(
        "SCIENTIFIC SUMMARY"
    )

    add("-" * 70)

    eta10 = experiment_4_df[
        experiment_4_df["min_n"] == 10
    ]

    if len(eta10):

        eta = float(
            eta10.iloc[0][
                "eta_position"
            ]
        )

        add(
            f"At n >= 10, the position "
            f"explains approximately "
            f"{eta:.2%} of reward variance."
        )

        add(
            f"The remaining "
            f"{1-eta:.2%} is within-position "
            f"variation / unexplained variance."
        )

    add()

    add(
        "NaN HANDLING:"
    )

    add(
        "NaN values in H/U/HU are treated as missing "
        "uncertainty measurements."
    )

    add(
        "They do not remove the corresponding reward "
        "observation from the analysis."
    )

    add(
        "H, U and HU are analysed independently, so an "
        "observation with valid U but NaN H still contributes "
        "to U-based statistics."
    )

    add()

    add(
        "IMPORTANT:"
    )

    add(
        "These results quantify reward informativeness."
    )

    add(
        "They do not by themselves prove that reward noise "
        "is the cause of RL failure."
    )

    add(
        "A causal claim requires a controlled experiment "
        "with an alternative reward signal."
    )

    add()

    add(
        "Recommended next experiment:"
    )

    add(
        "Compare the same architecture and RL pipeline "
        "using the terminal {-1,0,+1} reward against "
        "a richer position-level target."
    )

    add()

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Global reward statistics
    # --------------------------------------------------------

    global_stats = (
        analyze_global_results(
            df
        )
    )

    # --------------------------------------------------------
    # Group by FEN
    # --------------------------------------------------------

    position_df = (
        build_position_table(
            df
        )
    )

    print()
    print("=" * 70)
    print("GROUPING POSITIONS BY FEN")
    print("=" * 70)

    print(
        f"Unique FENs: "
        f"{len(position_df):,}"
    )

    repeated = position_df[
        position_df["n"] >= 10
    ]

    print()
    print(
        "Repeated positions:"
    )

    print(
        f"FENs with >= 10 observations: "
        f"{len(repeated):,}"
    )

    print(
        f"Median observations/FEN: "
        f"{position_df['n'].median():.1f}"
    )

    print(
        f"Mean observations/FEN: "
        f"{position_df['n'].mean():.1f}"
    )

    print(
        f"Maximum observations/FEN: "
        f"{position_df['n'].max():,}"
    )

    # --------------------------------------------------------
    # Starting position
    # --------------------------------------------------------

    start_stats = (
        analyze_starting_position(
            df,
            position_df,
        )
    )

    # --------------------------------------------------------
    # Main experiments
    # --------------------------------------------------------

    experiment_1_df = (
        experiment_1(
            position_df
        )
    )

    experiment_2_df = (
        experiment_2(
            position_df
        )
    )

    experiment_3_df = (
        experiment_3(
            position_df
        )
    )

    experiment_4_df = (
        experiment_4(
            df,
            position_df,
        )
    )

    # --------------------------------------------------------
    # H / U / HU
    # --------------------------------------------------------

    correlations_df = (
        analyze_uncertainty_relationships(
            position_df
        )
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    (
        coverage_df,
        position_coverage_df,
    ) = analyze_metric_coverage(
        df,
        position_df,
    )

    # --------------------------------------------------------
    # Extremes
    # --------------------------------------------------------

    print_extremes(
        position_df
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_csvs(
        df,
        position_df,
        experiment_1_df,
        experiment_2_df,
        experiment_3_df,
        experiment_4_df,
        correlations_df,
        coverage_df,
        position_coverage_df,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    make_plots(
        position_df,
        experiment_2_df,
        experiment_4_df,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        global_stats,
        start_stats,
        position_df,
        experiment_2_df,
        experiment_4_df,
        correlations_df,
        coverage_df,
        position_coverage_df,
    )

    report_path = (
        OUTPUT_DIR
        / "analysis_report.txt"
    )

    with open(
        report_path,
        "w",
    ) as f:

        f.write(
            report
        )

    print()
    print("=" * 70)
    print("REPORT")
    print("=" * 70)

    print(
        "Full report saved to:"
    )

    print(
        f"  {report_path}"
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Results written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()