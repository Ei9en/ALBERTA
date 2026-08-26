#!/usr/bin/env python3

"""
ALBERTA - Reward Information Analysis
=====================================

Analyse de data/uncertainty_stats_11-20.json.

Objectif scientifique
---------------------

Quantifier dans quelle mesure le reward terminal {-1, 0, +1}
est informatif sur une position donnée.

Expériences :

1. Reward noise
   - H(R | s)
   - E[R | s]
   - Var(R | s)

2. Sample-size effect
   - évolution de H(R | s)
   - évolution de E[R | s]
   - évolution des IC
   - évolution de la variance

3. Confidence intervals
   - IC 95% de E[R | s]

4. Variance explained
   - eta = Var(E[R | s]) / Var(R)

5. H / U / HU
   - corrélations avec le reward noise
   - corrélations avec |E[R|s]|

NaN
---

Les NaN de H/U/HU ne suppriment jamais l'observation
de reward correspondante.

Ils sont simplement ignorés pour la métrique concernée.

IMPORTANT
---------

Aucun CSV n'est généré.

Outputs :

    data/uncertainty_analysis/
        analysis_report.txt
        entropy_vs_sample_size.png
        mean_reward_vs_sample_size.png
        entropy_vs_U.png
        absolute_reward_vs_U.png
        variance_explained.png

Usage :

    python analyze_uncertainty.py
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
    "data/uncertainty_stats_1-10.json"
)

OUTPUT_DIR = Path(
    "data/uncertainty_analysis_1-10"
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
    H(R | s)
    """

    n = (
        wins
        + draws
        + losses
    )

    if n == 0:
        return np.nan

    return entropy_from_probs(
        [
            wins / n,
            draws / n,
            losses / n,
        ]
    )


def normal_ci_mean(
    mean,
    variance,
    n,
    z=1.959963984540054,
):
    """
    Approximate normal 95% confidence interval.
    """

    if n <= 1:
        return (
            np.nan,
            np.nan,
        )

    se = math.sqrt(
        max(variance, 0.0) / n
    )

    return (
        mean - z * se,
        mean + z * se,
    )


def rankdata(values):
    """
    Average-rank implementation for Spearman.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="mergesort",
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
    Pearson correlation with pairwise NaN removal.
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
            f"Input file not found: {INPUT_PATH}"
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

        fen = record.get("fen")
        result = record.get("result")

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
        f"Invalid records skipped: {invalid:,}"
    )

    print()
    print("METRIC COVERAGE")
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

        missing = n - valid

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
# Global results
# ============================================================

def analyze_global_results(df):

    print()
    print("=" * 70)
    print("GLOBAL RESULT DISTRIBUTION")
    print("=" * 70)

    counts = df["result"].value_counts()

    wins = int(
        counts.get("1-0", 0)
    )

    losses = int(
        counts.get("0-1", 0)
    )

    draws = int(
        counts.get("1/2-1/2", 0)
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
# Position aggregation
# ============================================================

def build_position_table(df):

    print()
    print(
        "Building position statistics..."
    )

    print(
        "Vectorized aggregation:"
    )

    # --------------------------------------------------------
    # Result counts
    # --------------------------------------------------------

    counts = pd.crosstab(
        df["fen"],
        df["result"],
    )

    for column in [
        "1-0",
        "1/2-1/2",
        "0-1",
    ]:

        if column not in counts.columns:

            counts[column] = 0

    counts = counts[
        [
            "1-0",
            "1/2-1/2",
            "0-1",
        ]
    ]

    counts.columns = [
        "W",
        "D",
        "L",
    ]

    counts = counts.astype(
        np.int64
    )

    counts["n"] = (
        counts["W"]
        + counts["D"]
        + counts["L"]
    )

    # --------------------------------------------------------
    # Reward statistics
    # --------------------------------------------------------

    counts["pW"] = (
        counts["W"]
        / counts["n"]
    )

    counts["pD"] = (
        counts["D"]
        / counts["n"]
    )

    counts["pL"] = (
        counts["L"]
        / counts["n"]
    )

    counts["mean_reward"] = (
        counts["pW"]
        - counts["pL"]
    )

    counts["reward_variance"] = (
        counts["pW"]
        + counts["pL"]
        - counts["mean_reward"] ** 2
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    pW = counts["pW"].to_numpy()
    pD = counts["pD"].to_numpy()
    pL = counts["pL"].to_numpy()

    entropy = np.zeros(
        len(counts),
        dtype=float,
    )

    for p in [
        pW,
        pD,
        pL,
    ]:

        mask = p > 0

        entropy[mask] -= (
            p[mask]
            * np.log2(
                p[mask]
            )
        )

    counts["result_entropy"] = entropy

    # --------------------------------------------------------
    # Confidence intervals
    # --------------------------------------------------------

    variance = counts[
        "reward_variance"
    ].to_numpy()

    mean_reward = counts[
        "mean_reward"
    ].to_numpy()

    n = counts[
        "n"
    ].to_numpy()

    se = np.sqrt(
        np.maximum(
            variance,
            0,
        )
        / n
    )

    z = 1.959963984540054

    counts["ci95_low"] = (
        mean_reward
        - z * se
    )

    counts["ci95_high"] = (
        mean_reward
        + z * se
    )

    counts["ci95_width"] = (
        2 * z * se
    )

    counts["ci_contains_zero"] = (
        (counts["ci95_low"] <= 0)
        &
        (counts["ci95_high"] >= 0)
    )

    # --------------------------------------------------------
    # H / U / HU means
    #
    # IMPORTANT:
    #
    # Each metric is aggregated independently.
    # --------------------------------------------------------

    print(
        "Aggregating H..."
    )

    H_stats = (
        df.groupby("fen")["H"]
        .agg(
            H_mean="mean",
            H_count="count",
        )
    )

    print(
        "Aggregating U..."
    )

    U_stats = (
        df.groupby("fen")["U"]
        .agg(
            U_mean="mean",
            U_count="count",
        )
    )

    print(
        "Aggregating HU..."
    )

    HU_stats = (
        df.groupby("fen")["HU"]
        .agg(
            HU_mean="mean",
            HU_count="count",
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    position_df = counts.join(
        H_stats,
        how="left",
    )

    position_df = position_df.join(
        U_stats,
        how="left",
    )

    position_df = position_df.join(
        HU_stats,
        how="left",
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    position_df["H_fraction"] = (
        position_df["H_count"]
        / position_df["n"]
    )

    position_df["U_fraction"] = (
        position_df["U_count"]
        / position_df["n"]
    )

    position_df["HU_fraction"] = (
        position_df["HU_count"]
        / position_df["n"]
    )

    position_df = (
        position_df
        .reset_index()
    )

    print(
        f"Position aggregation complete: "
        f"{len(position_df):,} unique FENs."
    )

    return position_df


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


def analyze_starting_position(df):

    print()
    print("=" * 70)
    print("SANITY CHECK: STARTING POSITION")
    print("=" * 70)

    subset = df[
        df["fen"] == START_FEN
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
        counts.get("1-0", 0)
    )

    losses = int(
        counts.get("0-1", 0)
    )

    draws = int(
        counts.get("1/2-1/2", 0)
    )

    n = len(subset)

    pW = wins / n
    pD = draws / n
    pL = losses / n

    H = entropy_from_probs(
        [pW, pD, pL]
    )

    mean = pW - pL

    variance = (
        pW
        + pL
        - mean ** 2
    )

    ci_low, ci_high = (
        normal_ci_mean(
            mean,
            variance,
            n,
        )
    )

    print(
        f"Observations: {n:,}"
    )

    print(
        f"Win:  {wins:,} "
        f"({pW:.2%})"
    )

    print(
        f"Draw: {draws:,} "
        f"({pD:.2%})"
    )

    print(
        f"Loss: {losses:,} "
        f"({pL:.2%})"
    )

    print(
        f"Entropy: {H:.4f} bits"
    )

    print(
        f"E[R | s]: {mean:+.4f}"
    )

    print(
        f"95% CI: "
        f"[{ci_low:+.4f}, {ci_high:+.4f}]"
    )

    return {
        "n": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "entropy": H,
        "mean_reward": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ============================================================
# Experiment 1
# ============================================================

def experiment_1(position_df):

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

    median_entropy = safe_median(
        repeated["result_entropy"]
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
        f"{median_entropy:.4f}"
    )

    for q in [
        0.90,
        0.95,
        0.99,
    ]:

        print(
            f"{int(q * 100)}th percentile: "
            f"{safe_quantile(repeated['result_entropy'], q):.4f}"
        )

    max_entropy = math.log2(3)

    print()
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

    for threshold in [
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
        1.25,
        1.50,
    ]:

        fraction = (
            repeated[
                "result_entropy"
            ] >= threshold
        ).mean()

        print(
            f"H >= {threshold:.2f}: "
            f"{fraction:.2%}"
        )

    return repeated


# ============================================================
# Experiment 2
# ============================================================

def experiment_2(position_df):

    print()
    print("=" * 70)
    print("EXPERIMENT 2 - SAMPLE SIZE EFFECT")
    print("=" * 70)

    rows = []

    for threshold in REPETITION_THRESHOLDS:

        subset = position_df[
            position_df["n"]
            >= threshold
        ]

        if len(subset) == 0:
            continue

        rows.append(
            {
                "min_n": threshold,
                "positions": len(subset),

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
                        subset["result_entropy"]
                    ),

                "median_entropy":
                    safe_median(
                        subset["result_entropy"]
                    ),

                "mean_abs_reward":
                    safe_mean(
                        subset["mean_reward"].abs()
                    ),

                "median_abs_reward":
                    safe_median(
                        subset["mean_reward"].abs()
                    ),

                "mean_ci_width":
                    safe_mean(
                        subset["ci95_width"]
                    ),

                "median_ci_width":
                    safe_median(
                        subset["ci95_width"]
                    ),

                "fraction_ci_contains_zero":
                    safe_mean(
                        subset["ci_contains_zero"]
                    ),
            }
        )

    result = pd.DataFrame(rows)

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
# Experiment 3
# ============================================================

def experiment_3(position_df):

    print()
    print("=" * 70)
    print("EXPERIMENT 3 - CONFIDENCE INTERVALS")
    print("=" * 70)

    subset = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ].copy()

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
        f"Mean CI width: "
        f"{safe_mean(subset['ci95_width']):.4f}"
    )

    print(
        f"Median CI width: "
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
# Experiment 4
# ============================================================

def experiment_4(
    df,
    position_df,
):

    print()
    print("=" * 70)
    print("EXPERIMENT 4 - VARIANCE EXPLAINED BY POSITION")
    print("=" * 70)

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

    global_variance = rewards.var(
        ddof=0
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

    for threshold in REPETITION_THRESHOLDS:

        subset = position_df[
            position_df["n"]
            >= threshold
        ].copy()

        if len(subset) == 0:
            continue

        weights = (
            subset["n"]
            / subset["n"].sum()
        )

        means = subset[
            "mean_reward"
        ]

        weighted_mean = np.sum(
            weights * means
        )

        between_variance = np.sum(
            weights
            * (
                means
                - weighted_mean
            ) ** 2
        )

        within_variance = np.sum(
            weights
            * subset[
                "reward_variance"
            ]
        )

        total_variance = (
            between_variance
            + within_variance
        )

        eta = (
            between_variance
            / total_variance
            if total_variance > 0
            else np.nan
        )

        results.append(
            {
                "min_n": threshold,
                "positions": len(subset),
                "weighted_mean_reward":
                    weighted_mean,
                "between_position_variance":
                    between_variance,
                "within_position_variance":
                    within_variance,
                "total_variance":
                    total_variance,
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
    print("Interpretation:")

    for _, row in result.iterrows():

        print(
            f"n >= {int(row['min_n']):>3}: "
            f"eta = {row['eta_position']:.2%} "
            f"| unexplained = "
            f"{row['fraction_unexplained']:.2%}"
        )

    return result


# ============================================================
# H / U / HU relationships
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
                "x": x_name,
                "y": y_name,
                "n_pairs": n_pairs,
                "coverage":
                    n_pairs / len(subset)
                    if len(subset)
                    else np.nan,
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

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return result


# ============================================================
# Coverage
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

    rows = []

    print()
    print(
        "Observation-level coverage:"
    )

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
                    count / n,
            }
        )

        print(
            f"{metric:>3}: "
            f"{count:,}/{n:,} "
            f"({count / n:.2%})"
        )

    coverage_df = pd.DataFrame(
        rows
    )

    repeated = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ]

    print()
    print(
        f"Position-level coverage "
        f"(FENs with n >= "
        f"{MIN_OBSERVATIONS_DEFAULT}):"
    )

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

        mean_fraction = safe_mean(
            repeated[
                fraction_column
            ]
        )

        position_rows.append(
            {
                "metric": metric,
                "positions_with_data":
                    int(valid_positions),
                "positions_with_complete_data":
                    int(complete_positions),
                "mean_fraction_valid":
                    mean_fraction,
            }
        )

        print(
            f"{metric:>3}: "
            f"{valid_positions:,} positions "
            f"with data | "
            f"{complete_positions:,} fully observed | "
            f"mean coverage = "
            f"{mean_fraction:.2%}"
        )

    return (
        coverage_df,
        pd.DataFrame(position_rows),
    )


# ============================================================
# Extremes
# ============================================================

def print_extremes(
    position_df,
):

    subset = position_df[
        position_df["n"]
        >= MIN_OBSERVATIONS_DEFAULT
    ]

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

    print()
    print("=" * 70)
    print("MOST NOISY POSITIONS")
    print("=" * 70)

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
    # 1. Entropy vs sample size
    # --------------------------------------------------------

    print(
        "Plot 1/5..."
    )

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
    # 2. Mean reward
    # --------------------------------------------------------

    print(
        "Plot 2/5..."
    )

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

    print(
        "Plot 3/5..."
    )

    valid = subset[
        np.isfinite(
            subset["U_mean"]
        )
        &
        np.isfinite(
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
    # 4. Absolute reward vs U
    # --------------------------------------------------------

    print(
        "Plot 4/5..."
    )

    valid = subset[
        np.isfinite(
            subset["U_mean"]
        )
        &
        np.isfinite(
            subset["mean_reward"]
        )
    ]

    plt.figure()

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
    # 5. Eta
    # --------------------------------------------------------

    print(
        "Plot 5/5..."
    )

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
        f"Records: {global_stats['n']:,}"
    )

    add()

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
        f"Entropy: {global_stats['entropy']:.6f} bits"
    )

    add()

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

    add(
        "POSITION COUNTS"
    )
    add("-" * 70)

    add(
        f"Unique FENs: "
        f"{len(position_df):,}"
    )

    for threshold in [
        5,
        10,
        50,
        100,
    ]:

        add(
            f"FENs >= {threshold} observations: "
            f"{(position_df['n'] >= threshold).sum():,}"
        )

    add(
        f"Maximum observations/FEN: "
        f"{position_df['n'].max():,}"
    )

    add()

    if start_stats:

        add(
            "STARTING POSITION"
        )
        add("-" * 70)

        add(
            f"Observations: {start_stats['n']:,}"
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

    median_entropy = safe_median(
        subset["result_entropy"]
    )

    add(
        f"Mean H(R|s): "
        f"{safe_mean(subset['result_entropy']):.6f}"
    )

    add(
        f"Median H(R|s): "
        f"{median_entropy:.6f}"
    )

    add(
        f"Maximum entropy: "
        f"{math.log2(3):.6f}"
    )

    add(
        f"Median / maximum: "
        f"{median_entropy / math.log2(3):.2%}"
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
            subset["result_entropy"]
            >= threshold
        ).mean()

        add(
            f"H >= {threshold:.2f}: "
            f"{fraction:.2%}"
        )

    add()

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
            f"At n >= 10, position explains "
            f"approximately {eta:.2%} "
            f"of reward variance."
        )

        add(
            f"The remaining {1 - eta:.2%} "
            f"is within-position variation."
        )

    add()

    add(
        "NaN HANDLING:"
    )

    add(
        "NaN values in H/U/HU are treated as "
        "missing uncertainty measurements."
    )

    add(
        "They never remove the corresponding "
        "reward observation."
    )

    add(
        "H, U and HU are analysed independently."
    )

    add()

    add(
        "IMPORTANT:"
    )

    add(
        "These results quantify reward informativeness."
    )

    add(
        "They do not by themselves prove that reward "
        "noise causes RL failure."
    )

    add(
        "A causal claim requires a controlled experiment "
        "with an alternative reward signal."
    )

    return "\n".join(lines)


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
    # Global
    # --------------------------------------------------------

    global_stats = (
        analyze_global_results(
            df
        )
    )

    # --------------------------------------------------------
    # Position aggregation
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

    print(
        f"FENs with >= 10 observations: "
        f"{(position_df['n'] >= 10).sum():,}"
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
            df
        )
    )

    # --------------------------------------------------------
    # Experiments
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
            position_df
        )
    )

    # --------------------------------------------------------
    # H/U/HU
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
    # Plots
    # --------------------------------------------------------

    make_plots(
        position_df,
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
        f"Full report saved to:"
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