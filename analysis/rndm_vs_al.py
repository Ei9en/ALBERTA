#!/usr/bin/env python3

"""
ALBERTA - Oracle Queue Comparison
==================================

Compare deux queues Oracle :

    - Random Oracle
    - AL(I)

Objectif
--------

Déterminer précisément ce que la sélection AL(I) sélectionne
différemment d'une sélection aléatoire, avec le même budget.

Variables analysées
-------------------

Signals:
    - H
    - U
    - HU
    - score
    - I_norm

Oracle annotation:
    - reward
    - |reward|
    - oracle_confidence
    - oracle_situation

Position:
    - side-to-move

Structure:
    - nombre de positions
    - FEN uniques
    - doublons
    - chevauchement entre les deux queues

Analyses
--------

1. Statistiques descriptives
2. Quantiles
3. Mann-Whitney U
4. Analyse des queues hautes
5. Reward distribution
6. Confidence distribution
7. Situation distribution
8. Side-to-move distribution
9. Overlap entre queues
10. Corrélations entre I_norm et les variables Oracle
11. Rapport texte
12. Graphiques

Outputs
-------

    data/uncertainty_analysis/queue_comparison/

        queue_comparison_report.txt

        signals_distribution.png
        reward_distribution.png
        confidence_distribution.png
        situation_distribution.png
        side_distribution.png

Usage
-----

Modifier simplement :

    RANDOM_QUEUE_PATH
    AL_QUEUE_PATH

puis :

    python compare_oracle_queues.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )



RANDOM_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_random.jsonl"
)

AL_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_AL_eq.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_analysis"
    / "queue_comparison"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "queue_comparison_report.txt"
)


# ============================================================
# Utilities
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return np.nan


def get_side_from_fen(fen):

    if not isinstance(
        fen,
        str,
    ):
        return None

    parts = fen.split()

    if len(parts) < 2:
        return None

    if parts[1] == "w":
        return "White"

    if parts[1] == "b":
        return "Black"

    return None


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


def safe_std(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) <= 1:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
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
# Load queue
# ============================================================

def load_queue(
    path,
    label,
):

    print()
    print("=" * 80)
    print(
        f"LOADING {label.upper()}"
    )
    print("=" * 80)

    print(
        f"Path: {path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Queue not found:\n{path}"
        )

    rows = []

    invalid = 0

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

                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                print(
                    f"WARNING: invalid JSON "
                    f"at line {line_number}"
                )

                invalid += 1
                continue

            if not isinstance(
                record,
                dict,
            ):

                invalid += 1
                continue

            fen = record.get(
                "fen"
            )

            if not isinstance(
                fen,
                str,
            ):

                print(
                    f"WARNING: missing FEN "
                    f"at line {line_number}"
                )

                invalid += 1
                continue

            H = safe_float(
                record.get("H")
            )

            U = safe_float(
                record.get("U")
            )

            HU = safe_float(
                record.get("HU")
            )

            score = safe_float(
                record.get("score")
            )

            I_norm = safe_float(
                record.get("I_norm")
            )

            reward = safe_float(
                record.get("reward")
            )

            confidence = (
                record.get(
                    "oracle_confidence"
                )
            )

            situation = (
                record.get(
                    "oracle_situation"
                )
            )

            side = get_side_from_fen(
                fen
            )

            rows.append(
                {
                    "fen":
                        fen,

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

                    "reward":
                        reward,

                    "abs_reward":
                        (
                            abs(reward)
                            if np.isfinite(
                                reward
                            )
                            else np.nan
                        ),

                    "oracle_confidence":
                        confidence,

                    "oracle_situation":
                        situation,

                    "side_to_move":
                        side,

                    "queue":
                        label,
                }
            )

    df = pd.DataFrame(
        rows
    )

    print(
        f"Valid records: {len(df):,}"
    )

    print(
        f"Invalid records: {invalid:,}"
    )

    return df


# ============================================================
# Queue structure
# ============================================================

def analyze_structure(
    random_df,
    al_df,
):

    print()
    print("=" * 80)
    print("QUEUE STRUCTURE")
    print("=" * 80)

    random_fens = set(
        random_df["fen"]
    )

    al_fens = set(
        al_df["fen"]
    )

    for label, df in [
        ("Random Oracle", random_df),
        ("AL(I)", al_df),
    ]:

        unique_fens = (
            df["fen"]
            .nunique()
        )

        duplicates = (
            len(df)
            - unique_fens
        )

        print()
        print(label)

        print(
            f"  Records: "
            f"{len(df):,}"
        )

        print(
            f"  Unique FENs: "
            f"{unique_fens:,}"
        )

        print(
            f"  Duplicate records: "
            f"{duplicates:,}"
        )

    intersection = (
        random_fens
        & al_fens
    )

    union = (
        random_fens
        | al_fens
    )

    print()
    print("Cross-queue overlap")

    print(
        f"  Shared FENs: "
        f"{len(intersection):,}"
    )

    print(
        f"  Union: "
        f"{len(union):,}"
    )

    if len(random_fens):

        print(
            f"  Random queue overlap: "
            f"{len(intersection) / len(random_fens):.2%}"
        )

    if len(al_fens):

        print(
            f"  AL(I) queue overlap: "
            f"{len(intersection) / len(al_fens):.2%}"
        )

    return {
        "random_n":
            len(random_df),

        "al_n":
            len(al_df),

        "random_unique":
            len(random_fens),

        "al_unique":
            len(al_fens),

        "shared":
            len(intersection),

        "union":
            len(union),
    }


# ============================================================
# Continuous variables
# ============================================================

CONTINUOUS_VARIABLES = [
    "H",
    "U",
    "HU",
    "score",
    "I_norm",
    "reward",
    "abs_reward",
]


def mann_whitney(
    x,
    y,
):

    try:

        from scipy.stats import mannwhitneyu

    except ImportError:

        return (
            np.nan,
            np.nan,
        )

    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    x = x[
        np.isfinite(x)
    ]

    y = y[
        np.isfinite(y)
    ]

    if len(x) == 0 or len(y) == 0:

        return (
            np.nan,
            np.nan,
        )

    result = mannwhitneyu(
        x,
        y,
        alternative="two-sided",
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def compare_continuous(
    random_df,
    al_df,
):

    rows = []

    for variable in CONTINUOUS_VARIABLES:

        random_values = (
            random_df[
                variable
            ]
        )

        al_values = (
            al_df[
                variable
            ]
        )

        U_stat, p_value = (
            mann_whitney(
                random_values,
                al_values,
            )
        )

        random_mean = safe_mean(
            random_values
        )

        al_mean = safe_mean(
            al_values
        )

        difference = (
            al_mean
            - random_mean
        )

        rows.append(
            {
                "variable":
                    variable,

                "random_n":
                    int(
                        random_values.notna().sum()
                    ),

                "al_n":
                    int(
                        al_values.notna().sum()
                    ),

                "random_mean":
                    random_mean,

                "al_mean":
                    al_mean,

                "AL_minus_Random":
                    difference,

                "random_std":
                    safe_std(
                        random_values
                    ),

                "al_std":
                    safe_std(
                        al_values
                    ),

                "random_median":
                    safe_median(
                        random_values
                    ),

                "al_median":
                    safe_median(
                        al_values
                    ),

                "random_q25":
                    safe_quantile(
                        random_values,
                        0.25,
                    ),

                "al_q25":
                    safe_quantile(
                        al_values,
                        0.25,
                    ),

                "random_q50":
                    safe_quantile(
                        random_values,
                        0.50,
                    ),

                "al_q50":
                    safe_quantile(
                        al_values,
                        0.50,
                    ),

                "random_q75":
                    safe_quantile(
                        random_values,
                        0.75,
                    ),

                "al_q75":
                    safe_quantile(
                        al_values,
                        0.75,
                    ),

                "random_q90":
                    safe_quantile(
                        random_values,
                        0.90,
                    ),

                "al_q90":
                    safe_quantile(
                        al_values,
                        0.90,
                    ),

                "random_q95":
                    safe_quantile(
                        random_values,
                        0.95,
                    ),

                "al_q95":
                    safe_quantile(
                        al_values,
                        0.95,
                    ),

                "random_q99":
                    safe_quantile(
                        random_values,
                        0.99,
                    ),

                "al_q99":
                    safe_quantile(
                        al_values,
                        0.99,
                    ),

                "random_min":
                    safe_quantile(
                        random_values,
                        0.0,
                    ),

                "al_min":
                    safe_quantile(
                        al_values,
                        0.0,
                    ),

                "random_max":
                    safe_quantile(
                        random_values,
                        1.0,
                    ),

                "al_max":
                    safe_quantile(
                        al_values,
                        1.0,
                    ),

                "mann_whitney_U":
                    U_stat,

                "p_value":
                    p_value,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Categorical variables
# ============================================================

CATEGORICAL_VARIABLES = [
    "oracle_confidence",
    "oracle_situation",
    "side_to_move",
]


def categorical_comparison(
    random_df,
    al_df,
    variable,
):

    random_counts = (
        random_df[
            variable
        ]
        .fillna("MISSING")
        .value_counts()
    )

    al_counts = (
        al_df[
            variable
        ]
        .fillna("MISSING")
        .value_counts()
    )

    categories = sorted(
        set(
            random_counts.index
        )
        |
        set(
            al_counts.index
        ),
        key=str,
    )

    rows = []

    for category in categories:

        random_n = int(
            random_counts.get(
                category,
                0,
            )
        )

        al_n = int(
            al_counts.get(
                category,
                0,
            )
        )

        random_fraction = (
            random_n
            / len(random_df)
            if len(random_df)
            else np.nan
        )

        al_fraction = (
            al_n
            / len(al_df)
            if len(al_df)
            else np.nan
        )

        rows.append(
            {
                "category":
                    category,

                "random_n":
                    random_n,

                "random_pct":
                    random_fraction
                    * 100,

                "al_n":
                    al_n,

                "al_pct":
                    al_fraction
                    * 100,

                "AL_minus_Random_pp":
                    (
                        al_fraction
                        - random_fraction
                    )
                    * 100,
            }
        )

    return pd.DataFrame(
        rows
    )


def compare_categorical(
    random_df,
    al_df,
):

    results = {}

    for variable in (
        CATEGORICAL_VARIABLES
    ):

        results[variable] = (
            categorical_comparison(
                random_df,
                al_df,
                variable,
            )
        )

    return results


# ============================================================
# Reward-specific analysis
# ============================================================

def reward_analysis(
    random_df,
    al_df,
):

    print()
    print("=" * 80)
    print("REWARD ANALYSIS")
    print("=" * 80)

    rows = []

    for label, df in [
        ("Random Oracle", random_df),
        ("AL(I)", al_df),
    ]:

        rewards = pd.to_numeric(
            df["reward"],
            errors="coerce",
        )

        n = len(
            rewards[
                np.isfinite(
                    rewards
                )
            ]
        )

        wins = int(
            (
                rewards == 1
            ).sum()
        )

        draws = int(
            (
                rewards == 0
            ).sum()
        )

        losses = int(
            (
                rewards == -1
            ).sum()
        )

        rows.append(
            {
                "queue":
                    label,

                "n":
                    n,

                "wins":
                    wins,

                "win_pct":
                    (
                        wins / n * 100
                        if n
                        else np.nan
                    ),

                "draws":
                    draws,

                "draw_pct":
                    (
                        draws / n * 100
                        if n
                        else np.nan
                    ),

                "losses":
                    losses,

                "loss_pct":
                    (
                        losses / n * 100
                        if n
                        else np.nan
                    ),

                "mean_reward":
                    safe_mean(
                        rewards
                    ),

                "mean_abs_reward":
                    safe_mean(
                        rewards.abs()
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
# Tail analysis
# ============================================================

def tail_analysis(
    random_df,
    al_df,
):

    print()
    print("=" * 80)
    print("TAIL ANALYSIS")
    print("=" * 80)

    rows = []

    variables = [
        "H",
        "U",
        "HU",
        "score",
        "I_norm",
    ]

    for variable in variables:

        random_values = pd.to_numeric(
            random_df[
                variable
            ],
            errors="coerce",
        ).dropna()

        al_values = pd.to_numeric(
            al_df[
                variable
            ],
            errors="coerce",
        ).dropna()

        for percentile in [
            90,
            95,
            99,
        ]:

            random_q = (
                random_values
                .quantile(
                    percentile / 100
                )
            )

            al_q = (
                al_values
                .quantile(
                    percentile / 100
                )
            )

            # How much of the other queue lies
            # above the selected queue's threshold?

            random_above_al = (
                (
                    random_values
                    >= al_q
                )
                .mean()
                * 100
            )

            al_above_random = (
                (
                    al_values
                    >= random_q
                )
                .mean()
                * 100
            )

            rows.append(
                {
                    "variable":
                        variable,

                    "percentile":
                        percentile,

                    "random_quantile":
                        random_q,

                    "al_quantile":
                        al_q,

                    "random_pct_above_AL_threshold":
                        random_above_al,

                    "al_pct_above_Random_threshold":
                        al_above_random,
                }
            )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    return result


# ============================================================
# Correlation with acquisition score
# ============================================================

def acquisition_correlations(
    random_df,
    al_df,
):

    print()
    print("=" * 80)
    print("I_NORM CORRELATIONS")
    print("=" * 80)

    variables = [
        "H",
        "U",
        "HU",
        "reward",
        "abs_reward",
    ]

    rows = []

    for label, df in [
        ("Random Oracle", random_df),
        ("AL(I)", al_df),
    ]:

        for variable in variables:

            subset = df[
                [
                    "I_norm",
                    variable,
                ]
            ].dropna()

            if len(subset) < 3:

                pearson = np.nan
                spearman = np.nan

            else:

                pearson = (
                    subset[
                        "I_norm"
                    ]
                    .corr(
                        subset[
                            variable
                        ],
                        method="pearson",
                    )
                )

                spearman = (
                    subset[
                        "I_norm"
                    ]
                    .corr(
                        subset[
                            variable
                        ],
                        method="spearman",
                    )
                )

            rows.append(
                {
                    "queue":
                        label,

                    "variable":
                        variable,

                    "pearson":
                        pearson,

                    "spearman":
                        spearman,
                }
            )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    return result


# ============================================================
# Signal / annotation cross-analysis
# ============================================================

def group_by_situation(
    random_df,
    al_df,
):

    print()
    print("=" * 80)
    print("SIGNALS BY ORACLE SITUATION")
    print("=" * 80)

    rows = []

    for label, df in [
        ("Random Oracle", random_df),
        ("AL(I)", al_df),
    ]:

        grouped = (
            df.groupby(
                "oracle_situation",
                dropna=False,
            )
        )

        for situation, group in grouped:

            rows.append(
                {
                    "queue":
                        label,

                    "situation":
                        situation,

                    "n":
                        len(group),

                    "mean_H":
                        safe_mean(
                            group["H"]
                        ),

                    "mean_U":
                        safe_mean(
                            group["U"]
                        ),

                    "mean_HU":
                        safe_mean(
                            group["HU"]
                        ),

                    "mean_score":
                        safe_mean(
                            group["score"]
                        ),

                    "mean_I_norm":
                        safe_mean(
                            group["I_norm"]
                        ),

                    "mean_reward":
                        safe_mean(
                            group["reward"]
                        ),

                    "mean_abs_reward":
                        safe_mean(
                            group["abs_reward"]
                        ),

                    "mean_confidence_numeric":
                        np.nan,
                }
            )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    return result


# ============================================================
# Confidence analysis
# ============================================================

def confidence_analysis(
    random_df,
    al_df,
):

    print()
    print("=" * 80)
    print("CONFIDENCE ANALYSIS")
    print("=" * 80)

    confidence_order = [
        "low",
        "medium",
        "high",
    ]

    rows = []

    for label, df in [
        ("Random Oracle", random_df),
        ("AL(I)", al_df),
    ]:

        for confidence in confidence_order:

            subset = df[
                df[
                    "oracle_confidence"
                ]
                .astype(str)
                .str.lower()
                == confidence
            ]

            if len(subset) == 0:
                continue

            rows.append(
                {
                    "queue":
                        label,

                    "confidence":
                        confidence,

                    "n":
                        len(subset),

                    "pct":
                        len(subset)
                        / len(df)
                        * 100,

                    "mean_I_norm":
                        safe_mean(
                            subset["I_norm"]
                        ),

                    "mean_H":
                        safe_mean(
                            subset["H"]
                        ),

                    "mean_U":
                        safe_mean(
                            subset["U"]
                        ),

                    "mean_HU":
                        safe_mean(
                            subset["HU"]
                        ),

                    "mean_abs_reward":
                        safe_mean(
                            subset["abs_reward"]
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
                f"{x:.6f}",
        )
    )

    return result


# ============================================================
# Plot: signals
# ============================================================

def plot_signals(
    random_df,
    al_df,
):

    print(
        "Plotting signal distributions..."
    )

    variables = [
        "H",
        "U",
        "HU",
        "score",
        "I_norm",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    axes = axes.flatten()

    for index, variable in enumerate(
        variables
    ):

        ax = axes[index]

        random_values = pd.to_numeric(
            random_df[
                variable
            ],
            errors="coerce",
        ).dropna()

        al_values = pd.to_numeric(
            al_df[
                variable
            ],
            errors="coerce",
        ).dropna()

        if len(random_values):

            ax.hist(
                random_values,
                bins=50,
                density=True,
                alpha=0.5,
                label="Random Oracle",
            )

        if len(al_values):

            ax.hist(
                al_values,
                bins=50,
                density=True,
                alpha=0.5,
                label="AL(I)",
            )

        ax.set_title(
            variable
        )

        ax.set_xlabel(
            variable
        )

        ax.set_ylabel(
            "Density"
        )

        ax.legend()

    # Remove unused sixth subplot.

    axes[-1].axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "signals_distribution.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot: reward
# ============================================================

def plot_reward(
    random_df,
    al_df,
):

    print(
        "Plotting reward distribution..."
    )

    categories = [
        -1,
        0,
        1,
    ]

    labels = [
        "Loss",
        "Draw",
        "Win",
    ]

    random_counts = [
        (
            random_df["reward"]
            == value
        ).sum()
        / len(random_df)
        * 100
        for value in categories
    ]

    al_counts = [
        (
            al_df["reward"]
            == value
        ).sum()
        / len(al_df)
        * 100
        for value in categories
    ]

    x = np.arange(
        len(categories)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        x - width / 2,
        random_counts,
        width,
        label="Random Oracle",
    )

    plt.bar(
        x + width / 2,
        al_counts,
        width,
        label="AL(I)",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Oracle reward distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "reward_distribution.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot: confidence
# ============================================================

def plot_confidence(
    random_df,
    al_df,
):

    print(
        "Plotting confidence distribution..."
    )

    categories = [
        "low",
        "medium",
        "high",
    ]

    random_values = []
    al_values = []

    for category in categories:

        random_values.append(
            (
                random_df[
                    "oracle_confidence"
                ]
                .astype(str)
                .str.lower()
                == category
            ).mean()
            * 100
        )

        al_values.append(
            (
                al_df[
                    "oracle_confidence"
                ]
                .astype(str)
                .str.lower()
                == category
            ).mean()
            * 100
        )

    x = np.arange(
        len(categories)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        x - width / 2,
        random_values,
        width,
        label="Random Oracle",
    )

    plt.bar(
        x + width / 2,
        al_values,
        width,
        label="AL(I)",
    )

    plt.xticks(
        x,
        [
            "Low",
            "Medium",
            "High",
        ],
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Oracle confidence distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "confidence_distribution.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot: situation
# ============================================================

def plot_situation(
    random_df,
    al_df,
):

    print(
        "Plotting situation distribution..."
    )

    categories = [
        "critical",
        "non_critical",
        "outcome_independent",
    ]

    random_values = []
    al_values = []

    for category in categories:

        random_values.append(
            (
                random_df[
                    "oracle_situation"
                ]
                .astype(str)
                .str.lower()
                == category
            ).mean()
            * 100
        )

        al_values.append(
            (
                al_df[
                    "oracle_situation"
                ]
                .astype(str)
                .str.lower()
                == category
            ).mean()
            * 100
        )

    x = np.arange(
        len(categories)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x - width / 2,
        random_values,
        width,
        label="Random Oracle",
    )

    plt.bar(
        x + width / 2,
        al_values,
        width,
        label="AL(I)",
    )

    plt.xticks(
        x,
        [
            "Critical",
            "Non-critical",
            "Outcome-independent",
        ],
        rotation=15,
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Oracle situation distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "situation_distribution.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot: side
# ============================================================

def plot_side(
    random_df,
    al_df,
):

    print(
        "Plotting side-to-move distribution..."
    )

    categories = [
        "White",
        "Black",
    ]

    random_values = [
        (
            random_df[
                "side_to_move"
            ]
            == category
        ).mean()
        * 100
        for category in categories
    ]

    al_values = [
        (
            al_df[
                "side_to_move"
            ]
            == category
        ).mean()
        * 100
        for category in categories
    ]

    x = np.arange(
        len(categories)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        x - width / 2,
        random_values,
        width,
        label="Random Oracle",
    )

    plt.bar(
        x + width / 2,
        al_values,
        width,
        label="AL(I)",
    )

    plt.xticks(
        x,
        categories,
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Side-to-move distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "side_distribution.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# Report
# ============================================================

def build_report(
    random_df,
    al_df,
    structure,
    continuous,
    categorical,
    tail,
    correlations,
    situation_df,
    confidence_df,
    reward_df,
):

    lines = []

    def add(text=""):
        lines.append(
            str(text)
        )

    add("=" * 80)
    add(
        "ALBERTA - ORACLE QUEUE COMPARISON"
    )
    add("=" * 80)
    add()

    add(
        f"Random queue: "
        f"{RANDOM_QUEUE_PATH}"
    )

    add(
        f"AL queue: "
        f"{AL_QUEUE_PATH}"
    )

    add()

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    add(
        "QUEUE STRUCTURE"
    )
    add("-" * 80)

    add(
        f"Random records: "
        f"{structure['random_n']:,}"
    )

    add(
        f"AL records: "
        f"{structure['al_n']:,}"
    )

    add(
        f"Random unique FENs: "
        f"{structure['random_unique']:,}"
    )

    add(
        f"AL unique FENs: "
        f"{structure['al_unique']:,}"
    )

    add(
        f"Shared FENs: "
        f"{structure['shared']:,}"
    )

    add(
        f"Union: "
        f"{structure['union']:,}"
    )

    add()

    # --------------------------------------------------------
    # Continuous
    # --------------------------------------------------------

    add(
        "CONTINUOUS VARIABLES"
    )
    add("-" * 80)

    add(
        continuous.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    for variable, table in (
        categorical.items()
    ):

        add(
            f"CATEGORICAL VARIABLE: "
            f"{variable}"
        )

        add(
            "-" * 80
        )

        add(
            table.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        add()

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    add(
        "REWARD ANALYSIS"
    )
    add("-" * 80)

    add(
        reward_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Tail
    # --------------------------------------------------------

    add(
        "TAIL ANALYSIS"
    )
    add("-" * 80)

    add(
        tail.to_string(
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
        "I_NORM CORRELATIONS"
    )
    add("-" * 80)

    add(
        correlations.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Situation
    # --------------------------------------------------------

    add(
        "SIGNALS BY ORACLE SITUATION"
    )
    add("-" * 80)

    add(
        situation_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    add(
        "CONFIDENCE ANALYSIS"
    )
    add("-" * 80)

    add(
        confidence_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    add(
        "INTERPRETATION GUIDE"
    )
    add("-" * 80)

    add(
        "Positive AL_minus_Random values indicate that "
        "AL(I) selects positions with higher values "
        "for the corresponding variable."
    )

    add(
        "For H, U, HU, score and I_norm, a positive "
        "difference means that the AL queue is shifted "
        "toward larger values."
    )

    add(
        "For abs_reward, a positive difference means "
        "that AL(I) selects more outcome-polarized positions."
    )

    add(
        "For reward itself, the sign must be interpreted "
        "carefully because it is from the side-to-move perspective."
    )

    add(
        "For categorical variables, AL_minus_Random_pp "
        "is expressed in percentage points."
    )

    add(
        "A small p-value in the Mann-Whitney test indicates "
        "that the distributions differ statistically; it does "
        "not by itself establish that AL(I) is more useful "
        "for learning."
    )

    add()

    add(
        "SCIENTIFIC QUESTION"
    )
    add("-" * 80)

    add(
        "The central question is whether AL(I) selects "
        "a systematically different type of position "
        "from Random Oracle, and whether those differences "
        "correspond to properties of the Oracle annotations "
        "that could plausibly increase their learning value."
    )

    add(
        "In particular, compare whether AL(I) enriches "
        "critical situations, high-confidence annotations, "
        "large |reward|, or other properties relative to "
        "the random baseline."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    random_df = load_queue(
        RANDOM_QUEUE_PATH,
        "Random Oracle",
    )

    al_df = load_queue(
        AL_QUEUE_PATH,
        "AL(I)",
    )

    if len(random_df) == 0:
        raise RuntimeError(
            "Random Oracle queue is empty."
        )

    if len(al_df) == 0:
        raise RuntimeError(
            "AL(I) queue is empty."
        )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    structure = analyze_structure(
        random_df,
        al_df,
    )

    # --------------------------------------------------------
    # Continuous
    # --------------------------------------------------------

    continuous = (
        compare_continuous(
            random_df,
            al_df,
        )
    )

    print()
    print("=" * 80)
    print("CONTINUOUS VARIABLES")
    print("=" * 80)

    print(
        continuous[
            [
                "variable",
                "random_n",
                "al_n",
                "random_mean",
                "al_mean",
                "AL_minus_Random",
                "random_median",
                "al_median",
                "random_q90",
                "al_q90",
                "random_q99",
                "al_q99",
                "p_value",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    categorical = (
        compare_categorical(
            random_df,
            al_df,
        )
    )

    for variable, table in (
        categorical.items()
    ):

        print()
        print("=" * 80)
        print(
            f"{variable.upper()}"
        )
        print("=" * 80)

        print(
            table.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    reward_df = reward_analysis(
        random_df,
        al_df,
    )

    # --------------------------------------------------------
    # Tail
    # --------------------------------------------------------

    tail = tail_analysis(
        random_df,
        al_df,
    )

    # --------------------------------------------------------
    # I correlations
    # --------------------------------------------------------

    correlations = (
        acquisition_correlations(
            random_df,
            al_df,
        )
    )

    # --------------------------------------------------------
    # Situation
    # --------------------------------------------------------

    situation_df = (
        group_by_situation(
            random_df,
            al_df,
        )
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_df = (
        confidence_analysis(
            random_df,
            al_df,
        )
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_signals(
        random_df,
        al_df,
    )

    plot_reward(
        random_df,
        al_df,
    )

    plot_confidence(
        random_df,
        al_df,
    )

    plot_situation(
        random_df,
        al_df,
    )

    plot_side(
        random_df,
        al_df,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        random_df,
        al_df,
        structure,
        continuous,
        categorical,
        tail,
        correlations,
        situation_df,
        confidence_df,
        reward_df,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            report
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Report:"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print()
    print(
        f"Plots:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()