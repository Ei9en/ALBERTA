#!/usr/bin/env python3

"""
ALBERTA - H/U Learning Progression Proxy Analysis
===================================================

Analyse si les signaux H et U peuvent servir de proxy statistique
de la progression d'un agent RL, sans aucun nouveau training.

Question scientifique
---------------------

Les signaux H et U ne semblent pas nécessairement identifier les
positions dont l'annotation apporte le plus de valeur au RL.

En revanche, ils peuvent organiser les positions selon des modes
de difficulté interprétables.

Cette analyse teste donc une hypothèse différente :

    H/U peuvent-ils servir de proxy statistique de la progression
    et/ou de diagnostic des lacunes de l'agent ?

IMPORTANT
---------

Le fichier uncertainty_stats_1-60.json ne contient PAS de champ
epoch/checkpoint.

En revanche, le pipeline de génération conserve l'ordre
chronologique des observations :

    RL10 observations
    RL11 observations
    ...
    RL60 observations

dans un même tableau cumulatif.

Le script exploite donc directement cet ordre et divise le fichier
en fenêtres chronologiques de taille égale.

Ces fenêtres ne sont PAS considérées comme des checkpoints exacts.
Elles servent uniquement à étudier l'évolution statistique des
signaux au cours du training.

Aucun nouveau training n'est effectué.

Expériences
-----------

1. Chronological H/U statistics
   - mean
   - median
   - std
   - quantiles
   - reward statistics

2. High-uncertainty tails
   - fractions above W1 reference thresholds
   - evolution of high-H and high-U populations

3. Reward-conditioned uncertainty
   - draw rate among high-H positions
   - decisive rate among high-U positions
   - mean reward
   - mean absolute reward

4. Chronological trends
   - monotonicity
   - relative change between first and last window
   - Spearman correlation with chronological position

5. H/U diagnostic quadrants
   - low H / low U
   - high H / low U
   - low H / high U
   - high H / high U

6. Distribution structure
   - number of records
   - unique FENs
   - evolution of repeated-FEN prevalence

Interpretation
--------------

The analysis does NOT prove that H/U cause learning progression.

It tests whether their statistical distributions evolve
systematically during an already completed training run.

A coherent temporal evolution is evidence compatible with their
use as diagnostic progression proxies.

It is NOT evidence that H/U are optimal acquisition functions.

The known RL10 -> RL60 performance progression is included only
as contextual evidence and is NOT statistically paired with the
chronological windows.

Outputs
-------

    data/uncertainty_progression/
        progression_report.txt

        chronological_statistics.csv
        uncertainty_tails.csv
        reward_conditioned_progression.csv
        diagnostic_quadrants.csv
        chronological_trends.csv

        hu_progression.png
        uncertainty_tail_progression.png
        reward_conditioned_progression.png
        diagnostic_quadrants.png

Usage
-----

    python analysis/proxy.py

or:

    python analysis/proxy.py \
        --input data/selfplay_jsons/uncertainty_stats_1-60.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-60.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_progression"
)

# ------------------------------------------------------------
# Chronological windows
#
# The JSON contains observations accumulated in chronological
# epoch order, but no explicit epoch field.
#
# Six equal windows provide a coarse temporal description.
# They are NOT exact RL checkpoints.
# ------------------------------------------------------------

N_WINDOWS = 6

WINDOW_LABELS = [
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
]

# ------------------------------------------------------------
# Known RL progression.
#
# Used ONLY as contextual reference in the report.
# It is deliberately NOT mapped one-to-one onto W1...W6.
# ------------------------------------------------------------

PERFORMANCE = {
    10: {
        "elo": 1324,
        "score": 26.0,
    },
    20: {
        "elo": 1344,
        "score": 28.6,
    },
    30: {
        "elo": 1412,
        "score": 38.0,
    },
    40: {
        "elo": 1511,
        "score": 52.2,
    },
    50: {
        "elo": 1655,
        "score": 71.8,
    },
    60: {
        "elo": 1754,
        "score": 83.4,
    },
}

# ------------------------------------------------------------
# Reference thresholds.
#
# W1 is the earliest chronological portion of the dataset.
# Its distribution is used as a descriptive baseline.
# ------------------------------------------------------------

REFERENCE_QUANTILES = [
    0.90,
    0.95,
    0.99,
]

# ------------------------------------------------------------
# Diagnostic quadrants use the W1 median.
# ------------------------------------------------------------

QUADRANT_REFERENCE_QUANTILE = 0.50


# ============================================================
# Utilities
# ============================================================

def finite_values(series):
    """
    Return finite numeric values from a Series.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    return values


def safe_mean(series):
    values = finite_values(
        series
    )

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def safe_median(series):
    values = finite_values(
        series
    )

    if len(values) == 0:
        return np.nan

    return float(
        values.median()
    )


def safe_std(series):
    values = finite_values(
        series
    )

    if len(values) <= 1:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


def safe_quantile(
    series,
    q,
):
    values = finite_values(
        series
    )

    if len(values) == 0:
        return np.nan

    return float(
        values.quantile(q)
    )


def safe_fraction(condition):
    condition = pd.Series(
        condition
    )

    if len(condition) == 0:
        return np.nan

    return float(
        condition.mean()
    )


def spearman_correlation(
    values
):
    """
    Spearman correlation between chronological position
    and window-level values.

    Only six windows are used, so this is descriptive rather
    than a high-powered inferential test.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    x = np.arange(
        len(values),
        dtype=float,
    )

    mask = (
        np.isfinite(values)
        & np.isfinite(x)
    )

    values = values[mask]
    x = x[mask]

    if len(values) < 3:
        return np.nan

    return float(
        pd.Series(x).corr(
            pd.Series(values),
            method="spearman",
        )
    )


# ============================================================
# Load
# ============================================================

def load_data(
    input_path
):

    print()
    print("=" * 72)
    print(
        "ALBERTA - H/U LEARNING PROGRESSION ANALYSIS"
    )
    print("=" * 72)

    print()
    print(
        "Loading uncertainty statistics"
    )
    print("-" * 72)

    print(
        f"File: {input_path}"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:

        raw = json.load(f)

    # --------------------------------------------------------
    # Accept either a direct list or a wrapped JSON structure.
    # --------------------------------------------------------

    if isinstance(
        raw,
        dict,
    ):

        for key in [
            "records",
            "data",
            "statistics",
            "observations",
        ]:

            if key in raw:

                raw = raw[key]
                break

    if not isinstance(
        raw,
        list,
    ):

        raise ValueError(
            "Expected a JSON list of records."
        )

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

        if not isinstance(
            fen,
            str,
        ):

            invalid += 1
            continue

        # ----------------------------------------------------
        # H
        # ----------------------------------------------------

        H = record.get(
            "H"
        )

        # ----------------------------------------------------
        # U
        # ----------------------------------------------------

        U = record.get(
            "U"
        )

        # ----------------------------------------------------
        # HU
        # ----------------------------------------------------

        HU = record.get(
            "HU"
        )

        try:

            H = float(H)
            U = float(U)

            if HU is None:

                HU = H * U

            else:

                HU = float(HU)

        except (
            TypeError,
            ValueError,
        ):

            invalid += 1
            continue

        if not (
            np.isfinite(H)
            and np.isfinite(U)
            and np.isfinite(HU)
        ):

            invalid += 1
            continue

        # ----------------------------------------------------
        # Reward
        #
        # Prefer explicit reward if available.
        # Otherwise reconstruct it from result.
        # ----------------------------------------------------

        reward = record.get(
            "reward"
        )

        if reward is not None:

            try:

                reward = float(
                    reward
                )

            except (
                TypeError,
                ValueError,
            ):

                reward = np.nan

        else:

            result = record.get(
                "result"
            )

            if result == "1-0":

                reward = 1.0

            elif result == "0-1":

                reward = -1.0

            elif result == "1/2-1/2":

                reward = 0.0

            else:

                reward = np.nan

        rows.append(
            {
                "order": len(rows),
                "fen": fen,
                "H": H,
                "U": U,
                "HU": HU,
                "reward": reward,
            }
        )

    df = pd.DataFrame(
        rows
    )

    print(
        f"Valid records: {len(df):,}"
    )

    print(
        f"Invalid records skipped: "
        f"{invalid:,}"
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid records."
        )

    return df


# ============================================================
# Chronological windows
# ============================================================

def assign_chronological_windows(
    df
):

    print()
    print("=" * 72)
    print(
        "CHRONOLOGICAL WINDOWS"
    )
    print("=" * 72)

    df = df.copy()

    n = len(df)

    indices = np.array_split(
        np.arange(n),
        N_WINDOWS,
    )

    df["window"] = ""

    for i, idx in enumerate(
        indices
    ):

        label = WINDOW_LABELS[i]

        df.loc[
            idx,
            "window",
        ] = label

        print(
            f"{label}: "
            f"records "
            f"{idx[0]:,} -> "
            f"{idx[-1]:,} "
            f"({len(idx):,})"
        )

    return df


# ============================================================
# Experiment 1
# Chronological statistics
# ============================================================

def chronological_statistics(
    df
):

    print()
    print("=" * 72)
    print(
        "EXPERIMENT 1 - CHRONOLOGICAL H/U STATISTICS"
    )
    print("=" * 72)

    rows = []

    for window in WINDOW_LABELS:

        subset = df[
            df["window"] == window
        ]

        row = {
            "window": window,

            "records":
                len(subset),

            "unique_fens":
                subset["fen"].nunique(),
        }

        # ----------------------------------------------------
        # Signal statistics
        # ----------------------------------------------------

        for signal in [
            "H",
            "U",
            "HU",
        ]:

            row[
                f"{signal}_mean"
            ] = safe_mean(
                subset[signal]
            )

            row[
                f"{signal}_median"
            ] = safe_median(
                subset[signal]
            )

            row[
                f"{signal}_std"
            ] = safe_std(
                subset[signal]
            )

            for q in [
                0.75,
                0.90,
                0.95,
                0.99,
            ]:

                row[
                    f"{signal}_q{int(q * 100)}"
                ] = safe_quantile(
                    subset[signal],
                    q,
                )

        # ----------------------------------------------------
        # Reward statistics
        # ----------------------------------------------------

        rewards = finite_values(
            subset["reward"]
        )

        row[
            "reward_mean"
        ] = safe_mean(
            rewards
        )

        row[
            "reward_abs_mean"
        ] = safe_mean(
            rewards.abs()
        )

        row[
            "draw_rate"
        ] = (
            safe_fraction(
                rewards == 0
            )
            if len(rewards)
            else np.nan
        )

        row[
            "decisive_rate"
        ] = (
            safe_fraction(
                rewards.abs() == 1
            )
            if len(rewards)
            else np.nan
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    print()

    print(
        result[
            [
                "window",
                "records",
                "unique_fens",
                "H_mean",
                "H_median",
                "U_mean",
                "U_median",
                "HU_mean",
                "HU_median",
                "draw_rate",
                "decisive_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    return result


# ============================================================
# Reference thresholds
# ============================================================

def build_reference_thresholds(
    df
):

    reference = df[
        df["window"]
        == WINDOW_LABELS[0]
    ]

    thresholds = {}

    for signal in [
        "H",
        "U",
    ]:

        thresholds[
            signal
        ] = {}

        for q in REFERENCE_QUANTILES:

            thresholds[
                signal
            ][q] = safe_quantile(
                reference[signal],
                q,
            )

    return thresholds


# ============================================================
# Experiment 2
# Uncertainty tails
# ============================================================

def analyze_uncertainty_tails(
    df,
    thresholds,
):

    print()
    print("=" * 72)
    print(
        "EXPERIMENT 2 - UNCERTAINTY TAILS"
    )
    print("=" * 72)

    rows = []

    for window in WINDOW_LABELS:

        subset = df[
            df["window"] == window
        ]

        row = {
            "window": window,
        }

        for signal in [
            "H",
            "U",
        ]:

            for q in REFERENCE_QUANTILES:

                threshold = thresholds[
                    signal
                ][q]

                row[
                    f"{signal}_above_ref_q{int(q * 100)}"
                ] = safe_fraction(
                    subset[signal]
                    >= threshold
                )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    return result


# ============================================================
# Experiment 3
# Reward-conditioned uncertainty
# ============================================================

def reward_conditioned_analysis(
    df,
    thresholds,
):

    print()
    print("=" * 72)
    print(
        "EXPERIMENT 3 - REWARD-CONDITIONED UNCERTAINTY"
    )
    print("=" * 72)

    H_threshold = thresholds[
        "H"
    ][0.90]

    U_threshold = thresholds[
        "U"
    ][0.90]

    rows = []

    for window in WINDOW_LABELS:

        subset = df[
            df["window"] == window
        ]

        high_H = subset[
            subset["H"]
            >= H_threshold
        ]

        high_U = subset[
            subset["U"]
            >= U_threshold
        ]

        high_H_rewards = finite_values(
            high_H["reward"]
        )

        high_U_rewards = finite_values(
            high_U["reward"]
        )

        row = {
            "window": window,

            # ------------------------------------------------
            # High H
            # ------------------------------------------------

            "high_H_n":
                len(high_H),

            "high_H_fraction":
                (
                    len(high_H)
                    / len(subset)
                    if len(subset)
                    else np.nan
                ),

            "high_H_draw_rate":
                (
                    safe_fraction(
                        high_H_rewards == 0
                    )
                    if len(high_H_rewards)
                    else np.nan
                ),

            "high_H_mean_abs_reward":
                safe_mean(
                    high_H_rewards.abs()
                ),

            "high_H_mean_reward":
                safe_mean(
                    high_H_rewards
                ),

            # ------------------------------------------------
            # High U
            # ------------------------------------------------

            "high_U_n":
                len(high_U),

            "high_U_fraction":
                (
                    len(high_U)
                    / len(subset)
                    if len(subset)
                    else np.nan
                ),

            "high_U_decisive_rate":
                (
                    safe_fraction(
                        high_U_rewards.abs()
                        == 1
                    )
                    if len(high_U_rewards)
                    else np.nan
                ),

            "high_U_mean_abs_reward":
                safe_mean(
                    high_U_rewards.abs()
                ),

            "high_U_mean_reward":
                safe_mean(
                    high_U_rewards
                ),
        }

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    return result


# ============================================================
# Experiment 4
# Chronological trends
# ============================================================

def trend_analysis(
    window_df
):

    print()
    print("=" * 72)
    print(
        "EXPERIMENT 4 - CHRONOLOGICAL TRENDS"
    )
    print("=" * 72)

    rows = []

    metrics = [
        "H_mean",
        "H_median",
        "H_q90",
        "H_q95",
        "H_q99",

        "U_mean",
        "U_median",
        "U_q90",
        "U_q95",
        "U_q99",

        "HU_mean",
        "HU_median",
        "HU_q90",
        "HU_q95",
        "HU_q99",

        "draw_rate",
        "decisive_rate",
    ]

    for metric in metrics:

        values = (
            window_df[metric]
            .to_numpy(
                dtype=float
            )
        )

        finite = np.isfinite(
            values
        )

        values = values[
            finite
        ]

        if len(values) < 2:
            continue

        first = values[0]
        last = values[-1]

        absolute_change = (
            last - first
        )

        relative_change = (
            absolute_change
            / abs(first)
            if first != 0
            else np.nan
        )

        differences = np.diff(
            values
        )

        monotonic_increasing = (
            np.all(
                differences >= 0
            )
        )

        monotonic_decreasing = (
            np.all(
                differences <= 0
            )
        )

        rho = spearman_correlation(
            values
        )

        rows.append(
            {
                "metric":
                    metric,

                "first":
                    first,

                "last":
                    last,

                "absolute_change":
                    absolute_change,

                "relative_change":
                    relative_change,

                "spearman_rho":
                    rho,

                "monotonic_increasing":
                    bool(
                        monotonic_increasing
                    ),

                "monotonic_decreasing":
                    bool(
                        monotonic_decreasing
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
                f"{x:.6f}",
        )
    )

    return result


# ============================================================
# Experiment 5
# Diagnostic H/U quadrants
# ============================================================

def diagnostic_quadrants(
    df
):

    print()
    print("=" * 72)
    print(
        "EXPERIMENT 5 - H/U DIAGNOSTIC QUADRANTS"
    )
    print("=" * 72)

    reference = df[
        df["window"]
        == WINDOW_LABELS[0]
    ]

    H_threshold = safe_quantile(
        reference["H"],
        QUADRANT_REFERENCE_QUANTILE,
    )

    U_threshold = safe_quantile(
        reference["U"],
        QUADRANT_REFERENCE_QUANTILE,
    )

    print(
        f"Reference H median: "
        f"{H_threshold:.6f}"
    )

    print(
        f"Reference U median: "
        f"{U_threshold:.6f}"
    )

    rows = []

    for window in WINDOW_LABELS:

        subset = df[
            df["window"] == window
        ]

        high_H = (
            subset["H"]
            >= H_threshold
        )

        high_U = (
            subset["U"]
            >= U_threshold
        )

        quadrants = {
            "low_H_low_U":
                (~high_H)
                & (~high_U),

            "high_H_low_U":
                high_H
                & (~high_U),

            "low_H_high_U":
                (~high_H)
                & high_U,

            "high_H_high_U":
                high_H
                & high_U,
        }

        for name, mask in (
            quadrants.items()
        ):

            q = subset[
                mask
            ]

            rewards = finite_values(
                q["reward"]
            )

            rows.append(
                {
                    "window":
                        window,

                    "quadrant":
                        name,

                    "n":
                        len(q),

                    "fraction":
                        (
                            len(q)
                            / len(subset)
                            if len(subset)
                            else np.nan
                        ),

                    "mean_H":
                        safe_mean(
                            q["H"]
                        ),

                    "mean_U":
                        safe_mean(
                            q["U"]
                        ),

                    "mean_HU":
                        safe_mean(
                            q["HU"]
                        ),

                    "mean_reward":
                        safe_mean(
                            rewards
                        ),

                    "draw_rate":
                        (
                            safe_fraction(
                                rewards == 0
                            )
                            if len(rewards)
                            else np.nan
                        ),

                    "decisive_rate":
                        (
                            safe_fraction(
                                rewards.abs()
                                == 1
                            )
                            if len(rewards)
                            else np.nan
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
                f"{x:.5f}",
        )
    )

    return result


# ============================================================
# Report
# ============================================================

def build_report(
    window_df,
    tail_df,
    reward_df,
    quadrant_df,
    trend_df,
):

    lines = []

    def add(
        text=""
    ):
        lines.append(
            str(text)
        )

    add("=" * 72)
    add(
        "ALBERTA - H/U LEARNING PROGRESSION ANALYSIS"
    )
    add("=" * 72)
    add()

    # --------------------------------------------------------
    # Purpose
    # --------------------------------------------------------

    add(
        "PURPOSE"
    )
    add("-" * 72)

    add(
        "This analysis evaluates whether H and U exhibit "
        "systematic statistical evolution during an already "
        "completed RL training run."
    )

    add(
        "The uncertainty dataset contains no explicit epoch "
        "identifier. Its records are nevertheless accumulated "
        "chronologically during RL10 -> RL60."
    )

    add(
        "The dataset is therefore divided into six chronological "
        "windows. These windows are not treated as exact "
        "checkpoint boundaries."
    )

    add(
        "No new training is performed."
    )

    add()

    # --------------------------------------------------------
    # Known performance
    # --------------------------------------------------------

    add(
        "KNOWN TRAINING PROGRESSION"
    )
    add("-" * 72)

    add(
        "The independently measured fixed round-robin progression "
        "was:"
    )

    for epoch in sorted(
        PERFORMANCE
    ):

        row = PERFORMANCE[
            epoch
        ]

        add(
            f"RL{epoch:02d}: "
            f"Elo={row['elo']}, "
            f"score={row['score']:.1f}%"
        )

    add()

    add(
        "These checkpoint values are used only as contextual "
        "evidence that the underlying agent improved throughout "
        "the training run."
    )

    add(
        "They are not assigned directly to the six chronological "
        "windows because the JSON does not encode exact epoch "
        "boundaries."
    )

    add()

    # --------------------------------------------------------
    # Chronological windows
    # --------------------------------------------------------

    add(
        "CHRONOLOGICAL WINDOWS"
    )
    add("-" * 72)

    for _, row in window_df.iterrows():

        add(
            f"{row['window']}: "
            f"records={int(row['records']):,}, "
            f"FENs={int(row['unique_fens']):,}, "
            f"H_mean={row['H_mean']:.6f}, "
            f"H_median={row['H_median']:.6f}, "
            f"U_mean={row['U_mean']:.6f}, "
            f"U_median={row['U_median']:.6f}, "
            f"HU_mean={row['HU_mean']:.6f}, "
            f"draw={row['draw_rate']:.2%}"
        )

    add()

    # --------------------------------------------------------
    # Trends
    # --------------------------------------------------------

    add(
        "CHRONOLOGICAL TRENDS"
    )
    add("-" * 72)

    for _, row in trend_df.iterrows():

        direction = (
            "non-monotonic"
        )

        if row[
            "monotonic_increasing"
        ]:

            direction = (
                "monotonic increase"
            )

        elif row[
            "monotonic_decreasing"
        ]:

            direction = (
                "monotonic decrease"
            )

        add(
            f"{row['metric']}: "
            f"{row['first']:.6f} -> "
            f"{row['last']:.6f}, "
            f"delta="
            f"{row['absolute_change']:+.6f}, "
            f"relative="
            f"{row['relative_change']:+.2%}, "
            f"Spearman rho="
            f"{row['spearman_rho']:+.4f}, "
            f"{direction}"
        )

    add()

    # --------------------------------------------------------
    # Tails
    # --------------------------------------------------------

    add(
        "HIGH-UNCERTAINTY TAILS"
    )
    add("-" * 72)

    for _, row in tail_df.iterrows():

        add(
            f"{row['window']}: "
            f"H>P90="
            f"{row['H_above_ref_q90']:.2%}, "
            f"H>P95="
            f"{row['H_above_ref_q95']:.2%}, "
            f"H>P99="
            f"{row['H_above_ref_q99']:.2%}, "
            f"U>P90="
            f"{row['U_above_ref_q90']:.2%}, "
            f"U>P95="
            f"{row['U_above_ref_q95']:.2%}, "
            f"U>P99="
            f"{row['U_above_ref_q99']:.2%}"
        )

    add()

    # --------------------------------------------------------
    # Reward-conditioned
    # --------------------------------------------------------

    add(
        "REWARD-CONDITIONED SIGNALS"
    )
    add("-" * 72)

    for _, row in reward_df.iterrows():

        add(
            f"{row['window']}: "
            f"high-H fraction="
            f"{row['high_H_fraction']:.2%}, "
            f"high-H draw rate="
            f"{row['high_H_draw_rate']:.2%}, "
            f"high-H |R|="
            f"{row['high_H_mean_abs_reward']:.4f}, "
            f"high-U fraction="
            f"{row['high_U_fraction']:.2%}, "
            f"high-U decisive rate="
            f"{row['high_U_decisive_rate']:.2%}, "
            f"high-U |R|="
            f"{row['high_U_mean_abs_reward']:.4f}"
        )

    add()

    # --------------------------------------------------------
    # Diagnostic quadrants
    # --------------------------------------------------------

    add(
        "DIAGNOSTIC QUADRANTS"
    )
    add("-" * 72)

    for window in WINDOW_LABELS:

        subset = quadrant_df[
            quadrant_df["window"]
            == window
        ]

        add(
            f"{window}:"
        )

        for _, row in subset.iterrows():

            add(
                f"  {row['quadrant']}: "
                f"{row['fraction']:.2%} "
                f"(draw="
                f"{row['draw_rate']:.2%}, "
                f"decisive="
                f"{row['decisive_rate']:.2%})"
            )

    add()

    # --------------------------------------------------------
    # Diagnostic interpretation
    # --------------------------------------------------------

    add(
        "DIAGNOSTIC INTERPRETATION"
    )
    add("-" * 72)

    add(
        "High-H populations are examined for their association "
        "with draw-heavy outcomes, consistent with a possible "
        "conversion-difficulty interpretation."
    )

    add(
        "High-U populations are examined for their association "
        "with decisive outcomes, consistent with a possible "
        "critical-decision interpretation."
    )

    add(
        "The four H/U quadrants provide a descriptive "
        "decomposition of these two dimensions."
    )

    add(
        "These interpretations are empirical hypotheses about "
        "the observed populations, not formal definitions of H "
        "or U."
    )

    add()

    # --------------------------------------------------------
    # Scientific conclusion
    # --------------------------------------------------------

    add(
        "SCIENTIFIC CONCLUSION"
    )
    add("-" * 72)

    add(
        "The purpose of this experiment is not to establish "
        "H or U as optimal acquisition functions."
    )

    add(
        "Instead, it asks whether their statistical distributions "
        "evolve systematically over the course of an already "
        "completed RL training run."
    )

    add(
        "A coherent chronological evolution of H and U, together "
        "with stable reward-conditioned relationships, provides "
        "evidence compatible with their use as statistical "
        "diagnostic proxies of learning progression."
    )

    add(
        "In particular, a decreasing H is consistent with a "
        "reduction of the population of positions exhibiting "
        "high outcome uncertainty, whereas an increasing U "
        "indicates that the distribution of model disagreement "
        "changes substantially during training."
    )

    add(
        "These observations suggest that H and U can provide "
        "an interpretable statistical interface for monitoring "
        "how the agent's difficulties evolve during learning."
    )

    add(
        "This role is distinct from active-learning acquisition: "
        "a signal may describe where and how an agent struggles "
        "without identifying the annotations that maximize "
        "marginal learning value."
    )

    add(
        "Consequently, diagnostic information and annotation "
        "utility should be treated as distinct quantities."
    )

    add()

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    add(
        "LIMITATIONS"
    )
    add("-" * 72)

    add(
        "The six chronological windows are approximate temporal "
        "aggregates rather than exact RL checkpoints."
    )

    add(
        "The analysis is observational and does not establish "
        "a causal relationship between H/U and learning."
    )

    add(
        "The chronological ordering is guaranteed at the epoch "
        "level by the data-generation pipeline, but individual "
        "records inside an epoch are unordered because "
        "self-play workers are processed with imap_unordered."
    )

    add(
        "The known RL10 -> RL60 performance progression is used "
        "as contextual reference rather than being statistically "
        "paired with the six windows."
    )

    add(
        "The trend analysis operates on window-level aggregates, "
        "avoiding pseudo-replication from treating millions of "
        "individual observations as independent measurements."
    )

    add(
        "The observed temporal association does not by itself "
        "establish that H/U are sufficient statistics for agent "
        "progression or that they would generalize to other "
        "training runs."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Plots
# ============================================================

def make_plots(
    window_df,
    tail_df,
    reward_df,
    quadrant_df,
):

    print()
    print("=" * 72)
    print(
        "GENERATING PLOTS"
    )
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = np.arange(
        len(window_df)
    )

    labels = window_df[
        "window"
    ].tolist()

    # --------------------------------------------------------
    # Plot 1
    # H/U progression
    # --------------------------------------------------------

    print(
        "Plot 1/4 - H/U progression..."
    )

    plt.figure()

    plt.plot(
        x,
        window_df["H_mean"],
        marker="o",
        label="H mean",
    )

    plt.plot(
        x,
        window_df["U_mean"],
        marker="o",
        label="U mean",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel(
        "Chronological window"
    )

    plt.ylabel(
        "Mean signal value"
    )

    plt.title(
        "H/U evolution across chronological windows"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "hu_progression.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2
    # Tail progression
    # --------------------------------------------------------

    print(
        "Plot 2/4 - uncertainty tails..."
    )

    plt.figure()

    for signal in [
        "H",
        "U",
    ]:

        column = (
            f"{signal}_above_ref_q90"
        )

        if column in tail_df:

            plt.plot(
                x,
                tail_df[column],
                marker="o",
                label=(
                    f"{signal} > W1 P90"
                ),
            )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel(
        "Chronological window"
    )

    plt.ylabel(
        "Fraction of positions"
    )

    plt.title(
        "Persistence of high-uncertainty populations"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "uncertainty_tail_progression.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3
    # Reward-conditioned progression
    # --------------------------------------------------------

    print(
        "Plot 3/4 - reward-conditioned..."
    )

    plt.figure()

    plt.plot(
        x,
        reward_df[
            "high_H_draw_rate"
        ],
        marker="o",
        label="Draw rate | high H",
    )

    plt.plot(
        x,
        reward_df[
            "high_U_decisive_rate"
        ],
        marker="o",
        label="Decisive rate | high U",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel(
        "Chronological window"
    )

    plt.ylabel(
        "Conditional fraction"
    )

    plt.title(
        "Reward characteristics of high-H and high-U populations"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "reward_conditioned_progression.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 4
    # Diagnostic quadrants
    # --------------------------------------------------------

    print(
        "Plot 4/4 - diagnostic quadrants..."
    )

    pivot = (
        quadrant_df
        .pivot(
            index="window",
            columns="quadrant",
            values="fraction",
        )
        .reindex(
            WINDOW_LABELS
        )
    )

    plt.figure()

    for column in pivot.columns:

        plt.plot(
            x,
            pivot[column],
            marker="o",
            label=column,
        )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel(
        "Chronological window"
    )

    plt.ylabel(
        "Fraction of positions"
    )

    plt.title(
        "Evolution of H/U diagnostic populations"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "diagnostic_quadrants.png",
        dpi=200,
    )

    plt.close()

    print(
        "Plots saved."
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze H/U as statistical learning "
            "progression proxies."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to uncertainty statistics JSON."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        args.input
    )

    # --------------------------------------------------------
    # Chronological windows
    # --------------------------------------------------------

    df = assign_chronological_windows(
        df
    )

    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    window_df = (
        chronological_statistics(
            df
        )
    )

    # --------------------------------------------------------
    # Reference thresholds
    # --------------------------------------------------------

    thresholds = (
        build_reference_thresholds(
            df
        )
    )

    print()
    print(
        "Reference thresholds from W1:"
    )

    for signal in [
        "H",
        "U",
    ]:

        for q in REFERENCE_QUANTILES:

            print(
                f"  {signal} P{int(q * 100)}: "
                f"{thresholds[signal][q]:.6f}"
            )

    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    tail_df = (
        analyze_uncertainty_tails(
            df,
            thresholds,
        )
    )

    # --------------------------------------------------------
    # Experiment 3
    # --------------------------------------------------------

    reward_df = (
        reward_conditioned_analysis(
            df,
            thresholds,
        )
    )

    # --------------------------------------------------------
    # Experiment 4
    # --------------------------------------------------------

    trend_df = (
        trend_analysis(
            window_df
        )
    )

    # --------------------------------------------------------
    # Experiment 5
    # --------------------------------------------------------

    quadrant_df = (
        diagnostic_quadrants(
            df
        )
    )

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------

    window_df.to_csv(
        OUTPUT_DIR
        / "chronological_statistics.csv",
        index=False,
    )

    tail_df.to_csv(
        OUTPUT_DIR
        / "uncertainty_tails.csv",
        index=False,
    )

    reward_df.to_csv(
        OUTPUT_DIR
        / "reward_conditioned_progression.csv",
        index=False,
    )

    quadrant_df.to_csv(
        OUTPUT_DIR
        / "diagnostic_quadrants.csv",
        index=False,
    )

    trend_df.to_csv(
        OUTPUT_DIR
        / "chronological_trends.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    make_plots(
        window_df,
        tail_df,
        reward_df,
        quadrant_df,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        window_df,
        tail_df,
        reward_df,
        quadrant_df,
        trend_df,
    )

    report_path = (
        OUTPUT_DIR
        / "progression_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            report
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 72)

    print()

    print(
        f"Chronological windows: "
        f"{len(window_df)}"
    )

    print(
        f"Total records: "
        f"{len(df):,}"
    )

    print(
        f"Unique FENs: "
        f"{df['fen'].nunique():,}"
    )

    print()

    print(
        "H/U first -> last:"
    )

    first = window_df.iloc[0]
    last = window_df.iloc[-1]

    H_change = (
        last["H_mean"]
        - first["H_mean"]
    )

    U_change = (
        last["U_mean"]
        - first["U_mean"]
    )

    HU_change = (
        last["HU_mean"]
        - first["HU_mean"]
    )

    print(
        f"  H: "
        f"{first['H_mean']:.6f} "
        f"-> "
        f"{last['H_mean']:.6f} "
        f"({H_change:+.6f})"
    )

    print(
        f"  U: "
        f"{first['U_mean']:.6f} "
        f"-> "
        f"{last['U_mean']:.6f} "
        f"({U_change:+.6f})"
    )

    print(
        f"  HU: "
        f"{first['HU_mean']:.6f} "
        f"-> "
        f"{last['HU_mean']:.6f} "
        f"({HU_change:+.6f})"
    )

    print()

    # --------------------------------------------------------
    # Main trend coefficients
    # --------------------------------------------------------

    trend_lookup = (
        trend_df
        .set_index("metric")
    )

    for metric in [
        "H_mean",
        "U_mean",
        "HU_mean",
    ]:

        if metric not in trend_lookup.index:
            continue

        row = trend_lookup.loc[
            metric
        ]

        print(
            f"  {metric}: "
            f"Spearman rho="
            f"{row['spearman_rho']:+.4f}"
        )

    print()

    print(
        "Report:"
    )

    print(
        f"  {report_path}"
    )

    print()

    print(
        "Results:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print("=" * 72)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":

    main()