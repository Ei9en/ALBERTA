#!/usr/bin/env python3

"""
ALBERTA - Oracle Queue Composition Analysis
===========================================

Compare two completed Oracle acquisition queues:

    - Random Oracle
    - Active-learning acquisition

Scientific question
-------------------

Does active acquisition select a systematically different
population of positions from a random acquisition with the
same annotation budget?

The analysis distinguishes:

1. Acquisition-space properties
   - H
   - U
   - HU
   - score
   - I_norm

2. Oracle properties not directly used by the acquisition
   - reward / |reward|
   - oracle_confidence
   - oracle_situation

3. Position metadata
   - side to move

4. Selection overlap
   - shared FENs
   - overlap fraction
   - Jaccard similarity

Important
---------

The purpose is descriptive.

A difference between Random and AL acquisition does not by
itself establish higher downstream learning value.

H/U/HU-dependent conclusions must be regenerated from queues
constructed with the corrected league uncertainty estimator.

Outputs
-------

data/analysis/queue_composition/

    queue_composition_report.txt

    continuous_comparison.csv
    reward_comparison.csv
    oracle_confidence_comparison.csv
    oracle_situation_comparison.csv
    side_to_move_comparison.csv
    situation_signal_summary.csv
    confidence_signal_summary.csv

    signal_H_distribution.png
    signal_U_distribution.png
    signal_HU_distribution.png
    reward_distribution.png
    confidence_distribution.png
    situation_distribution.png
    side_distribution.png

Example
-------

python analysis/analyze_queue_composition.py \
    --random-queue checkpoints/queue/oracle_queue_1-10_random.jsonl \
    --al-queue checkpoints/queue/oracle_queue_1-10_AL.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Defaults
# ============================================================

DEFAULT_RANDOM_QUEUE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_random.jsonl"
)

DEFAULT_AL_QUEUE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_AL.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "queue_composition"
)


# ============================================================
# Variables
# ============================================================

# Inferential continuous comparisons.
#
# score and I_norm are intentionally excluded here because the
# AL queue is selected directly from the acquisition score.
# Testing whether the AL queue has larger scores would therefore
# be mostly tautological.
CONTINUOUS_VARIABLES = (
    "H",
    "U",
    "HU",
    "abs_reward",
)

CATEGORICAL_VARIABLES = (
    "oracle_confidence",
    "oracle_situation",
    "side_to_move",
)

CONFIDENCE_ORDER = (
    "low",
    "medium",
    "high",
)

SITUATION_ORDER = (
    "critical",
    "non_critical",
    "outcome_independent",
)

SIDE_ORDER = (
    "White",
    "Black",
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Compare the composition of a Random Oracle queue "
            "with an active-learning Oracle queue."
        )
    )

    parser.add_argument(
        "--random-queue",
        type=Path,
        default=DEFAULT_RANDOM_QUEUE,
    )

    parser.add_argument(
        "--al-queue",
        type=Path,
        default=DEFAULT_AL_QUEUE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--random-label",
        type=str,
        default="Random Oracle",
    )

    parser.add_argument(
        "--al-label",
        type=str,
        default="AL(I)",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def safe_float(
    value,
) -> float:

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return np.nan

    if not math.isfinite(
        result
    ):

        return np.nan

    return result


def finite_series(
    series: pd.Series,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return values[
        np.isfinite(
            values
        )
    ]


def safe_mean(
    series: pd.Series,
) -> float:

    values = finite_series(
        series
    )

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        values.mean()
    )


def safe_std(
    series: pd.Series,
) -> float:

    values = finite_series(
        series
    )

    if len(
        values
    ) <= 1:

        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


def safe_median(
    series: pd.Series,
) -> float:

    values = finite_series(
        series
    )

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        values.median()
    )


def safe_quantile(
    series: pd.Series,
    q: float,
) -> float:

    values = finite_series(
        series
    )

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        values.quantile(
            q
        )
    )


def get_side_from_fen(
    fen: str,
) -> str | None:

    if not isinstance(
        fen,
        str,
    ):

        return None

    parts = fen.split()

    if len(
        parts
    ) < 2:

        return None

    if parts[
        1
    ] == "w":

        return "White"

    if parts[
        1
    ] == "b":

        return "Black"

    return None


# ============================================================
# Queue loading
# ============================================================

def load_queue(
    path: Path,
    label: str,
) -> pd.DataFrame:
    """
    Load completed Oracle annotations.

    The lifecycle status is not used as the source of truth.

    Explicitly discarded records are ignored.

    A usable record must contain:

        fen
        oracle_move
        oracle_confidence
        oracle_situation
        reward

    Legacy confidence / criticality names are accepted.
    """

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

    skipped_incomplete = 0
    skipped_discarded = 0
    invalid = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number} in:\n{path}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):

                invalid += 1
                continue

            if record.get(
                "status"
            ) == "discarded":

                skipped_discarded += 1
                continue

            fen = record.get(
                "fen"
            )

            oracle_move = record.get(
                "oracle_move"
            )

            confidence = record.get(
                "oracle_confidence",
                record.get(
                    "confidence"
                ),
            )

            situation = record.get(
                "oracle_situation",
                record.get(
                    "criticality"
                ),
            )

            reward = safe_float(
                record.get(
                    "reward"
                )
            )

            if (
                not isinstance(
                    fen,
                    str,
                )
                or not fen
            ):

                invalid += 1
                continue

            if (
                oracle_move is None
                or confidence is None
                or situation is None
                or not np.isfinite(
                    reward
                )
            ):

                skipped_incomplete += 1
                continue

            if reward not in {
                -1.0,
                0.0,
                1.0,
            }:

                raise ValueError(
                    f"Invalid Oracle reward at line "
                    f"{line_number}: {reward}"
                )

            if confidence not in CONFIDENCE_ORDER:

                raise ValueError(
                    f"Invalid Oracle confidence at line "
                    f"{line_number}: {confidence}"
                )

            if situation not in SITUATION_ORDER:

                raise ValueError(
                    f"Invalid Oracle situation at line "
                    f"{line_number}: {situation}"
                )

            side = get_side_from_fen(
                fen
            )

            if side is None:

                raise ValueError(
                    f"Invalid side-to-move in FEN at "
                    f"line {line_number}:\n{fen}"
                )

            H = safe_float(
                record.get(
                    "H"
                )
            )

            U = safe_float(
                record.get(
                    "U"
                )
            )

            HU = safe_float(
                record.get(
                    "HU"
                )
            )

            score = safe_float(
                record.get(
                    "score",
                    record.get(
                        "I"
                    ),
                )
            )

            I_norm = safe_float(
                record.get(
                    "I_norm"
                )
            )

            rows.append(
                {
                    "query_id":
                        record.get(
                            "query_id"
                        ),

                    "fen":
                        fen,

                    "oracle_move":
                        oracle_move,

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
                        abs(
                            reward
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
        f"Usable records: {len(df):,}"
    )

    print(
        f"Incomplete:     {skipped_incomplete:,}"
    )

    print(
        f"Discarded:      {skipped_discarded:,}"
    )

    print(
        f"Invalid:        {invalid:,}"
    )

    if len(
        df
    ) == 0:

        raise RuntimeError(
            f"No usable completed annotations in:\n{path}"
        )

    return df


# ============================================================
# Queue structure / overlap
# ============================================================

def analyze_structure(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
) -> dict:

    random_fens = set(
        random_df[
            "fen"
        ]
    )

    al_fens = set(
        al_df[
            "fen"
        ]
    )

    intersection = (
        random_fens
        & al_fens
    )

    union = (
        random_fens
        | al_fens
    )

    random_overlap_fraction = (
        len(
            intersection
        )
        / len(
            random_fens
        )
        if random_fens
        else np.nan
    )

    al_overlap_fraction = (
        len(
            intersection
        )
        / len(
            al_fens
        )
        if al_fens
        else np.nan
    )

    jaccard = (
        len(
            intersection
        )
        / len(
            union
        )
        if union
        else np.nan
    )

    return {
        "random_records":
            len(
                random_df
            ),

        "al_records":
            len(
                al_df
            ),

        "random_unique_fens":
            len(
                random_fens
            ),

        "al_unique_fens":
            len(
                al_fens
            ),

        "random_duplicates":
            (
                len(
                    random_df
                )
                - len(
                    random_fens
                )
            ),

        "al_duplicates":
            (
                len(
                    al_df
                )
                - len(
                    al_fens
                )
            ),

        "shared_fens":
            len(
                intersection
            ),

        "union_fens":
            len(
                union
            ),

        "random_overlap_fraction":
            random_overlap_fraction,

        "al_overlap_fraction":
            al_overlap_fraction,

        "jaccard":
            jaccard,
    }


# ============================================================
# Continuous comparison
# ============================================================

def rank_biserial_from_u(
    u_statistic: float,
    n_random: int,
    n_al: int,
) -> float:

    if (
        n_random <= 0
        or n_al <= 0
    ):

        return np.nan

    # scipy's U corresponds to the first sample.
    #
    # Positive effect here is defined as AL > Random.
    return float(
        1.0
        - (
            2.0
            * u_statistic
            / (
                n_random
                * n_al
            )
        )
    )


def compare_continuous(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for variable in CONTINUOUS_VARIABLES:

        random_values = finite_series(
            random_df[
                variable
            ]
        )

        al_values = finite_series(
            al_df[
                variable
            ]
        )

        if (
            len(
                random_values
            ) == 0
            or len(
                al_values
            ) == 0
        ):

            u_statistic = np.nan
            p_value = np.nan
            rank_biserial = np.nan

        else:

            test = mannwhitneyu(
                random_values,
                al_values,
                alternative="two-sided",
            )

            u_statistic = float(
                test.statistic
            )

            p_value = float(
                test.pvalue
            )

            rank_biserial = (
                rank_biserial_from_u(
                    u_statistic,
                    len(
                        random_values
                    ),
                    len(
                        al_values
                    ),
                )
            )

        random_mean = safe_mean(
            random_values
        )

        al_mean = safe_mean(
            al_values
        )

        rows.append(
            {
                "variable":
                    variable,

                "random_n":
                    len(
                        random_values
                    ),

                "al_n":
                    len(
                        al_values
                    ),

                "random_mean":
                    random_mean,

                "al_mean":
                    al_mean,

                "AL_minus_Random":
                    (
                        al_mean
                        - random_mean
                    ),

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

                "mann_whitney_U":
                    u_statistic,

                "p_value":
                    p_value,

                # Positive:
                #     AL tends to be larger than Random.
                #
                # Negative:
                #     Random tends to be larger than AL.
                "rank_biserial_AL_minus_Random":
                    rank_biserial,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Acquisition-score descriptive sanity check
# ============================================================

def acquisition_score_summary(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    score / I_norm are reported descriptively only.

    Because the active queue is selected using the acquisition
    score itself, inferential testing here would be largely
    tautological.
    """

    rows = []

    for variable in (
        "score",
        "I_norm",
    ):

        for label, df in (
            (
                "Random Oracle",
                random_df,
            ),
            (
                "AL(I)",
                al_df,
            ),
        ):

            values = finite_series(
                df[
                    variable
                ]
            )

            rows.append(
                {
                    "queue":
                        label,

                    "variable":
                        variable,

                    "n":
                        len(
                            values
                        ),

                    "mean":
                        safe_mean(
                            values
                        ),

                    "median":
                        safe_median(
                            values
                        ),

                    "q10":
                        safe_quantile(
                            values,
                            0.10,
                        ),

                    "q90":
                        safe_quantile(
                            values,
                            0.90,
                        ),

                    "min":
                        safe_quantile(
                            values,
                            0.00,
                        ),

                    "max":
                        safe_quantile(
                            values,
                            1.00,
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Reward analysis
# ============================================================

def reward_analysis(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for label, df in (
        (
            "Random Oracle",
            random_df,
        ),
        (
            "AL(I)",
            al_df,
        ),
    ):

        rewards = finite_series(
            df[
                "reward"
            ]
        )

        n = len(
            rewards
        )

        losses = int(
            (
                rewards == -1.0
            ).sum()
        )

        draws = int(
            (
                rewards == 0.0
            ).sum()
        )

        wins = int(
            (
                rewards == 1.0
            ).sum()
        )

        rows.append(
            {
                "queue":
                    label,

                "n":
                    n,

                "losses":
                    losses,

                "loss_pct":
                    (
                        losses
                        / n
                        * 100.0
                        if n
                        else np.nan
                    ),

                "draws":
                    draws,

                "draw_pct":
                    (
                        draws
                        / n
                        * 100.0
                        if n
                        else np.nan
                    ),

                "wins":
                    wins,

                "win_pct":
                    (
                        wins
                        / n
                        * 100.0
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

    return pd.DataFrame(
        rows
    )


# ============================================================
# Categorical comparison
# ============================================================

def categorical_comparison(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:

    random_values = (
        random_df[
            variable
        ]
        .fillna(
            "MISSING"
        )
    )

    al_values = (
        al_df[
            variable
        ]
        .fillna(
            "MISSING"
        )
    )

    categories = sorted(
        set(
            random_values.unique()
        )
        |
        set(
            al_values.unique()
        ),
        key=str,
    )

    random_counts = (
        random_values
        .value_counts()
    )

    al_counts = (
        al_values
        .value_counts()
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
            / len(
                random_df
            )
        )

        al_fraction = (
            al_n
            / len(
                al_df
            )
        )

        rows.append(
            {
                "category":
                    category,

                "random_n":
                    random_n,

                "random_pct":
                    (
                        100.0
                        * random_fraction
                    ),

                "al_n":
                    al_n,

                "al_pct":
                    (
                        100.0
                        * al_fraction
                    ),

                "AL_minus_Random_pp":
                    (
                        100.0
                        * (
                            al_fraction
                            - random_fraction
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def categorical_global_test(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
    variable: str,
) -> dict:
    """
    Pearson chi-square + Cramer's V.

    This is descriptive/inferential context only.
    """

    random_values = (
        random_df[
            variable
        ]
        .fillna(
            "MISSING"
        )
    )

    al_values = (
        al_df[
            variable
        ]
        .fillna(
            "MISSING"
        )
    )

    categories = sorted(
        set(
            random_values.unique()
        )
        |
        set(
            al_values.unique()
        ),
        key=str,
    )

    table = np.asarray(
        [
            [
                int(
                    (
                        random_values
                        == category
                    ).sum()
                )
                for category
                in categories
            ],
            [
                int(
                    (
                        al_values
                        == category
                    ).sum()
                )
                for category
                in categories
            ],
        ],
        dtype=np.int64,
    )

    # Remove categories absent from both rows.
    keep = (
        table.sum(
            axis=0
        )
        > 0
    )

    table = table[
        :,
        keep,
    ]

    if table.shape[
        1
    ] < 2:

        return {
            "variable":
                variable,

            "chi2":
                np.nan,

            "dof":
                np.nan,

            "p_value":
                np.nan,

            "cramers_v":
                np.nan,
        }

    chi2, p_value, dof, _ = (
        chi2_contingency(
            table
        )
    )

    n = int(
        table.sum()
    )

    min_dimension = min(
        table.shape[
            0
        ] - 1,
        table.shape[
            1
        ] - 1,
    )

    if (
        n <= 0
        or min_dimension <= 0
    ):

        cramers_v = np.nan

    else:

        cramers_v = math.sqrt(
            chi2
            / (
                n
                * min_dimension
            )
        )

    return {
        "variable":
            variable,

        "chi2":
            float(
                chi2
            ),

        "dof":
            int(
                dof
            ),

        "p_value":
            float(
                p_value
            ),

        "cramers_v":
            float(
                cramers_v
            ),
    }


# ============================================================
# Signal summaries by Oracle annotation
# ============================================================

def summarize_by_situation(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for label, df in (
        (
            "Random Oracle",
            random_df,
        ),
        (
            "AL(I)",
            al_df,
        ),
    ):

        for situation in SITUATION_ORDER:

            subset = df[
                df[
                    "oracle_situation"
                ]
                == situation
            ]

            if len(
                subset
            ) == 0:

                continue

            rows.append(
                {
                    "queue":
                        label,

                    "situation":
                        situation,

                    "n":
                        len(
                            subset
                        ),

                    "mean_H":
                        safe_mean(
                            subset[
                                "H"
                            ]
                        ),

                    "mean_U":
                        safe_mean(
                            subset[
                                "U"
                            ]
                        ),

                    "mean_HU":
                        safe_mean(
                            subset[
                                "HU"
                            ]
                        ),

                    "mean_abs_reward":
                        safe_mean(
                            subset[
                                "abs_reward"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def summarize_by_confidence(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for label, df in (
        (
            "Random Oracle",
            random_df,
        ),
        (
            "AL(I)",
            al_df,
        ),
    ):

        for confidence in CONFIDENCE_ORDER:

            subset = df[
                df[
                    "oracle_confidence"
                ]
                == confidence
            ]

            if len(
                subset
            ) == 0:

                continue

            rows.append(
                {
                    "queue":
                        label,

                    "confidence":
                        confidence,

                    "n":
                        len(
                            subset
                        ),

                    "mean_H":
                        safe_mean(
                            subset[
                                "H"
                            ]
                        ),

                    "mean_U":
                        safe_mean(
                            subset[
                                "U"
                            ]
                        ),

                    "mean_HU":
                        safe_mean(
                            subset[
                                "HU"
                            ]
                        ),

                    "mean_abs_reward":
                        safe_mean(
                            subset[
                                "abs_reward"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plots
# ============================================================

def plot_signal_distribution(
    random_df: pd.DataFrame,
    al_df: pd.DataFrame,
    variable: str,
    output_dir: Path,
    random_label: str,
    al_label: str,
) -> None:

    random_values = finite_series(
        random_df[
            variable
        ]
    )

    al_values = finite_series(
        al_df[
            variable
        ]
    )

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    if len(
        random_values
    ):

        plt.hist(
            random_values,
            bins=50,
            density=True,
            alpha=0.5,
            label=random_label,
        )

    if len(
        al_values
    ):

        plt.hist(
            al_values,
            bins=50,
            density=True,
            alpha=0.5,
            label=al_label,
        )

    plt.xlabel(
        variable
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        f"{variable} distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir
        / f"signal_{variable}_distribution.png",
        dpi=200,
    )

    plt.close()


def plot_categorical_distribution(
    table: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    random_label: str,
    al_label: str,
) -> None:

    x = np.arange(
        len(
            table
        )
    )

    width = 0.35

    plt.figure(
        figsize=(
            9,
            6,
        )
    )

    plt.bar(
        x
        - width
        / 2.0,
        table[
            "random_pct"
        ],
        width,
        label=random_label,
    )

    plt.bar(
        x
        + width
        / 2.0,
        table[
            "al_pct"
        ],
        width,
        label=al_label,
    )

    plt.xticks(
        x,
        table[
            "category"
        ],
        rotation=15,
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def plot_reward_distribution(
    reward_df: pd.DataFrame,
    output_path: Path,
    random_label: str,
    al_label: str,
) -> None:

    random_row = reward_df.iloc[
        0
    ]

    al_row = reward_df.iloc[
        1
    ]

    labels = (
        "Loss",
        "Draw",
        "Win",
    )

    random_values = (
        random_row[
            "loss_pct"
        ],
        random_row[
            "draw_pct"
        ],
        random_row[
            "win_pct"
        ],
    )

    al_values = (
        al_row[
            "loss_pct"
        ],
        al_row[
            "draw_pct"
        ],
        al_row[
            "win_pct"
        ],
    )

    x = np.arange(
        len(
            labels
        )
    )

    width = 0.35

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.bar(
        x
        - width
        / 2.0,
        random_values,
        width,
        label=random_label,
    )

    plt.bar(
        x
        + width
        / 2.0,
        al_values,
        width,
        label=al_label,
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
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Report
# ============================================================

def build_report(
    *,
    random_queue: Path,
    al_queue: Path,
    random_label: str,
    al_label: str,
    structure: dict,
    continuous_df: pd.DataFrame,
    acquisition_df: pd.DataFrame,
    categorical_tables: dict[str, pd.DataFrame],
    categorical_tests: pd.DataFrame,
    reward_df: pd.DataFrame,
    situation_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
) -> str:

    lines = []

    def add(
        text="",
    ) -> None:

        lines.append(
            str(
                text
            )
        )

    add(
        "=" * 80
    )

    add(
        "ALBERTA - ORACLE QUEUE COMPOSITION ANALYSIS"
    )

    add(
        "=" * 80
    )

    add()

    add(
        f"Random queue: {random_queue}"
    )

    add(
        f"AL queue:     {al_queue}"
    )

    add()

    # ========================================================
    # Structure
    # ========================================================

    add(
        "QUEUE STRUCTURE"
    )

    add(
        "-" * 80
    )

    add(
        f"{random_label} records: "
        f"{structure['random_records']:,}"
    )

    add(
        f"{al_label} records: "
        f"{structure['al_records']:,}"
    )

    add(
        f"{random_label} unique FENs: "
        f"{structure['random_unique_fens']:,}"
    )

    add(
        f"{al_label} unique FENs: "
        f"{structure['al_unique_fens']:,}"
    )

    add(
        f"{random_label} duplicates: "
        f"{structure['random_duplicates']:,}"
    )

    add(
        f"{al_label} duplicates: "
        f"{structure['al_duplicates']:,}"
    )

    add(
        f"Shared FENs: "
        f"{structure['shared_fens']:,}"
    )

    add(
        f"Union FENs: "
        f"{structure['union_fens']:,}"
    )

    add(
        f"Overlap / Random: "
        f"{structure['random_overlap_fraction']:.2%}"
    )

    add(
        f"Overlap / AL: "
        f"{structure['al_overlap_fraction']:.2%}"
    )

    add(
        f"Jaccard similarity: "
        f"{structure['jaccard']:.2%}"
    )

    add()

    # ========================================================
    # Continuous
    # ========================================================

    add(
        "CONTINUOUS COMPARISON"
    )

    add(
        "-" * 80
    )

    add(
        continuous_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    add()

    add(
        "Mann-Whitney tests are used for H, U, HU and |reward|."
    )

    add(
        "The rank-biserial effect is oriented so that positive "
        "values indicate larger values in the AL queue."
    )

    add()

    # ========================================================
    # Acquisition score
    # ========================================================

    add(
        "ACQUISITION SCORE - DESCRIPTIVE ONLY"
    )

    add(
        "-" * 80
    )

    add(
        acquisition_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    add()

    add(
        "score and I_norm are not subjected to significance "
        "tests because the active queue is selected directly "
        "from the acquisition score."
    )

    add()

    # ========================================================
    # Reward
    # ========================================================

    add(
        "ORACLE REWARD"
    )

    add(
        "-" * 80
    )

    add(
        reward_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    add()

    add(
        "Reward is the Oracle evaluation from the side-to-move "
        "perspective."
    )

    add()

    # ========================================================
    # Categorical
    # ========================================================

    for variable in CATEGORICAL_VARIABLES:

        add(
            f"CATEGORICAL VARIABLE: {variable}"
        )

        add(
            "-" * 80
        )

        add(
            categorical_tables[
                variable
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}",
            )
        )

        add()

    add(
        "CATEGORICAL GLOBAL TESTS"
    )

    add(
        "-" * 80
    )

    add(
        categorical_tests.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    add()

    add(
        "Cramer's V describes the magnitude of the association "
        "between queue identity and the categorical variable."
    )

    add()

    # ========================================================
    # Cross summaries
    # ========================================================

    add(
        "SIGNALS BY ORACLE SITUATION"
    )

    add(
        "-" * 80
    )

    add(
        situation_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    add()

    add(
        "SIGNALS BY ORACLE CONFIDENCE"
    )

    add(
        "-" * 80
    )

    add(
        confidence_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    add()

    # ========================================================
    # Interpretation
    # ========================================================

    add(
        "INTERPRETATION"
    )

    add(
        "-" * 80
    )

    add(
        "The central question is whether active acquisition "
        "changes the composition of the annotation budget "
        "relative to Random Oracle."
    )

    add()

    add(
        "Differences in H, U and HU describe which uncertainty "
        "regimes are preferentially selected."
    )

    add(
        "Differences in reward magnitude, Oracle confidence and "
        "Oracle situation are more informative because those "
        "properties are not directly part of the acquisition "
        "score."
    )

    add()

    add(
        "For example, enrichment of critical situations or "
        "high-confidence annotations would demonstrate that the "
        "acquisition score indirectly selects a different type "
        "of human supervision."
    )

    add()

    add(
        "However, selection enrichment is not equivalent to "
        "learning utility."
    )

    add(
        "The downstream value of the selected annotations must "
        "be established independently through RL+Oracle "
        "training and tournament evaluation."
    )

    add()

    add(
        "H/U/HU-dependent conclusions are valid only for queues "
        "constructed from the corrected league uncertainty "
        "estimator."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load
    # ========================================================

    random_df = load_queue(
        args.random_queue,
        args.random_label,
    )

    al_df = load_queue(
        args.al_queue,
        args.al_label,
    )

    # ========================================================
    # Structure
    # ========================================================

    structure = analyze_structure(
        random_df,
        al_df,
    )

    print()
    print("=" * 80)
    print("QUEUE STRUCTURE")
    print("=" * 80)

    print(
        f"Shared FENs: "
        f"{structure['shared_fens']:,}"
    )

    print(
        f"Overlap / Random: "
        f"{structure['random_overlap_fraction']:.2%}"
    )

    print(
        f"Overlap / AL: "
        f"{structure['al_overlap_fraction']:.2%}"
    )

    print(
        f"Jaccard: "
        f"{structure['jaccard']:.2%}"
    )

    # ========================================================
    # Continuous
    # ========================================================

    continuous_df = compare_continuous(
        random_df,
        al_df,
    )

    print()
    print("=" * 80)
    print("CONTINUOUS COMPARISON")
    print("=" * 80)

    print(
        continuous_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    # ========================================================
    # Acquisition score sanity check
    # ========================================================

    acquisition_df = (
        acquisition_score_summary(
            random_df,
            al_df,
        )
    )

    # ========================================================
    # Reward
    # ========================================================

    reward_df = reward_analysis(
        random_df,
        al_df,
    )

    print()
    print("=" * 80)
    print("REWARD")
    print("=" * 80)

    print(
        reward_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ========================================================
    # Categorical
    # ========================================================

    categorical_tables = {}

    categorical_test_rows = []

    for variable in CATEGORICAL_VARIABLES:

        table = categorical_comparison(
            random_df,
            al_df,
            variable,
        )

        categorical_tables[
            variable
        ] = table

        categorical_test_rows.append(
            categorical_global_test(
                random_df,
                al_df,
                variable,
            )
        )

        print()
        print("=" * 80)
        print(
            variable.upper()
        )
        print("=" * 80)

        print(
            table.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}",
            )
        )

    categorical_tests = pd.DataFrame(
        categorical_test_rows
    )

    # ========================================================
    # Cross summaries
    # ========================================================

    situation_df = summarize_by_situation(
        random_df,
        al_df,
    )

    confidence_df = summarize_by_confidence(
        random_df,
        al_df,
    )

    # ========================================================
    # Save tabular results
    # ========================================================

    continuous_df.to_csv(
        args.output_dir
        / "continuous_comparison.csv",
        index=False,
    )

    acquisition_df.to_csv(
        args.output_dir
        / "acquisition_score_summary.csv",
        index=False,
    )

    reward_df.to_csv(
        args.output_dir
        / "reward_comparison.csv",
        index=False,
    )

    categorical_tests.to_csv(
        args.output_dir
        / "categorical_global_tests.csv",
        index=False,
    )

    categorical_tables[
        "oracle_confidence"
    ].to_csv(
        args.output_dir
        / "oracle_confidence_comparison.csv",
        index=False,
    )

    categorical_tables[
        "oracle_situation"
    ].to_csv(
        args.output_dir
        / "oracle_situation_comparison.csv",
        index=False,
    )

    categorical_tables[
        "side_to_move"
    ].to_csv(
        args.output_dir
        / "side_to_move_comparison.csv",
        index=False,
    )

    situation_df.to_csv(
        args.output_dir
        / "situation_signal_summary.csv",
        index=False,
    )

    confidence_df.to_csv(
        args.output_dir
        / "confidence_signal_summary.csv",
        index=False,
    )

    # ========================================================
    # Plots
    # ========================================================

    print()
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    for variable in (
        "H",
        "U",
        "HU",
    ):

        plot_signal_distribution(
            random_df,
            al_df,
            variable,
            args.output_dir,
            args.random_label,
            args.al_label,
        )

    plot_reward_distribution(
        reward_df,
        args.output_dir
        / "reward_distribution.png",
        args.random_label,
        args.al_label,
    )

    plot_categorical_distribution(
        categorical_tables[
            "oracle_confidence"
        ],
        title="Oracle confidence distribution",
        ylabel="Percentage",
        output_path=(
            args.output_dir
            / "confidence_distribution.png"
        ),
        random_label=args.random_label,
        al_label=args.al_label,
    )

    plot_categorical_distribution(
        categorical_tables[
            "oracle_situation"
        ],
        title="Oracle situation distribution",
        ylabel="Percentage",
        output_path=(
            args.output_dir
            / "situation_distribution.png"
        ),
        random_label=args.random_label,
        al_label=args.al_label,
    )

    plot_categorical_distribution(
        categorical_tables[
            "side_to_move"
        ],
        title="Side-to-move distribution",
        ylabel="Percentage",
        output_path=(
            args.output_dir
            / "side_distribution.png"
        ),
        random_label=args.random_label,
        al_label=args.al_label,
    )

    # ========================================================
    # Report
    # ========================================================

    report = build_report(
        random_queue=args.random_queue,
        al_queue=args.al_queue,
        random_label=args.random_label,
        al_label=args.al_label,
        structure=structure,
        continuous_df=continuous_df,
        acquisition_df=acquisition_df,
        categorical_tables=categorical_tables,
        categorical_tests=categorical_tests,
        reward_df=reward_df,
        situation_df=situation_df,
        confidence_df=confidence_df,
    )

    report_path = (
        args.output_dir
        / "queue_composition_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Results: {args.output_dir}"
    )

    print(
        f"Report:  {report_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()