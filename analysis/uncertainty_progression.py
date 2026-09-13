#!/usr/bin/env python3

"""
ALBERTA - Uncertainty Progression Analysis
==========================================

Study how ALBERTA's uncertainty-related signals evolve during an
already completed RL training trajectory.

Scientific question
-------------------

Do the distributions of:

    H  = policy entropy
    U  = historical-league value disagreement
    HU = H * U

change systematically as training progresses?

This analysis is descriptive. It does NOT establish that H or U
cause learning progress, nor that they are optimal acquisition
functions.

Chronology
----------

The historical uncertainty-statistics file does not necessarily
contain explicit epoch identifiers.

The data-generation pipeline nevertheless appends observations in
chronological epoch order. The records are therefore divided into
equal chronological windows:

    W1, W2, ..., WK

These windows are coarse temporal aggregates, NOT exact checkpoint
boundaries.

Reward convention
-----------------

Reward is always interpreted from the side-to-move perspective:

    +1 : eventual win for the player to move
     0 : draw
    -1 : eventual loss for the player to move

If an explicit reward is present, it is used directly.
Otherwise it is reconstructed from:

    result + FEN side-to-move

Signal validity
---------------

H, U and HU are handled independently.

A missing U does NOT invalidate H.
A missing H does NOT invalidate U.

HU is reconstructed as H * U only when both H and U are finite and
no valid HU value is stored.

Experiments
-----------

1. Chronological signal statistics
2. High-signal tails relative to W1
3. Reward-conditioned high-H / high-U populations
4. Chronological trends
5. H/U diagnostic quadrants
6. Repeated-position prevalence

Outputs
-------

data/analysis/uncertainty_progression/

    chronological_statistics.csv
    uncertainty_tails.csv
    reward_conditioned_progression.csv
    diagnostic_quadrants.csv
    chronological_trends.csv

    H_progression.png
    U_progression.png
    uncertainty_tail_progression.png
    reward_conditioned_progression.png
    diagnostic_quadrants.png

    uncertainty_progression_report.txt

Example
-------

python analysis/analyze_uncertainty_progression.py \
    --input data/selfplay_jsons/uncertainty_stats_1-60.json \
    --windows 6
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Defaults
# ============================================================

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-60.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "uncertainty_progression"
)

DEFAULT_N_WINDOWS = 6

REFERENCE_QUANTILES = (
    0.90,
    0.95,
    0.99,
)

QUADRANT_REFERENCE_QUANTILE = 0.50


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the chronological evolution of ALBERTA "
            "policy entropy and league-value disagreement."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Uncertainty-statistics JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=DEFAULT_N_WINDOWS,
        help=(
            "Number of equal chronological windows used for "
            "descriptive temporal aggregation."
        ),
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def safe_float(
    value,
) -> float:

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return np.nan

    if not math.isfinite(
        value
    ):

        return np.nan

    return value


def finite_values(
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

    values = finite_values(
        series
    )

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        values.mean()
    )


def safe_median(
    series: pd.Series,
) -> float:

    values = finite_values(
        series
    )

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        values.median()
    )


def safe_std(
    series: pd.Series,
) -> float:

    values = finite_values(
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


def safe_quantile(
    series: pd.Series,
    q: float,
) -> float:

    values = finite_values(
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


def safe_fraction(
    condition,
) -> float:

    values = np.asarray(
        condition,
        dtype=bool,
    )

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def fraction_above(
    series: pd.Series,
    threshold: float,
) -> float:
    """
    Fraction above threshold among finite observations only.
    """

    values = finite_values(
        series
    )

    if (
        len(
            values
        ) == 0
        or not np.isfinite(
            threshold
        )
    ):

        return np.nan

    return float(
        (
            values
            >= threshold
        ).mean()
    )


def chronological_spearman(
    values,
) -> float:
    """
    Spearman correlation between original chronological window
    index and a window-level metric.

    Missing windows are removed without renumbering chronology.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    chronology = np.arange(
        len(
            values
        ),
        dtype=np.float64,
    )

    mask = np.isfinite(
        values
    )

    if mask.sum() < 3:

        return np.nan

    return float(
        pd.Series(
            chronology[
                mask
            ]
        ).corr(
            pd.Series(
                values[
                    mask
                ]
            ),
            method="spearman",
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

        return "w"

    if parts[
        1
    ] == "b":

        return "b"

    return None


# ============================================================
# Reward convention
# ============================================================

def reward_from_result(
    result,
    fen: str,
) -> float:
    """
    Convert a chess result to reward from the side-to-move
    perspective of the supplied FEN.
    """

    side = get_side_from_fen(
        fen
    )

    if side is None:

        return np.nan

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
            1.0
            if side == "b"
            else -1.0
        )

    return np.nan


def extract_reward(
    record: dict,
    fen: str,
) -> float:

    explicit_reward = safe_float(
        record.get(
            "reward"
        )
    )

    if np.isfinite(
        explicit_reward
    ):

        if explicit_reward not in {
            -1.0,
            0.0,
            1.0,
        }:

            return np.nan

        return explicit_reward

    return reward_from_result(
        record.get(
            "result"
        ),
        fen,
    )


# ============================================================
# Load
# ============================================================

def load_data(
    input_path: Path,
) -> pd.DataFrame:

    print()
    print("=" * 80)
    print("ALBERTA - UNCERTAINTY PROGRESSION ANALYSIS")
    print("=" * 80)

    print(
        f"Input: {input_path}"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        raw = json.load(
            file
        )

    # --------------------------------------------------------
    # Accept historical wrapped containers.
    # --------------------------------------------------------

    if isinstance(
        raw,
        dict,
    ):

        for key in (
            "records",
            "data",
            "statistics",
            "observations",
            "stats",
            "uncertainty_stats",
        ):

            value = raw.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                raw = value
                break

    if not isinstance(
        raw,
        list,
    ):

        raise ValueError(
            "Expected a JSON list of uncertainty records."
        )

    rows = []

    invalid_records = 0
    invalid_fens = 0

    for raw_index, record in enumerate(
        raw
    ):

        if not isinstance(
            record,
            dict,
        ):

            invalid_records += 1
            continue

        fen = record.get(
            "fen"
        )

        if (
            not isinstance(
                fen,
                str,
            )
            or not fen
            or get_side_from_fen(
                fen
            )
            is None
        ):

            invalid_fens += 1
            continue

        # ----------------------------------------------------
        # Signals are independent.
        # --------------------------------------------------------

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

        if (
            not np.isfinite(
                HU
            )
            and np.isfinite(
                H
            )
            and np.isfinite(
                U
            )
        ):

            HU = (
                H
                * U
            )

        reward = extract_reward(
            record,
            fen,
        )

        rows.append(
            {
                "raw_order":
                    raw_index,

                "fen":
                    fen,

                "side_to_move":
                    get_side_from_fen(
                        fen
                    ),

                "H":
                    H,

                "U":
                    U,

                "HU":
                    HU,

                "reward":
                    reward,
            }
        )

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ) == 0:

        raise RuntimeError(
            "No valid FEN records found."
        )

    print(
        f"Raw records:        {len(raw):,}"
    )

    print(
        f"Usable FEN records: {len(df):,}"
    )

    print(
        f"Invalid records:    {invalid_records:,}"
    )

    print(
        f"Invalid FENs:       {invalid_fens:,}"
    )

    print()

    for signal in (
        "H",
        "U",
        "HU",
        "reward",
    ):

        n_valid = int(
            np.isfinite(
                pd.to_numeric(
                    df[
                        signal
                    ],
                    errors="coerce",
                )
            ).sum()
        )

        print(
            f"Finite {signal:6s}: {n_valid:,}"
        )

    return df


# ============================================================
# Chronological windows
# ============================================================

def make_window_labels(
    n_windows: int,
) -> list[str]:

    return [
        f"W{i + 1}"
        for i in range(
            n_windows
        )
    ]


def assign_chronological_windows(
    df: pd.DataFrame,
    n_windows: int,
) -> tuple[pd.DataFrame, list[str]]:

    if n_windows < 2:

        raise ValueError(
            "--windows must be >= 2."
        )

    if n_windows > len(
        df
    ):

        raise ValueError(
            "Number of windows exceeds number of records."
        )

    labels = make_window_labels(
        n_windows
    )

    result = df.copy()

    index_groups = np.array_split(
        np.arange(
            len(
                result
            )
        ),
        n_windows,
    )

    result[
        "window"
    ] = ""

    print()
    print("=" * 80)
    print("CHRONOLOGICAL WINDOWS")
    print("=" * 80)

    for (
        label,
        indices,
    ) in zip(
        labels,
        index_groups,
    ):

        result.loc[
            indices,
            "window",
        ] = label

        print(
            f"{label}: "
            f"{indices[0]:,} -> {indices[-1]:,} "
            f"({len(indices):,} records)"
        )

    return (
        result,
        labels,
    )


# ============================================================
# Repetition statistics
# ============================================================

def repetition_statistics(
    subset: pd.DataFrame,
) -> dict:

    counts = subset[
        "fen"
    ].value_counts()

    repeated_counts = counts[
        counts > 1
    ]

    repeated_fens = int(
        len(
            repeated_counts
        )
    )

    repeated_observations = int(
        repeated_counts.sum()
    )

    return {
        "unique_fens":
            int(
                len(
                    counts
                )
            ),

        "repeated_fens":
            repeated_fens,

        "repeated_observations":
            repeated_observations,

        "repeated_observation_fraction":
            (
                repeated_observations
                / len(
                    subset
                )
                if len(
                    subset
                )
                else np.nan
            ),
    }


# ============================================================
# Experiment 1
# Chronological statistics
# ============================================================

def chronological_statistics(
    df: pd.DataFrame,
    window_labels: list[str],
) -> pd.DataFrame:

    rows = []

    for window in window_labels:

        subset = df[
            df[
                "window"
            ]
            == window
        ]

        repetition = repetition_statistics(
            subset
        )

        row = {
            "window":
                window,

            "records":
                len(
                    subset
                ),

            **repetition,
        }

        # ----------------------------------------------------
        # Signals
        # --------------------------------------------------------

        for signal in (
            "H",
            "U",
            "HU",
        ):

            values = finite_values(
                subset[
                    signal
                ]
            )

            row[
                f"{signal}_n"
            ] = len(
                values
            )

            row[
                f"{signal}_mean"
            ] = safe_mean(
                values
            )

            row[
                f"{signal}_median"
            ] = safe_median(
                values
            )

            row[
                f"{signal}_std"
            ] = safe_std(
                values
            )

            for quantile in (
                0.75,
                0.90,
                0.95,
                0.99,
            ):

                row[
                    f"{signal}_q{int(100 * quantile)}"
                ] = safe_quantile(
                    values,
                    quantile,
                )

        # ----------------------------------------------------
        # Reward
        # --------------------------------------------------------

        rewards = finite_values(
            subset[
                "reward"
            ]
        )

        row[
            "reward_n"
        ] = len(
            rewards
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
            float(
                (
                    rewards
                    == 0.0
                ).mean()
            )
            if len(
                rewards
            )
            else np.nan
        )

        row[
            "decisive_rate"
        ] = (
            float(
                (
                    rewards.abs()
                    == 1.0
                ).mean()
            )
            if len(
                rewards
            )
            else np.nan
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Reference thresholds
# ============================================================

def build_reference_thresholds(
    df: pd.DataFrame,
    first_window: str,
) -> dict[str, dict[float, float]]:

    reference = df[
        df[
            "window"
        ]
        == first_window
    ]

    thresholds = {}

    for signal in (
        "H",
        "U",
    ):

        thresholds[
            signal
        ] = {}

        for quantile in REFERENCE_QUANTILES:

            thresholds[
                signal
            ][
                quantile
            ] = safe_quantile(
                reference[
                    signal
                ],
                quantile,
            )

    return thresholds


# ============================================================
# Experiment 2
# High-signal tails
# ============================================================

def analyze_uncertainty_tails(
    df: pd.DataFrame,
    thresholds: dict,
    window_labels: list[str],
) -> pd.DataFrame:

    rows = []

    for window in window_labels:

        subset = df[
            df[
                "window"
            ]
            == window
        ]

        row = {
            "window":
                window,
        }

        for signal in (
            "H",
            "U",
        ):

            values = finite_values(
                subset[
                    signal
                ]
            )

            row[
                f"{signal}_n"
            ] = len(
                values
            )

            for quantile in REFERENCE_QUANTILES:

                threshold = thresholds[
                    signal
                ][
                    quantile
                ]

                row[
                    f"{signal}_above_ref_q"
                    f"{int(100 * quantile)}"
                ] = fraction_above(
                    values,
                    threshold,
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Experiment 3
# Reward-conditioned signal populations
# ============================================================

def reward_conditioned_analysis(
    df: pd.DataFrame,
    thresholds: dict,
    window_labels: list[str],
) -> pd.DataFrame:

    H_threshold = thresholds[
        "H"
    ][
        0.90
    ]

    U_threshold = thresholds[
        "U"
    ][
        0.90
    ]

    rows = []

    for window in window_labels:

        subset = df[
            df[
                "window"
            ]
            == window
        ]

        valid_H = subset[
            np.isfinite(
                pd.to_numeric(
                    subset[
                        "H"
                    ],
                    errors="coerce",
                )
            )
        ]

        valid_U = subset[
            np.isfinite(
                pd.to_numeric(
                    subset[
                        "U"
                    ],
                    errors="coerce",
                )
            )
        ]

        high_H = valid_H[
            valid_H[
                "H"
            ]
            >= H_threshold
        ]

        high_U = valid_U[
            valid_U[
                "U"
            ]
            >= U_threshold
        ]

        high_H_rewards = finite_values(
            high_H[
                "reward"
            ]
        )

        high_U_rewards = finite_values(
            high_U[
                "reward"
            ]
        )

        rows.append(
            {
                "window":
                    window,

                "valid_H_n":
                    len(
                        valid_H
                    ),

                "high_H_n":
                    len(
                        high_H
                    ),

                "high_H_fraction":
                    (
                        len(
                            high_H
                        )
                        / len(
                            valid_H
                        )
                        if len(
                            valid_H
                        )
                        else np.nan
                    ),

                "high_H_reward_n":
                    len(
                        high_H_rewards
                    ),

                "high_H_draw_rate":
                    (
                        float(
                            (
                                high_H_rewards
                                == 0.0
                            ).mean()
                        )
                        if len(
                            high_H_rewards
                        )
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

                "valid_U_n":
                    len(
                        valid_U
                    ),

                "high_U_n":
                    len(
                        high_U
                    ),

                "high_U_fraction":
                    (
                        len(
                            high_U
                        )
                        / len(
                            valid_U
                        )
                        if len(
                            valid_U
                        )
                        else np.nan
                    ),

                "high_U_reward_n":
                    len(
                        high_U_rewards
                    ),

                "high_U_decisive_rate":
                    (
                        float(
                            (
                                high_U_rewards.abs()
                                == 1.0
                            ).mean()
                        )
                        if len(
                            high_U_rewards
                        )
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
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Experiment 4
# Chronological trends
# ============================================================

def trend_analysis(
    window_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = (
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
        "repeated_observation_fraction",
    )

    rows = []

    for metric in metrics:

        if metric not in window_df.columns:

            continue

        values = window_df[
            metric
        ].to_numpy(
            dtype=np.float64,
        )

        finite_indices = np.flatnonzero(
            np.isfinite(
                values
            )
        )

        if len(
            finite_indices
        ) < 2:

            continue

        first_index = finite_indices[
            0
        ]

        last_index = finite_indices[
            -1
        ]

        first = values[
            first_index
        ]

        last = values[
            last_index
        ]

        absolute_change = (
            last
            - first
        )

        relative_change = (
            absolute_change
            / abs(
                first
            )
            if abs(
                first
            )
            > 1e-12
            else np.nan
        )

        finite_values_array = values[
            finite_indices
        ]

        differences = np.diff(
            finite_values_array
        )

        monotonic_increasing = bool(
            np.all(
                differences
                >= 0.0
            )
        )

        monotonic_decreasing = bool(
            np.all(
                differences
                <= 0.0
            )
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
                    chronological_spearman(
                        values
                    ),

                "monotonic_increasing":
                    monotonic_increasing,

                "monotonic_decreasing":
                    monotonic_decreasing,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Experiment 5
# H/U quadrants
# ============================================================

def diagnostic_quadrants(
    df: pd.DataFrame,
    first_window: str,
    window_labels: list[str],
) -> tuple[pd.DataFrame, float, float]:

    reference = df[
        df[
            "window"
        ]
        == first_window
    ]

    H_threshold = safe_quantile(
        reference[
            "H"
        ],
        QUADRANT_REFERENCE_QUANTILE,
    )

    U_threshold = safe_quantile(
        reference[
            "U"
        ],
        QUADRANT_REFERENCE_QUANTILE,
    )

    if (
        not np.isfinite(
            H_threshold
        )
        or not np.isfinite(
            U_threshold
        )
    ):

        raise RuntimeError(
            "Unable to construct W1 H/U quadrant thresholds."
        )

    rows = []

    for window in window_labels:

        subset = df[
            df[
                "window"
            ]
            == window
        ].copy()

        valid_mask = (
            np.isfinite(
                pd.to_numeric(
                    subset[
                        "H"
                    ],
                    errors="coerce",
                )
            )
            &
            np.isfinite(
                pd.to_numeric(
                    subset[
                        "U"
                    ],
                    errors="coerce",
                )
            )
        )

        subset = subset[
            valid_mask
        ]

        high_H = (
            subset[
                "H"
            ]
            >= H_threshold
        )

        high_U = (
            subset[
                "U"
            ]
            >= U_threshold
        )

        quadrants = {
            "low_H_low_U":
                (
                    ~high_H
                    & ~high_U
                ),

            "high_H_low_U":
                (
                    high_H
                    & ~high_U
                ),

            "low_H_high_U":
                (
                    ~high_H
                    & high_U
                ),

            "high_H_high_U":
                (
                    high_H
                    & high_U
                ),
        }

        for (
            name,
            mask,
        ) in quadrants.items():

            quadrant = subset[
                mask
            ]

            rewards = finite_values(
                quadrant[
                    "reward"
                ]
            )

            rows.append(
                {
                    "window":
                        window,

                    "quadrant":
                        name,

                    "valid_HU_n":
                        len(
                            subset
                        ),

                    "n":
                        len(
                            quadrant
                        ),

                    "fraction":
                        (
                            len(
                                quadrant
                            )
                            / len(
                                subset
                            )
                            if len(
                                subset
                            )
                            else np.nan
                        ),

                    "mean_H":
                        safe_mean(
                            quadrant[
                                "H"
                            ]
                        ),

                    "mean_U":
                        safe_mean(
                            quadrant[
                                "U"
                            ]
                        ),

                    "mean_HU":
                        safe_mean(
                            quadrant[
                                "HU"
                            ]
                        ),

                    "reward_n":
                        len(
                            rewards
                        ),

                    "mean_reward":
                        safe_mean(
                            rewards
                        ),

                    "mean_abs_reward":
                        safe_mean(
                            rewards.abs()
                        ),

                    "draw_rate":
                        (
                            float(
                                (
                                    rewards
                                    == 0.0
                                ).mean()
                            )
                            if len(
                                rewards
                            )
                            else np.nan
                        ),

                    "decisive_rate":
                        (
                            float(
                                (
                                    rewards.abs()
                                    == 1.0
                                ).mean()
                            )
                            if len(
                                rewards
                            )
                            else np.nan
                        ),
                }
            )

    return (
        pd.DataFrame(
            rows
        ),
        H_threshold,
        U_threshold,
    )


# ============================================================
# Plots
# ============================================================

def plot_signal_progression(
    window_df: pd.DataFrame,
    *,
    signal: str,
    output_path: Path,
) -> None:

    x = np.arange(
        len(
            window_df
        )
    )

    labels = window_df[
        "window"
    ].tolist()

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    plt.plot(
        x,
        window_df[
            f"{signal}_mean"
        ],
        marker="o",
        label=f"{signal} mean",
    )

    plt.plot(
        x,
        window_df[
            f"{signal}_median"
        ],
        marker="o",
        label=f"{signal} median",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel(
        "Chronological window"
    )

    plt.ylabel(
        signal
    )

    plt.title(
        f"{signal} across chronological windows"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def plot_tail_progression(
    tail_df: pd.DataFrame,
    output_path: Path,
) -> None:

    x = np.arange(
        len(
            tail_df
        )
    )

    labels = tail_df[
        "window"
    ].tolist()

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

    for signal in (
        "H",
        "U",
    ):

        column = (
            f"{signal}_above_ref_q90"
        )

        plt.plot(
            x,
            tail_df[
                column
            ],
            marker="o",
            label=f"{signal} > W1 P90",
        )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel(
        "Chronological window"
    )

    plt.ylabel(
        "Fraction among finite signal observations"
    )

    plt.title(
        "Persistence of W1 high-signal populations"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def plot_reward_conditioned(
    reward_df: pd.DataFrame,
    output_path: Path,
) -> None:

    x = np.arange(
        len(
            reward_df
        )
    )

    labels = reward_df[
        "window"
    ].tolist()

    plt.figure(
        figsize=(
            8,
            6,
        )
    )

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
        "Conditional outcome fraction"
    )

    plt.title(
        "Outcome characteristics of W1-relative high-H/high-U states"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def plot_quadrants(
    quadrant_df: pd.DataFrame,
    window_labels: list[str],
    output_path: Path,
) -> None:

    pivot = (
        quadrant_df
        .pivot(
            index="window",
            columns="quadrant",
            values="fraction",
        )
        .reindex(
            window_labels
        )
    )

    x = np.arange(
        len(
            window_labels
        )
    )

    plt.figure(
        figsize=(
            9,
            6,
        )
    )

    for column in pivot.columns:

        plt.plot(
            x,
            pivot[
                column
            ],
            marker="o",
            label=column,
        )

    plt.xticks(
        x,
        window_labels,
    )

    plt.xlabel(
        "Chronological window"
    )

    plt.ylabel(
        "Fraction among positions with finite H and U"
    )

    plt.title(
        "Evolution of W1-relative H/U diagnostic regimes"
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
    input_path: Path,
    window_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    quadrant_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    H_quadrant_threshold: float,
    U_quadrant_threshold: float,
    window_labels: list[str],
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
        "ALBERTA - UNCERTAINTY PROGRESSION ANALYSIS"
    )

    add(
        "=" * 80
    )

    add()

    add(
        f"Input: {input_path}"
    )

    add(
        f"Chronological windows: {len(window_labels)}"
    )

    add()

    # ========================================================
    # Purpose
    # ========================================================

    add(
        "SCIENTIFIC QUESTION"
    )

    add(
        "-" * 80
    )

    add(
        "Do policy entropy H and historical-league value "
        "disagreement U exhibit systematic distributional "
        "changes over the course of an already completed RL run?"
    )

    add()

    add(
        "This is a temporal diagnostic analysis. It does not "
        "establish H or U as measures of playing strength, "
        "learning value, or optimal acquisition utility."
    )

    add()

    # ========================================================
    # Chronology
    # ========================================================

    add(
        "CHRONOLOGICAL APPROXIMATION"
    )

    add(
        "-" * 80
    )

    add(
        "The source does not provide exact epoch identifiers."
    )

    add(
        "Records are therefore divided into equal chronological "
        "windows according to their append order."
    )

    add(
        "These windows are coarse temporal aggregates and must "
        "not be interpreted as exact RL checkpoints."
    )

    add()

    # ========================================================
    # Window statistics
    # ========================================================

    add(
        "CHRONOLOGICAL STATISTICS"
    )

    add(
        "-" * 80
    )

    for _, row in window_df.iterrows():

        add(
            f"{row['window']}: "
            f"records={int(row['records']):,}, "
            f"unique_FENs={int(row['unique_fens']):,}, "
            f"repeat_obs={row['repeated_observation_fraction']:.2%}, "
            f"H={row['H_mean']:.6f}, "
            f"U={row['U_mean']:.6f}, "
            f"HU={row['HU_mean']:.6f}, "
            f"draw={row['draw_rate']:.2%}, "
            f"decisive={row['decisive_rate']:.2%}"
        )

    add()

    # ========================================================
    # Trends
    # ========================================================

    add(
        "CHRONOLOGICAL TRENDS"
    )

    add(
        "-" * 80
    )

    for _, row in trend_df.iterrows():

        if row[
            "monotonic_increasing"
        ]:

            direction = "monotonic increase"

        elif row[
            "monotonic_decreasing"
        ]:

            direction = "monotonic decrease"

        else:

            direction = "non-monotonic"

        add(
            f"{row['metric']}: "
            f"{row['first']:.6f} -> {row['last']:.6f}, "
            f"delta={row['absolute_change']:+.6f}, "
            f"relative={row['relative_change']:+.2%}, "
            f"rho={row['spearman_rho']:+.4f}, "
            f"{direction}"
        )

    add()

    # ========================================================
    # W1 reference thresholds
    # ========================================================

    add(
        "W1-RELATIVE DIAGNOSTIC THRESHOLDS"
    )

    add(
        "-" * 80
    )

    add(
        f"H median threshold: {H_quadrant_threshold:.6f}"
    )

    add(
        f"U median threshold: {U_quadrant_threshold:.6f}"
    )

    add()

    add(
        "The thresholds are reference-relative descriptors, "
        "not universal definitions of high or low uncertainty."
    )

    add()

    # ========================================================
    # High tails
    # ========================================================

    add(
        "HIGH-SIGNAL TAILS"
    )

    add(
        "-" * 80
    )

    for _, row in tail_df.iterrows():

        add(
            f"{row['window']}: "
            f"H>P90={row['H_above_ref_q90']:.2%}, "
            f"H>P95={row['H_above_ref_q95']:.2%}, "
            f"H>P99={row['H_above_ref_q99']:.2%}, "
            f"U>P90={row['U_above_ref_q90']:.2%}, "
            f"U>P95={row['U_above_ref_q95']:.2%}, "
            f"U>P99={row['U_above_ref_q99']:.2%}"
        )

    add()

    # ========================================================
    # Reward-conditioned
    # ========================================================

    add(
        "OUTCOME-CONDITIONED SIGNAL POPULATIONS"
    )

    add(
        "-" * 80
    )

    for _, row in reward_df.iterrows():

        add(
            f"{row['window']}: "
            f"high-H={row['high_H_fraction']:.2%}, "
            f"draw|high-H={row['high_H_draw_rate']:.2%}, "
            f"|R||high-H={row['high_H_mean_abs_reward']:.4f}, "
            f"high-U={row['high_U_fraction']:.2%}, "
            f"decisive|high-U={row['high_U_decisive_rate']:.2%}, "
            f"|R||high-U={row['high_U_mean_abs_reward']:.4f}"
        )

    add()

    # ========================================================
    # Quadrants
    # ========================================================

    add(
        "H/U DIAGNOSTIC REGIMES"
    )

    add(
        "-" * 80
    )

    for window in window_labels:

        add(
            f"{window}:"
        )

        subset = quadrant_df[
            quadrant_df[
                "window"
            ]
            == window
        ]

        for _, row in subset.iterrows():

            add(
                f"  {row['quadrant']}: "
                f"{row['fraction']:.2%} "
                f"(draw={row['draw_rate']:.2%}, "
                f"decisive={row['decisive_rate']:.2%})"
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
        "H is policy entropy: larger H means that the model's "
        "legal-action distribution is more diffuse."
    )

    add(
        "A decrease in H is therefore consistent with increasingly "
        "concentrated action distributions on encountered states."
    )

    add()

    add(
        "U measures disagreement among trained historical value "
        "estimators in the league."
    )

    add(
        "Changes in U therefore indicate changes in the structure "
        "of historical value disagreement encountered during "
        "training."
    )

    add()

    add(
        "Associations between high H and draws, or high U and "
        "decisive outcomes, are empirical population-level "
        "relationships rather than definitions of those signals."
    )

    add()

    add(
        "The diagnostic role of H/U is distinct from acquisition "
        "utility: a signal can describe how the agent's state "
        "distribution changes without identifying annotations "
        "that improve downstream performance."
    )

    add()

    # ========================================================
    # Limitations
    # ========================================================

    add(
        "LIMITATIONS"
    )

    add(
        "-" * 80
    )

    add(
        "The chronological windows are approximate and are not "
        "statistically paired with checkpoint playing strength."
    )

    add(
        "The analysis is observational and does not establish "
        "causality."
    )

    add(
        "Window-level trend coefficients are based on a small "
        "number of temporal aggregates and are descriptive."
    )

    add(
        "Individual H/U/HU observations are not treated as "
        "independent temporal replicates."
    )

    add(
        "Any U/HU result must be produced using the corrected "
        "league uncertainty estimator that excludes untrained "
        "BC value heads."
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

    df = load_data(
        args.input
    )

    (
        df,
        window_labels,
    ) = assign_chronological_windows(
        df,
        args.windows,
    )

    # ========================================================
    # Chronological statistics
    # ========================================================

    window_df = chronological_statistics(
        df,
        window_labels,
    )

    # ========================================================
    # W1 reference thresholds
    # ========================================================

    thresholds = build_reference_thresholds(
        df,
        window_labels[
            0
        ],
    )

    print()
    print("W1 reference thresholds")
    print("-" * 80)

    for signal in (
        "H",
        "U",
    ):

        for quantile in REFERENCE_QUANTILES:

            print(
                f"{signal} P{int(100 * quantile):02d}: "
                f"{thresholds[signal][quantile]:.6f}"
            )

    # ========================================================
    # Tails
    # ========================================================

    tail_df = analyze_uncertainty_tails(
        df,
        thresholds,
        window_labels,
    )

    # ========================================================
    # Reward-conditioned populations
    # ========================================================

    reward_df = reward_conditioned_analysis(
        df,
        thresholds,
        window_labels,
    )

    # ========================================================
    # Trends
    # ========================================================

    trend_df = trend_analysis(
        window_df
    )

    # ========================================================
    # Quadrants
    # ========================================================

    (
        quadrant_df,
        H_quadrant_threshold,
        U_quadrant_threshold,
    ) = diagnostic_quadrants(
        df,
        window_labels[
            0
        ],
        window_labels,
    )

    # ========================================================
    # Save tabular outputs
    # ========================================================

    window_df.to_csv(
        args.output_dir
        / "chronological_statistics.csv",
        index=False,
    )

    tail_df.to_csv(
        args.output_dir
        / "uncertainty_tails.csv",
        index=False,
    )

    reward_df.to_csv(
        args.output_dir
        / "reward_conditioned_progression.csv",
        index=False,
    )

    quadrant_df.to_csv(
        args.output_dir
        / "diagnostic_quadrants.csv",
        index=False,
    )

    trend_df.to_csv(
        args.output_dir
        / "chronological_trends.csv",
        index=False,
    )

    # ========================================================
    # Plots
    # ========================================================

    print()
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_signal_progression(
        window_df,
        signal="H",
        output_path=(
            args.output_dir
            / "H_progression.png"
        ),
    )

    plot_signal_progression(
        window_df,
        signal="U",
        output_path=(
            args.output_dir
            / "U_progression.png"
        ),
    )

    plot_tail_progression(
        tail_df,
        args.output_dir
        / "uncertainty_tail_progression.png",
    )

    plot_reward_conditioned(
        reward_df,
        args.output_dir
        / "reward_conditioned_progression.png",
    )

    plot_quadrants(
        quadrant_df,
        window_labels,
        args.output_dir
        / "diagnostic_quadrants.png",
    )

    # ========================================================
    # Report
    # ========================================================

    report = build_report(
        input_path=args.input,
        window_df=window_df,
        tail_df=tail_df,
        reward_df=reward_df,
        quadrant_df=quadrant_df,
        trend_df=trend_df,
        H_quadrant_threshold=H_quadrant_threshold,
        U_quadrant_threshold=U_quadrant_threshold,
        window_labels=window_labels,
    )

    report_path = (
        args.output_dir
        / "uncertainty_progression_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Records:     {len(df):,}"
    )

    print(
        f"Unique FENs: {df['fen'].nunique():,}"
    )

    print(
        f"Windows:     {len(window_labels)}"
    )

    print()

    first = window_df.iloc[
        0
    ]

    last = window_df.iloc[
        -1
    ]

    for signal in (
        "H",
        "U",
        "HU",
    ):

        first_value = first[
            f"{signal}_mean"
        ]

        last_value = last[
            f"{signal}_mean"
        ]

        print(
            f"{signal} mean: "
            f"{first_value:.6f} -> "
            f"{last_value:.6f} "
            f"({last_value - first_value:+.6f})"
        )

    print()

    trend_lookup = trend_df.set_index(
        "metric"
    )

    for metric in (
        "H_mean",
        "U_mean",
        "HU_mean",
    ):

        if metric not in trend_lookup.index:

            continue

        row = trend_lookup.loc[
            metric
        ]

        print(
            f"{metric:8s}: "
            f"Spearman rho="
            f"{row['spearman_rho']:+.4f}"
        )

    print()
    print(
        f"Results: {args.output_dir}"
    )

    print(
        f"Report:  {report_path}"
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()