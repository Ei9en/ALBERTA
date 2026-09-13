from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_analysis"
)


# ============================================================
# Configuration
# ============================================================

SIGNALS = [
    "H",
    "U",
    "HU",
]

MIN_REPEATED_OBSERVATIONS = 10

REPETITION_THRESHOLDS = [
    5,
    10,
    20,
    50,
    100,
    200,
]

N_QUANTILES = 10

CI95_Z = 1.959963984540054


START_FEN = (
    "rnbqkbnr/"
    "pppppppp/"
    "8/8/8/8/"
    "PPPPPPPP/"
    "RNBQKBNR "
    "w KQkq - 0 1"
)


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


def safe_mean(
    values,
) -> float:

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def safe_median(
    values,
) -> float:

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.median()
    )


def safe_quantile(
    values,
    q: float,
) -> float:

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.quantile(
            q
        )
    )


def entropy_from_probabilities(
    probabilities,
) -> float:
    """
    Shannon entropy in bits.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    if len(probabilities) == 0:
        return np.nan

    return float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )


def reward_entropy(
    rewards,
) -> float:
    """
    Entropy of rewards in {-1, 0, +1}.
    """

    rewards = np.asarray(
        rewards,
        dtype=float,
    )

    rewards = rewards[
        np.isfinite(
            rewards
        )
    ]

    if len(rewards) == 0:
        return np.nan

    probabilities = [
        np.mean(
            rewards == 1.0
        ),
        np.mean(
            rewards == 0.0
        ),
        np.mean(
            rewards == -1.0
        ),
    ]

    return entropy_from_probabilities(
        probabilities
    )


def result_to_reward(
    result: str,
    side_to_move: str,
) -> float:
    """
    Convert terminal result to reward from the perspective
    of the player to move in the recorded FEN.

    White to move:
        1-0 -> +1
        0-1 -> -1

    Black to move:
        0-1 -> +1
        1-0 -> -1

    Draw:
        0
    """

    if result == "1/2-1/2":
        return 0.0

    if side_to_move == "w":

        if result == "1-0":
            return 1.0

        if result == "0-1":
            return -1.0

    elif side_to_move == "b":

        if result == "0-1":
            return 1.0

        if result == "1-0":
            return -1.0

    return np.nan


def pearson_corr(
    x,
    y,
) -> float:

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

    x = x[
        mask
    ]

    y = y[
        mask
    ]

    if len(x) < 2:
        return np.nan

    if (
        np.std(x) == 0.0
        or np.std(y) == 0.0
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
) -> float:

    x = pd.Series(
        x,
        dtype=float,
    )

    y = pd.Series(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[
        mask
    ]

    y = y[
        mask
    ]

    if len(x) < 2:
        return np.nan

    if (
        x.nunique() < 2
        or y.nunique() < 2
    ):
        return np.nan

    return float(
        x.corr(
            y,
            method="spearman",
        )
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze terminal reward informativeness and its "
            "relationship with ALBERTA uncertainty signals."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input uncertainty statistics JSON.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for report and plots.",
    )

    parser.add_argument(
        "--min-observations",
        type=int,
        default=MIN_REPEATED_OBSERVATIONS,
        help=(
            "Minimum number of observations required for "
            "repeated-position analyses."
        ),
    )

    parser.add_argument(
        "--quantiles",
        type=int,
        default=N_QUANTILES,
        help="Number of quantile bins for signal analysis.",
    )

    return parser.parse_args()


# ============================================================
# Loading
# ============================================================

def load_data(
    input_path: Path,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("ALBERTA - REWARD SIGNAL ANALYSIS")
    print("=" * 70)

    print()
    print(
        f"Input: {input_path}"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        raw = json.load(
            file
        )

    if not isinstance(
        raw,
        list,
    ):

        raise ValueError(
            "Expected JSON root to be a list."
        )

    rows = []

    invalid_record = 0
    invalid_fen = 0
    invalid_reward = 0

    for record in raw:

        if not isinstance(
            record,
            dict,
        ):

            invalid_record += 1
            continue

        fen = record.get(
            "fen"
        )

        result = record.get(
            "result"
        )

        if not isinstance(
            fen,
            str,
        ):

            invalid_fen += 1
            continue

        fen_parts = fen.split()

        if len(
            fen_parts
        ) < 2:

            invalid_fen += 1
            continue

        side_to_move = fen_parts[
            1
        ]

        if side_to_move not in {
            "w",
            "b",
        }:

            invalid_fen += 1
            continue

        reward = result_to_reward(
            result,
            side_to_move,
        )

        if not math.isfinite(
            reward
        ):

            invalid_reward += 1
            continue

        rows.append(
            {
                "fen":
                    fen,

                "side_to_move":
                    side_to_move,

                "result":
                    result,

                "reward":
                    reward,

                "abs_reward":
                    abs(
                        reward
                    ),

                "H":
                    safe_float(
                        record.get(
                            "H"
                        )
                    ),

                "U":
                    safe_float(
                        record.get(
                            "U"
                        )
                    ),

                "HU":
                    safe_float(
                        record.get(
                            "HU"
                        )
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid observations found."
        )

    print(
        f"Raw records:       {len(raw):,}"
    )

    print(
        f"Valid observations:{len(df):,}"
    )

    print(
        f"Invalid records:   {invalid_record:,}"
    )

    print(
        f"Invalid FENs:      {invalid_fen:,}"
    )

    print(
        f"Invalid rewards:   {invalid_reward:,}"
    )

    print(
        f"Unique FENs:       {df['fen'].nunique():,}"
    )

    print()
    print(
        "Signal coverage:"
    )

    for signal in SIGNALS:

        valid = int(
            np.isfinite(
                df[
                    signal
                ]
            ).sum()
        )

        print(
            f"  {signal:<2}: "
            f"{valid:,}/{len(df):,} "
            f"({valid / len(df):.2%})"
        )

    return df


# ============================================================
# Global reward analysis
# ============================================================

def analyze_global_reward(
    df: pd.DataFrame,
) -> dict:

    print()
    print("=" * 70)
    print("GLOBAL REWARD DISTRIBUTION")
    print("=" * 70)

    total = len(
        df
    )

    losses = int(
        (
            df["reward"]
            == -1.0
        ).sum()
    )

    draws = int(
        (
            df["reward"]
            == 0.0
        ).sum()
    )

    wins = int(
        (
            df["reward"]
            == 1.0
        ).sum()
    )

    entropy = reward_entropy(
        df[
            "reward"
        ]
    )

    mean_reward = float(
        df[
            "reward"
        ].mean()
    )

    variance = float(
        df[
            "reward"
        ].var(
            ddof=0
        )
    )

    print(
        f"Loss: {losses:>10,} "
        f"({losses / total:.2%})"
    )

    print(
        f"Draw: {draws:>10,} "
        f"({draws / total:.2%})"
    )

    print(
        f"Win:  {wins:>10,} "
        f"({wins / total:.2%})"
    )

    print()

    print(
        f"H(R):   {entropy:.6f} bits"
    )

    print(
        f"E[R]:   {mean_reward:+.6f}"
    )

    print(
        f"Var(R): {variance:.6f}"
    )

    return {
        "n":
            total,

        "wins":
            wins,

        "draws":
            draws,

        "losses":
            losses,

        "entropy":
            entropy,

        "mean_reward":
            mean_reward,

        "variance":
            variance,
    }


# ============================================================
# Signal distributions
# ============================================================

def analyze_signal_distributions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("SIGNAL DISTRIBUTIONS")
    print("=" * 70)

    rows = []

    for signal in SIGNALS:

        values = df[
            signal
        ]

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values) == 0:
            continue

        row = {
            "signal":
                signal,

            "count":
                len(values),

            "min":
                float(
                    values.min()
                ),

            "mean":
                float(
                    values.mean()
                ),

            "std":
                float(
                    values.std(
                        ddof=0
                    )
                ),

            "p01":
                float(
                    values.quantile(
                        0.01
                    )
                ),

            "p05":
                float(
                    values.quantile(
                        0.05
                    )
                ),

            "p10":
                float(
                    values.quantile(
                        0.10
                    )
                ),

            "p25":
                float(
                    values.quantile(
                        0.25
                    )
                ),

            "p50":
                float(
                    values.quantile(
                        0.50
                    )
                ),

            "p75":
                float(
                    values.quantile(
                        0.75
                    )
                ),

            "p90":
                float(
                    values.quantile(
                        0.90
                    )
                ),

            "p95":
                float(
                    values.quantile(
                        0.95
                    )
                ),

            "p99":
                float(
                    values.quantile(
                        0.99
                    )
                ),

            "max":
                float(
                    values.max()
                ),
        }

        rows.append(
            row
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
# Observation-level signal / reward relationships
# ============================================================

def analyze_observation_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("OBSERVATION-LEVEL SIGNAL / REWARD CORRELATIONS")
    print("=" * 70)

    rows = []

    for signal in SIGNALS:

        for target in [
            "reward",
            "abs_reward",
        ]:

            mask = (
                np.isfinite(
                    df[
                        signal
                    ]
                )
                & np.isfinite(
                    df[
                        target
                    ]
                )
            )

            rows.append(
                {
                    "signal":
                        signal,

                    "target":
                        target,

                    "n_pairs":
                        int(
                            mask.sum()
                        ),

                    "pearson":
                        pearson_corr(
                            df[
                                signal
                            ],
                            df[
                                target
                            ],
                        ),

                    "spearman":
                        spearman_corr(
                            df[
                                signal
                            ],
                            df[
                                target
                            ],
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
                f"{x:+.6f}",
        )
    )

    return result


# ============================================================
# Quantile analysis
# ============================================================

def analyze_signal_quantiles(
    df: pd.DataFrame,
    n_quantiles: int,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("SIGNAL QUANTILE ANALYSIS")
    print("=" * 70)

    rows = []

    for signal in SIGNALS:

        valid = df[
            np.isfinite(
                df[
                    signal
                ]
            )
        ].copy()

        if len(
            valid
        ) < n_quantiles:

            continue

        ranked = valid[
            signal
        ].rank(
            method="first"
        )

        valid[
            "quantile"
        ] = (
            pd.qcut(
                ranked,
                q=n_quantiles,
                labels=False,
            )
            + 1
        )

        for quantile in range(
            1,
            n_quantiles + 1,
        ):

            subset = valid[
                valid[
                    "quantile"
                ]
                == quantile
            ]

            if len(
                subset
            ) == 0:

                continue

            rows.append(
                {
                    "signal":
                        signal,

                    "quantile":
                        quantile,

                    "n":
                        len(
                            subset
                        ),

                    "mean_signal":
                        float(
                            subset[
                                signal
                            ].mean()
                        ),

                    "mean_reward":
                        float(
                            subset[
                                "reward"
                            ].mean()
                        ),

                    "mean_abs_reward":
                        float(
                            subset[
                                "abs_reward"
                            ].mean()
                        ),

                    "reward_entropy":
                        reward_entropy(
                            subset[
                                "reward"
                            ]
                        ),

                    "win_fraction":
                        float(
                            (
                                subset[
                                    "reward"
                                ]
                                == 1.0
                            ).mean()
                        ),

                    "draw_fraction":
                        float(
                            (
                                subset[
                                    "reward"
                                ]
                                == 0.0
                            ).mean()
                        ),

                    "loss_fraction":
                        float(
                            (
                                subset[
                                    "reward"
                                ]
                                == -1.0
                            ).mean()
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
# Signal redundancy
# ============================================================

def analyze_signal_redundancy(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("SIGNAL REDUNDANCY")
    print("=" * 70)

    pairs = [
        (
            "H",
            "U",
        ),
        (
            "H",
            "HU",
        ),
        (
            "U",
            "HU",
        ),
    ]

    rows = []

    for a, b in pairs:

        rows.append(
            {
                "a":
                    a,

                "b":
                    b,

                "pearson":
                    pearson_corr(
                        df[
                            a
                        ],
                        df[
                            b
                        ],
                    ),

                "spearman":
                    spearman_corr(
                        df[
                            a
                        ],
                        df[
                            b
                        ],
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
                f"{x:+.6f}",
        )
    )

    return result


# ============================================================
# Side-to-move analysis
# ============================================================

def analyze_side_to_move(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("SIDE-TO-MOVE ANALYSIS")
    print("=" * 70)

    rows = []

    for side in [
        "w",
        "b",
    ]:

        subset = df[
            df[
                "side_to_move"
            ]
            == side
        ]

        row = {
            "side":
                side,

            "n":
                len(
                    subset
                ),

            "win_fraction":
                float(
                    (
                        subset[
                            "reward"
                        ]
                        == 1.0
                    ).mean()
                ),

            "draw_fraction":
                float(
                    (
                        subset[
                            "reward"
                        ]
                        == 0.0
                    ).mean()
                ),

            "loss_fraction":
                float(
                    (
                        subset[
                            "reward"
                        ]
                        == -1.0
                    ).mean()
                ),

            "mean_reward":
                float(
                    subset[
                        "reward"
                    ].mean()
                ),
        }

        for signal in SIGNALS:

            row[
                f"{signal}_mean"
            ] = safe_mean(
                subset[
                    signal
                ]
            )

        rows.append(
            row
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
# Position aggregation
# ============================================================

def build_position_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("POSITION AGGREGATION")
    print("=" * 70)

    counts = pd.crosstab(
        df[
            "fen"
        ],
        df[
            "reward"
        ],
    )

    for reward in [
        -1.0,
        0.0,
        1.0,
    ]:

        if reward not in counts.columns:

            counts[
                reward
            ] = 0

    counts = counts[
        [
            1.0,
            0.0,
            -1.0,
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

    counts[
        "n"
    ] = (
        counts[
            "W"
        ]
        + counts[
            "D"
        ]
        + counts[
            "L"
        ]
    )

    counts[
        "pW"
    ] = (
        counts[
            "W"
        ]
        / counts[
            "n"
        ]
    )

    counts[
        "pD"
    ] = (
        counts[
            "D"
        ]
        / counts[
            "n"
        ]
    )

    counts[
        "pL"
    ] = (
        counts[
            "L"
        ]
        / counts[
            "n"
        ]
    )

    counts[
        "mean_reward"
    ] = (
        counts[
            "pW"
        ]
        - counts[
            "pL"
        ]
    )

    counts[
        "abs_mean_reward"
    ] = counts[
        "mean_reward"
    ].abs()

    counts[
        "reward_variance"
    ] = (
        counts[
            "pW"
        ]
        + counts[
            "pL"
        ]
        - counts[
            "mean_reward"
        ]
        ** 2
    )

    probabilities = counts[
        [
            "pW",
            "pD",
            "pL",
        ]
    ].to_numpy(
        dtype=float
    )

    entropy = np.zeros(
        len(
            counts
        ),
        dtype=float,
    )

    for column in range(
        probabilities.shape[
            1
        ]
    ):

        values = probabilities[
            :,
            column
        ]

        mask = (
            values > 0.0
        )

        entropy[
            mask
        ] -= (
            values[
                mask
            ]
            * np.log2(
                values[
                    mask
                ]
            )
        )

    counts[
        "reward_entropy"
    ] = entropy

    variance = counts[
        "reward_variance"
    ].to_numpy(
        dtype=float
    )

    n = counts[
        "n"
    ].to_numpy(
        dtype=float
    )

    se = np.sqrt(
        np.maximum(
            variance,
            0.0,
        )
        / n
    )

    counts[
        "ci95_low"
    ] = (
        counts[
            "mean_reward"
        ]
        - CI95_Z
        * se
    )

    counts[
        "ci95_high"
    ] = (
        counts[
            "mean_reward"
        ]
        + CI95_Z
        * se
    )

    counts[
        "ci95_width"
    ] = (
        2.0
        * CI95_Z
        * se
    )

    counts[
        "ci95_contains_zero"
    ] = (
        (
            counts[
                "ci95_low"
            ]
            <= 0.0
        )
        & (
            counts[
                "ci95_high"
            ]
            >= 0.0
        )
    )

    # --------------------------------------------------------
    # Aggregate H/U/HU independently.
    # Missing one signal never removes another signal or reward.
    # --------------------------------------------------------

    for signal in SIGNALS:

        stats = (
            df.groupby(
                "fen"
            )[
                signal
            ]
            .agg(
                [
                    "mean",
                    "count",
                ]
            )
        )

        stats.columns = [
            f"{signal}_mean",
            f"{signal}_count",
        ]

        counts = counts.join(
            stats,
            how="left",
        )

        counts[
            f"{signal}_fraction"
        ] = (
            counts[
                f"{signal}_count"
            ]
            / counts[
                "n"
            ]
        )

    position_df = counts.reset_index()

    print(
        f"Unique FENs: "
        f"{len(position_df):,}"
    )

    print(
        f"FENs with >= {MIN_REPEATED_OBSERVATIONS} observations: "
        f"{(position_df['n'] >= MIN_REPEATED_OBSERVATIONS).sum():,}"
    )

    print(
        f"Maximum observations/FEN: "
        f"{position_df['n'].max():,}"
    )

    return position_df


# ============================================================
# Starting position
# ============================================================

def analyze_starting_position(
    position_df: pd.DataFrame,
) -> dict | None:

    subset = position_df[
        position_df[
            "fen"
        ]
        == START_FEN
    ]

    if len(
        subset
    ) == 0:

        return None

    row = subset.iloc[
        0
    ]

    return {
        "n":
            int(
                row[
                    "n"
                ]
            ),

        "wins":
            int(
                row[
                    "W"
                ]
            ),

        "draws":
            int(
                row[
                    "D"
                ]
            ),

        "losses":
            int(
                row[
                    "L"
                ]
            ),

        "reward_entropy":
            float(
                row[
                    "reward_entropy"
                ]
            ),

        "mean_reward":
            float(
                row[
                    "mean_reward"
                ]
            ),

        "ci95_low":
            float(
                row[
                    "ci95_low"
                ]
            ),

        "ci95_high":
            float(
                row[
                    "ci95_high"
                ]
            ),
    }


# ============================================================
# Repeated-position reward ambiguity
# ============================================================

def analyze_repeated_positions(
    position_df: pd.DataFrame,
    min_observations: int,
) -> dict:

    print()
    print("=" * 70)
    print("REPEATED-POSITION REWARD AMBIGUITY")
    print("=" * 70)

    repeated = position_df[
        position_df[
            "n"
        ]
        >= min_observations
    ].copy()

    if len(
        repeated
    ) == 0:

        raise RuntimeError(
            "No positions satisfy the repeated-position threshold."
        )

    max_entropy = math.log2(
        3.0
    )

    stats = {
        "positions":
            len(
                repeated
            ),

        "mean_entropy":
            safe_mean(
                repeated[
                    "reward_entropy"
                ]
            ),

        "median_entropy":
            safe_median(
                repeated[
                    "reward_entropy"
                ]
            ),

        "p90_entropy":
            safe_quantile(
                repeated[
                    "reward_entropy"
                ],
                0.90,
            ),

        "p95_entropy":
            safe_quantile(
                repeated[
                    "reward_entropy"
                ],
                0.95,
            ),

        "mean_abs_reward":
            safe_mean(
                repeated[
                    "abs_mean_reward"
                ]
            ),

        "median_abs_reward":
            safe_median(
                repeated[
                    "abs_mean_reward"
                ]
            ),

        "mean_ci_width":
            safe_mean(
                repeated[
                    "ci95_width"
                ]
            ),

        "median_ci_width":
            safe_median(
                repeated[
                    "ci95_width"
                ]
            ),

        "fraction_ci_contains_zero":
            float(
                repeated[
                    "ci95_contains_zero"
                ].mean()
            ),

        "max_entropy":
            max_entropy,
    }

    for key, value in stats.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:<30}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:<30}: "
                f"{value:,}"
            )

    return {
        "summary":
            stats,

        "table":
            repeated,
    }


# ============================================================
# Repetition strata
# ============================================================

def analyze_repetition_strata(
    position_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive statistics for increasingly repeated positions.

    IMPORTANT:
    These thresholds select different populations of FENs.
    They must not be interpreted as a controlled sample-size
    experiment on a fixed population.
    """

    print()
    print("=" * 70)
    print("REPETITION STRATA")
    print("=" * 70)

    rows = []

    for threshold in REPETITION_THRESHOLDS:

        subset = position_df[
            position_df[
                "n"
            ]
            >= threshold
        ]

        if len(
            subset
        ) == 0:

            continue

        rows.append(
            {
                "min_n":
                    threshold,

                "positions":
                    len(
                        subset
                    ),

                "mean_n":
                    safe_mean(
                        subset[
                            "n"
                        ]
                    ),

                "median_n":
                    safe_median(
                        subset[
                            "n"
                        ]
                    ),

                "mean_reward_entropy":
                    safe_mean(
                        subset[
                            "reward_entropy"
                        ]
                    ),

                "median_reward_entropy":
                    safe_median(
                        subset[
                            "reward_entropy"
                        ]
                    ),

                "mean_abs_reward":
                    safe_mean(
                        subset[
                            "abs_mean_reward"
                        ]
                    ),

                "median_abs_reward":
                    safe_median(
                        subset[
                            "abs_mean_reward"
                        ]
                    ),

                "mean_ci_width":
                    safe_mean(
                        subset[
                            "ci95_width"
                        ]
                    ),

                "fraction_ci_contains_zero":
                    float(
                        subset[
                            "ci95_contains_zero"
                        ].mean()
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
# Variance decomposition
# ============================================================

def analyze_variance_decomposition(
    position_df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("REWARD VARIANCE DECOMPOSITION")
    print("=" * 70)

    rows = []

    for threshold in REPETITION_THRESHOLDS:

        subset = position_df[
            position_df[
                "n"
            ]
            >= threshold
        ].copy()

        if len(
            subset
        ) == 0:

            continue

        weights = (
            subset[
                "n"
            ]
            / subset[
                "n"
            ].sum()
        )

        means = subset[
            "mean_reward"
        ]

        weighted_mean = float(
            np.sum(
                weights
                * means
            )
        )

        between_variance = float(
            np.sum(
                weights
                * (
                    means
                    - weighted_mean
                )
                ** 2
            )
        )

        within_variance = float(
            np.sum(
                weights
                * subset[
                    "reward_variance"
                ]
            )
        )

        total_variance = (
            between_variance
            + within_variance
        )

        explained_fraction = (
            between_variance
            / total_variance
            if total_variance > 0.0
            else np.nan
        )

        rows.append(
            {
                "min_n":
                    threshold,

                "positions":
                    len(
                        subset
                    ),

                "weighted_mean_reward":
                    weighted_mean,

                "between_position_variance":
                    between_variance,

                "within_position_variance":
                    within_variance,

                "total_variance":
                    total_variance,

                "fraction_explained_by_position":
                    explained_fraction,

                "fraction_within_position":
                    (
                        1.0
                        - explained_fraction
                        if np.isfinite(
                            explained_fraction
                        )
                        else np.nan
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
# Position-level uncertainty relationships
# ============================================================

def analyze_position_correlations(
    position_df: pd.DataFrame,
    min_observations: int,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("POSITION-LEVEL UNCERTAINTY / REWARD RELATIONSHIPS")
    print("=" * 70)

    repeated = position_df[
        position_df[
            "n"
        ]
        >= min_observations
    ].copy()

    rows = []

    for signal in SIGNALS:

        signal_column = (
            f"{signal}_mean"
        )

        for target in [
            "reward_entropy",
            "abs_mean_reward",
        ]:

            mask = (
                np.isfinite(
                    repeated[
                        signal_column
                    ]
                )
                & np.isfinite(
                    repeated[
                        target
                    ]
                )
            )

            rows.append(
                {
                    "signal":
                        signal,

                    "target":
                        target,

                    "n_pairs":
                        int(
                            mask.sum()
                        ),

                    "coverage":
                        (
                            float(
                                mask.mean()
                            )
                            if len(
                                repeated
                            )
                            else np.nan
                        ),

                    "pearson":
                        pearson_corr(
                            repeated[
                                signal_column
                            ],
                            repeated[
                                target
                            ],
                        ),

                    "spearman":
                        spearman_corr(
                            repeated[
                                signal_column
                            ],
                            repeated[
                                target
                            ],
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
                f"{x:+.6f}",
        )
    )

    return result


# ============================================================
# Plots
# ============================================================

def make_plots(
    position_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    output_dir: Path,
    min_observations: int,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repeated = position_df[
        position_df[
            "n"
        ]
        >= min_observations
    ].copy()

    # --------------------------------------------------------
    # Reward entropy vs repetition
    # --------------------------------------------------------

    plt.figure()

    plt.scatter(
        repeated[
            "n"
        ],
        repeated[
            "reward_entropy"
        ],
        alpha=0.35,
    )

    plt.xscale(
        "log"
    )

    plt.xlabel(
        "Observations per FEN"
    )

    plt.ylabel(
        "Reward entropy H_R(s) [bits]"
    )

    plt.title(
        "Reward ambiguity across repeated positions"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "reward_entropy_vs_repetition.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # U vs reward entropy
    # --------------------------------------------------------

    valid = repeated[
        np.isfinite(
            repeated[
                "U_mean"
            ]
        )
        & np.isfinite(
            repeated[
                "reward_entropy"
            ]
        )
    ]

    plt.figure()

    if len(
        valid
    ):

        plt.scatter(
            valid[
                "U_mean"
            ],
            valid[
                "reward_entropy"
            ],
            alpha=0.35,
        )

    plt.xlabel(
        "Mean value uncertainty U"
    )

    plt.ylabel(
        "Reward entropy H_R(s) [bits]"
    )

    plt.title(
        "Reward ambiguity vs value uncertainty"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "reward_entropy_vs_U.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # U vs absolute mean reward
    # --------------------------------------------------------

    valid = repeated[
        np.isfinite(
            repeated[
                "U_mean"
            ]
        )
        & np.isfinite(
            repeated[
                "abs_mean_reward"
            ]
        )
    ]

    plt.figure()

    if len(
        valid
    ):

        plt.scatter(
            valid[
                "U_mean"
            ],
            valid[
                "abs_mean_reward"
            ],
            alpha=0.35,
        )

    plt.xlabel(
        "Mean value uncertainty U"
    )

    plt.ylabel(
        "|E[R | s]|"
    )

    plt.title(
        "Outcome decisiveness vs value uncertainty"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "absolute_mean_reward_vs_U.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Variance decomposition
    # --------------------------------------------------------

    if len(
        variance_df
    ):

        plt.figure()

        plt.plot(
            variance_df[
                "min_n"
            ],
            variance_df[
                "fraction_explained_by_position"
            ],
            marker="o",
        )

        plt.xlabel(
            "Minimum observations per FEN"
        )

        plt.ylabel(
            "Between-position variance fraction"
        )

        plt.ylim(
            0.0,
            1.0,
        )

        plt.title(
            "Reward variance attributable to state identity"
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / "reward_variance_decomposition.png",
            dpi=200,
        )

        plt.close()


# ============================================================
# Report
# ============================================================

def build_report(
    input_path: Path,
    global_stats: dict,
    signal_distribution_df: pd.DataFrame,
    observation_corr_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    side_df: pd.DataFrame,
    position_df: pd.DataFrame,
    start_stats: dict | None,
    repeated_summary: dict,
    strata_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    position_corr_df: pd.DataFrame,
    min_observations: int,
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
        "="
        * 70
    )

    add(
        "ALBERTA - REWARD SIGNAL ANALYSIS"
    )

    add(
        "="
        * 70
    )

    add()

    add(
        f"Input: {input_path}"
    )

    add(
        f"Observations: {global_stats['n']:,}"
    )

    add(
        f"Unique FENs: {len(position_df):,}"
    )

    add()

    add(
        "REWARD DEFINITION"
    )

    add(
        "-"
        * 70
    )

    add(
        "Reward is expressed from the perspective of the "
        "player to move in the recorded FEN:"
    )

    add(
        "+1 = eventual win, 0 = draw, -1 = eventual loss."
    )

    add()

    add(
        "GLOBAL REWARD"
    )

    add(
        "-"
        * 70
    )

    add(
        f"Wins:   {global_stats['wins']:,}"
    )

    add(
        f"Draws:  {global_stats['draws']:,}"
    )

    add(
        f"Losses: {global_stats['losses']:,}"
    )

    add(
        f"H(R):   {global_stats['entropy']:.6f} bits"
    )

    add(
        f"E[R]:   {global_stats['mean_reward']:+.6f}"
    )

    add(
        f"Var(R): {global_stats['variance']:.6f}"
    )

    add()

    add(
        "SIGNAL DISTRIBUTIONS"
    )

    add(
        "-"
        * 70
    )

    add(
        signal_distribution_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    add(
        "OBSERVATION-LEVEL CORRELATIONS"
    )

    add(
        "-"
        * 70
    )

    add(
        observation_corr_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.6f}",
        )
    )

    add()

    add(
        "SIGNAL REDUNDANCY"
    )

    add(
        "-"
        * 70
    )

    add(
        redundancy_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.6f}",
        )
    )

    add()

    add(
        "SIDE-TO-MOVE ANALYSIS"
    )

    add(
        "-"
        * 70
    )

    add(
        side_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    add(
        f"REPEATED POSITIONS (n >= {min_observations})"
    )

    add(
        "-"
        * 70
    )

    for key, value in repeated_summary.items():

        add(
            f"{key}: {value}"
        )

    add()

    if start_stats is not None:

        add(
            "STARTING POSITION"
        )

        add(
            "-"
            * 70
        )

        for key, value in start_stats.items():

            add(
                f"{key}: {value}"
            )

        add()

    add(
        "REPETITION STRATA"
    )

    add(
        "-"
        * 70
    )

    add(
        "These rows describe different subsets of increasingly "
        "repeated positions; they are not a controlled sample-size "
        "experiment."
    )

    add()

    add(
        strata_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    add(
        "REWARD VARIANCE DECOMPOSITION"
    )

    add(
        "-"
        * 70
    )

    add(
        variance_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    add(
        "POSITION-LEVEL UNCERTAINTY RELATIONSHIPS"
    )

    add(
        "-"
        * 70
    )

    add(
        position_corr_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.6f}",
        )
    )

    add()

    add(
        "QUANTILE ANALYSIS"
    )

    add(
        "-"
        * 70
    )

    add(
        quantile_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    add()

    add(
        "INTERPRETATION NOTES"
    )

    add(
        "-"
        * 70
    )

    add(
        "1. H_R(s) = H(R | s) denotes terminal reward entropy "
        "and must not be confused with policy entropy H."
    )

    add(
        "2. Missing H/U/HU measurements never remove an otherwise "
        "valid reward observation."
    )

    add(
        "3. Signal correlations are descriptive and do not establish "
        "causality."
    )

    add(
        "4. U- and HU-dependent results must be regenerated after "
        "changing the league uncertainty estimator."
    )

    add(
        "5. Reward ambiguity does not by itself demonstrate that "
        "terminal reward noise causes PPO failure."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.min_observations < 1:

        raise ValueError(
            "--min-observations must be >= 1."
        )

    if args.quantiles < 2:

        raise ValueError(
            "--quantiles must be >= 2."
        )

    args.output_dir.mkdir(
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
    # Observation-level analyses
    # --------------------------------------------------------

    global_stats = analyze_global_reward(
        df
    )

    signal_distribution_df = (
        analyze_signal_distributions(
            df
        )
    )

    observation_corr_df = (
        analyze_observation_correlations(
            df
        )
    )

    quantile_df = (
        analyze_signal_quantiles(
            df,
            args.quantiles,
        )
    )

    redundancy_df = (
        analyze_signal_redundancy(
            df
        )
    )

    side_df = (
        analyze_side_to_move(
            df
        )
    )

    # --------------------------------------------------------
    # Position-level analyses
    # --------------------------------------------------------

    position_df = build_position_table(
        df
    )

    start_stats = (
        analyze_starting_position(
            position_df
        )
    )

    repeated = (
        analyze_repeated_positions(
            position_df,
            args.min_observations,
        )
    )

    strata_df = (
        analyze_repetition_strata(
            position_df
        )
    )

    variance_df = (
        analyze_variance_decomposition(
            position_df
        )
    )

    position_corr_df = (
        analyze_position_correlations(
            position_df,
            args.min_observations,
        )
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    make_plots(
        position_df,
        variance_df,
        args.output_dir,
        args.min_observations,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        input_path=args.input,
        global_stats=global_stats,
        signal_distribution_df=signal_distribution_df,
        observation_corr_df=observation_corr_df,
        quantile_df=quantile_df,
        redundancy_df=redundancy_df,
        side_df=side_df,
        position_df=position_df,
        start_stats=start_stats,
        repeated_summary=repeated["summary"],
        strata_df=strata_df,
        variance_df=variance_df,
        position_corr_df=position_corr_df,
        min_observations=args.min_observations,
    )

    report_path = (
        args.output_dir
        / "reward_signal_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()