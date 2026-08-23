from pathlib import Path
from collections import Counter

import json
import math

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reward_signal_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SIGNALS = [
    "H",
    "U",
    "HU",
]

N_QUANTILES = 10


# ============================================================
# Utilities
# ============================================================

def safe_float(value):
    """
    Convert a value to float.

    Returns NaN if conversion fails.
    """

    try:
        value = float(value)

        if not math.isfinite(value):
            return np.nan

        return value

    except (
        TypeError,
        ValueError,
    ):
        return np.nan


def result_to_reward(
    result,
):
    """
    Convert the game result into a fixed reward based on
    the winning color.

    White win:
        1-0 -> +1

    Draw:
        1/2-1/2 -> 0

    Black win:
        0-1 -> -1

    IMPORTANT:
        This reward is NOT from the perspective of the
        player to move in the FEN.

        It is a fixed encoding of the game outcome:
            White win = +1
            Draw      =  0
            Black win = -1
    """

    if result == "1-0":
        return 1.0

    if result == "0-1":
        return -1.0

    if result in (
        "1/2-1/2",
        "1/2",
        "draw",
        "Draw",
    ):
        return 0.0

    return np.nan


def reward_entropy(
    rewards,
):
    """
    Shannon entropy of the empirical reward distribution.

    Rewards are assumed to be:
        -1
         0
        +1

    Returned in bits.
    """

    rewards = np.asarray(
        rewards,
    )

    if len(rewards) == 0:
        return np.nan

    counts = Counter(
        rewards.astype(int)
    )

    total = len(rewards)

    entropy = 0.0

    for count in counts.values():

        p = count / total

        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def percentile_table(
    series,
):
    """
    Return standard descriptive statistics.
    """

    series = pd.Series(
        series
    ).dropna()

    if len(series) == 0:
        return {}

    percentiles = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    result = {
        "count": len(series),
        "min": series.min(),
        "mean": series.mean(),
        "std": series.std(),
        "max": series.max(),
    }

    for p in percentiles:

        result[
            f"p{p}"
        ] = np.percentile(
            series,
            p,
        )

    return result


# ============================================================
# Loading
# ============================================================

def load_data():
    print()
    print("=" * 70)
    print("ALBERTA - REWARD SIGNAL ANALYSIS")
    print("=" * 70)

    print()
    print("Loading uncertainty statistics")
    print("-" * 70)

    print(
        f"File: {INPUT_FILE}"
    )

    with open(
        INPUT_FILE,
        "r",
    ) as f:

        data = json.load(f)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Expected the JSON root to be a list."
        )

    print(
        f"Raw records: {len(data):,}"
    )

    return data


# ============================================================
# Build dataframe
# ============================================================

def build_dataframe(
    data,
):

    rows = []

    invalid_reward = 0
    invalid_signal = 0

    for record in data:

        if not isinstance(
            record,
            dict,
        ):
            continue

        fen = record.get(
            "fen"
        )

        result = record.get(
            "result"
        )

        H = safe_float(
            record.get("H")
        )

        U = safe_float(
            record.get("U")
        )

        HU = safe_float(
            record.get("HU")
        )

        reward = result_to_reward(
            result,
        )

        if not math.isfinite(
            reward
        ):
            invalid_reward += 1
            continue

        if not all(
            math.isfinite(x)
            for x in (
                H,
                U,
                HU,
            )
        ):
            invalid_signal += 1
            continue

        side_to_move = fen.split()[1]

        rows.append(
            {
                "fen": fen,
                "side_to_move": side_to_move,
                "result": result,
                "reward": reward,
                "abs_reward": abs(reward),
                "H": H,
                "U": U,
                "HU": HU,
            }
        )

    df = pd.DataFrame(
        rows
    )

    print()
    print("DATASET")
    print("-" * 70)

    print(
        f"Valid observations: {len(df):,}"
    )

    print(
        f"Invalid rewards skipped: {invalid_reward:,}"
    )

    print(
        f"Invalid signal records skipped: {invalid_signal:,}"
    )

    print(
        f"Unique FENs: {df['fen'].nunique():,}"
    )

    return df


# ============================================================
# Result distribution
# ============================================================

def print_result_distribution(
    df,
):

    print()
    print("RESULT DISTRIBUTION")
    print("-" * 70)

    counts = df[
        "reward"
    ].value_counts().sort_index()

    total = len(df)

    for reward, label in [
        (-1.0, "Loss"),
        (0.0, "Draw"),
        (1.0, "Win"),
    ]:

        count = int(
            counts.get(
                reward,
                0,
            )
        )

        pct = (
            100.0
            * count
            / total
        )

        print(
            f"{label:<6}: "
            f"{count:>8,} "
            f"({pct:6.2f}%)"
        )

    entropy = reward_entropy(
        df["reward"].values
    )

    print(
        f"\nReward entropy: "
        f"{entropy:.4f} bits"
    )

    print(
        f"E[R]: "
        f"{df['reward'].mean():+.6f}"
    )


# ============================================================
# Signal distributions
# ============================================================

def analyze_signal_distributions(
    df,
):

    print()
    print("SIGNAL DISTRIBUTIONS")
    print("-" * 70)

    rows = []

    for signal in SIGNALS:

        stats = percentile_table(
            df[signal]
        )

        stats[
            "signal"
        ] = signal

        rows.append(
            stats
        )

        print()
        print(
            f"{signal}"
        )

        print(
            f"  min    : {stats['min']:.8f}"
        )

        print(
            f"  mean   : {stats['mean']:.8f}"
        )

        print(
            f"  median : {stats['p50']:.8f}"
        )

        print(
            f"  p90    : {stats['p90']:.8f}"
        )

        print(
            f"  p95    : {stats['p95']:.8f}"
        )

        print(
            f"  p99    : {stats['p99']:.8f}"
        )

        print(
            f"  max    : {stats['max']:.8f}"
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_DIR
        / "signal_distributions.csv",
        index=False,
    )

    return output


# ============================================================
# Correlations
# ============================================================

def analyze_correlations(
    df,
):

    print()
    print("CORRELATIONS WITH REWARD")
    print("-" * 70)

    rows = []

    for signal in SIGNALS:

        pearson_reward = df[
            signal
        ].corr(
            df["reward"],
            method="pearson",
        )

        spearman_reward = df[
            signal
        ].corr(
            df["reward"],
            method="spearman",
        )

        pearson_abs = df[
            signal
        ].corr(
            df["abs_reward"],
            method="pearson",
        )

        spearman_abs = df[
            signal
        ].corr(
            df["abs_reward"],
            method="spearman",
        )

        rows.append(
            {
                "signal": signal,
                "pearson_reward":
                    pearson_reward,
                "spearman_reward":
                    spearman_reward,
                "pearson_abs_reward":
                    pearson_abs,
                "spearman_abs_reward":
                    spearman_abs,
            }
        )

        print(
            f"{signal:<4} | "
            f"Pearson(R)={pearson_reward:+.4f} | "
            f"Spearman(R)={spearman_reward:+.4f} | "
            f"Pearson(|R|)={pearson_abs:+.4f} | "
            f"Spearman(|R|)={spearman_abs:+.4f}"
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_DIR
        / "signal_correlations.csv",
        index=False,
    )

    return output


# ============================================================
# Quantile analysis
# ============================================================

def analyze_quantiles(
    df,
):

    print()
    print("QUANTILE ANALYSIS")
    print("-" * 70)

    all_rows = []

    for signal in SIGNALS:

        print()
        print(
            f"{signal}"
        )

        # rank(method="first") guarantees that qcut
        # can produce exactly N_QUANTILES even when
        # many values are identical.
        ranked = (
            df[signal]
            .rank(
                method="first"
            )
        )

        quantile = pd.qcut(
            ranked,
            q=N_QUANTILES,
            labels=False,
        ) + 1

        temp = df.copy()

        temp[
            "quantile"
        ] = quantile

        for q in range(
            1,
            N_QUANTILES + 1,
        ):

            subset = temp[
                temp["quantile"] == q
            ]

            if len(subset) == 0:
                continue

            wins = (
                subset["reward"] == 1
            ).sum()

            draws = (
                subset["reward"] == 0
            ).sum()

            losses = (
                subset["reward"] == -1
            ).sum()

            row = {
                "signal": signal,
                "quantile": q,
                "n": len(subset),
                "signal_mean":
                    subset[signal].mean(),
                "signal_median":
                    subset[signal].median(),
                "win_pct":
                    100.0
                    * wins
                    / len(subset),
                "draw_pct":
                    100.0
                    * draws
                    / len(subset),
                "loss_pct":
                    100.0
                    * losses
                    / len(subset),
                "mean_reward":
                    subset["reward"].mean(),
                "mean_abs_reward":
                    subset["abs_reward"].mean(),
                "reward_entropy":
                    reward_entropy(
                        subset[
                            "reward"
                        ].values
                    ),
            }

            all_rows.append(
                row
            )

            print(
                f"Q{q:02d} | "
                f"N={len(subset):>7,} | "
                f"Win={row['win_pct']:6.2f}% | "
                f"Draw={row['draw_pct']:6.2f}% | "
                f"Loss={row['loss_pct']:6.2f}% | "
                f"E[R]={row['mean_reward']:+.4f} | "
                f"E[|R|]={row['mean_abs_reward']:.4f}"
            )

    output = pd.DataFrame(
        all_rows
    )

    output.to_csv(
        OUTPUT_DIR
        / "signal_quantiles.csv",
        index=False,
    )

    return output


# ============================================================
# Top-percentile selection
# ============================================================

def analyze_top_selection(
    df,
):

    print()
    print("TOP-QUANTILE SELECTION POWER")
    print("-" * 70)

    rows = []

    thresholds = [
        0.50,
        0.25,
        0.10,
        0.05,
        0.01,
    ]

    for signal in SIGNALS:

        print()
        print(
            signal
        )

        for fraction in thresholds:

            threshold = df[
                signal
            ].quantile(
                1.0 - fraction
            )

            subset = df[
                df[signal] >= threshold
            ]

            mean_abs_reward = subset[
                "abs_reward"
            ].mean()

            entropy = reward_entropy(
                subset[
                    "reward"
                ].values
            )

            row = {
                "signal": signal,
                "fraction": fraction,
                "threshold": threshold,
                "n": len(subset),
                "mean_reward":
                    subset["reward"].mean(),
                "mean_abs_reward":
                    mean_abs_reward,
                "reward_entropy":
                    entropy,
                "win_pct":
                    100.0
                    * (
                        subset["reward"] == 1
                    ).mean(),
                "draw_pct":
                    100.0
                    * (
                        subset["reward"] == 0
                    ).mean(),
                "loss_pct":
                    100.0
                    * (
                        subset["reward"] == -1
                    ).mean(),
            }

            rows.append(
                row
            )

            print(
                f"Top {100*fraction:>5.1f}% | "
                f"N={len(subset):>7,} | "
                f"E[|R|]={mean_abs_reward:.4f} | "
                f"H(R)={entropy:.4f}"
            )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_DIR
        / "top_quantile_selection.csv",
        index=False,
    )

    return output


# ============================================================
# H x U interaction
# ============================================================

def analyze_h_u_interaction(
    df,
):

    print()
    print("H × U INTERACTION")
    print("-" * 70)

    temp = df.copy()

    temp[
        "H_bin"
    ] = pd.qcut(
        temp["H"].rank(
            method="first"
        ),
        q=3,
        labels=[
            "low",
            "medium",
            "high",
        ],
    )

    temp[
        "U_bin"
    ] = pd.qcut(
        temp["U"].rank(
            method="first"
        ),
        q=3,
        labels=[
            "low",
            "medium",
            "high",
        ],
    )

    rows = []

    for h_bin in [
        "low",
        "medium",
        "high",
    ]:

        for u_bin in [
            "low",
            "medium",
            "high",
        ]:

            subset = temp[
                (temp["H_bin"] == h_bin)
                &
                (temp["U_bin"] == u_bin)
            ]

            if len(subset) == 0:
                continue

            row = {
                "H_bin": h_bin,
                "U_bin": u_bin,
                "n": len(subset),
                "mean_H":
                    subset["H"].mean(),
                "mean_U":
                    subset["U"].mean(),
                "mean_HU":
                    subset["HU"].mean(),
                "mean_reward":
                    subset["reward"].mean(),
                "mean_abs_reward":
                    subset["abs_reward"].mean(),
                "reward_entropy":
                    reward_entropy(
                        subset[
                            "reward"
                        ].values
                    ),
                "win_pct":
                    100.0
                    * (
                        subset["reward"] == 1
                    ).mean(),
                "draw_pct":
                    100.0
                    * (
                        subset["reward"] == 0
                    ).mean(),
                "loss_pct":
                    100.0
                    * (
                        subset["reward"] == -1
                    ).mean(),
            }

            rows.append(
                row
            )

    output = pd.DataFrame(
        rows
    )

    print()

    for _, row in output.iterrows():

        print(
            f"H={row['H_bin']:<6} "
            f"U={row['U_bin']:<6} | "
            f"N={int(row['n']):>7,} | "
            f"E[R]={row['mean_reward']:+.4f} | "
            f"E[|R|]={row['mean_abs_reward']:.4f} | "
            f"H(R)={row['reward_entropy']:.4f}"
        )

    output.to_csv(
        OUTPUT_DIR
        / "h_u_interaction.csv",
        index=False,
    )

    return output


# ============================================================
# HU validation
# ============================================================

def analyze_hu_redundancy(
    df,
):

    print()
    print("HU REDUNDANCY / INFORMATION")
    print("-" * 70)

    rows = []

    pairs = [
        ("H", "U"),
        ("H", "HU"),
        ("U", "HU"),
    ]

    for a, b in pairs:

        pearson = df[
            a
        ].corr(
            df[b],
            method="pearson",
        )

        spearman = df[
            a
        ].corr(
            df[b],
            method="spearman",
        )

        rows.append(
            {
                "signal_a": a,
                "signal_b": b,
                "pearson": pearson,
                "spearman": spearman,
            }
        )

        print(
            f"{a:<2} vs {b:<2} | "
            f"Pearson={pearson:+.4f} | "
            f"Spearman={spearman:+.4f}"
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_DIR
        / "signal_redundancy.csv",
        index=False,
    )

    return output


# ============================================================
# Side-to-move sanity check
# ============================================================

def analyze_winner_color(
    df,
):
    """
    Analyze the empirical distribution of game outcomes
    by winning color.

    This does NOT assume that the player to move is the agent.

    reward encoding:
        +1 = White win
         0 = Draw
        -1 = Black win
    """

    print()
    print("WINNER COLOR ANALYSIS")
    print("-" * 70)

    rows = []

    total = len(df)

    white_wins = (
        df["reward"] == 1
    ).sum()

    draws = (
        df["reward"] == 0
    ).sum()

    black_wins = (
        df["reward"] == -1
    ).sum()

    outcomes = [
        (1.0, "White win"),
        (0.0, "Draw"),
        (-1.0, "Black win"),
    ]

    for reward, label in outcomes:

        count = int(
            (
                df["reward"] == reward
            ).sum()
        )

        pct = (
            100.0
            * count
            / total
        )

        rows.append(
            {
                "outcome": label,
                "count": count,
                "percentage": pct,
            }
        )

        print(
            f"{label:<10} | "
            f"N={count:>8,} | "
            f"{pct:6.2f}%"
        )

    print()

    decisive = white_wins + black_wins

    if decisive > 0:

        white_decisive_pct = (
            100.0
            * white_wins
            / decisive
        )

        black_decisive_pct = (
            100.0
            * black_wins
            / decisive
        )

        print(
            "DECISIVE GAMES"
        )
        print(
            f"White | "
            f"{white_decisive_pct:6.2f}%"
        )

        print(
            f"Black | "
            f"{black_decisive_pct:6.2f}%"
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_DIR
        / "winner_color.csv",
        index=False,
    )

    return output

# ============================================================
# Repeated FEN analysis
# ============================================================

def analyze_repeated_positions(
    df,
):

    print()
    print("REPEATED POSITION ANALYSIS")
    print("-" * 70)

    counts = (
        df[
            "fen"
        ]
        .value_counts()
    )

    print(
        f"Unique FENs: {len(counts):,}"
    )

    print(
        f"FENs seen >= 2 times: "
        f"{(counts >= 2).sum():,}"
    )

    print(
        f"FENs seen >= 5 times: "
        f"{(counts >= 5).sum():,}"
    )

    print(
        f"FENs seen >= 10 times: "
        f"{(counts >= 10).sum():,}"
    )

    rows = []

    repeated_fens = counts[
        counts >= 2
    ].index

    for fen in repeated_fens:

        subset = df[
            df["fen"] == fen
        ]

        rows.append(
            {
                "fen": fen,
                "n": len(subset),
                "mean_H":
                    subset["H"].mean(),
                "mean_U":
                    subset["U"].mean(),
                "mean_HU":
                    subset["HU"].mean(),
                "mean_reward":
                    subset["reward"].mean(),
                "reward_entropy":
                    reward_entropy(
                        subset[
                            "reward"
                        ].values
                    ),
                "win_pct":
                    100.0
                    * (
                        subset["reward"] == 1
                    ).mean(),
                "draw_pct":
                    100.0
                    * (
                        subset["reward"] == 0
                    ).mean(),
                "loss_pct":
                    100.0
                    * (
                        subset["reward"] == -1
                    ).mean(),
            }
        )

    output = pd.DataFrame(
        rows
    )

    output = output.sort_values(
        "n",
        ascending=False,
    )

    output.to_csv(
        OUTPUT_DIR
        / "repeated_positions.csv",
        index=False,
    )

    return output


# ============================================================
# Main
# ============================================================

def main():

    data = load_data()

    df = build_dataframe(
        data
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid observations."
        )

    print_result_distribution(
        df
    )

    analyze_signal_distributions(
        df
    )

    analyze_correlations(
        df
    )

    analyze_quantiles(
        df
    )

    analyze_top_selection(
        df
    )

    analyze_h_u_interaction(
        df
    )

    analyze_hu_redundancy(
        df
    )

    analyze_winner_color(
        df
    )

    analyze_repeated_positions(
        df
    )

    # --------------------------------------------------------
    # Save cleaned observations
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_DIR
        / "all_observations.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results written to:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================

if __name__ == "__main__":
    main()