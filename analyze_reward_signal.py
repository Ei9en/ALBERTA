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
    / "uncertainty_stats_1-100.json"
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
    try:
        value = float(value)

        if not math.isfinite(value):
            return np.nan

        return value

    except (TypeError, ValueError):
        return np.nan


def result_to_reward(result):

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


def reward_entropy(rewards):

    rewards = np.asarray(rewards)

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


def percentile_table(series):

    series = pd.Series(series).dropna()

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
        result[f"p{p}"] = np.percentile(
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
    print("ALBERTA - FINAL PRE-AL REWARD SIGNAL ANALYSIS")
    print("=" * 70)

    print()
    print("Loading uncertainty statistics")
    print("-" * 70)

    print(f"File: {INPUT_FILE}")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
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

def build_dataframe(data):

    rows = []

    invalid_reward = 0
    invalid_signal = 0
    invalid_fen = 0

    for record in data:

        if not isinstance(record, dict):
            continue

        fen = record.get("fen")
        result = record.get("result")

        H = safe_float(record.get("H"))
        U = safe_float(record.get("U"))
        HU = safe_float(record.get("HU"))

        reward = result_to_reward(result)

        if not math.isfinite(reward):
            invalid_reward += 1
            continue

        if not all(
            math.isfinite(x)
            for x in (H, U, HU)
        ):
            invalid_signal += 1
            continue

        if not isinstance(fen, str):
            invalid_fen += 1
            continue

        fen_parts = fen.split()

        if len(fen_parts) < 2:
            invalid_fen += 1
            continue

        side_to_move = fen_parts[1]

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

    df = pd.DataFrame(rows)

    print()
    print("DATASET")
    print("-" * 70)

    print(
        f"Valid observations:       {len(df):,}"
    )

    print(
        f"Invalid rewards skipped:   {invalid_reward:,}"
    )

    print(
        f"Invalid signals skipped:   {invalid_signal:,}"
    )

    print(
        f"Invalid FENs skipped:       {invalid_fen:,}"
    )

    print(
        f"Unique FENs:               {df['fen'].nunique():,}"
    )

    return df


# ============================================================
# Result distribution
# ============================================================

def analyze_results(df):

    print()
    print("RESULT DISTRIBUTION")
    print("-" * 70)

    total = len(df)

    for reward, label in [
        (-1.0, "Loss"),
        (0.0, "Draw"),
        (1.0, "Win"),
    ]:

        count = int(
            (df["reward"] == reward).sum()
        )

        pct = 100.0 * count / total

        print(
            f"{label:<6}: "
            f"{count:>10,} "
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

def analyze_signal_distributions(df):

    print()
    print("SIGNAL DISTRIBUTIONS")
    print("-" * 70)

    for signal in SIGNALS:

        stats = percentile_table(
            df[signal]
        )

        print()
        print(signal)

        print(
            f"  min    : {stats['min']:.8f}"
        )

        print(
            f"  mean   : {stats['mean']:.8f}"
        )

        print(
            f"  std    : {stats['std']:.8f}"
        )

        print(
            f"  p01    : {stats['p1']:.8f}"
        )

        print(
            f"  p05    : {stats['p5']:.8f}"
        )

        print(
            f"  p10    : {stats['p10']:.8f}"
        )

        print(
            f"  p25    : {stats['p25']:.8f}"
        )

        print(
            f"  median : {stats['p50']:.8f}"
        )

        print(
            f"  p75    : {stats['p75']:.8f}"
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


# ============================================================
# Correlations
# ============================================================

def analyze_correlations(df):

    print()
    print("CORRELATIONS WITH REWARD")
    print("-" * 70)

    for signal in SIGNALS:

        pearson_reward = df[signal].corr(
            df["reward"],
            method="pearson",
        )

        spearman_reward = df[signal].corr(
            df["reward"],
            method="spearman",
        )

        pearson_abs = df[signal].corr(
            df["abs_reward"],
            method="pearson",
        )

        spearman_abs = df[signal].corr(
            df["abs_reward"],
            method="spearman",
        )

        print(
            f"{signal:<4} | "
            f"Pearson(R)={pearson_reward:+.4f} | "
            f"Spearman(R)={spearman_reward:+.4f} | "
            f"Pearson(|R|)={pearson_abs:+.4f} | "
            f"Spearman(|R|)={spearman_abs:+.4f}"
        )


# ============================================================
# Quantile analysis
# ============================================================

def analyze_quantiles(df):

    print()
    print("QUANTILE ANALYSIS")
    print("-" * 70)

    for signal in SIGNALS:

        print()
        print(signal)

        ranked = df[signal].rank(
            method="first"
        )

        quantile = pd.qcut(
            ranked,
            q=N_QUANTILES,
            labels=False,
        ) + 1

        temp = df.copy()
        temp["quantile"] = quantile

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

            win_pct = (
                100.0 * wins / len(subset)
            )

            draw_pct = (
                100.0 * draws / len(subset)
            )

            loss_pct = (
                100.0 * losses / len(subset)
            )

            mean_reward = subset[
                "reward"
            ].mean()

            mean_abs_reward = subset[
                "abs_reward"
            ].mean()

            entropy = reward_entropy(
                subset["reward"].values
            )

            print(
                f"Q{q:02d} | "
                f"N={len(subset):>7,} | "
                f"Win={win_pct:6.2f}% | "
                f"Draw={draw_pct:6.2f}% | "
                f"Loss={loss_pct:6.2f}% | "
                f"E[R]={mean_reward:+.4f} | "
                f"E[|R|]={mean_abs_reward:.4f} | "
                f"H(R)={entropy:.4f}"
            )


# ============================================================
# Top-percentile selection
# ============================================================

def analyze_top_selection(df):

    print()
    print("TOP-QUANTILE SELECTION POWER")
    print("-" * 70)

    thresholds = [
        0.50,
        0.25,
        0.10,
        0.05,
        0.01,
    ]

    for signal in SIGNALS:

        print()
        print(signal)

        for fraction in thresholds:

            threshold = df[signal].quantile(
                1.0 - fraction
            )

            subset = df[
                df[signal] >= threshold
            ]

            mean_reward = subset[
                "reward"
            ].mean()

            mean_abs_reward = subset[
                "abs_reward"
            ].mean()

            entropy = reward_entropy(
                subset["reward"].values
            )

            win_pct = 100.0 * (
                subset["reward"] == 1
            ).mean()

            draw_pct = 100.0 * (
                subset["reward"] == 0
            ).mean()

            loss_pct = 100.0 * (
                subset["reward"] == -1
            ).mean()

            print(
                f"Top {100*fraction:>5.1f}% | "
                f"N={len(subset):>7,} | "
                f"Threshold={threshold:.6f} | "
                f"E[R]={mean_reward:+.4f} | "
                f"E[|R|]={mean_abs_reward:.4f} | "
                f"H(R)={entropy:.4f} | "
                f"Win={win_pct:6.2f}% | "
                f"Draw={draw_pct:6.2f}% | "
                f"Loss={loss_pct:6.2f}%"
            )


# ============================================================
# H x U interaction
# ============================================================

def analyze_h_u_interaction(df):

    print()
    print("H × U INTERACTION")
    print("-" * 70)

    temp = df.copy()

    temp["H_bin"] = pd.qcut(
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

    temp["U_bin"] = pd.qcut(
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

            mean_reward = subset[
                "reward"
            ].mean()

            mean_abs_reward = subset[
                "abs_reward"
            ].mean()

            entropy = reward_entropy(
                subset["reward"].values
            )

            win_pct = 100.0 * (
                subset["reward"] == 1
            ).mean()

            draw_pct = 100.0 * (
                subset["reward"] == 0
            ).mean()

            loss_pct = 100.0 * (
                subset["reward"] == -1
            ).mean()

            print(
                f"H={h_bin:<6} "
                f"U={u_bin:<6} | "
                f"N={len(subset):>7,} | "
                f"E[R]={mean_reward:+.4f} | "
                f"E[|R|]={mean_abs_reward:.4f} | "
                f"H(R)={entropy:.4f} | "
                f"Win={win_pct:6.2f}% | "
                f"Draw={draw_pct:6.2f}% | "
                f"Loss={loss_pct:6.2f}%"
            )


# ============================================================
# HU redundancy
# ============================================================

def analyze_hu_redundancy(df):

    print()
    print("HU REDUNDANCY / INFORMATION")
    print("-" * 70)

    pairs = [
        ("H", "U"),
        ("H", "HU"),
        ("U", "HU"),
    ]

    for a, b in pairs:

        pearson = df[a].corr(
            df[b],
            method="pearson",
        )

        spearman = df[a].corr(
            df[b],
            method="spearman",
        )

        print(
            f"{a:<2} vs {b:<2} | "
            f"Pearson={pearson:+.4f} | "
            f"Spearman={spearman:+.4f}"
        )


# ============================================================
# Winner color
# ============================================================

def analyze_winner_color(df):

    print()
    print("WINNER COLOR ANALYSIS")
    print("-" * 70)

    total = len(df)

    white_wins = int(
        (df["reward"] == 1).sum()
    )

    draws = int(
        (df["reward"] == 0).sum()
    )

    black_wins = int(
        (df["reward"] == -1).sum()
    )

    for count, label in [
        (white_wins, "White win"),
        (draws, "Draw"),
        (black_wins, "Black win"),
    ]:

        pct = 100.0 * count / total

        print(
            f"{label:<10} | "
            f"N={count:>8,} | "
            f"{pct:6.2f}%"
        )

    decisive = white_wins + black_wins

    if decisive > 0:

        print()
        print("DECISIVE GAMES")

        print(
            f"White | "
            f"{100.0 * white_wins / decisive:6.2f}%"
        )

        print(
            f"Black | "
            f"{100.0 * black_wins / decisive:6.2f}%"
        )


# ============================================================
# Repeated positions
# ============================================================

def analyze_repeated_positions(df):

    print()
    print("REPEATED POSITION ANALYSIS")
    print("-" * 70)

    counts = df["fen"].value_counts()

    print(
        f"Unique FENs: "
        f"{len(counts):,}"
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

    repeated = counts[counts >= 2]

    if len(repeated) == 0:
        return

    print()
    print("MOST REPEATED POSITIONS")
    print("-" * 70)

    for fen, count in repeated.head(10).items():

        subset = df[
            df["fen"] == fen
        ]

        print(
            f"N={count:>5} | "
            f"H={subset['H'].mean():.5f} | "
            f"U={subset['U'].mean():.8f} | "
            f"HU={subset['HU'].mean():.8f} | "
            f"E[R]={subset['reward'].mean():+.4f} | "
            f"H(R)={reward_entropy(subset['reward'].values):.4f}"
        )

        print(
            f"    {fen}"
        )


# ============================================================
# Main
# ============================================================

def main():

    data = load_data()

    df = build_dataframe(data)

    if len(df) == 0:
        raise RuntimeError(
            "No valid observations."
        )

    analyze_results(df)

    analyze_signal_distributions(df)

    analyze_correlations(df)

    analyze_quantiles(df)

    analyze_top_selection(df)

    analyze_h_u_interaction(df)

    analyze_hu_redundancy(df)

    analyze_winner_color(df)

    analyze_repeated_positions(df)

    print()
    print("=" * 70)
    print("FINAL PRE-AL ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(
        f"Analyzed observations: {len(df):,}"
    )
    print(
        f"Input: {INPUT_FILE}"
    )
    print()


# ============================================================

if __name__ == "__main__":
    main()