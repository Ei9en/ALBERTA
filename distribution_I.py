import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats_1-10.json"
)


# ============================================================
# Weights
# ============================================================

W_H = 0
W_U = 1
W_HU = 0


# ============================================================
# Percentile-rank normalization
# ============================================================

def percentile_rank(values):
    """
    Convert values to empirical percentile ranks in [0, 1].

    Ties receive the same mid-rank.

    This function is used independently for White-to-move and
    Black-to-move positions.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n < 2:
        raise ValueError(
            "At least two values are required."
        )

    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[order]

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        rank = (
            (start + end - 1)
            / 2.0
        )

        percentile = (
            rank
            / (n - 1)
        )

        ranks[order[start:end]] = percentile

        start = end

    return ranks


# ============================================================
# Extract side to move
# ============================================================

def extract_side_to_move(fens):

    sides = []

    for fen in fens:

        try:

            side = fen.split()[1]

            if side not in ("w", "b"):
                raise ValueError

            sides.append(side)

        except (
            IndexError,
            ValueError,
        ):

            raise ValueError(
                f"Invalid FEN: {fen}"
            )

    return np.array(
        sides,
        dtype="<U1",
    )


# ============================================================
# Normalize one signal side-aware
# ============================================================

def normalize_side_aware(
    values,
    white_mask,
    black_mask,
):
    """
    Empirical percentile-rank normalization performed
    independently for White-to-move and Black-to-move.
    """

    normalized = np.empty_like(
        values,
        dtype=np.float64,
    )

    normalized[white_mask] = percentile_rank(
        values[white_mask]
    )

    normalized[black_mask] = percentile_rank(
        values[black_mask]
    )

    return normalized


# ============================================================
# Min-max normalization
# ============================================================

def min_max_normalize(values):
    """
    Normalize values to [0, 1]:

        x_norm = (x - xmin) / (xmax - xmin)

    This is a strictly increasing affine transformation as long
    as xmax > xmin, therefore it preserves the ranking of all
    distinct values.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    xmin = np.min(values)
    xmax = np.max(values)

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise RuntimeError(
            "ERROR: non-finite I_min or I_max."
        )

    if xmax <= xmin:
        raise RuntimeError(
            "ERROR: I_max must be strictly greater than I_min."
        )

    normalized = (
        values - xmin
    ) / (
        xmax - xmin
    )

    # Numerical safety.
    #
    # The theoretical range is exactly [0, 1], but floating-point
    # arithmetic can produce values such as 1.0000000000000002.
    # Clipping does not alter the ranking.
    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return normalized, xmin, xmax


# ============================================================
# Ranking verification
# ============================================================

def verify_ranking_preserved(
    original,
    normalized,
):
    """
    Verify that min-max normalization preserves ordering.

    We do NOT compare argsort arrays directly because:
      - ties can have arbitrary stable ordering;
      - floating-point roundoff can affect exact equality.

    Instead, we verify the monotonic relationship:

        x_i < x_j  =>  f(x_i) <= f(x_j)

    and:

        x_i == x_j  =>  f(x_i) == f(x_j)

    For the present affine transformation, this should always
    hold up to floating-point tolerance.

    The implementation checks the ordering through sorting.
    """

    original = np.asarray(
        original,
        dtype=np.float64,
    )

    normalized = np.asarray(
        normalized,
        dtype=np.float64,
    )

    if len(original) != len(normalized):
        return False

    order = np.argsort(
        original,
        kind="stable",
    )

    sorted_original = original[order]
    sorted_normalized = normalized[order]

    # --------------------------------------------------------
    # Differences between consecutive original values.
    # --------------------------------------------------------

    original_diff = np.diff(
        sorted_original
    )

    normalized_diff = np.diff(
        sorted_normalized
    )

    # --------------------------------------------------------
    # Where the original values are strictly increasing,
    # normalized values must never decrease.
    #
    # We use a tolerance scaled to the numerical precision
    # of the normalized interval.
    # --------------------------------------------------------

    strict_mask = (
        original_diff
        > 0.0
    )

    if np.any(strict_mask):

        if np.any(
            normalized_diff[strict_mask]
            < -1e-14
        ):
            return False

    # --------------------------------------------------------
    # Where original values are exactly equal, the affine
    # transformation must give exactly equal results.
    # --------------------------------------------------------

    equal_mask = (
        original_diff
        == 0.0
    )

    if np.any(equal_mask):

        if np.any(
            np.abs(
                normalized_diff[equal_mask]
            )
            > 1e-14
        ):
            return False

    return True


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "ALBERTA - ACTIVE LEARNING SCORE ANALYSIS"
    )
    print("=" * 70)

    # ========================================================
    # Load data
    # ========================================================

    print()
    print(
        "Loading uncertainty statistics"
    )
    print("-" * 70)

    print(
        f"File: {DATA_FILE}"
    )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    print(
        f"Raw records: {len(data):,}"
    )

    # ========================================================
    # Extract signals
    # ========================================================

    fens = np.array(
        [
            record["fen"]
            for record in data
        ],
        dtype=object,
    )

    H = np.array(
        [
            record["H"]
            for record in data
        ],
        dtype=np.float64,
    )

    U = np.array(
        [
            record["U"]
            for record in data
        ],
        dtype=np.float64,
    )

    HU = np.array(
        [
            record["HU"]
            for record in data
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Validate raw data
    # ========================================================

    if not (
        np.all(np.isfinite(H))
        and np.all(np.isfinite(U))
        and np.all(np.isfinite(HU))
    ):

        raise RuntimeError(
            "ERROR: H, U or HU contains non-finite values."
        )

    sides = extract_side_to_move(
        fens
    )

    white_mask = (
        sides == "w"
    )

    black_mask = (
        sides == "b"
    )

    # ========================================================
    # Side statistics
    # ========================================================

    print()
    print(
        "SIDE TO MOVE"
    )
    print("-" * 70)

    print(
        f"White : "
        f"{np.sum(white_mask):,}"
    )

    print(
        f"Black : "
        f"{np.sum(black_mask):,}"
    )

    # ========================================================
    # Raw signal statistics
    # ========================================================

    print()
    print(
        "RAW SIGNALS"
    )
    print("-" * 70)

    for name, values in [
        ("H", H),
        ("U", U),
        ("HU", HU),
    ]:

        print()
        print(name)

        print(
            f"White mean   : "
            f"{np.mean(values[white_mask]):.9f}"
        )

        print(
            f"Black mean   : "
            f"{np.mean(values[black_mask]):.9f}"
        )

        print(
            f"White median : "
            f"{np.median(values[white_mask]):.9f}"
        )

        print(
            f"Black median : "
            f"{np.median(values[black_mask]):.9f}"
        )

        print(
            f"White max    : "
            f"{np.max(values[white_mask]):.9f}"
        )

        print(
            f"Black max    : "
            f"{np.max(values[black_mask]):.9f}"
        )

    # ========================================================
    # Side-aware percentile normalization
    # ========================================================

    print()
    print(
        "SIDE-AWARE PERCENTILE NORMALIZATION"
    )
    print("-" * 70)

    H_norm = normalize_side_aware(
        H,
        white_mask,
        black_mask,
    )

    U_norm = normalize_side_aware(
        U,
        white_mask,
        black_mask,
    )

    HU_norm = normalize_side_aware(
        HU,
        white_mask,
        black_mask,
    )

    print(
        "Normalization completed."
    )

    # ========================================================
    # Normalized signal diagnostics
    # ========================================================

    print()
    print(
        "NORMALIZED SIGNALS"
    )
    print("-" * 70)

    for name, values in [
        ("H", H_norm),
        ("U", U_norm),
        ("HU", HU_norm),
    ]:

        print(
            f"{name:<3} "
            f"mean={np.mean(values):.6f} | "
            f"median={np.median(values):.6f} | "
            f"p90={np.percentile(values, 90):.6f} | "
            f"p99={np.percentile(values, 99):.6f} | "
            f"max={np.max(values):.6f}"
        )

    # ========================================================
    # Colour balance after normalization
    # ========================================================

    print()
    print(
        "COLOUR BALANCE AFTER NORMALIZATION"
    )
    print("-" * 70)

    for name, values in [
        ("H", H_norm),
        ("U", U_norm),
        ("HU", HU_norm),
    ]:

        white_mean = np.mean(
            values[white_mask]
        )

        black_mean = np.mean(
            values[black_mask]
        )

        white_median = np.median(
            values[white_mask]
        )

        black_median = np.median(
            values[black_mask]
        )

        print()
        print(name)

        print(
            f"White mean   : "
            f"{white_mean:.6f}"
        )

        print(
            f"Black mean   : "
            f"{black_mean:.6f}"
        )

        print(
            f"White median : "
            f"{white_median:.6f}"
        )

        print(
            f"Black median : "
            f"{black_median:.6f}"
        )

        if black_mean > 0:

            print(
                f"Mean ratio W/B : "
                f"{white_mean / black_mean:.3f}x"
            )

    # ========================================================
    # Raw active-learning score
    # ========================================================
    #
    # No W_0 anymore.
    #
    # I_raw = W_H * H_norm
    #       + W_U * U_norm
    #       + W_HU * HU_norm
    #
    # Note that W_H is negative.
    # ========================================================

    I_raw = (
        W_H * H_norm
        +
        W_U * U_norm
        +
        W_HU * HU_norm
    )

    if not np.all(
        np.isfinite(I_raw)
    ):

        raise RuntimeError(
            "ERROR: raw I contains non-finite values."
        )

    # ========================================================
    # Raw I range
    # ========================================================

    I_min = np.min(
        I_raw
    )

    I_max = np.max(
        I_raw
    )

    print()
    print(
        "RAW I RANGE"
    )
    print("-" * 70)

    print(
        f"I_min : "
        f"{I_min:.12f}"
    )

    print(
        f"I_max : "
        f"{I_max:.12f}"
    )

    # ========================================================
    # Min-max normalization
    # ========================================================
    #
    # I_norm = (I_raw - I_min)
    #          / (I_max - I_min)
    #
    # This changes the numerical scale but NOT the ranking.
    # ========================================================

    I, I_min_check, I_max_check = min_max_normalize(
        I_raw
    )

    # ========================================================
    # Ranking preservation
    # ========================================================

    ranking_preserved = verify_ranking_preserved(
        I_raw,
        I,
    )

    print()
    print(
        "MIN-MAX NORMALIZATION"
    )
    print("-" * 70)

    print(
        f"I_min : "
        f"{I_min_check:.12f}"
    )

    print(
        f"I_max : "
        f"{I_max_check:.12f}"
    )

    print(
        f"Normalized min : "
        f"{np.min(I):.12f}"
    )

    print(
        f"Normalized max : "
        f"{np.max(I):.12f}"
    )

    print(
        f"Ranking preserved : "
        f"{ranking_preserved}"
    )

    if not ranking_preserved:

        raise RuntimeError(
            "ERROR: min-max normalization failed the "
            "monotonicity check."
        )

    # ========================================================
    # Score sanity check
    # ========================================================

    if (
        np.min(I) < 0.0
        or np.max(I) > 1.0
    ):

        raise RuntimeError(
            "ERROR: normalized I is outside [0, 1]."
        )

    # ========================================================
    # Saturation diagnostic
    # ========================================================

    exact_one = np.sum(
        I == 1.0
    )

    near_one = np.sum(
        I >= 0.999
    )

    print()
    print(
        "SATURATION DIAGNOSTIC"
    )
    print("-" * 70)

    print(
        f"I == 1.000000 : "
        f"{exact_one:,}"
    )

    print(
        f"I >= 0.999000 : "
        f"{near_one:,}"
    )

    # ========================================================
    # Weights
    # ========================================================

    print()
    print(
        "WEIGHTS"
    )
    print("-" * 70)

    print(
        f"H   : {W_H:.4f}"
    )

    print(
        f"U   : {W_U:.4f}"
    )

    print(
        f"HU  : {W_HU:.4f}"
    )

    print(
        f"Sum : "
        f"{W_H + W_U + W_HU:.4f}"
    )

    # ========================================================
    # I distribution
    # ========================================================

    print()
    print(
        "I SCORE DISTRIBUTION"
    )
    print("-" * 70)

    percentiles = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        97.5,
        99,
        99.5,
        99.9,
        99.99,
        100,
    ]

    for percentile in percentiles:

        value = np.percentile(
            I,
            percentile,
        )

        print(
            f"P{percentile:<6} : "
            f"{value:.9f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "SUMMARY"
    )
    print("-" * 70)

    print(
        f"N      : "
        f"{len(I):,}"
    )

    print(
        f"Mean   : "
        f"{np.mean(I):.9f}"
    )

    print(
        f"Std    : "
        f"{np.std(I):.9f}"
    )

    print(
        f"Median : "
        f"{np.median(I):.9f}"
    )

    print(
        f"Min    : "
        f"{np.min(I):.9f}"
    )

    print(
        f"Max    : "
        f"{np.max(I):.9f}"
    )

    # ========================================================
    # Candidate thresholds
    # ========================================================

    print()
    print(
        "ACTIVE LEARNING BUDGET THRESHOLDS"
    )
    print("-" * 70)

    for budget in [
        0.10,
        0.05,
        0.02,
        0.01,
        0.005,
        0.0025,
        0.001,
    ]:

        percentile = (
            100.0
            * (1.0 - budget)
        )

        threshold = np.percentile(
            I,
            percentile,
        )

        selected_mask = (
            I >= threshold
        )

        selected = np.sum(
            selected_mask
        )

        fraction = (
            selected
            / len(I)
        )

        selected_white = np.sum(
            selected_mask
            & white_mask
        )

        selected_black = np.sum(
            selected_mask
            & black_mask
        )

        print(
            f"Budget {100 * budget:>6.2f}% "
            f"| Q{percentile:>6.2f} "
            f"| threshold={threshold:.9f} "
            f"| selected={selected:>7,} "
            f"| actual={100 * fraction:.3f}% "
            f"| W/B={selected_white}/{selected_black}"
        )

    # ========================================================
    # Explicit 0.02% colour diagnostic
    # ========================================================

    threshold_9998 = np.percentile(
        I,
        99.98,
    )

    selected_mask = (
        I >= threshold_9998
    )

    selected_total = np.sum(
        selected_mask
    )

    selected_white = np.sum(
        selected_mask
        & white_mask
    )

    selected_black = np.sum(
        selected_mask
        & black_mask
    )

    global_white_fraction = np.mean(
        white_mask
    )

    selected_white_fraction = (
        selected_white
        / selected_total
    )

    print()
    print("=" * 70)
    print(
        "0.02% SELECTION — COLOUR BALANCE"
    )
    print("=" * 70)

    print(
        f"Threshold : "
        f"{threshold_9998:.9f}"
    )

    print(
        f"Selected  : "
        f"{selected_total:,}"
    )

    print()

    print(
        f"White : "
        f"{selected_white:,} "
        f"({100 * selected_white_fraction:.3f}%)"
    )

    print(
        f"Black : "
        f"{selected_black:,} "
        f"({100 * (1 - selected_white_fraction):.3f}%)"
    )

    print()

    print(
        f"Global White : "
        f"{100 * global_white_fraction:.3f}%"
    )

    print(
        f"Global Black : "
        f"{100 * (1 - global_white_fraction):.3f}%"
    )

    print()

    print(
        f"White enrichment : "
        f"{selected_white_fraction / global_white_fraction:.3f}x"
    )

    print("=" * 70)

    # ========================================================
    # Histogram
    # ========================================================

    print()
    print(
        "I HISTOGRAM"
    )
    print("-" * 70)

    counts, edges = np.histogram(
        I,
        bins=20,
        range=(0.0, 1.0),
    )

    max_count = np.max(
        counts
    )

    for i, count in enumerate(
        counts
    ):

        left = edges[i]
        right = edges[i + 1]

        bar_length = int(
            50
            * count
            / max_count
        )

        bar = "#" * bar_length

        print(
            f"{left:5.2f} - "
            f"{right:5.2f} | "
            f"{bar:<50} "
            f"{count:>7,}"
        )

    # ========================================================
    # PNG distribution plot
    # ========================================================

    print()
    print(
        "GENERATING DISTRIBUTION PLOT"
    )
    print("-" * 70)

    histogram_counts, histogram_bins, _ = plt.hist(
        I,
        bins=100,
        alpha=0.6,
        label="Positions",
    )

    mean = np.mean(I)
    std = np.std(I)

    if std > 0:

        x = np.linspace(
            np.min(I),
            np.max(I),
            500,
        )

        gaussian = (
            1.0
            / (
                std
                * np.sqrt(2 * np.pi)
            )
            * np.exp(
                -0.5
                * (
                    (x - mean)
                    / std
                ) ** 2
            )
        )

        bin_width = (
            histogram_bins[1]
            - histogram_bins[0]
        )

        gaussian *= (
            len(I)
            * bin_width
        )

        plt.plot(
            x,
            gaussian,
            linewidth=2,
            label=(
                f"Gaussienne "
                f"(μ={mean:.3f}, "
                f"σ={std:.3f})"
            ),
        )

    plt.xlim(
        0.0,
        1.0,
    )

    plt.axvline(
        threshold_9998,
        linestyle="--",
        linewidth=2,
        label=(
            f"P99.98 = "
            f"{threshold_9998:.3f}"
        ),
    )

    plt.xlabel(
        "Score I normalisé"
    )

    plt.ylabel(
        "Nombre de positions"
    )

    plt.title(
        "ALBERTA - Distribution du score I"
    )

    plt.legend()

    plt.grid(
        alpha=0.2,
    )

    plt.tight_layout()

    output_path = (
        PROJECT_ROOT
        / "data"
        / "I_distribution_1-10.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Plot saved to: {output_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()