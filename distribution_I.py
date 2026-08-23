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
    / "uncertainty_stats.json"
)


# ============================================================
# Initial weights
# ============================================================

W_H = 0.079
W_U = 0.591
W_HU = 0.330


# ============================================================
# Calibration ranges
#
# p01 / p99 from the previous reward-signal analysis.
# Values outside the range are clipped to [0, 1].
# ============================================================

H_LOW = 0.0
U_LOW = 0.0
HU_LOW = 0.0

# ============================================================
# Normalization
# ============================================================

def normalize(
    values,
    minimum,
    maximum,
):
    """
    Min-max normalization with clipping to [0, 1].
    """

    normalized = (
        values - minimum
    ) / (
        maximum - minimum
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ALBERTA - ACTIVE LEARNING SCORE ANALYSIS")
    print("=" * 70)

    print()
    print("Loading uncertainty statistics")
    print("-" * 70)
    print(f"File: {DATA_FILE}")

    with open(
        DATA_FILE,
        "r",
    ) as f:

        data = json.load(f)

    print(f"Raw records: {len(data):,}")

    #
    # Extract signals
    #
    H = np.array(
        [record["H"] for record in data],
        dtype=np.float64,
    )

    U = np.array(
        [record["U"] for record in data],
        dtype=np.float64,
    )

    HU = np.array(
        [record["HU"] for record in data],
        dtype=np.float64,
    )

    #
    # Historical calibration maxima.
    #
    # These maxima define the reference scale for the
    # current calibration dataset.
    #
    H_HIGH = np.max(H)
    U_HIGH = np.max(U)
    HU_HIGH = np.max(HU)

    print()
    print("CALIBRATION RANGES")
    print("-" * 70)

    print(f"H   : [{H_LOW:.6f}, {H_HIGH:.6f}]")
    print(f"U   : [{U_LOW:.6f}, {U_HIGH:.6f}]")
    print(f"HU  : [{HU_LOW:.6f}, {HU_HIGH:.6f}]")

    #
    # Normalize
    #
    H_norm = normalize(
        H,
        H_LOW,
        H_HIGH,
    )

    U_norm = normalize(
        U,
        U_LOW,
        U_HIGH,
    )

    HU_norm = normalize(
        HU,
        HU_LOW,
        HU_HIGH,
    )

    #
    # Multilinear score
    #
    I = (
        W_H * H_norm
        +
        W_U * U_norm
        +
        W_HU * HU_norm
    )

    # ========================================================
    # Distribution
    # ========================================================

    print()
    print("WEIGHTS")
    print("-" * 70)

    print(f"H   : {W_H:.4f}")
    print(f"U   : {W_U:.4f}")
    print(f"HU  : {W_HU:.4f}")
    print(f"Sum : {W_H + W_U + W_HU:.4f}")

    print()
    print("NORMALIZED SIGNALS")
    print("-" * 70)

    for name, values in [
        ("H", H_norm),
        ("U", U_norm),
        ("HU", HU_norm),
    ]:

        print(
            f"{name:<3} "
            f"mean={np.mean(values):.4f} | "
            f"median={np.median(values):.4f} | "
            f"p90={np.percentile(values, 90):.4f} | "
            f"p99={np.percentile(values, 99):.4f} | "
            f"max={np.max(values):.4f}"
        )

    print()
    print("I SCORE DISTRIBUTION")
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
        100,
    ]

    for percentile in percentiles:

        value = np.percentile(
            I,
            percentile,
        )

        print(
            f"P{percentile:<5} : {value:.6f}"
        )

    print()
    print("SUMMARY")
    print("-" * 70)

    print(f"N      : {len(I):,}")
    print(f"Mean   : {np.mean(I):.6f}")
    print(f"Std    : {np.std(I):.6f}")
    print(f"Median : {np.median(I):.6f}")
    print(f"Min    : {np.min(I):.6f}")
    print(f"Max    : {np.max(I):.6f}")

    # ========================================================
    # Candidate thresholds
    # ========================================================

    print()
    print("COACHING BUDGET THRESHOLDS")
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

        selected = np.sum(
            I >= threshold
        )

        fraction = (
            selected / len(I)
        )

        print(
            f"Budget {100 * budget:>6.2f}% "
            f"| Q{percentile:>6.2f} "
            f"| threshold={threshold:.6f} "
            f"| selected={selected:>7,} "
            f"| actual={100 * fraction:.3f}%"
        )

    # ========================================================
    # Histogram
    # ========================================================

    print()
    print("I HISTOGRAM")
    print("-" * 70)

    counts, edges = np.histogram(
        I,
        bins=20,
        range=(0.0, 1.0),
    )

    max_count = np.max(counts)

    for i, count in enumerate(counts):

        left = edges[i]
        right = edges[i + 1]

        bar_length = int(
            50 * count / max_count
        )

        bar = "#" * bar_length

        print(
            f"{left:5.2f} - {right:5.2f} | "
            f"{bar:<50} "
            f"{count:>7,}"
        )

    # ========================================================
    # PNG distribution plot
    # ========================================================

    print()
    print("GENERATING DISTRIBUTION PLOT")
    print("-" * 70)

    #
    # Histogram with raw counts.
    #
    histogram_counts, histogram_bins, _ = plt.hist(
        I,
        bins=100,
        alpha=0.6,
        label="Positions",
    )

    #
    # Mean and standard deviation.
    #
    mean = np.mean(I)
    std = np.std(I)

    #
    # X-axis for Gaussian.
    #
    x = np.linspace(
        np.min(I),
        np.max(I),
        500,
    )

    #
    # Gaussian probability density.
    #
    gaussian = (
        1.0
        / (std * np.sqrt(2 * np.pi))
        * np.exp(
            -0.5
            * ((x - mean) / std) ** 2
        )
    )

    #
    # Convert density into expected number
    # of observations per histogram bin.
    #
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
            f"(μ={mean:.3f}, σ={std:.3f})"
        ),
    )

    plt.xlim(
        0.0,
        np.max(I),
    )

    #
    # P99.5 threshold.
    #
    threshold_995 = np.percentile(
        I,
        99.5,
    )

    plt.axvline(
        threshold_995,
        linestyle="--",
        linewidth=2,
        label=(
            f"P99.5 = "
            f"{threshold_995:.3f}"
        ),
    )

    #
    # Labels.
    #
    plt.xlabel("Score I")
    plt.ylabel("Nombre de positions")

    plt.title(
        "ALBERTA - Distribution du score I"
    )

    plt.legend()

    plt.grid(
        alpha=0.2,
    )

    plt.tight_layout()

    #
    # Save.
    #
    output_path = (
        PROJECT_ROOT
        / "data"
        / "I_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Plot saved to: {output_path}"
    )


if __name__ == "__main__":
    main()