import json
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats_1-100.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "I_is_function_of_ply_strata.json"
)


# ============================================================
# Weights
# ============================================================

W_H = -0.277
W_U = 0.441
W_HU = 0.282


# ============================================================
# Ply strata
# ============================================================

def build_strata(max_ply):
    """
    Build progressively wider ply strata.

    1-100   : 10-ply strata
    101-400 : 20-ply strata
    401-600 : 50-ply strata
    601-910 : 50-ply strata

    The final stratum is automatically clipped to max_ply.
    """

    strata = []

    # --------------------------------------------------------
    # 1-100 : width 10
    # --------------------------------------------------------

    for low in range(1, 101, 10):

        high = min(
            low + 9,
            max_ply,
        )

        if low > max_ply:
            break

        strata.append(
            (
                f"{low}-{high}",
                low,
                high,
            )
        )

    # --------------------------------------------------------
    # 101-400 : width 20
    # --------------------------------------------------------

    for low in range(101, 401, 20):

        high = min(
            low + 19,
            max_ply,
        )

        if low > max_ply:
            break

        strata.append(
            (
                f"{low}-{high}",
                low,
                high,
            )
        )

    # --------------------------------------------------------
    # 401-600 : width 50
    # --------------------------------------------------------

    for low in range(401, 601, 50):

        high = min(
            low + 49,
            max_ply,
        )

        if low > max_ply:
            break

        strata.append(
            (
                f"{low}-{high}",
                low,
                high,
            )
        )

    # --------------------------------------------------------
    # 601+ : width 50
    # --------------------------------------------------------

    for low in range(601, max_ply + 1, 50):

        high = min(
            low + 49,
            max_ply,
        )

        strata.append(
            (
                f"{low}-{high}",
                low,
                high,
            )
        )

    return strata


# ============================================================
# Percentile rank
# ============================================================

def percentile_rank(values):
    """
    Empirical percentile-rank normalization.

    Smallest value -> 0
    Largest value  -> 1

    Ties receive the same mid-rank.

    Performed independently for White-to-move and
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
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        # Mid-rank
        rank = (
            (start + end - 1)
            / 2.0
        )

        percentile = (
            rank
            / (n - 1)
        )

        ranks[
            order[start:end]
        ] = percentile

        start = end

    return ranks


# ============================================================
# Side-aware normalization
# ============================================================

def normalize_side_aware(
    values,
    white_mask,
    black_mask,
):
    """
    Percentile-rank normalization independently for
    White-to-move and Black-to-move positions.
    """

    normalized = np.empty_like(
        values,
        dtype=np.float64,
    )

    normalized[
        white_mask
    ] = percentile_rank(
        values[
            white_mask
        ]
    )

    normalized[
        black_mask
    ] = percentile_rank(
        values[
            black_mask
        ]
    )

    return normalized


# ============================================================
# Extract ply from FEN
# ============================================================

def extract_ply(fens):
    """
    Reconstruct ply from FEN.

    FEN fullmove numbering starts at 1.

    White to move:
        ply = 2 * (fullmove - 1)

    Black to move:
        ply = 2 * (fullmove - 1) + 1
    """

    plies = np.empty(
        len(fens),
        dtype=np.int32,
    )

    for i, fen in enumerate(fens):

        parts = fen.split()

        if len(parts) < 6:
            raise ValueError(
                f"Invalid FEN: {fen}"
            )

        side = parts[1]

        try:
            fullmove = int(
                parts[5]
            )
        except ValueError:
            raise ValueError(
                f"Invalid fullmove number in FEN: {fen}"
            )

        if fullmove < 1:
            raise ValueError(
                f"Invalid fullmove number in FEN: {fen}"
            )

        if side == "w":

            plies[i] = (
                2 * (fullmove - 1)
            )

        elif side == "b":

            plies[i] = (
                2 * (fullmove - 1)
                + 1
            )

        else:

            raise ValueError(
                f"Invalid side-to-move in FEN: {fen}"
            )

    return plies


# ============================================================
# Min-max normalization
# ============================================================

def minmax_normalize(values):
    """
    Global min-max normalization.

        I = (I_raw - I_min)
            / (I_max - I_min)

    This transformation is strictly monotonic and therefore
    preserves the ordering of I_raw.

    Numerical protection is applied only after normalization.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    minimum = np.min(values)
    maximum = np.max(values)

    span = maximum - minimum

    if not np.isfinite(minimum):
        raise RuntimeError(
            "I_min is not finite."
        )

    if not np.isfinite(maximum):
        raise RuntimeError(
            "I_max is not finite."
        )

    if span <= 0.0:
        raise RuntimeError(
            "I_max <= I_min."
        )

    normalized = (
        values - minimum
    ) / span

    # Protect against microscopic floating-point excursions.
    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return normalized


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "ALBERTA - I = f(PLY) BY STRATA"
    )
    print("=" * 78)

    # ========================================================
    # Load data
    # ========================================================

    print()
    print(
        "Loading uncertainty statistics"
    )
    print("-" * 78)

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
        f"Records: {len(data):,}"
    )

    # ========================================================
    # Extract arrays
    # ========================================================

    print()
    print(
        "Extracting signals..."
    )

    fens = np.asarray(
        [
            record["fen"]
            for record in data
        ],
        dtype=object,
    )

    H = np.asarray(
        [
            record["H"]
            for record in data
        ],
        dtype=np.float64,
    )

    U = np.asarray(
        [
            record["U"]
            for record in data
        ],
        dtype=np.float64,
    )

    HU = np.asarray(
        [
            record["HU"]
            for record in data
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Side
    # ========================================================

    print()
    print(
        "Extracting side to move..."
    )

    sides = np.asarray(
        [
            fen.split()[1]
            for fen in fens
        ],
        dtype="<U1",
    )

    white_mask = (
        sides == "w"
    )

    black_mask = (
        sides == "b"
    )

    print(
        f"White: {np.sum(white_mask):,}"
    )

    print(
        f"Black: {np.sum(black_mask):,}"
    )

    # ========================================================
    # Ply
    # ========================================================

    print()
    print(
        "Extracting ply..."
    )

    ply = extract_ply(
        fens
    )

    min_ply = int(
        np.min(ply)
    )

    max_ply = int(
        np.max(ply)
    )

    print(
        f"Minimum ply: {min_ply}"
    )

    print(
        f"Maximum ply: {max_ply}"
    )

    # ========================================================
    # Build strata
    # ========================================================

    STRATA = build_strata(
        max_ply
    )

    print()
    print(
        "PLY STRATA"
    )
    print("-" * 78)

    for name, low, high in STRATA:

        count = np.sum(
            (ply >= low)
            &
            (ply <= high)
        )

        print(
            f"{name:>10} : "
            f"{count:>10,}"
        )

    # ========================================================
    # Side-aware normalization
    # ========================================================

    print()
    print(
        "SIDE-AWARE PERCENTILE NORMALIZATION"
    )
    print("-" * 78)

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
        "H / U / HU normalized."
    )

    # ========================================================
    # Raw I
    # ========================================================

    print()
    print(
        "COMPUTING RAW I"
    )
    print("-" * 78)

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
            "I_raw contains non-finite values."
        )

    I_min = float(
        np.min(I_raw)
    )

    I_max = float(
        np.max(I_raw)
    )

    print(
        f"I_raw min : {I_min:.12f}"
    )

    print(
        f"I_raw max : {I_max:.12f}"
    )

    # ========================================================
    # Global min-max normalization
    # ========================================================

    print()
    print(
        "GLOBAL MIN-MAX NORMALIZATION"
    )
    print("-" * 78)

    I = minmax_normalize(
        I_raw
    )

    print(
        f"I min : {np.min(I):.12f}"
    )

    print(
        f"I max : {np.max(I):.12f}"
    )

    # ========================================================
    # Statistics by ply stratum
    # ========================================================

    print()
    print("=" * 78)
    print(
        "I = f(PLY)"
    )
    print("=" * 78)

    header = (
        f"{'PLY':>10} "
        f"{'N':>10} "
        f"{'MEAN':>10} "
        f"{'MEDIAN':>10} "
        f"{'STD':>10} "
        f"{'P10':>10} "
        f"{'P25':>10} "
        f"{'P75':>10} "
        f"{'P90':>10} "
        f"{'P95':>10} "
        f"{'P99':>10} "
        f"{'MIN':>10} "
        f"{'MAX':>10}"
    )

    print()
    print(header)
    print("-" * len(header))

    summary = {}

    for name, low, high in STRATA:

        mask = (
            (ply >= low)
            &
            (ply <= high)
        )

        values = I[mask]

        if len(values) == 0:
            continue

        stats = {
            "n": int(len(values)),

            "mean": float(
                np.mean(values)
            ),

            "median": float(
                np.median(values)
            ),

            "std": float(
                np.std(values)
            ),

            "p10": float(
                np.percentile(
                    values,
                    10,
                )
            ),

            "p25": float(
                np.percentile(
                    values,
                    25,
                )
            ),

            "p50": float(
                np.percentile(
                    values,
                    50,
                )
            ),

            "p75": float(
                np.percentile(
                    values,
                    75,
                )
            ),

            "p90": float(
                np.percentile(
                    values,
                    90,
                )
            ),

            "p95": float(
                np.percentile(
                    values,
                    95,
                )
            ),

            "p99": float(
                np.percentile(
                    values,
                    99,
                )
            ),

            "min": float(
                np.min(values)
            ),

            "max": float(
                np.max(values)
            ),
        }

        summary[name] = stats

        print(
            f"{name:>10} "
            f"{stats['n']:>10,} "
            f"{stats['mean']:>10.6f} "
            f"{stats['median']:>10.6f} "
            f"{stats['std']:>10.6f} "
            f"{stats['p10']:>10.6f} "
            f"{stats['p25']:>10.6f} "
            f"{stats['p75']:>10.6f} "
            f"{stats['p90']:>10.6f} "
            f"{stats['p95']:>10.6f} "
            f"{stats['p99']:>10.6f} "
            f"{stats['min']:>10.6f} "
            f"{stats['max']:>10.6f}"
        )

    # ========================================================
    # Save
    # ========================================================

    output = {
        "weights": {
            "W_H": W_H,
            "W_U": W_U,
            "W_HU": W_HU,
        },

        "normalization": {
            "signals": (
                "side-aware percentile rank"
            ),
            "I": (
                "global min-max normalization"
            ),
        },

        "ply": {
            "min": min_ply,
            "max": max_ply,
        },

        "I_raw": {
            "min": I_min,
            "max": I_max,
        },

        "strata": summary,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()