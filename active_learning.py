from dataclasses import dataclass
from typing import Iterable


# ============================================================
# Correlation-derived baseline weights
# ============================================================

def correlation_weight(
    pearson: float,
    spearman: float,
) -> float:
    """
    Measure the strength of association between a signal
    and the reward.

    The sign is intentionally ignored because active learning
    only cares about the strength of the relationship.
    """

    return (
        pearson ** 2
        +
        spearman ** 2
    ) ** 0.5


def compute_baseline_weights():
    """
    Compute baseline weights from the reward-signal analysis.

    Weight:
        sqrt(Pearson^2 + Spearman^2)

    The three weights are normalized so that:

        w_H + w_U + w_HU = 1
    """

    scores = {
        "H": correlation_weight(
            -0.0173,
            -0.0190,
        ),
        "U": correlation_weight(
            0.1297,
            0.1405,
        ),
        "HU": correlation_weight(
            0.0772,
            0.0737,
        ),
    }

    total = sum(scores.values())

    if total <= 0.0:
        raise ValueError(
            "Correlation scores must have a positive sum."
        )

    return {
        name: value / total
        for name, value in scores.items()
    }


# ============================================================
# Configuration
# ============================================================

@dataclass
class ActiveLearningConfig:

    #
    # Baseline weights.
    #
    w_h: float = 0.079
    w_u: float = 0.591
    w_hu: float = 0.330

    #
    # Normalization ranges.
    #
    # These are calibrated automatically from the reference
    # observations using their empirical maxima.
    #
    h_low: float = 0.0
    h_high: float = 1.0

    u_low: float = 0.0
    u_high: float = 1.0

    hu_low: float = 0.0
    hu_high: float = 1.0

    #
    # Fraction of positions sent to the oracle.
    #
    # 0.005 = 0.5%
    #
    query_budget: float = 0.005


# ============================================================
# Calibration
# ============================================================

def calibrate_config(
    observations: Iterable[dict],
    query_budget: float = 0.005,
) -> ActiveLearningConfig:
    """
    Build an ActiveLearningConfig from reference observations.

    The calibration maxima are the empirical maxima of:

        H
        U
        HU

    observed in the reference dataset.

    Lower bounds are fixed at zero because all three signals
    are non-negative.

    Values exceeding these historical maxima are later clipped
    to 1.0 by normalize().

    Example:

        H_high  = max(H)
        U_high  = max(U)
        HU_high = max(HU)
    """

    if not 0.0 < query_budget <= 1.0:
        raise ValueError(
            "query_budget must be in (0, 1]."
        )

    max_h = None
    max_u = None
    max_hu = None

    count = 0

    for observation in observations:

        try:
            H = float(observation["H"])
            U = float(observation["U"])
            HU = float(observation["HU"])

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        #
        # Ignore non-finite values.
        #
        if not all(
            value == value
            and abs(value) != float("inf")
            for value in (H, U, HU)
        ):
            continue

        #
        # Signals are expected to be non-negative.
        #
        if H < 0.0 or U < 0.0 or HU < 0.0:
            raise ValueError(
                "H, U and HU must be non-negative."
            )

        if max_h is None or H > max_h:
            max_h = H

        if max_u is None or U > max_u:
            max_u = U

        if max_hu is None or HU > max_hu:
            max_hu = HU

        count += 1

    if count == 0:
        raise ValueError(
            "No valid observations available for calibration."
        )

    if max_h <= 0.0:
        raise ValueError(
            "Maximum H must be greater than zero."
        )

    if max_u <= 0.0:
        raise ValueError(
            "Maximum U must be greater than zero."
        )

    if max_hu <= 0.0:
        raise ValueError(
            "Maximum HU must be greater than zero."
        )

    #
    # Start with the correlation-derived baseline weights.
    #
    weights = compute_baseline_weights()

    return ActiveLearningConfig(
        w_h=weights["H"],
        w_u=weights["U"],
        w_hu=weights["HU"],

        h_low=0.0,
        h_high=max_h,

        u_low=0.0,
        u_high=max_u,

        hu_low=0.0,
        hu_high=max_hu,

        query_budget=query_budget,
    )


# ============================================================
# Normalization
# ============================================================

def normalize(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Normalize a signal to [0, 1].

    Values outside the calibration range are clipped.

    Therefore:

        value <= minimum -> 0
        value >= maximum -> 1

    This means future values above the historical calibration
    maximum are safely saturated at 1.0.
    """

    if maximum <= minimum:
        raise ValueError(
            "Normalization maximum must be greater than minimum."
        )

    normalized = (
        value - minimum
    ) / (
        maximum - minimum
    )

    return max(
        0.0,
        min(
            1.0,
            normalized,
        ),
    )


# ============================================================
# Multilinear AL score
# ============================================================

def compute_score(
    H: float,
    U: float,
    HU: float,
    config: ActiveLearningConfig,
) -> float:
    """
    Compute the multilinear active-learning score I.

    Each signal is independently normalized to [0, 1]:

        H_norm
        U_norm
        HU_norm

    Then:

        I = w_H * H_norm
          + w_U * U_norm
          + w_HU * HU_norm

    Since the weights sum to 1 and every normalized signal
    is in [0, 1], I is also in [0, 1].
    """

    H_norm = normalize(
        H,
        config.h_low,
        config.h_high,
    )

    U_norm = normalize(
        U,
        config.u_low,
        config.u_high,
    )

    HU_norm = normalize(
        HU,
        config.hu_low,
        config.hu_high,
    )

    return (
        config.w_h * H_norm
        +
        config.w_u * U_norm
        +
        config.w_hu * HU_norm
    )


# ============================================================
# Threshold calibration
# ============================================================

def compute_threshold(
    scores: Iterable[float],
    config: ActiveLearningConfig,
) -> float:
    """
    Compute the empirical active-learning threshold.

    For a query budget of 0.5%:

        threshold = Q99.5%(I)

    More generally:

        threshold = Q(1 - query_budget)

    The threshold is therefore always determined by the
    reference score distribution rather than by an arbitrary
    absolute I value.
    """

    scores = sorted(scores)

    if not scores:
        raise ValueError(
            "Cannot compute threshold from an empty score set."
        )

    if not 0.0 < config.query_budget <= 1.0:
        raise ValueError(
            "query_budget must be in (0, 1]."
        )

    #
    # Example:
    #
    # 0.005 -> 0.995 -> Q99.5
    #
    quantile = (
        1.0
        - config.query_budget
    )

    position = (
        quantile
        * (len(scores) - 1)
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(scores) - 1,
    )

    fraction = (
        position
        - lower
    )

    threshold = (
        scores[lower]
        +
        fraction
        * (
            scores[upper]
            - scores[lower]
        )
    )

    return threshold


# ============================================================
# Selection
# ============================================================

def should_query(
    H: float,
    U: float,
    HU: float,
    threshold: float,
    config: ActiveLearningConfig,
) -> bool:
    """
    Decide whether the current position should trigger
    an active-learning query.

    A query is triggered when:

        I >= threshold
    """

    score = compute_score(
        H,
        U,
        HU,
        config,
    )

    return score >= threshold


# ============================================================
# Position evaluation
# ============================================================

def evaluate_position(
    H: float,
    U: float,
    HU: float,
    threshold: float,
    config: ActiveLearningConfig,
) -> dict:
    """
    Return the complete AL decision for one position.

    The returned dictionary contains:

        raw signals
        normalized signals
        final score
        threshold
        selection decision
    """

    H_norm = normalize(
        H,
        config.h_low,
        config.h_high,
    )

    U_norm = normalize(
        U,
        config.u_low,
        config.u_high,
    )

    HU_norm = normalize(
        HU,
        config.hu_low,
        config.hu_high,
    )

    score = (
        config.w_h * H_norm
        +
        config.w_u * U_norm
        +
        config.w_hu * HU_norm
    )

    return {
        "H": H,
        "U": U,
        "HU": HU,
        "H_normalized": H_norm,
        "U_normalized": U_norm,
        "HU_normalized": HU_norm,
        "score": score,
        "threshold": threshold,
        "selected": score >= threshold,
    }