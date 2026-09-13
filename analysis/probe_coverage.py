#!/usr/bin/env python3

"""
ALBERTA - Calibration Probe Coverage Analysis
=============================================

Measure whether a fixed calibration-probe set continues to cover
the H/U/HU feature distribution encountered during later RL
training.

Two complementary diagnostics are performed.

A. STATIC PROBE
---------------

The H/U/HU values stored when the probe was originally created are
kept fixed:

    Q_probe,0

and compared with later RL strata:

    P_t

This measures whether the original feature coordinates of the
probe become stale relative to the evolving RL distribution.

B. DYNAMIC PROBE
----------------

The exact same probe FENs are reevaluated under later learners:

    H_t(s)
    U_t(s)
    HU_t(s)

and compared with the corresponding contemporary RL stratum:

    Q_probe,t  vs  P_t

This separates:

    stale learner-dependent measurements

from:

    stale state support.

Scientific question
-------------------

If static coverage degrades but dynamic coverage remains stable,
the same FENs may remain useful anchors even though their original
H/U measurements became obsolete.

If both static and dynamic coverage degrade, the fixed FEN set
itself no longer covers the contemporary H/U/HU distribution well.

Diagnostics
-----------

1. Marginal range coverage
2. k-NN support distance
3. Relative support-distance ratio
4. Two-sample classifier distinguishability

Important
---------

This is a support diagnostic, not a proof of statistical
representativeness.

The dynamic U reconstruction uses ONLY trained historical RL value
heads plus the current model.

BC6 / BC7 remain league opponents but are explicitly excluded from
uncertainty estimation.

Therefore:

    U_t(s) = Var[
        trained historical RL critics,
        current critic
    ]

and never includes randomly initialized BC value heads.

The contemporary self-play statistics and the dynamically
recomputed probe must have been generated with the same corrected
uncertainty definition.

Outputs
-------

data/analysis/probe_coverage/

    static_probe_coverage.csv
    dynamic_probe_coverage.csv
    probe_coverage_report.txt

    static_support_progression.png
    dynamic_support_progression.png

Example
-------

python analysis/analyze_probe_coverage.py \
    --probe checkpoints/queue/oracle_queue_1-10_pareto_probe.jsonl \
    --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import chess.variant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# ALBERTA imports
# ============================================================

import training.train_al as al

from src.actions_space import ACTIONS, ACTION_TO_INDEX
from src.encoding import encode_boards
from src.models.actor_critic import ActorCritic
from src.models.resnet import ChessResNet
from src.selfplay.league import League


# ============================================================
# Defaults
# ============================================================

DEFAULT_PROBE_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)

DEFAULT_STRATA = {
    "RL01-10":
        PROJECT_ROOT
        / "data"
        / "selfplay_jsons"
        / "uncertainty_stats_1-10.json",

    "RL11-20":
        PROJECT_ROOT
        / "data"
        / "selfplay_jsons"
        / "uncertainty_stats_11-20.json",

    "RL21-30":
        PROJECT_ROOT
        / "data"
        / "selfplay_jsons"
        / "uncertainty_stats_21-30.json",

    "RL31-40":
        PROJECT_ROOT
        / "data"
        / "selfplay_jsons"
        / "uncertainty_stats_31-40.json",

    "RL41-50":
        PROJECT_ROOT
        / "data"
        / "selfplay_jsons"
        / "uncertainty_stats_41-50.json",

    "RL51-60":
        PROJECT_ROOT
        / "data"
        / "selfplay_jsons"
        / "uncertainty_stats_51-60.json",
}

DEFAULT_DYNAMIC_CHECKPOINTS = {
    10: "RL01-10",
    20: "RL11-20",
    30: "RL21-30",
    40: "RL31-40",
    50: "RL41-50",
    60: "RL51-60",
}

DEFAULT_BC_EPOCH_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
)

DEFAULT_RL_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
)

DEFAULT_LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "probe_coverage"
)

DEFAULT_DEVICE = "cpu"

DEFAULT_SEED = 42

DEFAULT_K_NEIGHBORS = 5
DEFAULT_MAX_POOL_SAMPLE = 20000
DEFAULT_DYNAMIC_BATCH_SIZE = 64

DEFAULT_CLASSIFIER_REPEATS = 10
DEFAULT_CLASSIFIER_FOLDS = 5

# Canonical current RL configuration.
DEFAULT_LEAGUE_MAX_AGENTS = 12

DEFAULT_BC_PRIOR_EPOCH = 7
DEFAULT_BC_ANCHOR_EPOCHS = (
    6,
    7,
)

DEFAULT_OPENING_PRIOR_PLIES = 6
DEFAULT_OPENING_PRIOR_STRENGTH = 1.0

FEATURE_NAMES = (
    "H",
    "U",
    "HU",
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze static and dynamic H/U/HU support coverage "
            "of a fixed ALBERTA calibration probe."
        )
    )

    parser.add_argument(
        "--probe",
        type=Path,
        default=DEFAULT_PROBE_FILE,
    )

    parser.add_argument(
        "--bc-dir",
        type=Path,
        default=DEFAULT_BC_EPOCH_DIR,
    )

    parser.add_argument(
        "--rl-dir",
        type=Path,
        default=DEFAULT_RL_CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--league-dir",
        type=Path,
        default=DEFAULT_LEAGUE_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
    )

    parser.add_argument(
        "--max-pool-sample",
        type=int,
        default=DEFAULT_MAX_POOL_SAMPLE,
    )

    parser.add_argument(
        "--dynamic-batch-size",
        type=int,
        default=DEFAULT_DYNAMIC_BATCH_SIZE,
    )

    parser.add_argument(
        "--classifier-repeats",
        type=int,
        default=DEFAULT_CLASSIFIER_REPEATS,
    )

    parser.add_argument(
        "--classifier-folds",
        type=int,
        default=DEFAULT_CLASSIFIER_FOLDS,
    )

    parser.add_argument(
        "--league-max-agents",
        type=int,
        default=DEFAULT_LEAGUE_MAX_AGENTS,
    )

    parser.add_argument(
        "--opening-prior-plies",
        type=int,
        default=DEFAULT_OPENING_PRIOR_PLIES,
    )

    parser.add_argument(
        "--opening-prior-strength",
        type=float,
        default=DEFAULT_OPENING_PRIOR_STRENGTH,
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    if hasattr(
        torch.backends,
        "cudnn",
    ):

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Generic JSON / JSONL loading
# ============================================================

def load_records(
    path: Path,
) -> list[dict]:

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    # ========================================================
    # JSONL
    # ========================================================

    if path.suffix.lower() == ".jsonl":

        records = []

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

                    raise ValueError(
                        f"Expected JSON object at line "
                        f"{line_number} in:\n{path}"
                    )

                records.append(
                    record
                )

        return records

    # ========================================================
    # JSON
    # ========================================================

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if isinstance(
        data,
        list,
    ):

        return data

    if isinstance(
        data,
        dict,
    ):

        for key in (
            "data",
            "records",
            "positions",
            "samples",
            "stats",
            "statistics",
            "observations",
            "uncertainty_stats",
        ):

            value = data.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                return value

    raise ValueError(
        f"Unsupported JSON structure:\n{path}"
    )


# ============================================================
# Feature extraction
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

    if not np.isfinite(
        value
    ):

        return np.nan

    return value


def build_features(
    records: list[dict],
) -> tuple[np.ndarray, int]:
    """
    Build the canonical raw feature space:

        [H, U, H*U]

    HU is recomputed from H and U rather than trusting a stored
    field, ensuring one identical definition across all sources.

    Because this is a joint support analysis, a record must have
    both finite H and finite U.
    """

    features = []

    skipped = 0

    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            skipped += 1
            continue

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

        if (
            not np.isfinite(
                H
            )
            or not np.isfinite(
                U
            )
            or U < 0.0
        ):

            skipped += 1
            continue

        features.append(
            [
                H,
                U,
                H * U,
            ]
        )

    if not features:

        raise RuntimeError(
            "No records with finite H and U."
        )

    return (
        np.asarray(
            features,
            dtype=np.float64,
        ),
        skipped,
    )


# ============================================================
# Deterministic pool sampling
# ============================================================

def make_rng(
    seed: int,
    offset: int,
) -> np.random.Generator:

    return np.random.default_rng(
        seed
        + offset
    )


def subsample_pool(
    X: np.ndarray,
    *,
    max_pool_sample: int,
    rng: np.random.Generator,
) -> np.ndarray:

    if len(
        X
    ) <= max_pool_sample:

        return X

    indices = rng.choice(
        len(
            X
        ),
        size=max_pool_sample,
        replace=False,
    )

    return X[
        indices
    ]


# ============================================================
# Marginal range coverage
# ============================================================

def compute_feature_coverage(
    X_probe: np.ndarray,
    X_pool: np.ndarray,
) -> dict:

    result = {}

    for (
        feature_index,
        feature_name,
    ) in enumerate(
        FEATURE_NAMES
    ):

        probe_values = X_probe[
            :,
            feature_index
        ]

        pool_values = X_pool[
            :,
            feature_index
        ]

        low = float(
            np.min(
                probe_values
            )
        )

        high = float(
            np.max(
                probe_values
            )
        )

        inside = (
            (
                pool_values
                >= low
            )
            &
            (
                pool_values
                <= high
            )
        )

        result[
            f"coverage_{feature_name}"
        ] = float(
            np.mean(
                inside
            )
        )

        result[
            f"probe_min_{feature_name}"
        ] = low

        result[
            f"probe_max_{feature_name}"
        ] = high

    return result


# ============================================================
# k-NN support distances
# ============================================================

def compute_probe_reference_distances(
    X_probe_scaled: np.ndarray,
    *,
    k_neighbors: int,
) -> np.ndarray:

    if len(
        X_probe_scaled
    ) <= 1:

        raise RuntimeError(
            "Probe must contain at least two states."
        )

    n_neighbors = min(
        k_neighbors + 1,
        len(
            X_probe_scaled
        ),
    )

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    model.fit(
        X_probe_scaled
    )

    distances, _ = model.kneighbors(
        X_probe_scaled
    )

    # First neighbor is the sample itself.
    distances = distances[
        :,
        1:
    ]

    if distances.shape[
        1
    ] == 0:

        raise RuntimeError(
            "Unable to compute leave-one-out probe distances."
        )

    return np.mean(
        distances,
        axis=1,
    )


def compute_pool_to_probe_distances(
    X_probe_scaled: np.ndarray,
    X_pool_scaled: np.ndarray,
    *,
    k_neighbors: int,
) -> np.ndarray:

    n_neighbors = min(
        k_neighbors,
        len(
            X_probe_scaled
        ),
    )

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    model.fit(
        X_probe_scaled
    )

    distances, _ = model.kneighbors(
        X_pool_scaled
    )

    return np.mean(
        distances,
        axis=1,
    )


# ============================================================
# Two-sample classifier
# ============================================================

def compute_two_sample_auc(
    X_probe: np.ndarray,
    X_pool: np.ndarray,
    *,
    seed: int,
    classifier_repeats: int,
    classifier_folds: int,
) -> dict:
    """
    Estimate feature-space distinguishability.

        class 0 = probe
        class 1 = contemporary RL pool

    The returned score is symmetric distinguishability:

        max(AUC, 1 - AUC)

    so 0.5 indicates weak separability and values approaching
    1 indicate strong separability.
    """

    n_probe = len(
        X_probe
    )

    if len(
        X_pool
    ) < n_probe:

        raise RuntimeError(
            "RL pool is smaller than probe: "
            f"{len(X_pool)} < {n_probe}"
        )

    if classifier_folds < 2:

        raise ValueError(
            "classifier_folds must be >= 2."
        )

    if classifier_folds > n_probe:

        raise ValueError(
            "classifier_folds cannot exceed probe size."
        )

    repeat_aucs = []

    for repeat in range(
        classifier_repeats
    ):

        rng = make_rng(
            seed,
            10000
            + repeat,
        )

        pool_indices = rng.choice(
            len(
                X_pool
            ),
            size=n_probe,
            replace=False,
        )

        X_pool_balanced = X_pool[
            pool_indices
        ]

        X = np.concatenate(
            [
                X_probe,
                X_pool_balanced,
            ],
            axis=0,
        )

        y = np.concatenate(
            [
                np.zeros(
                    n_probe,
                    dtype=np.int64,
                ),
                np.ones(
                    n_probe,
                    dtype=np.int64,
                ),
            ]
        )

        cv = StratifiedKFold(
            n_splits=classifier_folds,
            shuffle=True,
            random_state=(
                seed
                + repeat
            ),
        )

        fold_aucs = []

        for fold_index, (
            train_indices,
            test_indices,
        ) in enumerate(
            cv.split(
                X,
                y,
            )
        ):

            classifier = ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                max_features=1.0,
                class_weight="balanced",
                n_jobs=-1,
                random_state=(
                    seed
                    + 1000
                    * repeat
                    + fold_index
                ),
            )

            classifier.fit(
                X[
                    train_indices
                ],
                y[
                    train_indices
                ],
            )

            probabilities = classifier.predict_proba(
                X[
                    test_indices
                ]
            )[
                :,
                1
            ]

            auc = float(
                roc_auc_score(
                    y[
                        test_indices
                    ],
                    probabilities,
                )
            )

            fold_aucs.append(
                max(
                    auc,
                    1.0
                    - auc,
                )
            )

        repeat_aucs.append(
            float(
                np.mean(
                    fold_aucs
                )
            )
        )

    return {
        "domain_auc_mean":
            float(
                np.mean(
                    repeat_aucs
                )
            ),

        "domain_auc_std":
            float(
                np.std(
                    repeat_aucs,
                    ddof=0,
                )
            ),
    }


# ============================================================
# Distance summary
# ============================================================

def add_distance_summary(
    prefix: str,
    distances: np.ndarray,
    result: dict,
) -> None:

    result[
        f"{prefix}_median"
    ] = float(
        np.median(
            distances
        )
    )

    result[
        f"{prefix}_p90"
    ] = float(
        np.percentile(
            distances,
            90,
        )
    )

    result[
        f"{prefix}_p95"
    ] = float(
        np.percentile(
            distances,
            95,
        )
    )


# ============================================================
# Generic support analysis
# ============================================================

def analyze_support(
    X_probe: np.ndarray,
    X_pool: np.ndarray,
    *,
    seed: int,
    k_neighbors: int,
    max_pool_sample: int,
    classifier_repeats: int,
    classifier_folds: int,
) -> dict:
    """
    Compare a probe feature cloud with a contemporary RL pool.

    Scaling is fitted on the probe because distances are defined
    relative to the probe's own feature geometry.
    """

    scaler = StandardScaler()

    X_probe_scaled = scaler.fit_transform(
        X_probe
    )

    pool_rng = make_rng(
        seed,
        1,
    )

    X_pool_subsample = subsample_pool(
        X_pool,
        max_pool_sample=max_pool_sample,
        rng=pool_rng,
    )

    X_pool_scaled = scaler.transform(
        X_pool_subsample
    )

    # --------------------------------------------------------
    # Probe internal distance scale
    # --------------------------------------------------------

    reference_distances = (
        compute_probe_reference_distances(
            X_probe_scaled,
            k_neighbors=k_neighbors,
        )
    )

    reference_median = float(
        np.median(
            reference_distances
        )
    )

    if reference_median <= 1e-12:

        raise RuntimeError(
            "Probe reference distance is approximately zero."
        )

    # --------------------------------------------------------
    # Marginal range coverage
    # --------------------------------------------------------

    result = compute_feature_coverage(
        X_probe,
        X_pool_subsample,
    )

    # --------------------------------------------------------
    # Multivariate support distance
    # --------------------------------------------------------

    pool_distances = (
        compute_pool_to_probe_distances(
            X_probe_scaled,
            X_pool_scaled,
            k_neighbors=k_neighbors,
        )
    )

    pool_distance_median = float(
        np.median(
            pool_distances
        )
    )

    result[
        "probe_reference_knn_median"
    ] = reference_median

    result[
        "knn_support_ratio"
    ] = (
        pool_distance_median
        / reference_median
    )

    # --------------------------------------------------------
    # Two-sample distinguishability
    # --------------------------------------------------------

    result.update(
        compute_two_sample_auc(
            X_probe,
            X_pool,
            seed=seed,
            classifier_repeats=classifier_repeats,
            classifier_folds=classifier_folds,
        )
    )

    add_distance_summary(
        "pool_probe_distance",
        pool_distances,
        result,
    )

    return result


# ============================================================
# ActorCritic loading
# ============================================================

def load_actor_critic_checkpoint(
    path: Path,
    device: torch.device,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            f"Checkpoint has no model_state_dict:\n{path}"
        )

    checkpoint_actions = checkpoint.get(
        "actions"
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(
            ACTIONS
        )
    ):

        raise ValueError(
            "Action-space mismatch: "
            f"{checkpoint_actions} != {len(ACTIONS)}"
        )

    model = al.build_actor_critic(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# BC policy loading
# ============================================================

def load_bc_policy(
    *,
    epoch: int,
    bc_dir: Path,
    device: torch.device,
):

    path = (
        bc_dir
        / f"bc_epoch_{epoch}.pt"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"BC checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            f"BC checkpoint has no model_state_dict:\n{path}"
        )

    model = ChessResNet(
        num_actions=len(
            ACTIONS
        ),
        channels=32,
        blocks=4,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# BC opponent loading
# ============================================================

def load_bc_opponent(
    *,
    epoch: int,
    bc_dir: Path,
    device: torch.device,
):
    """
    Build an ActorCritic wrapper around a BC policy.

    The new value head is untrained, so this object is valid as
    a policy opponent but MUST NEVER contribute to U.
    """

    path = (
        bc_dir
        / f"bc_epoch_{epoch}.pt"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"BC checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    base_model = ChessResNet(
        num_actions=len(
            ACTIONS
        ),
        channels=32,
        blocks=4,
    )

    base_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = ActorCritic(
        base_model
    ).to(
        device
    )

    model.eval()

    return model


# ============================================================
# League reconstruction
# ============================================================

def load_league_for_epoch(
    *,
    epoch: int,
    bc_dir: Path,
    league_dir: Path,
    device: torch.device,
    league_max_agents: int,
) -> League:
    """
    Reconstruct the historical league available at RL epoch t.

    BC6 / BC7:
        opponents = yes
        uncertainty = NO

    RL snapshots:
        opponents = yes
        uncertainty = yes
    """

    protected_names = [
        f"bc_epoch_{epoch_id}"
        for epoch_id
        in DEFAULT_BC_ANCHOR_EPOCHS
    ]

    league = League(
        max_agents=league_max_agents,
        protected_agents=protected_names,
    )

    # ========================================================
    # BC anchors
    # ========================================================

    for bc_epoch in DEFAULT_BC_ANCHOR_EPOCHS:

        name = (
            f"bc_epoch_{bc_epoch}"
        )

        agent = load_bc_opponent(
            epoch=bc_epoch,
            bc_dir=bc_dir,
            device=device,
        )

        league.add_agent(
            name,
            agent,
            use_for_uncertainty=False,
        )

    # ========================================================
    # Trained historical RL critics
    # ========================================================

    loaded_snapshots = 0

    for snapshot_epoch in range(
        1,
        epoch + 1,
    ):

        name = (
            f"league_epoch_{snapshot_epoch:03d}"
        )

        path = (
            league_dir
            / f"{name}.pt"
        )

        if not path.exists():

            continue

        snapshot = (
            load_actor_critic_checkpoint(
                path,
                device,
            )
        )

        league.add_agent(
            name,
            snapshot,
            use_for_uncertainty=True,
        )

        loaded_snapshots += 1

    print(
        f"League RL{epoch}: "
        f"{len(league)} opponents, "
        f"{len(league.uncertainty_names())} "
        f"historical uncertainty critics."
    )

    print(
        "  U contributors:"
    )

    for name in league.uncertainty_names():

        print(
            f"    - {name}"
        )

    if loaded_snapshots == 0:

        print(
            "  NOTE: no historical RL snapshot was loaded; "
            "U may collapse to zero when only the current "
            "critic contributes."
        )

    return league


# ============================================================
# BC opening prior
# ============================================================

def opening_prior_strength(
    ply: int,
    *,
    prior_plies: int,
    prior_strength: float,
) -> float:

    if prior_plies <= 0:

        return 0.0

    if ply >= prior_plies:

        return 0.0

    return (
        prior_strength
        * (
            1.0
            - (
                ply
                / prior_plies
            )
        )
    )


# ============================================================
# Dynamic entropy H_t
# ============================================================

@torch.no_grad()
def compute_entropy_batch(
    boards: list[chess.variant.AtomicBoard],
    *,
    current_model,
    bc_policy,
    device: torch.device,
    opening_prior_plies: int,
    opening_prior_strength_value: float,
) -> torch.Tensor:
    """
    Reproduce ALBERTA rollout entropy semantics:

        RL legal logits
        + BC opening prior where active

    Entropy is measured BEFORE rollout sampling temperature.
    """

    if not boards:

        return torch.empty(
            0,
            dtype=torch.float32,
        )

    encoded = encode_boards(
        boards
    ).to(
        device
    )

    policy_logits, _ = current_model(
        encoded
    )

    # ========================================================
    # BC prior only where active
    # ========================================================

    opening_indices = [
        index
        for (
            index,
            board,
        ) in enumerate(
            boards
        )
        if opening_prior_strength(
            board.ply(),
            prior_plies=opening_prior_plies,
            prior_strength=opening_prior_strength_value,
        )
        > 0.0
    ]

    bc_logits_lookup = {}

    if opening_indices:

        opening_boards = [
            boards[
                index
            ]
            for index
            in opening_indices
        ]

        opening_encoded = encode_boards(
            opening_boards
        ).to(
            device
        )

        bc_logits = bc_policy(
            opening_encoded
        )

        if isinstance(
            bc_logits,
            tuple,
        ):

            bc_logits = bc_logits[
                0
            ]

        for (
            local_index,
            global_index,
        ) in enumerate(
            opening_indices
        ):

            bc_logits_lookup[
                global_index
            ] = bc_logits[
                local_index
            ]

    # ========================================================
    # Per-position legal entropy
    # ========================================================

    entropies = []

    for (
        index,
        board,
    ) in enumerate(
        boards
    ):

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:

            entropies.append(
                0.0
            )

            continue

        legal_indices = [
            ACTION_TO_INDEX[
                move.uci()
            ]
            for move
            in legal_moves
        ]

        legal_index_tensor = torch.tensor(
            legal_indices,
            dtype=torch.long,
            device=device,
        )

        legal_logits = policy_logits[
            index,
            legal_index_tensor,
        ]

        alpha = opening_prior_strength(
            board.ply(),
            prior_plies=opening_prior_plies,
            prior_strength=opening_prior_strength_value,
        )

        if (
            alpha > 0.0
            and index in bc_logits_lookup
        ):

            bc_legal_logits = (
                bc_logits_lookup[
                    index
                ][
                    legal_index_tensor
                ]
            )

            bc_log_probs = F.log_softmax(
                bc_legal_logits,
                dim=0,
            )

            legal_logits = (
                legal_logits
                + alpha
                * bc_log_probs
            )

        log_probs = F.log_softmax(
            legal_logits,
            dim=0,
        )

        probs = torch.exp(
            log_probs
        )

        entropy = -(
            probs
            * log_probs
        ).sum()

        entropies.append(
            float(
                entropy.item()
            )
        )

    return torch.tensor(
        entropies,
        dtype=torch.float32,
    )


# ============================================================
# Dynamic probe recomputation
# ============================================================

@torch.no_grad()
def recompute_dynamic_probe_features(
    *,
    probe_records: list[dict],
    current_model,
    league: League,
    bc_policy,
    device: torch.device,
    batch_size: int,
    opening_prior_plies: int,
    opening_prior_strength_value: float,
) -> np.ndarray:

    fens = []

    for record in probe_records:

        fen = record.get(
            "fen"
        )

        if (
            not isinstance(
                fen,
                str,
            )
            or not fen
        ):

            raise ValueError(
                "Probe record has no valid FEN."
            )

        # Validate now rather than failing halfway through.
        chess.variant.AtomicBoard(
            fen
        )

        fens.append(
            fen
        )

    features = []

    for start in range(
        0,
        len(
            fens
        ),
        batch_size,
    ):

        batch_fens = fens[
            start:
            start
            + batch_size
        ]

        boards = [
            chess.variant.AtomicBoard(
                fen
            )
            for fen
            in batch_fens
        ]

        # ====================================================
        # H_t
        # ========================================================

        H_values = compute_entropy_batch(
            boards,
            current_model=current_model,
            bc_policy=bc_policy,
            device=device,
            opening_prior_plies=opening_prior_plies,
            opening_prior_strength_value=(
                opening_prior_strength_value
            ),
        )

        # ====================================================
        # U_t
        #
        # League itself decides which agents are eligible.
        # BC anchors are explicitly excluded.
        # ========================================================

        encoded = encode_boards(
            boards
        ).to(
            device
        )

        U_values = league.uncertainty_batch(
            encoded,
            current_model=current_model,
        )

        H_values = (
            H_values
            .detach()
            .cpu()
            .numpy()
        )

        U_values = (
            U_values
            .detach()
            .cpu()
            .numpy()
        )

        for (
            H,
            U,
        ) in zip(
            H_values,
            U_values,
        ):

            H = float(
                H
            )

            U = float(
                U
            )

            if (
                not np.isfinite(
                    H
                )
                or not np.isfinite(
                    U
                )
                or U < 0.0
            ):

                raise RuntimeError(
                    "Non-finite dynamic H/U value."
                )

            features.append(
                [
                    H,
                    U,
                    H * U,
                ]
            )

    X = np.asarray(
        features,
        dtype=np.float64,
    )

    if len(
        X
    ) != len(
        probe_records
    ):

        raise RuntimeError(
            "Dynamic probe size mismatch: "
            f"{len(X)} != {len(probe_records)}"
        )

    return X


# ============================================================
# Plotting
# ============================================================

def plot_support_progression(
    dataframe: pd.DataFrame,
    *,
    x_column: str,
    output_path: Path,
    title: str,
) -> None:

    if len(
        dataframe
    ) == 0:

        return

    x = np.arange(
        len(
            dataframe
        )
    )

    labels = dataframe[
        x_column
    ].astype(
        str
    ).tolist()

    plt.figure(
        figsize=(
            9,
            6,
        )
    )

    plt.plot(
        x,
        dataframe[
            "knn_support_ratio"
        ],
        marker="o",
        label="k-NN support ratio",
    )

    plt.plot(
        x,
        dataframe[
            "domain_auc_mean"
        ],
        marker="o",
        label="Domain distinguishability AUC",
    )

    plt.xticks(
        x,
        labels,
        rotation=20,
    )

    plt.xlabel(
        x_column
    )

    plt.ylabel(
        "Diagnostic value"
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


# ============================================================
# Report
# ============================================================

def build_report(
    *,
    probe_path: Path,
    static_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
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
        "ALBERTA - CALIBRATION PROBE COVERAGE ANALYSIS"
    )

    add(
        "=" * 80
    )

    add()

    add(
        f"Probe: {probe_path}"
    )

    add()

    # ========================================================
    # Scientific meaning
    # ========================================================

    add(
        "SCIENTIFIC QUESTION"
    )

    add(
        "-" * 80
    )

    add(
        "Does a fixed set of calibration FENs continue to cover "
        "the H/U/HU regions encountered by later RL agents?"
    )

    add()

    add(
        "This is a feature-support diagnostic. It does not prove "
        "that the probe is statistically representative of the "
        "full state distribution."
    )

    add()

    # ========================================================
    # Static
    # ========================================================

    add(
        "STATIC PROBE"
    )

    add(
        "-" * 80
    )

    if len(
        static_df
    ):

        add(
            static_df[
                [
                    "stratum",
                    "coverage_H",
                    "coverage_U",
                    "coverage_HU",
                    "knn_support_ratio",
                    "domain_auc_mean",
                    "domain_auc_std",
                ]
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.5f}",
            )
        )

    add()

    add(
        "The static analysis holds the probe's original H/U/HU "
        "coordinates fixed."
    )

    add(
        "Degradation therefore mixes state-distribution drift "
        "with staleness of learner-dependent measurements."
    )

    add()

    # ========================================================
    # Dynamic
    # ========================================================

    add(
        "DYNAMIC PROBE"
    )

    add(
        "-" * 80
    )

    if len(
        dynamic_df
    ):

        add(
            dynamic_df[
                [
                    "epoch",
                    "reference_stratum",
                    "mean_H",
                    "mean_U",
                    "mean_HU",
                    "coverage_H",
                    "coverage_U",
                    "coverage_HU",
                    "knn_support_ratio",
                    "domain_auc_mean",
                    "domain_auc_std",
                ]
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.5f}",
            )
        )

    add()

    add(
        "The dynamic analysis reevaluates the exact same FENs "
        "using the contemporary learner and historical trained "
        "RL critics."
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
        "Static degradation with comparatively stable dynamic "
        "coverage is consistent with stale H/U measurements "
        "while the underlying FEN anchors remain useful."
    )

    add()

    add(
        "Degradation of both static and dynamic coverage is "
        "consistent with the fixed FEN set itself covering the "
        "contemporary H/U/HU distribution less well."
    )

    add()

    add(
        "Lower marginal coverage, larger k-NN support ratios and "
        "larger two-sample distinguishability all indicate "
        "greater feature-space mismatch."
    )

    add()

    # ========================================================
    # U definition
    # ========================================================

    add(
        "UNCERTAINTY DEFINITION"
    )

    add(
        "-" * 80
    )

    add(
        "Dynamic U uses only trained historical RL value heads "
        "plus the current model."
    )

    add(
        "BC6 and BC7 may remain policy opponents in the league "
        "but their randomly initialized ActorCritic value heads "
        "are excluded from uncertainty estimation."
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
        "A dynamic checkpoint is compared against a coarse "
        "10-epoch self-play stratum rather than an exact "
        "checkpoint-specific state distribution."
    )

    add(
        "For example, RL20 is compared against the aggregate "
        "RL11-20 statistics."
    )

    add()

    add(
        "The min/max marginal coverage diagnostic is deliberately "
        "simple and should be interpreted together with the "
        "multivariate k-NN and classifier diagnostics."
    )

    add()

    add(
        "The domain classifier measures feature-space "
        "distinguishability, not task usefulness."
    )

    add()

    add(
        "Static H/U/HU results are scientifically meaningful "
        "only if the probe file itself was generated with the "
        "corrected uncertainty estimator."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.k_neighbors <= 0:

        raise ValueError(
            "--k-neighbors must be positive."
        )

    if args.max_pool_sample <= 0:

        raise ValueError(
            "--max-pool-sample must be positive."
        )

    if args.dynamic_batch_size <= 0:

        raise ValueError(
            "--dynamic-batch-size must be positive."
        )

    if args.classifier_repeats <= 0:

        raise ValueError(
            "--classifier-repeats must be positive."
        )

    if args.classifier_folds < 2:

        raise ValueError(
            "--classifier-folds must be >= 2."
        )

    if args.league_max_agents < 1:

        raise ValueError(
            "--league-max-agents must be positive."
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    static_output_path = (
        args.output_dir
        / "static_probe_coverage.csv"
    )

    dynamic_output_path = (
        args.output_dir
        / "dynamic_probe_coverage.csv"
    )

    report_path = (
        args.output_dir
        / "probe_coverage_report.txt"
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 80)
    print("ALBERTA - CALIBRATION PROBE COVERAGE")
    print("=" * 80)

    print(
        f"Probe:  {args.probe}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Seed:   {args.seed}"
    )

    print()
    print(
        "Feature space: [H, U, H*U]"
    )

    # ========================================================
    # Original probe
    # ========================================================

    probe_records = load_records(
        args.probe
    )

    (
        X_original_probe,
        skipped_probe,
    ) = build_features(
        probe_records
    )

    print()
    print("=" * 80)
    print("ORIGINAL PROBE")
    print("=" * 80)

    print(
        f"Raw records:   {len(probe_records):,}"
    )

    print(
        f"Valid H/U:     {len(X_original_probe):,}"
    )

    print(
        f"Skipped:       {skipped_probe:,}"
    )

    if len(
        X_original_probe
    ) < 20:

        raise RuntimeError(
            "Probe set is too small for this diagnostic."
        )

    # ========================================================
    # Load contemporary strata
    # ========================================================

    stratum_feature_cache = {}

    print()
    print("=" * 80)
    print("LOADING RL STRATA")
    print("=" * 80)

    for (
        stratum_index,
        (
            stratum_name,
            stratum_path,
        ),
    ) in enumerate(
        DEFAULT_STRATA.items()
    ):

        records = load_records(
            stratum_path
        )

        (
            X_pool,
            skipped,
        ) = build_features(
            records
        )

        stratum_feature_cache[
            stratum_name
        ] = X_pool

        print(
            f"{stratum_name}: "
            f"{len(X_pool):,} valid, "
            f"{skipped:,} skipped"
        )

    # ========================================================
    # A. Static probe
    # ========================================================

    static_results = []

    print()
    print("=" * 80)
    print("STATIC PROBE ANALYSIS")
    print("=" * 80)

    for stratum_index, (
        stratum_name,
        X_pool,
    ) in enumerate(
        stratum_feature_cache.items()
    ):

        result = analyze_support(
            X_original_probe,
            X_pool,
            seed=(
                args.seed
                + 100
                * stratum_index
            ),
            k_neighbors=args.k_neighbors,
            max_pool_sample=args.max_pool_sample,
            classifier_repeats=args.classifier_repeats,
            classifier_folds=args.classifier_folds,
        )

        result[
            "stratum"
        ] = stratum_name

        static_results.append(
            result
        )

        print()
        print(
            stratum_name
        )

        print(
            f"  H range coverage : "
            f"{result['coverage_H']:.2%}"
        )

        print(
            f"  U range coverage : "
            f"{result['coverage_U']:.2%}"
        )

        print(
            f"  HU range coverage: "
            f"{result['coverage_HU']:.2%}"
        )

        print(
            f"  k-NN ratio       : "
            f"{result['knn_support_ratio']:.3f}x"
        )

        print(
            f"  Domain AUC       : "
            f"{result['domain_auc_mean']:.4f} "
            f"+/- {result['domain_auc_std']:.4f}"
        )

    static_df = pd.DataFrame(
        static_results
    )

    static_df.to_csv(
        static_output_path,
        index=False,
    )

    # ========================================================
    # B. Dynamic probe
    # ========================================================

    print()
    print("=" * 80)
    print("DYNAMIC PROBE ANALYSIS")
    print("=" * 80)

    bc_policy = load_bc_policy(
        epoch=DEFAULT_BC_PRIOR_EPOCH,
        bc_dir=args.bc_dir,
        device=device,
    )

    dynamic_results = []

    for dynamic_index, (
        epoch,
        reference_stratum,
    ) in enumerate(
        DEFAULT_DYNAMIC_CHECKPOINTS.items()
    ):

        print()
        print("-" * 80)
        print(
            f"RL{epoch} -> {reference_stratum}"
        )
        print("-" * 80)

        checkpoint_path = (
            args.rl_dir
            / f"rl_epoch_{epoch}.pt"
        )

        if not checkpoint_path.exists():

            print(
                f"WARNING: missing checkpoint:\n"
                f"{checkpoint_path}"
            )

            continue

        if reference_stratum not in stratum_feature_cache:

            print(
                f"WARNING: missing stratum "
                f"{reference_stratum}"
            )

            continue

        current_model = (
            load_actor_critic_checkpoint(
                checkpoint_path,
                device,
            )
        )

        league = load_league_for_epoch(
            epoch=epoch,
            bc_dir=args.bc_dir,
            league_dir=args.league_dir,
            device=device,
            league_max_agents=(
                args.league_max_agents
            ),
        )

        X_dynamic_probe = (
            recompute_dynamic_probe_features(
                probe_records=probe_records,
                current_model=current_model,
                league=league,
                bc_policy=bc_policy,
                device=device,
                batch_size=args.dynamic_batch_size,
                opening_prior_plies=(
                    args.opening_prior_plies
                ),
                opening_prior_strength_value=(
                    args.opening_prior_strength
                ),
            )
        )

        X_pool = stratum_feature_cache[
            reference_stratum
        ]

        result = analyze_support(
            X_dynamic_probe,
            X_pool,
            seed=(
                args.seed
                + 10000
                + 100
                * dynamic_index
            ),
            k_neighbors=args.k_neighbors,
            max_pool_sample=args.max_pool_sample,
            classifier_repeats=args.classifier_repeats,
            classifier_folds=args.classifier_folds,
        )

        result[
            "epoch"
        ] = epoch

        result[
            "reference_stratum"
        ] = reference_stratum

        result[
            "mean_H"
        ] = float(
            np.mean(
                X_dynamic_probe[
                    :,
                    0
                ]
            )
        )

        result[
            "mean_U"
        ] = float(
            np.mean(
                X_dynamic_probe[
                    :,
                    1
                ]
            )
        )

        result[
            "mean_HU"
        ] = float(
            np.mean(
                X_dynamic_probe[
                    :,
                    2
                ]
            )
        )

        dynamic_results.append(
            result
        )

        print(
            f"Dynamic mean H : "
            f"{result['mean_H']:.6e}"
        )

        print(
            f"Dynamic mean U : "
            f"{result['mean_U']:.6e}"
        )

        print(
            f"Dynamic mean HU: "
            f"{result['mean_HU']:.6e}"
        )

        print(
            f"H range coverage : "
            f"{result['coverage_H']:.2%}"
        )

        print(
            f"U range coverage : "
            f"{result['coverage_U']:.2%}"
        )

        print(
            f"HU range coverage: "
            f"{result['coverage_HU']:.2%}"
        )

        print(
            f"k-NN ratio       : "
            f"{result['knn_support_ratio']:.3f}x"
        )

        print(
            f"Domain AUC       : "
            f"{result['domain_auc_mean']:.4f} "
            f"+/- {result['domain_auc_std']:.4f}"
        )

        del current_model
        del league

        if device.type == "cuda":

            torch.cuda.empty_cache()

    dynamic_df = pd.DataFrame(
        dynamic_results
    )

    if len(
        dynamic_df
    ):

        dynamic_df = dynamic_df.sort_values(
            "epoch"
        ).reset_index(
            drop=True
        )

        dynamic_df.to_csv(
            dynamic_output_path,
            index=False,
        )

    # ========================================================
    # Plots
    # ========================================================

    plot_support_progression(
        static_df,
        x_column="stratum",
        output_path=(
            args.output_dir
            / "static_support_progression.png"
        ),
        title=(
            "Static calibration-probe support "
            "across RL strata"
        ),
    )

    if len(
        dynamic_df
    ):

        dynamic_plot_df = (
            dynamic_df.copy()
        )

        dynamic_plot_df[
            "checkpoint"
        ] = (
            "RL"
            + dynamic_plot_df[
                "epoch"
            ].astype(
                str
            )
        )

        plot_support_progression(
            dynamic_plot_df,
            x_column="checkpoint",
            output_path=(
                args.output_dir
                / "dynamic_support_progression.png"
            ),
            title=(
                "Dynamic calibration-probe support "
                "across RL checkpoints"
            ),
        )

    # ========================================================
    # Report
    # ========================================================

    report = build_report(
        probe_path=args.probe,
        static_df=static_df,
        dynamic_df=dynamic_df,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 80)
    print("STATIC SUMMARY")
    print("=" * 80)

    print(
        static_df[
            [
                "stratum",
                "coverage_H",
                "coverage_U",
                "coverage_HU",
                "knn_support_ratio",
                "domain_auc_mean",
                "domain_auc_std",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    if len(
        dynamic_df
    ):

        print()
        print("=" * 80)
        print("DYNAMIC SUMMARY")
        print("=" * 80)

        print(
            dynamic_df[
                [
                    "epoch",
                    "reference_stratum",
                    "mean_H",
                    "mean_U",
                    "mean_HU",
                    "coverage_H",
                    "coverage_U",
                    "coverage_HU",
                    "knn_support_ratio",
                    "domain_auc_mean",
                    "domain_auc_std",
                ]
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}",
            )
        )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Static CSV:  {static_output_path}"
    )

    if len(
        dynamic_df
    ):

        print(
            f"Dynamic CSV: {dynamic_output_path}"
        )

    print(
        f"Report:      {report_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()