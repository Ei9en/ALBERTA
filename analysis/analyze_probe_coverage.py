# ============================================================
# analyze_probe_coverage.py
#
# ALBERTA
#
# Goal
# ----
# Determine whether the original 200 calibration probes remain
# representative of later RL generations.
#
# Two complementary diagnostics are performed:
#
#   A. STATIC PROBE
#
#      Original H / U / HU stored at probe creation time are
#      kept fixed and compared with later RL strata.
#
#      Q_probe,10  vs  P_t
#
#
#   B. DYNAMIC PROBE
#
#      The SAME 200 FENs are reevaluated under later learners:
#
#          H_t(s)
#          U_t(s)
#          HU_t(s)
#
#      and compared with the corresponding contemporary RL
#      stratum.
#
#      Q_probe,t  vs  P_t
#
#
# Diagnostics:
#
#   1. Univariate feature coverage
#   2. Multivariate k-NN distance to probe support
#   3. Relative support-distance ratio
#   4. Two-sample classifier AUC
#
#
# Interpretation
# --------------
#
# Static degrades, dynamic stays stable:
#
#     -> same 200 FENs remain useful anchors
#     -> learner-dependent measurements became stale
#
#
# Static AND dynamic degrade:
#
#     -> original 200 FENs themselves become
#        non-representative
#     -> fresh calibration states would eventually be needed
#
#
# No additional human annotation is required.
# ============================================================


import sys
import json
from pathlib import Path

import chess
import chess.variant

import numpy as np
import pandas as pd

import torch

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
#
# IMPORTANT:
# Adapt ONLY THESE IMPORT PATHS if necessary.
#
# The imported objects must correspond to the same
# implementation used by train_rl_league_colab.py.
# ============================================================

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.encoding import encode_boards
from src.actions_space import ACTIONS, ACTION_TO_INDEX

from src.selfplay.league import League


# ============================================================
# Device
#
# CPU is enough for 200 probes and avoids GPU dependency.
# ============================================================

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# Original probe
# ============================================================

PROBE_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)


# ============================================================
# RL strata
# ============================================================

STRATA_FILES = {

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


# ============================================================
# Model checkpoints
# ============================================================

BC_EPOCH_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
)


RL_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
)


LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league"
)


# ============================================================
# Dynamic probe checkpoints
#
# Each learner is compared against the closest available
# self-play stratum.
#
# Note:
#
# RL15 and RL20 both use RL11-20 because stats are stored
# only in 10-epoch strata.
#
# RL25 uses RL21-30.
# ============================================================

DYNAMIC_CHECKPOINTS = {
    10: "RL01-10",
    20: "RL11-20",
    30: "RL21-30",
    40: "RL31-40",
    50: "RL41-50",
    60: "RL51-60",
}


# ============================================================
# League configuration
# ============================================================

LEAGUE_MAX_AGENTS = 22


# ============================================================
# BC opening prior
#
# MUST match training.
#
# From your training comments:
#
#   ply 0 -> 100 %
#   ply 1 -> 83 %
#   ...
#   ply 5 -> 17 %
#   ply 6 -> 0 %
#
# => OPENING_PRIOR_PLIES = 6
#
# Verify OPENING_PRIOR_STRENGTH against the training config.
# ============================================================

OPENING_PRIOR_PLIES = 6

OPENING_PRIOR_STRENGTH = 1.0


# ============================================================
# Feature configuration
#
# Historical ALBERTA stats use:
#
#     H
#     U
#     HU = H * U
#
# Therefore USE_LOG_U stays False unless you explicitly
# decide to analyze a transformed feature space.
# ============================================================

USE_LOG_U = False

TAU = 0.5


FEATURE_NAMES = [
    "H",
    "U_feature",
    "HU",
]


# ============================================================
# k-NN configuration
# ============================================================

K_NEIGHBORS = 5


# ============================================================
# Pool subsampling
# ============================================================

MAX_POOL_SAMPLE = 20000


# ============================================================
# Dynamic inference batch size
# ============================================================

DYNAMIC_BATCH_SIZE = 64


# ============================================================
# Two-sample classifier
# ============================================================

CLASSIFIER_REPEATS = 10

CLASSIFIER_FOLDS = 5

RANDOM_SEED = 42


# ============================================================
# Outputs
# ============================================================

STATIC_OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "probe_coverage_diagnostics.csv"
)


DYNAMIC_OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "probe_dynamic_coverage_diagnostics.csv"
)


# ============================================================
# Generic JSON / JSONL loader
# ============================================================

def load_records(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n"
            f"{path}"
        )

    # ========================================================
    # JSONL
    # ========================================================

    if (
        path.suffix.lower()
        ==
        ".jsonl"
    ):

        records = []

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            for (
                line_number,
                line,
            ) in enumerate(
                f,
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
                        f"{line_number} in:\n"
                        f"{path}"
                    ) from exc

                records.append(
                    record
                )

        return records

    # ========================================================
    # JSON
    # ========================================================

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    # --------------------------------------------------------
    # Direct list
    # --------------------------------------------------------

    if isinstance(
        data,
        list,
    ):

        return data

    # --------------------------------------------------------
    # Common containers
    # --------------------------------------------------------

    if isinstance(
        data,
        dict,
    ):

        candidate_keys = (

            "data",
            "records",
            "positions",
            "samples",
            "stats",
            "uncertainty_stats",
        )

        for key in candidate_keys:

            value = data.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                return value

    raise ValueError(
        "Unsupported JSON structure in:\n"
        f"{path}"
    )


# ============================================================
# Feature extraction
# ============================================================

def extract_H_U(
    record,
):

    H = float(
        record["H"]
    )

    U = float(
        record["U"]
    )

    return H, U


# ============================================================
# U transformation
# ============================================================

def transform_U(
    U,
):

    if USE_LOG_U:

        return np.log1p(
            U
            /
            TAU
        )

    return U


# ============================================================
# Build stored feature matrix
# ============================================================

def build_features(
    records,
):

    features = []

    skipped = 0

    for record in records:

        try:

            H, U = extract_H_U(
                record
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            skipped += 1
            continue

        if not (
            np.isfinite(H)
            and
            np.isfinite(U)
        ):

            skipped += 1
            continue

        if U < 0:

            skipped += 1
            continue

        U_feature = transform_U(
            U
        )

        HU = (
            H
            *
            U_feature
        )

        features.append(
            [
                H,
                U_feature,
                HU,
            ]
        )

    if not features:

        raise RuntimeError(
            "No valid feature records found."
        )

    X = np.asarray(
        features,
        dtype=np.float64,
    )

    return (
        X,
        skipped,
    )


# ============================================================
# Pool subsampling
# ============================================================

def subsample_pool(
    X,
    rng,
):

    if (
        len(X)
        <=
        MAX_POOL_SAMPLE
    ):

        return X

    indices = rng.choice(
        len(X),
        size=MAX_POOL_SAMPLE,
        replace=False,
    )

    return X[
        indices
    ]


# ============================================================
# Feature coverage
# ============================================================

def compute_feature_coverage(
    X_probe,
    X_pool,
):

    result = {}

    for (
        feature_index,
        feature_name,
    ) in enumerate(
        FEATURE_NAMES
    ):

        probe_values = (
            X_probe[
                :,
                feature_index
            ]
        )

        pool_values = (
            X_pool[
                :,
                feature_index
            ]
        )

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
            (pool_values >= low)
            &
            (pool_values <= high)
        )

        coverage = float(
            np.mean(
                inside
            )
        )

        result[
            f"coverage_{feature_name}"
        ] = coverage

        result[
            f"probe_min_{feature_name}"
        ] = low

        result[
            f"probe_max_{feature_name}"
        ] = high

    return result


# ============================================================
# Probe leave-one-out self-distance
# ============================================================

def compute_probe_reference_distances(
    X_probe_scaled,
):

    if (
        len(
            X_probe_scaled
        )
        <=
        1
    ):

        raise RuntimeError(
            "Probe must contain at least 2 states."
        )

    n_neighbors = min(
        K_NEIGHBORS + 1,
        len(
            X_probe_scaled
        ),
    )

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    knn.fit(
        X_probe_scaled
    )

    distances, _ = knn.kneighbors(
        X_probe_scaled
    )

    # --------------------------------------------------------
    # Remove self-neighbor.
    # --------------------------------------------------------

    distances = distances[
        :,
        1:
    ]

    return np.mean(
        distances,
        axis=1,
    )


# ============================================================
# Pool -> probe distances
# ============================================================

def compute_pool_to_probe_distances(
    X_probe_scaled,
    X_pool_scaled,
):

    n_neighbors = min(
        K_NEIGHBORS,
        len(
            X_probe_scaled
        ),
    )

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )

    knn.fit(
        X_probe_scaled
    )

    distances, _ = knn.kneighbors(
        X_pool_scaled
    )

    return np.mean(
        distances,
        axis=1,
    )


# ============================================================
# Two-sample classifier
#
# class 0 = probe
# class 1 = contemporary RL pool
# ============================================================

def compute_two_sample_auc(
    X_probe,
    X_pool,
    rng,
):

    n_probe = len(
        X_probe
    )

    if (
        len(
            X_pool
        )
        <
        n_probe
    ):

        raise RuntimeError(
            "RL pool smaller than probe:\n"
            f"pool={len(X_pool)}, "
            f"probe={n_probe}"
        )

    repeat_aucs = []

    for repeat in range(
        CLASSIFIER_REPEATS
    ):

        pool_indices = rng.choice(
            len(
                X_pool
            ),
            size=n_probe,
            replace=False,
        )

        X_pool_balanced = (
            X_pool[
                pool_indices
            ]
        )

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
            n_splits=CLASSIFIER_FOLDS,
            shuffle=True,
            random_state=(
                RANDOM_SEED
                +
                repeat
            ),
        )

        fold_aucs = []

        for (
            train_indices,
            test_indices,
        ) in cv.split(
            X,
            y,
        ):

            classifier = ExtraTreesClassifier(

                n_estimators=300,

                min_samples_leaf=3,

                max_features=1.0,

                class_weight="balanced",

                n_jobs=-1,

                random_state=(
                    RANDOM_SEED
                    +
                    repeat
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

            probabilities = (
                classifier.predict_proba(
                    X[
                        test_indices
                    ]
                )[
                    :,
                    1
                ]
            )

            auc = roc_auc_score(
                y[
                    test_indices
                ],
                probabilities,
            )

            # ------------------------------------------------
            # Distinguishability is symmetric.
            # ------------------------------------------------

            auc = max(
                auc,
                1.0 - auc,
            )

            fold_aucs.append(
                auc
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
                    repeat_aucs
                )
            ),
    }


# ============================================================
# Distance summary
# ============================================================

def add_distance_summary(
    prefix,
    distances,
    result,
):

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
# Model construction
#
# Same architecture as training.
# ============================================================

def make_actor_critic():

    base_model = ChessResNet(

        num_actions=len(
            ACTIONS
        ),

        channels=32,

        blocks=4,
    )

    model = ActorCritic(
        base_model
    )

    model.to(
        DEVICE
    )

    return model


# ============================================================
# Load ActorCritic checkpoint
# ============================================================

def load_actor_critic_checkpoint(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model = make_actor_critic()

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# ============================================================
# Load BC policy model
#
# This follows your RL initialization:
#
#     bc_model = ChessResNet(...)
#     bc_model.load_state_dict(checkpoint["model_state_dict"])
#
# Used only for the BC opening prior.
# ============================================================

def load_bc_policy(
    epoch,
):

    path = (
        BC_EPOCH_DIR
        /
        f"bc_epoch_{epoch}.pt"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"BC checkpoint missing:\n"
            f"{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
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
        ]
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# Load BC as league agent
#
# Your league expects:
#
#     policy, value = model(x)
#
# Therefore BC snapshots used by the league must be wrapped
# into ActorCritic exactly as during RL initialization.
#
# NOTE:
# If your existing load_bc_agent() initializes the value head
# differently, replace this function by that existing helper.
# ============================================================

def load_bc_league_agent(
    epoch,
):

    path = (
        BC_EPOCH_DIR
        /
        f"bc_epoch_{epoch}.pt"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"BC checkpoint missing:\n"
            f"{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
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
        ]
    )

    model = ActorCritic(
        base_model
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# Reconstruct league at RL epoch t
# ============================================================

def load_league_for_epoch(
    epoch,
):

    league = League(
        max_agents=LEAGUE_MAX_AGENTS
    )

    # --------------------------------------------------------
    # Fixed BC anchors
    # --------------------------------------------------------

    bc6 = load_bc_league_agent(
        6
    )

    bc7 = load_bc_league_agent(
        7
    )

    league.add_agent(
        "bc_epoch_6",
        bc6,
    )

    league.add_agent(
        "bc_epoch_7",
        bc7,
    )

    loaded_snapshots = 0

    # --------------------------------------------------------
    # Historical RL snapshots
    # --------------------------------------------------------

    for snapshot_epoch in range(
        1,
        epoch + 1,
    ):

        name = (
            f"league_epoch_"
            f"{snapshot_epoch:03d}"
        )

        path = (
            LEAGUE_DIR
            /
            f"{name}.pt"
        )

        if not path.exists():

            continue

        snapshot = (
            load_actor_critic_checkpoint(
                path
            )
        )

        league.add_agent(
            name,
            snapshot,
        )

        loaded_snapshots += 1

    print(
        f"League RL{epoch}: "
        f"{len(league)} agents "
        f"({loaded_snapshots} RL snapshots)"
    )

    return league


# ============================================================
# Opening prior strength
#
# Exact formula from training.
# ============================================================

def opening_prior_strength(
    ply,
):

    if (
        OPENING_PRIOR_PLIES
        <=
        0
    ):

        return 0.0

    if (
        ply
        >=
        OPENING_PRIOR_PLIES
    ):

        return 0.0

    progress = (
        ply
        /
        OPENING_PRIOR_PLIES
    )

    return (
        OPENING_PRIOR_STRENGTH
        *
        (
            1.0
            -
            progress
        )
    )


# ============================================================
# Current entropy H_t
#
# Reproduces choose_moves():
#
#   - RL logits
#   - legal moves only
#   - BC prior if opening
#   - entropy BEFORE sampling temperature
# ============================================================

@torch.no_grad()
def compute_entropy_batch(
    boards,
    current_model,
    bc_policy,
):

    if not boards:

        return torch.empty(
            0,
            dtype=torch.float32,
        )

    # ========================================================
    # RL forward
    # ========================================================

    x = encode_boards(
        boards
    ).to(
        DEVICE
    )

    policies, _ = current_model(
        x
    )

    # ========================================================
    # Evaluate BC only where opening prior is active
    # ========================================================

    opening_indices = [

        i
        for (
            i,
            board,
        ) in enumerate(
            boards
        )

        if opening_prior_strength(
            board.ply()
        )
        >
        0.0
    ]

    bc_policy_lookup = {}

    if opening_indices:

        opening_boards = [

            boards[i]

            for i in opening_indices
        ]

        opening_x = encode_boards(
            opening_boards
        ).to(
            DEVICE
        )

        bc_policies = bc_policy(
            opening_x
        )

        # ----------------------------------------------------
        # Defensive handling:
        #
        # If ChessResNet returns a tuple, keep first element.
        # ----------------------------------------------------

        if isinstance(
            bc_policies,
            tuple,
        ):

            bc_policies = (
                bc_policies[0]
            )

        for (
            local_index,
            global_index,
        ) in enumerate(
            opening_indices
        ):

            bc_policy_lookup[
                global_index
            ] = (
                bc_policies[
                    local_index
                ]
            )

    # ========================================================
    # Position-wise legal entropy
    # ========================================================

    entropies = []

    for (
        i,
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

            for move in legal_moves
        ]

        legal_index_tensor = torch.tensor(
            legal_indices,
            dtype=torch.long,
            device=DEVICE,
        )

        legal_logits = (
            policies[
                i
            ][
                legal_index_tensor
            ]
        )

        # ====================================================
        # BC opening prior
        # ====================================================

        alpha = opening_prior_strength(
            board.ply()
        )

        if (
            alpha > 0.0
            and
            i in bc_policy_lookup
        ):

            bc_logits = (
                bc_policy_lookup[
                    i
                ]
            )

            bc_legal_logits = (
                bc_logits[
                    legal_index_tensor
                ]
            )

            bc_log_probs = (
                torch.log_softmax(
                    bc_legal_logits,
                    dim=0,
                )
            )

            legal_logits = (
                legal_logits
                +
                alpha
                *
                bc_log_probs
            )

        # ====================================================
        # Entropy before temperature
        # ====================================================

        entropy_log_probs = (
            torch.log_softmax(
                legal_logits,
                dim=0,
            )
        )

        entropy_probs = torch.exp(
            entropy_log_probs
        )

        entropy = -(
            entropy_probs
            *
            entropy_log_probs
        ).sum()

        entropies.append(
            entropy.item()
        )

    return torch.tensor(
        entropies,
        dtype=torch.float32,
    )


# ============================================================
# Recompute H_t, U_t, HU_t for SAME probe FENs
# ============================================================

@torch.no_grad()
def recompute_dynamic_probe_features(
    probe_records,
    epoch,
    current_model,
    league,
    bc_policy,
):

    fens = []

    for record in probe_records:

        fen = record.get(
            "fen"
        )

        if fen is None:

            raise KeyError(
                "Probe record has no 'fen' field."
            )

        fens.append(
            fen
        )

    all_features = []

    for start in range(
        0,
        len(fens),
        DYNAMIC_BATCH_SIZE,
    ):

        batch_fens = fens[
            start:
            start + DYNAMIC_BATCH_SIZE
        ]

        boards = [

            chess.variant.AtomicBoard(
                fen
            )

            for fen in batch_fens
        ]

        # ====================================================
        # H_t
        # ====================================================

        H_values = compute_entropy_batch(
            boards=boards,
            current_model=current_model,
            bc_policy=bc_policy,
        )

        # ====================================================
        # U_t
        # ====================================================

        x = encode_boards(
            boards
        ).to(
            DEVICE
        )

        U_values = league.uncertainty_batch(
            x,
            current_model=current_model,
        )

        H_values = (
            H_values
            .detach()
            .cpu()
            .tolist()
        )

        U_values = (
            U_values
            .detach()
            .cpu()
            .tolist()
        )

        # ====================================================
        # HU_t
        # ====================================================

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

            U_feature = transform_U(
                U
            )

            HU = (
                H
                *
                U_feature
            )

            all_features.append(
                [
                    H,
                    U_feature,
                    HU,
                ]
            )

    X = np.asarray(
        all_features,
        dtype=np.float64,
    )

    if (
        len(X)
        !=
        len(
            probe_records
        )
    ):

        raise RuntimeError(
            f"RL{epoch}: dynamic probe "
            f"returned {len(X)} positions "
            f"instead of "
            f"{len(probe_records)}."
        )

    return X


# ============================================================
# Generic support analysis
# ============================================================

def analyze_support(
    X_probe,
    X_pool,
    rng,
):

    # ========================================================
    # Scale relative to current probe
    # ========================================================

    scaler = StandardScaler()

    X_probe_scaled = (
        scaler.fit_transform(
            X_probe
        )
    )

    X_pool_subsample = subsample_pool(
        X_pool,
        rng,
    )

    X_pool_scaled = (
        scaler.transform(
            X_pool_subsample
        )
    )

    # ========================================================
    # Probe internal reference
    # ========================================================

    reference_distances = (
        compute_probe_reference_distances(
            X_probe_scaled
        )
    )

    reference_median = float(
        np.median(
            reference_distances
        )
    )

    if (
        reference_median
        <=
        1e-12
    ):

        raise RuntimeError(
            "Probe reference distance "
            "is approximately zero."
        )

    # ========================================================
    # Marginal coverage
    # ========================================================

    coverage_result = (
        compute_feature_coverage(
            X_probe,
            X_pool_subsample,
        )
    )

    # ========================================================
    # Multivariate distance
    # ========================================================

    pool_distances = (
        compute_pool_to_probe_distances(
            X_probe_scaled,
            X_pool_scaled,
        )
    )

    pool_distance_median = float(
        np.median(
            pool_distances
        )
    )

    support_ratio = (
        pool_distance_median
        /
        reference_median
    )

    # ========================================================
    # Domain classifier
    # ========================================================

    auc_result = (
        compute_two_sample_auc(
            X_probe,
            X_pool,
            rng,
        )
    )

    result = {

        "probe_reference_knn_median":
            reference_median,

        "knn_support_ratio":
            float(
                support_ratio
            ),
    }

    result.update(
        coverage_result
    )

    result.update(
        auc_result
    )

    add_distance_summary(
        "pool_probe_distance",
        pool_distances,
        result,
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ALBERTA - PROBE SUPPORT / "
        "DISTRIBUTION SHIFT ANALYSIS"
    )

    print(
        "=" * 72
    )

    print(
        f"Probe file:\n"
        f"{PROBE_FILE}"
    )

    print()

    print(
        "Features:"
    )

    for name in FEATURE_NAMES:

        print(
            f"  - {name}"
        )

    print()

    print(
        f"Use log(U): "
        f"{USE_LOG_U}"
    )

    print(
        f"k-NN: "
        f"{K_NEIGHBORS}"
    )

    print(
        f"Max pool sample: "
        f"{MAX_POOL_SAMPLE}"
    )


    # ========================================================
    # Original probe
    # ========================================================

    probe_records = load_records(
        PROBE_FILE
    )

    (
        X_original_probe,
        skipped_probe,
    ) = build_features(
        probe_records
    )

    print()
    print(
        "-" * 72
    )

    print(
        "ORIGINAL PROBE"
    )

    print(
        "-" * 72
    )

    print(
        f"Raw records   : "
        f"{len(probe_records)}"
    )

    print(
        f"Valid records : "
        f"{len(X_original_probe)}"
    )

    print(
        f"Skipped       : "
        f"{skipped_probe}"
    )

    if (
        len(
            X_original_probe
        )
        <
        20
    ):

        raise RuntimeError(
            "Probe set too small."
        )


    # ========================================================
    # Load all strata once
    # ========================================================

    stratum_feature_cache = {}

    for (
        stratum_name,
        stratum_file,
    ) in STRATA_FILES.items():

        print()
        print(
            f"Loading {stratum_name}..."
        )

        records = load_records(
            stratum_file
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
            f"  records = "
            f"{len(X_pool)}"
        )

        print(
            f"  skipped = "
            f"{skipped}"
        )


    # ========================================================
    # A. STATIC ANALYSIS
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "STATIC PROBE ANALYSIS"
    )

    print(
        "=" * 72
    )

    static_results = []

    for (
        stratum_name,
        X_pool,
    ) in stratum_feature_cache.items():

        print()
        print(
            "-" * 72
        )

        print(
            stratum_name
        )

        print(
            "-" * 72
        )

        result = analyze_support(
            X_probe=X_original_probe,
            X_pool=X_pool,
            rng=rng,
        )

        result[
            "stratum"
        ] = stratum_name

        static_results.append(
            result
        )

        print(
            f"Coverage H  : "
            f"{result['coverage_H']:.2%}"
        )

        print(
            f"Coverage U  : "
            f"{result['coverage_U_feature']:.2%}"
        )

        print(
            f"Coverage HU : "
            f"{result['coverage_HU']:.2%}"
        )

        print(
            f"kNN ratio   : "
            f"{result['knn_support_ratio']:.3f}x"
        )

        print(
            f"Domain AUC  : "
            f"{result['domain_auc_mean']:.4f}"
            " +/- "
            f"{result['domain_auc_std']:.4f}"
        )


    static_dataframe = pd.DataFrame(
        static_results
    )

    static_dataframe.to_csv(
        STATIC_OUTPUT_CSV,
        index=False,
    )


    print()
    print(
        "=" * 72
    )

    print(
        "STATIC SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        static_dataframe[
            [
                "stratum",

                "coverage_H",

                "coverage_U_feature",

                "coverage_HU",

                "knn_support_ratio",

                "domain_auc_mean",

                "domain_auc_std",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # ========================================================
    # B. DYNAMIC ANALYSIS
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "DYNAMIC PROBE ANALYSIS"
    )

    print(
        "=" * 72
    )

    print(
        "Same 200 FENs are reevaluated "
        "under each learner."
    )


    # --------------------------------------------------------
    # BC7 policy used by opening prior
    # --------------------------------------------------------

    bc_policy = load_bc_policy(
        7
    )


    dynamic_results = []


    for (
        epoch,
        reference_stratum,
    ) in DYNAMIC_CHECKPOINTS.items():

        print()
        print(
            "=" * 72
        )

        print(
            f"DYNAMIC RL{epoch}"
        )

        print(
            "=" * 72
        )

        checkpoint_path = (
            RL_CHECKPOINT_DIR
            /
            f"rl_epoch_{epoch}.pt"
        )

        if not checkpoint_path.exists():

            print(
                "WARNING: missing checkpoint:"
            )

            print(
                checkpoint_path
            )

            continue

        if (
            reference_stratum
            not in
            stratum_feature_cache
        ):

            print(
                "WARNING: missing reference "
                f"stratum {reference_stratum}"
            )

            continue


        # ====================================================
        # Learner
        # ========================================================

        current_model = (
            load_actor_critic_checkpoint(
                checkpoint_path
            )
        )


        # ====================================================
        # League
        # ========================================================

        league = load_league_for_epoch(
            epoch
        )


        # ====================================================
        # Same 200 FENs, current features
        # ========================================================

        X_dynamic_probe = (
            recompute_dynamic_probe_features(

                probe_records=probe_records,

                epoch=epoch,

                current_model=current_model,

                league=league,

                bc_policy=bc_policy,
            )
        )


        # ====================================================
        # Contemporary pool
        # ========================================================

        X_pool = (
            stratum_feature_cache[
                reference_stratum
            ]
        )


        # ====================================================
        # Support diagnostics
        # ========================================================

        result = analyze_support(

            X_probe=X_dynamic_probe,

            X_pool=X_pool,

            rng=rng,
        )


        result[
            "epoch"
        ] = epoch

        result[
            "reference_stratum"
        ] = reference_stratum


        # ====================================================
        # Track dynamic probe means
        # ========================================================

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
            f"Reference stratum : "
            f"{reference_stratum}"
        )

        print()

        print(
            "Dynamic probe means"
        )

        print(
            f"  H  = "
            f"{result['mean_H']:.6e}"
        )

        print(
            f"  U  = "
            f"{result['mean_U']:.6e}"
        )

        print(
            f"  HU = "
            f"{result['mean_HU']:.6e}"
        )

        print()

        print(
            "Coverage"
        )

        print(
            f"  H  : "
            f"{result['coverage_H']:.2%}"
        )

        print(
            f"  U  : "
            f"{result['coverage_U_feature']:.2%}"
        )

        print(
            f"  HU : "
            f"{result['coverage_HU']:.2%}"
        )

        print()

        print(
            f"kNN support ratio : "
            f"{result['knn_support_ratio']:.3f}x"
        )

        print(
            f"Domain AUC        : "
            f"{result['domain_auc_mean']:.4f}"
            " +/- "
            f"{result['domain_auc_std']:.4f}"
        )


        # ====================================================
        # Free memory
        # ========================================================

        del current_model

        del league


    # ========================================================
    # Dynamic summary
    # ========================================================

    if dynamic_results:

        dynamic_dataframe = pd.DataFrame(
            dynamic_results
        )

        dynamic_dataframe = (
            dynamic_dataframe.sort_values(
                "epoch"
            )
        )

        dynamic_dataframe.to_csv(
            DYNAMIC_OUTPUT_CSV,
            index=False,
        )


        print()
        print(
            "=" * 72
        )

        print(
            "DYNAMIC SUMMARY"
        )

        print(
            "=" * 72
        )

        print(
            dynamic_dataframe[
                [
                    "epoch",

                    "reference_stratum",

                    "mean_H",

                    "mean_U",

                    "mean_HU",

                    "coverage_H",

                    "coverage_U_feature",

                    "coverage_HU",

                    "knn_support_ratio",

                    "domain_auc_mean",

                    "domain_auc_std",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )


    # ========================================================
    # Interpretation
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 72
    )

    print(
        """
STATIC diagnostic
-----------------

The original probe H/U/HU values remain frozen.

If:

    static kNN ratio ↑
    static AUC       ↑

the original calibration domain becomes increasingly
different from later RL state distributions.


DYNAMIC diagnostic
------------------

The SAME 200 probe FENs are reevaluated under learner t:

    H_t(s)
    U_t(s)
    HU_t(s)

If STATIC support degrades while DYNAMIC support remains
approximately stable:

    -> the 200 FENs remain useful calibration anchors
    -> their learner-dependent measurements became stale
    -> recycling the same human annotations may be viable


If BOTH STATIC and DYNAMIC support degrade:

    -> the FENs themselves cease to represent the evolving
       learner distribution
    -> merely recomputing their response is insufficient
    -> new calibration states would eventually be needed


Important:
RL15 and RL20 are both compared against RL11-20 because the
available self-play statistics are stored by 10-epoch strata.
Therefore this is an exploratory support diagnostic rather
than an exact per-epoch distribution comparison.
"""
    )


    print(
        "=" * 72
    )

    print(
        "Saved:"
    )

    print(
        STATIC_OUTPUT_CSV
    )

    print(
        DYNAMIC_OUTPUT_CSV
    )

    print(
        "=" * 72
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()