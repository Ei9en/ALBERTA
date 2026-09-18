#!/usr/bin/env python3

"""
ALBERTA - Dynamic Pareto Acquisition
====================================

Learn a two-objective local-response surrogate from the fixed
human-annotated Pareto calibration probe, refresh the candidate
self-play pool under the common RL10 learner, then select positions
in predicted response space.

Pipeline
--------

    historical RL1-10 self-play pool
                |
                v
    unique eligible FENs
                |
                v
    refresh H_10 / U_10 / HU_10
                |
                v
    ExtraTrees response surrogate
                |
                v
    predicted Delta_KL / |Delta_V|
                |
                v
    exact 2D Pareto sorting + crowding
                |
                v
    final Oracle queue

Dynamic features
----------------

At the branch point t = 10:

    H_10
        policy entropy of current RL10
        with the same BC6 opening prior used by ALBERTA

    U_10
        population variance over:
            RL1 ... RL9 + current RL10

        BC6 is excluded from U.
        RL10 contributes exactly once.

    HU_10
        H_10 * U_10

Surrogate
---------

Predictors:

    H_t
    U_t
    H_t * U_t

Targets:

    Delta_KL =
        KL(pi_before || pi_after)

    Delta_V =
        |V_after - V_before|

Selection
---------

Jointly maximize:

    predicted Delta_KL
    predicted |Delta_V|

using exact 2D non-dominated sorting followed by
crowding-distance truncation.

There is deliberately:

    - no scalarization of the two response objectives;
    - no I score in the Pareto selector;
    - no log transform of U;
    - no BC value-head contribution to U.

The log1p transform applies ONLY to the positive response targets
fitted by ExtraTrees.

Budget
------

Canonical ALBERTA sparse-supervision budget:

    0.0001 = 0.01%

The denominator is the raw RL1-10 self-play pool before candidate
filtering.

For the canonical 1,205,323-position pool this gives 121 queries.

Outputs
-------

    data/queue/oracle_queue_dynamic_pareto_10.jsonl
    data/pareto/pareto_selection_10.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path

import chess.variant
import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import spearmanr

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
)


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# ALBERTA imports
# ============================================================

import training.train_al as al
import training.train_rl as rl

from src.actions_space import ACTIONS, ACTION_TO_INDEX
from src.encoding import encode_boards
from src.models.actor_critic import ActorCritic
from src.models.resnet import ChessResNet
from src.selfplay.league import League


# ============================================================
# Defaults
# ============================================================

DEFAULT_LEARNER_EPOCH = 10

DEFAULT_PROBE_RESPONSE_PATH = (
    PROJECT_ROOT
    / "data"
    / "pareto"
    / "pareto_local_responses.jsonl"
)

DEFAULT_PROBE_QUEUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)

DEFAULT_POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_BASELINE_LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league"
)

DEFAULT_BC_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
)

DEFAULT_QUEUE_DIR = (
    PROJECT_ROOT
    / "data"
    / "queue"
)

DEFAULT_DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "pareto"
)

DEFAULT_BUDGET_FRACTION = 0.0001

DEFAULT_SEED = 42

DEFAULT_CV_FOLDS = 5
DEFAULT_N_TREES = 300

DEFAULT_REFRESH_BATCH_SIZE = 256
DEFAULT_PREDICT_CHUNK_SIZE = 100000

DEFAULT_CHANNELS = 32
DEFAULT_BLOCKS = 4

DEFAULT_LEAGUE_MAX_AGENTS = 12

DEFAULT_BC_PRIOR_EPOCH = 6
DEFAULT_BC_ANCHOR_EPOCH = 6

DEFAULT_OPENING_PRIOR_PLIES = getattr(
    rl,
    "DEFAULT_OPENING_PRIOR_PLIES",
    6,
)

DEFAULT_OPENING_PRIOR_STRENGTH = getattr(
    rl,
    "DEFAULT_OPENING_PRIOR_STRENGTH",
    1.0,
)

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
            "Build an ALBERTA Oracle queue using a learned "
            "two-objective local-response Pareto criterion."
        )
    )

    parser.add_argument(
        "--learner-epoch",
        type=int,
        default=DEFAULT_LEARNER_EPOCH,
    )

    parser.add_argument(
        "--probe-responses",
        type=Path,
        default=DEFAULT_PROBE_RESPONSE_PATH,
    )

    parser.add_argument(
        "--probe-queue",
        type=Path,
        default=DEFAULT_PROBE_QUEUE_PATH,
    )

    parser.add_argument(
        "--pool",
        type=Path,
        default=DEFAULT_POOL_PATH,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--baseline-league-dir",
        type=Path,
        default=DEFAULT_BASELINE_LEAGUE_DIR,
    )

    parser.add_argument(
        "--bc-dir",
        type=Path,
        default=DEFAULT_BC_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=DEFAULT_BUDGET_FRACTION,
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
    )

    parser.add_argument(
        "--trees",
        type=int,
        default=DEFAULT_N_TREES,
    )

    parser.add_argument(
        "--refresh-batch-size",
        type=int,
        default=DEFAULT_REFRESH_BATCH_SIZE,
    )

    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=DEFAULT_PREDICT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=DEFAULT_BLOCKS,
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

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Generic helpers
# ============================================================

def safe_float(value) -> float:

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def query_id_from_fen(fen: str) -> str:

    return uuid.uuid5(
        uuid.NAMESPACE_DNS,
        fen,
    ).hex


# ============================================================
# JSON loading
# ============================================================

def load_json_records(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):

        for key in (
            "data",
            "records",
            "positions",
            "samples",
            "stats",
            "uncertainty_stats",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        f"Unsupported JSON structure:\n{path}"
    )


def load_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    rows = []

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
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line "
                    f"{line_number} in:\n{path}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object at line "
                    f"{line_number} in:\n{path}"
                )

            rows.append(row)

    return rows


# ============================================================
# Probe response dataset
# ============================================================

def load_probe_dataset(
    path: Path,
    *,
    learner_epoch: int,
) -> list[dict]:

    rows = load_jsonl(path)

    cleaned = []

    for index, row in enumerate(rows):

        for required in (
            "query_id",
            "fen",
            "H",
            "U",
            "delta_kl",
            "delta_v",
        ):

            if required not in row:
                raise ValueError(
                    f"Probe response {index} is missing "
                    f"'{required}'."
                )

        H = safe_float(row.get("H"))
        U = safe_float(row.get("U"))

        delta_kl = safe_float(
            row.get("delta_kl")
        )

        delta_v = safe_float(
            row.get("delta_v")
        )

        if not all(
            np.isfinite(value)
            for value in (
                H,
                U,
                delta_kl,
                delta_v,
            )
        ):
            raise ValueError(
                f"Non-finite probe response at index {index}."
            )

        if U < 0.0:
            raise ValueError(
                f"Negative U at probe index {index}: {U}"
            )

        if delta_kl < 0.0 or delta_v < 0.0:
            raise ValueError(
                "Pareto targets must be response magnitudes. "
                f"Probe index {index}: "
                f"delta_kl={delta_kl}, delta_v={delta_v}"
            )

        fen = row.get("fen")

        if not isinstance(fen, str) or not fen:
            raise ValueError(
                f"Invalid FEN at probe index {index}."
            )

        chess.variant.AtomicBoard(fen)

        row_epoch = row.get(
            "learner_epoch"
        )

        if (
            row_epoch is not None
            and int(row_epoch) != learner_epoch
        ):
            raise ValueError(
                "Probe-response epoch mismatch: "
                f"row {index} has learner_epoch={row_epoch}, "
                f"expected {learner_epoch}."
            )

        normalized = dict(row)

        normalized["H"] = H
        normalized["U"] = U
        normalized["HU"] = H * U
        normalized["delta_kl"] = delta_kl
        normalized["delta_v"] = delta_v

        cleaned.append(normalized)

    if len(cleaned) < 20:
        raise RuntimeError(
            "Too few Pareto probe responses: "
            f"{len(cleaned)}."
        )

    return cleaned


# ============================================================
# Feature matrix
# ============================================================

def make_feature_matrix(
    rows: list[dict],
) -> np.ndarray:

    return np.asarray(
        [
            [
                float(row["H"]),
                float(row["U"]),
                float(row["H"])
                * float(row["U"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


# ============================================================
# Surrogate
# ============================================================

def create_surrogate(
    *,
    n_trees: int,
    seed: int,
) -> ExtraTreesRegressor:

    return ExtraTreesRegressor(
        n_estimators=n_trees,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=1.0,
    )


def safe_spearman(
    y_true,
    y_pred,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return np.nan

    if (
        np.std(y_true) < 1e-12
        or np.std(y_pred) < 1e-12
    ):
        return np.nan

    rho = spearmanr(
        y_true,
        y_pred,
    ).statistic

    if not np.isfinite(rho):
        return np.nan

    return float(rho)


def target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    return {
        "r2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),

        "mae":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "spearman":
            safe_spearman(
                y_true,
                y_pred,
            ),
    }


def cross_validate_surrogate(
    rows: list[dict],
    *,
    cv_folds: int,
    n_trees: int,
    seed: int,
) -> tuple[dict, np.ndarray]:

    X = make_feature_matrix(rows)

    delta_kl = np.asarray(
        [
            row["delta_kl"]
            for row in rows
        ],
        dtype=np.float64,
    )

    delta_v = np.asarray(
        [
            row["delta_v"]
            for row in rows
        ],
        dtype=np.float64,
    )

    if cv_folds > len(rows):
        raise ValueError(
            "--cv-folds cannot exceed the number "
            "of probe responses."
        )

    Y_log = np.column_stack(
        [
            np.log1p(delta_kl),
            np.log1p(delta_v),
        ]
    )

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    predicted_log = cross_val_predict(
        create_surrogate(
            n_trees=n_trees,
            seed=seed,
        ),
        X,
        Y_log,
        cv=cv,
        n_jobs=1,
    )

    predicted = np.maximum(
        np.expm1(predicted_log),
        0.0,
    )

    metrics = {
        "delta_kl":
            target_metrics(
                delta_kl,
                predicted[:, 0],
            ),

        "delta_v":
            target_metrics(
                delta_v,
                predicted[:, 1],
            ),
    }

    return metrics, predicted


def fit_final_surrogate(
    rows: list[dict],
    *,
    n_trees: int,
    seed: int,
) -> ExtraTreesRegressor:

    X = make_feature_matrix(rows)

    Y = np.asarray(
        [
            [
                math.log1p(
                    float(row["delta_kl"])
                ),
                math.log1p(
                    float(row["delta_v"])
                ),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    model = create_surrogate(
        n_trees=n_trees,
        seed=seed,
    )

    model.fit(X, Y)

    return model


# ============================================================
# Probe leakage exclusion
# ============================================================

def load_probe_query_ids(
    path: Path,
) -> set[str]:

    rows = load_jsonl(path)

    query_ids = set()

    for row in rows:

        fen = row.get("fen")

        if isinstance(fen, str) and fen:

            query_ids.add(
                query_id_from_fen(fen)
            )

            continue

        stored_id = row.get("query_id")

        if stored_id:
            query_ids.add(
                str(stored_id)
            )

    return query_ids


# ============================================================
# Candidate pool
# ============================================================

def build_candidate_pool(
    records: list[dict],
    excluded_ids: set[str],
) -> tuple[list[dict], dict]:

    print()
    print(
        "Filtering candidate pool...",
        flush=True,
    )

    candidates = []
    seen_ids = set()

    counters = {
        "excluded_probe": 0,
        "duplicate": 0,
        "ineligible": 0,
        "invalid": 0,
    }

    matched_probe_ids = set()

    for index, record in enumerate(records):

        if (
            index > 0
            and index % 100000 == 0
        ):
            print(
                f"  checked "
                f"{index:,} / {len(records):,}",
                flush=True,
            )

        if not isinstance(record, dict):
            counters["invalid"] += 1
            continue

        fen = record.get("fen")

        if (
            not isinstance(fen, str)
            or not fen
        ):
            counters["invalid"] += 1
            continue

        query_id = query_id_from_fen(fen)

        if query_id in excluded_ids:

            counters["excluded_probe"] += 1
            matched_probe_ids.add(
                query_id
            )

            continue

        if query_id in seen_ids:

            counters["duplicate"] += 1
            continue

        seen_ids.add(query_id)

        try:
            board = chess.variant.AtomicBoard(
                fen
            )

        except Exception:
            counters["invalid"] += 1
            continue

        if board.legal_moves.count() <= 1:
            counters["ineligible"] += 1
            continue

        candidates.append(
            {
                "source_index":
                    index,

                "query_id":
                    query_id,

                "fen":
                    fen,

                # Refreshed later.
                "H":
                    None,

                "U":
                    None,

                "HU":
                    None,

                "model":
                    record.get("model"),

                "epoch":
                    record.get(
                        "epoch",
                        -1,
                    ),

                "game_id":
                    record.get(
                        "game_id",
                        -1,
                    ),

                "ply":
                    record.get(
                        "ply",
                        -1,
                    ),
            }
        )

    diagnostics = {
        **counters,

        "matched_probe_ids":
            len(matched_probe_ids),

        "total_probe_ids":
            len(excluded_ids),

        "eligible_candidates":
            len(candidates),
    }

    return candidates, diagnostics


# ============================================================
# Model loading
# ============================================================

def load_current_model(
    *,
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Current learner checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"No model_state_dict in:\n{path}"
        )

    checkpoint_actions = checkpoint.get(
        "actions"
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(ACTIONS)
    ):
        raise ValueError(
            "Action-space mismatch: "
            f"{checkpoint_actions} != {len(ACTIONS)}"
        )

    model = al.build_actor_critic(
        args,
        device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model


def load_actor_critic_checkpoint(
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
):

    if not path.exists():
        raise FileNotFoundError(
            f"ActorCritic checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"No model_state_dict in:\n{path}"
        )

    checkpoint_actions = checkpoint.get(
        "actions"
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(ACTIONS)
    ):
        raise ValueError(
            "Action-space mismatch: "
            f"{checkpoint_actions} != {len(ACTIONS)}"
        )

    model = al.build_actor_critic(
        args,
        device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model


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
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def load_bc_opponent(
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

    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = ActorCritic(
        base_model
    ).to(device)

    model.eval()

    return model


# ============================================================
# Historical league
# ============================================================

def load_current_league(
    *,
    current_epoch: int,
    baseline_league_dir: Path,
    bc_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    max_agents: int,
) -> League:

    if current_epoch < 1:
        raise ValueError(
            "current_epoch must be >= 1."
        )

    league = League(
        max_agents=max_agents,
        protected_agents={
            "bc_epoch_6",
        },
    )

    print()
    print("=" * 72)
    print("HISTORICAL LEAGUE")
    print("=" * 72)

    bc6 = load_bc_opponent(
        epoch=DEFAULT_BC_ANCHOR_EPOCH,
        bc_dir=bc_dir,
        device=device,
    )

    league.add_agent(
        "bc_epoch_6",
        bc6,
        use_for_uncertainty=False,
    )

    print(
        "Loaded bc_epoch_6 "
        "(protected opponent, excluded from U)"
    )

    baseline_loaded = 0

    for epoch in range(
        1,
        current_epoch,
    ):

        name = (
            f"league_epoch_{epoch:03d}"
        )

        path = (
            baseline_league_dir
            / f"{name}.pt"
        )

        if not path.exists():
            raise FileNotFoundError(
                "Missing required baseline league snapshot:\n"
                f"{path}"
            )

        model = load_actor_critic_checkpoint(
            path,
            args,
            device,
        )

        league.add_agent(
            name,
            model,
            use_for_uncertainty=True,
        )

        baseline_loaded += 1

        print(
            f"Loaded baseline {name}"
        )

    print()

    print(
        f"Baseline snapshots inserted: "
        f"{baseline_loaded}"
    )

    print(
        f"Final historical league size: "
        f"{len(league)}"
    )

    print(
        f"League members: "
        f"{league.names()}"
    )

    print(
        f"Historical U contributors: "
        f"{league.uncertainty_names()}"
    )

    expected_uncertainty_count = (
        len(
            league.uncertainty_names()
        )
        + 1
    )

    print(
        "U critics including current learner: "
        f"{expected_uncertainty_count}"
    )

    current_name = (
        f"league_epoch_{current_epoch:03d}"
    )

    if current_name in league.names():
        raise RuntimeError(
            "Current learner is already present in the "
            "historical league. This would duplicate it in U."
        )

    return league


# ============================================================
# Opening prior
# ============================================================

def opening_prior_alpha(
    board: chess.variant.AtomicBoard,
    *,
    prior_plies: int,
    prior_strength: float,
) -> float:

    ply = board.ply()

    if (
        prior_plies <= 0
        or prior_strength <= 0.0
        or ply >= prior_plies
    ):
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
# Refresh candidate H/U/HU
# ============================================================

@torch.no_grad()
def refresh_candidate_features(
    *,
    candidates: list[dict],
    current_model,
    bc_policy,
    league: League,
    device: torch.device,
    batch_size: int,
    opening_prior_plies: int,
    opening_prior_strength: float,
) -> None:

    print()
    print("=" * 72)
    print("REFRESHING CANDIDATE FEATURES AT RL10")
    print("=" * 72)

    current_model.eval()
    bc_policy.eval()

    total = len(candidates)

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        batch_candidates = candidates[
            start:end
        ]

        boards = [
            chess.variant.AtomicBoard(
                candidate["fen"]
            )
            for candidate
            in batch_candidates
        ]

        encoded = encode_boards(
            boards
        ).to(device)

        # ----------------------------------------------------
        # Current RL10 policy
        # ----------------------------------------------------

        logits, _ = current_model(
            encoded
        )

        # ----------------------------------------------------
        # BC6 policy
        # ----------------------------------------------------

        bc_logits = bc_policy(
            encoded
        )

        if isinstance(
            bc_logits,
            tuple,
        ):
            bc_logits = bc_logits[0]

        # ----------------------------------------------------
        # U10 = RL1..RL9 + current RL10
        # ----------------------------------------------------

        U_tensor = league.uncertainty_batch(
            encoded,
            current_model=current_model,
        )

        U_values = (
            U_tensor
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # H10
        # ----------------------------------------------------

        for local_index, board in enumerate(
            boards
        ):

            legal_moves = list(
                board.legal_moves
            )

            if not legal_moves:
                raise RuntimeError(
                    "Candidate position has no legal moves."
                )

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

            legal_logits = logits[
                local_index,
                legal_index_tensor,
            ]

            alpha = opening_prior_alpha(
                board,
                prior_plies=opening_prior_plies,
                prior_strength=(
                    opening_prior_strength
                ),
            )

            if alpha > 0.0:

                bc_legal_logits = bc_logits[
                    local_index,
                    legal_index_tensor,
                ]

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

            H = float(
                entropy.item()
            )

            U = float(
                U_values[
                    local_index
                ]
            )

            if (
                not np.isfinite(H)
                or not np.isfinite(U)
                or U < 0.0
            ):
                raise RuntimeError(
                    "Non-finite refreshed H/U."
                )

            candidate = batch_candidates[
                local_index
            ]

            candidate["H"] = H
            candidate["U"] = U
            candidate["HU"] = H * U

        print(
            f"  refreshed {end:,} / {total:,}",
            flush=True,
        )


# ============================================================
# Chunked surrogate prediction
# ============================================================

def predict_candidates(
    model: ExtraTreesRegressor,
    candidates: list[dict],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:

    n = len(candidates)

    predicted_kl = np.empty(
        n,
        dtype=np.float64,
    )

    predicted_v = np.empty(
        n,
        dtype=np.float64,
    )

    print()
    print(
        "Projecting candidate pool into response space...",
        flush=True,
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            n,
        )

        X = make_feature_matrix(
            candidates[start:end]
        )

        prediction_log = model.predict(X)

        prediction = np.maximum(
            np.expm1(prediction_log),
            0.0,
        )

        predicted_kl[
            start:end
        ] = prediction[:, 0]

        predicted_v[
            start:end
        ] = prediction[:, 1]

        print(
            f"  predicted {end:,} / {n:,}",
            flush=True,
        )

    return predicted_kl, predicted_v


# ============================================================
# Fenwick tree
# ============================================================

class FenwickMax:

    def __init__(
        self,
        size: int,
    ) -> None:

        self.size = size

        self.tree = np.zeros(
            size + 1,
            dtype=np.int32,
        )

    def query(
        self,
        index: int,
    ) -> int:

        result = 0

        while index > 0:

            result = max(
                result,
                int(
                    self.tree[index]
                ),
            )

            index -= (
                index
                & -index
            )

        return result

    def update(
        self,
        index: int,
        value: int,
    ) -> None:

        while index <= self.size:

            if value > self.tree[index]:
                self.tree[index] = value

            index += (
                index
                & -index
            )


# ============================================================
# Exact 2D Pareto ranks
# ============================================================

def pareto_ranks_2d(
    objective_1,
    objective_2,
) -> np.ndarray:

    x = np.asarray(
        objective_1,
        dtype=np.float64,
    )

    y = np.asarray(
        objective_2,
        dtype=np.float64,
    )

    if len(x) != len(y):
        raise ValueError(
            "Objective arrays have different lengths."
        )

    if len(x) == 0:
        raise ValueError(
            "No objective values."
        )

    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
    ):
        raise ValueError(
            "Pareto objectives contain non-finite values."
        )

    print()
    print(
        "Computing exact 2D Pareto layers...",
        flush=True,
    )

    unique_y = np.unique(y)

    y_index = (
        len(unique_y)
        - np.searchsorted(
            unique_y,
            y,
            side="left",
        )
    )

    order = np.lexsort(
        (
            -y,
            -x,
        )
    )

    ranks = np.empty(
        len(x),
        dtype=np.int32,
    )

    fenwick = FenwickMax(
        len(unique_y)
    )

    position = 0

    while position < len(order):

        index = int(
            order[position]
        )

        current_x = x[index]
        current_y = y[index]

        end = position + 1

        while end < len(order):

            other = int(
                order[end]
            )

            if (
                x[other] != current_x
                or y[other] != current_y
            ):
                break

            end += 1

        yi = int(
            y_index[index]
        )

        previous_depth = fenwick.query(
            yi
        )

        rank = previous_depth + 1

        duplicate_group = order[
            position:end
        ]

        ranks[
            duplicate_group
        ] = rank

        fenwick.update(
            yi,
            rank,
        )

        position = end

    return ranks


# ============================================================
# Crowding distance
# ============================================================

def crowding_distance(
    indices,
    objective_1: np.ndarray,
    objective_2: np.ndarray,
) -> np.ndarray:

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    n = len(indices)

    if n == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if n <= 2:
        return np.full(
            n,
            np.inf,
            dtype=np.float64,
        )

    distance = np.zeros(
        n,
        dtype=np.float64,
    )

    for values in (
        objective_1[indices],
        objective_2[indices],
    ):

        order = np.argsort(
            values,
            kind="stable",
        )

        minimum = values[
            order[0]
        ]

        maximum = values[
            order[-1]
        ]

        span = maximum - minimum

        if span <= 0.0:
            continue

        distance[
            order[0]
        ] = np.inf

        distance[
            order[-1]
        ] = np.inf

        interior = order[1:-1]

        distance[
            interior
        ] += (
            values[
                order[2:]
            ]
            - values[
                order[:-2]
            ]
        ) / span

    return distance


# ============================================================
# Budgeted Pareto selection
# ============================================================

def select_budget(
    *,
    pareto_ranks: np.ndarray,
    predicted_kl: np.ndarray,
    predicted_v: np.ndarray,
    budget: int,
) -> np.ndarray:

    if budget <= 0:
        raise ValueError(
            "Budget must be positive."
        )

    if budget > len(pareto_ranks):
        raise ValueError(
            "Budget exceeds candidate count."
        )

    selected = []

    max_rank = int(
        np.max(pareto_ranks)
    )

    print()
    print(
        "Pareto front occupancy:",
        flush=True,
    )

    for rank in range(
        1,
        max_rank + 1,
    ):

        front = np.flatnonzero(
            pareto_ranks == rank
        )

        if len(front) == 0:
            continue

        remaining = (
            budget
            - len(selected)
        )

        print(
            f"  Front {rank:3d}: "
            f"{len(front):,}",
            flush=True,
        )

        if remaining <= 0:
            break

        if len(front) <= remaining:

            selected.extend(
                front.tolist()
            )

            continue

        distances = crowding_distance(
            front,
            predicted_kl,
            predicted_v,
        )

        local_order = sorted(
            range(len(front)),
            key=lambda local_index: (
                -distances[
                    local_index
                ],
                -predicted_kl[
                    front[
                        local_index
                    ]
                ],
                -predicted_v[
                    front[
                        local_index
                    ]
                ],
                int(
                    front[
                        local_index
                    ]
                ),
            ),
        )

        selected.extend(
            [
                int(
                    front[
                        local_index
                    ]
                )
                for local_index
                in local_order[
                    :remaining
                ]
            ]
        )

        break

    if len(selected) != budget:
        raise RuntimeError(
            f"Expected {budget} selected positions, "
            f"obtained {len(selected)}."
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


# ============================================================
# Queue writer
# ============================================================

def write_hmi_queue(
    *,
    candidates: list[dict],
    selected_indices: np.ndarray,
    output_path: Path,
    force: bool,
) -> None:

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not force
    ):
        raise FileExistsError(
            f"Output queue already exists:\n"
            f"{output_path}\n\n"
            "Use --force to overwrite it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for selected_index in selected_indices:

            candidate = candidates[
                int(selected_index)
            ]

            item = {
                "query_id":
                    candidate["query_id"],

                "fen":
                    candidate["fen"],

                "H":
                    float(candidate["H"]),

                "U":
                    float(candidate["U"]),

                "HU":
                    float(candidate["HU"]),

                "score":
                    None,

                "I_norm":
                    None,

                "threshold":
                    None,

                "model":
                    candidate.get(
                        "model"
                    ),

                "epoch":
                    candidate.get(
                        "epoch",
                        -1,
                    ),

                "game_id":
                    candidate.get(
                        "game_id",
                        -1,
                    ),

                "ply":
                    candidate.get(
                        "ply",
                        -1,
                    ),

                "created_at":
                    created_at,

                "status":
                    "pending",

                "oracle_move":
                    None,

                "oracle_confidence":
                    None,

                "oracle_situation":
                    None,

                "reward":
                    None,

                "answered_at":
                    None,
            }

            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Quantiles / diagnostics
# ============================================================

def quantile_summary(values) -> dict:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "min":
            float(np.min(values)),

        "p10":
            float(
                np.percentile(
                    values,
                    10,
                )
            ),

        "p25":
            float(
                np.percentile(
                    values,
                    25,
                )
            ),

        "median":
            float(np.median(values)),

        "p75":
            float(
                np.percentile(
                    values,
                    75,
                )
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90,
                )
            ),

        "max":
            float(np.max(values)),
    }


def print_quantiles(
    name: str,
    values,
) -> None:

    summary = quantile_summary(
        values
    )

    print()
    print(name)

    for key in (
        "min",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
        "max",
    ):
        print(
            f"  {key:>6s}: "
            f"{summary[key]:.6e}"
        )


def probe_correlations(
    rows: list[dict],
) -> dict:

    delta_kl = np.asarray(
        [
            row["delta_kl"]
            for row in rows
        ],
        dtype=np.float64,
    )

    delta_v = np.asarray(
        [
            row["delta_v"]
            for row in rows
        ],
        dtype=np.float64,
    )

    output = {}

    for feature_name in FEATURE_NAMES:

        if feature_name == "HU":

            values = np.asarray(
                [
                    row["H"]
                    * row["U"]
                    for row in rows
                ],
                dtype=np.float64,
            )

        else:

            values = np.asarray(
                [
                    row[feature_name]
                    for row in rows
                ],
                dtype=np.float64,
            )

        output[
            feature_name
        ] = {
            "delta_kl_spearman":
                safe_spearman(
                    values,
                    delta_kl,
                ),

            "delta_v_spearman":
                safe_spearman(
                    values,
                    delta_v,
                ),
        }

    return output


def write_diagnostics(
    *,
    path: Path,
    learner_epoch: int,
    seed: int,
    budget_fraction: float,
    raw_pool_size: int | None,
    budget: int | None,
    probe_count: int,
    cv_metrics: dict,
    correlations: dict,
    candidate_diagnostics: dict | None,
    selected_diagnostics: dict | None,
) -> None:

    payload = {
        "learner_epoch":
            learner_epoch,

        "seed":
            seed,

        "budget_fraction":
            budget_fraction,

        "raw_pool_size":
            raw_pool_size,

        "budget":
            budget,

        "probe_count":
            probe_count,

        "features":
            list(FEATURE_NAMES),

        "targets": [
            "delta_kl",
            "delta_v",
        ],

        "target_semantics": {
            "delta_kl":
                "KL(policy_before || policy_after)",

            "delta_v":
                "abs(value_after - value_before)",
        },

        "uncertainty_semantics": (
            "RL1..RL9 + current RL10 exactly once; "
            "BC6 excluded"
        ),

        "cv_metrics":
            cv_metrics,

        "probe_correlations":
            correlations,

        "candidate_filtering":
            candidate_diagnostics,

        "selected":
            selected_diagnostics,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.learner_epoch < 1:
        raise ValueError(
            "--learner-epoch must be >= 1."
        )

    if not (
        0.0
        < args.budget_fraction
        <= 1.0
    ):
        raise ValueError(
            "--budget-fraction must be in (0, 1]."
        )

    if args.cv_folds < 2:
        raise ValueError(
            "--cv-folds must be >= 2."
        )

    if args.trees <= 0:
        raise ValueError(
            "--trees must be positive."
        )

    if args.refresh_batch_size <= 0:
        raise ValueError(
            "--refresh-batch-size must be positive."
        )

    if args.predict_chunk_size <= 0:
        raise ValueError(
            "--predict-chunk-size must be positive."
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    output_path = args.output

    if output_path is None:

        output_path = (
            DEFAULT_QUEUE_DIR
            / (
                "oracle_queue_dynamic_pareto_"
                f"{args.learner_epoch}.jsonl"
            )
        )

    diagnostic_path = args.diagnostics

    if diagnostic_path is None:

        diagnostic_path = (
            DEFAULT_DIAGNOSTIC_DIR
            / (
                "pareto_selection_"
                f"{args.learner_epoch}.json"
            )
        )

    print()
    print("=" * 72)
    print("ALBERTA - DYNAMIC PARETO ACQUISITION")
    print("=" * 72)

    print(
        f"Learner epoch:   {args.learner_epoch}"
    )

    print(
        f"Probe responses: {args.probe_responses}"
    )

    print(
        f"Probe queue:     {args.probe_queue}"
    )

    print(
        f"Candidate pool:  {args.pool}"
    )

    print(
        f"Checkpoint:      {args.checkpoint}"
    )

    print(
        f"Output queue:    {output_path}"
    )

    print(
        f"Budget fraction: "
        f"{args.budget_fraction:.6f} "
        f"({args.budget_fraction * 100:.4f}%)"
    )

    print(
        f"Features:        {FEATURE_NAMES}"
    )

    print(
        f"Trees:           {args.trees}"
    )

    print(
        f"CV folds:        {args.cv_folds}"
    )

    print(
        f"Refresh batch:   {args.refresh_batch_size}"
    )

    print(
        f"Device:          {device}"
    )

    print(
        f"Seed:            {args.seed}"
    )

    # ========================================================
    # Probe responses
    # ========================================================

    probes = load_probe_dataset(
        args.probe_responses,
        learner_epoch=args.learner_epoch,
    )

    print()
    print(
        f"Probe responses loaded: {len(probes):,}"
    )

    correlations = probe_correlations(
        probes
    )

    print()
    print("=" * 72)
    print("PROBE RESPONSE CORRELATIONS")
    print("=" * 72)

    for feature_name, values in correlations.items():

        print(
            f"{feature_name:<4s} "
            f"| rho(KL)="
            f"{values['delta_kl_spearman']:+.4f} "
            f"| rho(|dV|)="
            f"{values['delta_v_spearman']:+.4f}"
        )

    # ========================================================
    # Cross-validation
    # ========================================================

    cv_metrics, _ = cross_validate_surrogate(
        probes,
        cv_folds=args.cv_folds,
        n_trees=args.trees,
        seed=args.seed,
    )

    print()
    print("=" * 72)
    print("SURROGATE CROSS-VALIDATION")
    print("=" * 72)

    for target_name in (
        "delta_kl",
        "delta_v",
    ):

        metrics = cv_metrics[
            target_name
        ]

        print(
            f"{target_name:<10s} "
            f"R2={metrics['r2']:+.4f} | "
            f"MAE={metrics['mae']:.6e} | "
            f"Spearman={metrics['spearman']:+.4f}"
        )

    if args.diagnostic_only:

        write_diagnostics(
            path=diagnostic_path,
            learner_epoch=args.learner_epoch,
            seed=args.seed,
            budget_fraction=args.budget_fraction,
            raw_pool_size=None,
            budget=None,
            probe_count=len(probes),
            cv_metrics=cv_metrics,
            correlations=correlations,
            candidate_diagnostics=None,
            selected_diagnostics=None,
        )

        print()
        print(
            "Diagnostic-only mode: no queue generated."
        )

        print(
            f"Diagnostics: {diagnostic_path}"
        )

        return

    # ========================================================
    # Fit final response surrogate
    # ========================================================

    surrogate = fit_final_surrogate(
        probes,
        n_trees=args.trees,
        seed=args.seed,
    )

    # ========================================================
    # Raw historical pool
    # ========================================================

    records = load_json_records(
        args.pool
    )

    raw_pool_size = len(records)

    # Same sparse-budget semantics as the canonical experiment.
    budget = max(
        1,
        int(
            math.ceil(
                raw_pool_size
                * args.budget_fraction
            )
        ),
    )

    print()
    print("=" * 72)
    print("CANDIDATE POOL")
    print("=" * 72)

    print(
        f"Raw pool size: {raw_pool_size:,}"
    )

    print(
        f"Budget:        {budget:,}"
    )

    if (
        args.budget_fraction
        == DEFAULT_BUDGET_FRACTION
        and raw_pool_size == 1_205_323
        and budget != 121
    ):
        raise RuntimeError(
            "Canonical ALBERTA pool should produce "
            "exactly 121 queries."
        )

    # ========================================================
    # Exclude calibration probe + deduplicate
    # ========================================================

    excluded_ids = load_probe_query_ids(
        args.probe_queue
    )

    (
        candidates,
        candidate_diagnostics,
    ) = build_candidate_pool(
        records,
        excluded_ids,
    )

    print()
    print(
        f"Eligible unique candidates: "
        f"{len(candidates):,}"
    )

    print(
        f"Calibration records removed: "
        f"{candidate_diagnostics['excluded_probe']:,}"
    )

    print(
        f"Matched calibration FENs: "
        f"{candidate_diagnostics['matched_probe_ids']:,}"
        f" / "
        f"{candidate_diagnostics['total_probe_ids']:,}"
    )

    print(
        f"Duplicate FENs removed: "
        f"{candidate_diagnostics['duplicate']:,}"
    )

    print(
        f"Ineligible positions removed: "
        f"{candidate_diagnostics['ineligible']:,}"
    )

    print(
        f"Invalid records removed: "
        f"{candidate_diagnostics['invalid']:,}"
    )

    if len(candidates) < budget:
        raise RuntimeError(
            "Eligible candidate pool is smaller "
            "than the annotation budget."
        )

    # ========================================================
    # Current RL10 learner
    # ========================================================

    current_model = load_current_model(
        path=args.checkpoint,
        args=args,
        device=device,
    )

    # ========================================================
    # BC6 opening prior
    # ========================================================

    bc_policy = load_bc_policy(
        epoch=DEFAULT_BC_PRIOR_EPOCH,
        bc_dir=args.bc_dir,
        device=device,
    )

    # ========================================================
    # Historical RL1..RL9 league
    # ========================================================

    league = load_current_league(
        current_epoch=args.learner_epoch,
        baseline_league_dir=(
            args.baseline_league_dir
        ),
        bc_dir=args.bc_dir,
        args=args,
        device=device,
        max_agents=args.league_max_agents,
    )

    # ========================================================
    # Refresh ALL candidate features at common RL10 state
    # ========================================================

    refresh_candidate_features(
        candidates=candidates,
        current_model=current_model,
        bc_policy=bc_policy,
        league=league,
        device=device,
        batch_size=args.refresh_batch_size,
        opening_prior_plies=(
            args.opening_prior_plies
        ),
        opening_prior_strength=(
            args.opening_prior_strength
        ),
    )

    candidate_H = np.asarray(
        [
            row["H"]
            for row in candidates
        ],
        dtype=np.float64,
    )

    candidate_U = np.asarray(
        [
            row["U"]
            for row in candidates
        ],
        dtype=np.float64,
    )

    candidate_HU = (
        candidate_H
        * candidate_U
    )

    print_quantiles(
        "Refreshed H10 - eligible pool",
        candidate_H,
    )

    print_quantiles(
        "Refreshed U10 - eligible pool",
        candidate_U,
    )

    print_quantiles(
        "Refreshed HU10 - eligible pool",
        candidate_HU,
    )

    # ========================================================
    # Predict local response
    # ========================================================

    (
        predicted_kl,
        predicted_v,
    ) = predict_candidates(
        surrogate,
        candidates,
        chunk_size=args.predict_chunk_size,
    )

    print_quantiles(
        "Predicted Delta KL - eligible pool",
        predicted_kl,
    )

    print_quantiles(
        "Predicted |Delta V| - eligible pool",
        predicted_v,
    )

    # ========================================================
    # Pareto sorting
    # ========================================================

    pareto_ranks = pareto_ranks_2d(
        predicted_kl,
        predicted_v,
    )

    selected = select_budget(
        pareto_ranks=pareto_ranks,
        predicted_kl=predicted_kl,
        predicted_v=predicted_v,
        budget=budget,
    )

    # ========================================================
    # Selected distributions
    # ========================================================

    selected_kl = predicted_kl[
        selected
    ]

    selected_v = predicted_v[
        selected
    ]

    selected_ranks = pareto_ranks[
        selected
    ]

    selected_H = candidate_H[
        selected
    ]

    selected_U = candidate_U[
        selected
    ]

    selected_HU = candidate_HU[
        selected
    ]

    print()
    print("=" * 72)
    print("FINAL PARETO SELECTION")
    print("=" * 72)

    print(
        f"Selected:          {len(selected):,}"
    )

    print(
        f"Pareto rank range: "
        f"{selected_ranks.min()} -> "
        f"{selected_ranks.max()}"
    )

    print_quantiles(
        "Selected H10",
        selected_H,
    )

    print_quantiles(
        "Selected U10",
        selected_U,
    )

    print_quantiles(
        "Selected HU10",
        selected_HU,
    )

    print_quantiles(
        "Selected predicted Delta KL",
        selected_kl,
    )

    print_quantiles(
        "Selected predicted |Delta V|",
        selected_v,
    )

    pool_response_rho = safe_spearman(
        predicted_kl,
        predicted_v,
    )

    selected_response_rho = safe_spearman(
        selected_kl,
        selected_v,
    )

    print()
    print(
        "Predicted response correlation:"
    )

    print(
        f"  eligible pool : "
        f"{pool_response_rho:+.4f}"
    )

    print(
        f"  selected set  : "
        f"{selected_response_rho:+.4f}"
    )

    # ========================================================
    # HMI queue
    # ========================================================

    write_hmi_queue(
        candidates=candidates,
        selected_indices=selected,
        output_path=output_path,
        force=args.force,
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    selected_diagnostics = {
        "n_selected":
            len(selected),

        "pareto_rank_min":
            int(
                selected_ranks.min()
            ),

        "pareto_rank_max":
            int(
                selected_ranks.max()
            ),

        "pool_response_spearman":
            pool_response_rho,

        "selected_response_spearman":
            selected_response_rho,

        "H":
            quantile_summary(
                selected_H
            ),

        "U":
            quantile_summary(
                selected_U
            ),

        "HU":
            quantile_summary(
                selected_HU
            ),

        "predicted_delta_kl":
            quantile_summary(
                selected_kl
            ),

        "predicted_delta_v":
            quantile_summary(
                selected_v
            ),

        # Full provenance stays here rather than in the strict
        # OracleQuery JSONL schema.
        "positions": [
            {
                "query_id":
                    candidates[
                        int(index)
                    ][
                        "query_id"
                    ],

                "predicted_delta_kl":
                    float(
                        predicted_kl[
                            int(index)
                        ]
                    ),

                "predicted_delta_v":
                    float(
                        predicted_v[
                            int(index)
                        ]
                    ),

                "pareto_rank":
                    int(
                        pareto_ranks[
                            int(index)
                        ]
                    ),
            }
            for index in selected
        ],
    }

    write_diagnostics(
        path=diagnostic_path,
        learner_epoch=args.learner_epoch,
        seed=args.seed,
        budget_fraction=args.budget_fraction,
        raw_pool_size=raw_pool_size,
        budget=budget,
        probe_count=len(probes),
        cv_metrics=cv_metrics,
        correlations=correlations,
        candidate_diagnostics=(
            candidate_diagnostics
        ),
        selected_diagnostics=(
            selected_diagnostics
        ),
    )

    print()
    print("=" * 72)
    print("DYNAMIC PARETO QUEUE CREATED")
    print("=" * 72)

    print(
        f"Queue:       {output_path}"
    )

    print(
        f"Diagnostics: {diagnostic_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()