#!/usr/bin/env python3

"""
ALBERTA - Local Oracle Annotation Response
==========================================

Measure the immediate local response of a fixed ActorCritic model
to individual Oracle annotations.

Scientific question
-------------------

For an Oracle annotation i, starting from the exact same model
parameters theta_0 each time:

    theta_0 --Oracle annotation i--> theta_i

how strongly does the model respond locally?

The analysis measures:

Policy response
    - Oracle-action probability change
    - Oracle surprise reduction
    - policy entropy change
    - KL(pi_before || pi_after)

Value response
    - value prediction change
    - Oracle value-error reduction

Gradient / update response
    - raw Oracle gradient norm before clipping
    - realized one-step displacement under the diagnostic
      fresh-Adam micro-update

Important distinction
---------------------

This script measures LOCAL SUSCEPTIBILITY.

It does NOT measure:
    - downstream playing-strength improvement
    - long-horizon annotation utility
    - PPO update direction
    - gradient alignment with future RL learning

Each annotation is processed independently from the exact same
starting ActorCritic checkpoint.

The Oracle objective is delegated directly to:

    training.train_al.compute_oracle_loss()

BatchNorm running statistics are frozen during the micro-update.

Single-sample weighting
-----------------------

The canonical Oracle loss uses normalized annotation weights:

    sum_i w_i L_i / sum_i w_i

For a one-annotation diagnostic batch:

    w_1 L_1 / w_1 = L_1

Therefore confidence × situation does NOT change the magnitude of
an individual diagnostic micro-update. Those labels are retained
for descriptive subgroup analysis only.

Optimizer semantics
-------------------

A fresh Adam optimizer is deliberately created for every
annotation.

Therefore:

    gradient_norm_before_clip

is the direct local gradient-magnitude diagnostic, whereas:

    adam_step_norm

is the realized parameter displacement under this particular
diagnostic micro-update.

The two quantities must not be interpreted as equivalent.

Outputs
-------

data/analysis/annotation_response/

    <checkpoint>__<queue>.csv
    <checkpoint>__<queue>_report.txt

Example
-------

python analysis/annotation_response.py \
    --checkpoint checkpoints/rl_epoch/rl_epoch_10.pt \
    --queue checkpoints/queue/oracle_queue_1-10_random.jsonl \
    --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import chess.variant
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import training.train_al as al

from src.actions_space import ACTIONS, ACTION_TO_INDEX
from src.encoding import encode_boards


# ============================================================
# Defaults
# ============================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_AL.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "annotation_response"
)

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DEFAULT_SEED = 42

DEFAULT_MICRO_LR = 1e-4
DEFAULT_GRAD_CLIP = 1.0


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Measure the local ActorCritic response to "
            "individual Oracle annotations."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "ActorCritic checkpoint used as the common "
            "starting point."
        ),
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help="Oracle queue JSONL.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and report outputs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--micro-lr",
        type=float,
        default=DEFAULT_MICRO_LR,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )

    parser.add_argument(
        "--oracle-policy-coef",
        type=float,
        default=al.DEFAULT_ORACLE_POLICY_COEF,
    )

    parser.add_argument(
        "--oracle-value-coef",
        type=float,
        default=al.DEFAULT_ORACLE_VALUE_COEF,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
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
# BatchNorm
# ============================================================

def freeze_batchnorm(
    model: torch.nn.Module,
) -> None:
    """
    Freeze BatchNorm running statistics while preserving
    gradients through trainable affine parameters.
    """

    for module in model.modules():

        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):

            module.eval()


# ============================================================
# Generic utilities
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


# ============================================================
# Queue loading
# ============================================================

def load_oracle_queue(
    path: Path,
) -> list[dict]:
    """
    Load complete Oracle annotations.

    Required annotation fields:

        oracle_move
        oracle_confidence
        oracle_situation
        reward

    Explicitly discarded records are ignored.

    Legacy confidence / criticality field names are accepted
    for historical queue compatibility.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Oracle queue not found:\n{path}"
        )

    annotations = []

    skipped_incomplete = 0
    skipped_discarded = 0

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
                    f"Invalid JSON at line {line_number} "
                    f"in:\n{path}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):

                raise ValueError(
                    f"Expected JSON object at line "
                    f"{line_number}."
                )

            if record.get(
                "status"
            ) == "discarded":

                skipped_discarded += 1
                continue

            oracle_move = record.get(
                "oracle_move"
            )

            confidence = record.get(
                "oracle_confidence",
                record.get(
                    "confidence"
                ),
            )

            situation = record.get(
                "oracle_situation",
                record.get(
                    "criticality"
                ),
            )

            reward_raw = record.get(
                "reward"
            )

            if (
                oracle_move is None
                or confidence is None
                or situation is None
                or reward_raw is None
            ):

                skipped_incomplete += 1
                continue

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
                    f"Missing FEN at line "
                    f"{line_number}."
                )

            try:

                reward = float(
                    reward_raw
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"Invalid reward at line "
                    f"{line_number}: {reward_raw}"
                ) from exc

            if reward not in {
                -1.0,
                0.0,
                1.0,
            }:

                raise ValueError(
                    f"Invalid reward at line "
                    f"{line_number}: {reward}"
                )

            if confidence not in al.CONFIDENCE_WEIGHTS:

                raise ValueError(
                    f"Invalid confidence at line "
                    f"{line_number}: {confidence}"
                )

            if situation not in al.SITUATION_WEIGHTS:

                raise ValueError(
                    f"Invalid situation at line "
                    f"{line_number}: {situation}"
                )

            if oracle_move not in ACTION_TO_INDEX:

                raise ValueError(
                    f"Unknown Oracle move at line "
                    f"{line_number}: {oracle_move}"
                )

            try:

                board = chess.variant.AtomicBoard(
                    fen
                )

            except Exception as exc:

                raise ValueError(
                    f"Invalid Atomic FEN at line "
                    f"{line_number}:\n{fen}"
                ) from exc

            legal_moves = {
                move.uci()
                for move in board.legal_moves
            }

            if oracle_move not in legal_moves:

                raise ValueError(
                    f"Illegal Oracle move at line "
                    f"{line_number}:\n"
                    f"FEN:  {fen}\n"
                    f"Move: {oracle_move}"
                )

            annotations.append(
                {
                    "query_id":
                        record.get(
                            "query_id"
                        ),

                    "fen":
                        fen,

                    "oracle_move":
                        oracle_move,

                    "confidence":
                        confidence,

                    "situation":
                        situation,

                    "reward":
                        reward,

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

                    "score":
                        safe_float(
                            record.get(
                                "score",
                                record.get(
                                    "I"
                                ),
                            )
                        ),

                    "I_norm":
                        safe_float(
                            record.get(
                                "I_norm"
                            )
                        ),
                }
            )

    if not annotations:

        raise RuntimeError(
            "No complete Oracle annotations found."
        )

    print()
    print("=" * 70)
    print("ORACLE QUEUE")
    print("=" * 70)

    print(
        f"Path:       {path}"
    )

    print(
        f"Usable:     {len(annotations):,}"
    )

    print(
        f"Incomplete: {skipped_incomplete:,}"
    )

    print(
        f"Discarded:  {skipped_discarded:,}"
    )

    return annotations


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"RL checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    checkpoint_actions = checkpoint.get(
        "actions"
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(ACTIONS)
    ):

        raise ValueError(
            "Action-space mismatch in checkpoint: "
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

    print()
    print("=" * 70)
    print("STARTING MODEL")
    print("=" * 70)

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Epoch:      "
        f"{checkpoint.get('epoch', '?')}"
    )

    return model


# ============================================================
# Position evaluation
# ============================================================

@torch.no_grad()
def evaluate_annotation(
    model,
    record: dict,
    device: torch.device,
) -> dict:
    """
    Evaluate intrinsic legal-action policy and value.

    No rollout temperature and no BC opening prior are applied.
    """

    board = chess.variant.AtomicBoard(
        record[
            "fen"
        ]
    )

    encoded = encode_boards(
        [
            board
        ]
    ).to(
        device
    )

    logits, values = model(
        encoded
    )

    legal_moves = list(
        board.legal_moves
    )

    if not legal_moves:

        raise RuntimeError(
            "Annotation position has no legal moves."
        )

    legal_indices = [
        ACTION_TO_INDEX[
            move.uci()
        ]
        for move in legal_moves
    ]

    legal_index_tensor = torch.tensor(
        legal_indices,
        dtype=torch.long,
        device=device,
    )

    legal_logits = logits[
        0,
        legal_index_tensor,
    ]

    legal_log_probs = F.log_softmax(
        legal_logits,
        dim=0,
    )

    legal_probs = torch.exp(
        legal_log_probs
    )

    oracle_action = ACTION_TO_INDEX[
        record[
            "oracle_move"
        ]
    ]

    try:

        oracle_position = legal_indices.index(
            oracle_action
        )

    except ValueError as exc:

        raise RuntimeError(
            "Oracle action is not present in legal action list."
        ) from exc

    oracle_log_prob = legal_log_probs[
        oracle_position
    ]

    oracle_probability = legal_probs[
        oracle_position
    ]

    predicted_value = values[
        0,
        0,
    ]

    reward = float(
        record[
            "reward"
        ]
    )

    policy_loss = (
        -oracle_log_prob
    )

    value_loss = (
        predicted_value
        - reward
    ) ** 2

    entropy = -(
        legal_probs
        * legal_log_probs
    ).sum()

    return {
        "legal_indices":
            legal_indices,

        "log_probs":
            legal_log_probs
            .detach()
            .clone(),

        "oracle_probability":
            float(
                oracle_probability.item()
            ),

        "oracle_log_prob":
            float(
                oracle_log_prob.item()
            ),

        "surprise":
            float(
                -oracle_log_prob.item()
            ),

        "predicted_value":
            float(
                predicted_value.item()
            ),

        "policy_loss":
            float(
                policy_loss.item()
            ),

        "value_loss":
            float(
                value_loss.item()
            ),

        "entropy":
            float(
                entropy.item()
            ),
    }


# ============================================================
# Policy KL
# ============================================================

def policy_kl_before_after(
    before: dict,
    after: dict,
) -> float:
    """
    KL(pi_before || pi_after) over legal actions.
    """

    if (
        before[
            "legal_indices"
        ]
        != after[
            "legal_indices"
        ]
    ):

        raise RuntimeError(
            "Legal action support changed between "
            "before and after evaluations."
        )

    before_log_probs = before[
        "log_probs"
    ]

    after_log_probs = after[
        "log_probs"
    ]

    before_probs = torch.exp(
        before_log_probs
    )

    kl = (
        before_probs
        * (
            before_log_probs
            - after_log_probs
        )
    ).sum()

    if not torch.isfinite(
        kl
    ):

        raise RuntimeError(
            "Non-finite policy KL."
        )

    return float(
        kl.item()
    )


# ============================================================
# Parameter displacement
# ============================================================

def clone_parameters(
    model,
) -> dict[str, torch.Tensor]:

    return {
        name:
            parameter
            .detach()
            .clone()

        for (
            name,
            parameter,
        ) in model.named_parameters()
    }


def parameter_displacement_norm(
    before_parameters: dict[str, torch.Tensor],
    model,
) -> float:

    squared_sum = 0.0

    for (
        name,
        parameter,
    ) in model.named_parameters():

        before = before_parameters[
            name
        ]

        delta = (
            parameter.detach()
            - before
        )

        squared_sum += float(
            delta.pow(
                2
            ).sum().item()
        )

    return math.sqrt(
        squared_sum
    )


# ============================================================
# Single-annotation analysis
# ============================================================

def analyze_annotation(
    *,
    base_model,
    record: dict,
    index: int,
    device: torch.device,
    micro_lr: float,
    grad_clip: float,
    policy_coef: float,
    value_coef: float,
) -> dict:
    """
    Apply one independent Oracle diagnostic micro-update.

    Every annotation starts from the exact same base model.

    The canonical Oracle objective is delegated directly to
    training.train_al.compute_oracle_loss().
    """

    model = copy.deepcopy(
        base_model
    ).to(
        device
    )

    # ========================================================
    # Before
    # ========================================================

    model.eval()

    before = evaluate_annotation(
        model,
        record,
        device,
    )

    before_parameters = clone_parameters(
        model
    )

    # ========================================================
    # Fresh diagnostic optimizer
    # ========================================================

    optimizer = Adam(
        model.parameters(),
        lr=micro_lr,
    )

    # ========================================================
    # Oracle micro-update
    # ========================================================

    model.train()

    freeze_batchnorm(
        model
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_data = al.compute_oracle_loss(
        model,
        [
            record
        ],
        device=device,
        policy_coef=policy_coef,
        value_coef=value_coef,
    )

    loss = loss_data[
        "loss"
    ]

    loss.backward()

    gradient_norm_before_clip = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )
    )

    optimizer.step()

    # ========================================================
    # After
    # ========================================================

    model.eval()

    after = evaluate_annotation(
        model,
        record,
        device,
    )

    # ========================================================
    # Response metrics
    # ========================================================

    delta_probability = (
        after[
            "oracle_probability"
        ]
        - before[
            "oracle_probability"
        ]
    )

    surprise_reduction = (
        before[
            "surprise"
        ]
        - after[
            "surprise"
        ]
    )

    delta_value = (
        after[
            "predicted_value"
        ]
        - before[
            "predicted_value"
        ]
    )

    reward = float(
        record[
            "reward"
        ]
    )

    value_error_before = abs(
        before[
            "predicted_value"
        ]
        - reward
    )

    value_error_after = abs(
        after[
            "predicted_value"
        ]
        - reward
    )

    value_error_reduction = (
        value_error_before
        - value_error_after
    )

    kl = policy_kl_before_after(
        before,
        after,
    )

    adam_step_norm = parameter_displacement_norm(
        before_parameters,
        model,
    )

    confidence_weight = al.CONFIDENCE_WEIGHTS[
        record[
            "confidence"
        ]
    ]

    situation_weight = al.SITUATION_WEIGHTS[
        record[
            "situation"
        ]
    ]

    supervision_weight = (
        confidence_weight
        * situation_weight
    )

    return {
        "index":
            index,

        "query_id":
            record.get(
                "query_id"
            ),

        "fen":
            record[
                "fen"
            ],

        "oracle_move":
            record[
                "oracle_move"
            ],

        "confidence":
            record[
                "confidence"
            ],

        "situation":
            record[
                "situation"
            ],

        "reward":
            reward,

        "H":
            record.get(
                "H",
                np.nan,
            ),

        "U":
            record.get(
                "U",
                np.nan,
            ),

        "HU":
            record.get(
                "HU",
                np.nan,
            ),

        "score":
            record.get(
                "score",
                np.nan,
            ),

        "I_norm":
            record.get(
                "I_norm",
                np.nan,
            ),

        # Descriptive only:
        # for one annotation it cancels in the normalized loss.
        "supervision_weight":
            supervision_weight,

        "oracle_probability_before":
            before[
                "oracle_probability"
            ],

        "oracle_probability_after":
            after[
                "oracle_probability"
            ],

        "delta_oracle_probability":
            delta_probability,

        "surprise_before":
            before[
                "surprise"
            ],

        "surprise_after":
            after[
                "surprise"
            ],

        "surprise_reduction":
            surprise_reduction,

        "entropy_before":
            before[
                "entropy"
            ],

        "entropy_after":
            after[
                "entropy"
            ],

        "delta_entropy":
            (
                after[
                    "entropy"
                ]
                - before[
                    "entropy"
                ]
            ),

        "value_before":
            before[
                "predicted_value"
            ],

        "value_after":
            after[
                "predicted_value"
            ],

        "delta_value":
            delta_value,

        "value_error_before":
            value_error_before,

        "value_error_after":
            value_error_after,

        "value_error_reduction":
            value_error_reduction,

        "policy_loss_before":
            before[
                "policy_loss"
            ],

        "policy_loss_after":
            after[
                "policy_loss"
            ],

        "value_loss_before":
            before[
                "value_loss"
            ],

        "value_loss_after":
            after[
                "value_loss"
            ],

        "oracle_loss_before":
            (
                policy_coef
                * before[
                    "policy_loss"
                ]
                + value_coef
                * before[
                    "value_loss"
                ]
            ),

        "oracle_loss_after":
            (
                policy_coef
                * after[
                    "policy_loss"
                ]
                + value_coef
                * after[
                    "value_loss"
                ]
            ),

        "policy_kl_before_after":
            kl,

        # Direct local gradient magnitude.
        "gradient_norm_before_clip":
            float(
                gradient_norm_before_clip.item()
            ),

        # Realized displacement under a fresh one-step Adam
        # diagnostic optimizer. This is NOT the gradient norm.
        "adam_step_norm":
            adam_step_norm,
    }


# ============================================================
# Correlations
# ============================================================

def correlation_table(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:

    rows = []

    for signal in (
        "H",
        "U",
        "HU",
    ):

        subset = (
            df[
                [
                    signal,
                    target,
                ]
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if len(
            subset
        ) < 3:

            rows.append(
                {
                    "signal":
                        signal,

                    "target":
                        target,

                    "n":
                        len(
                            subset
                        ),

                    "pearson":
                        np.nan,

                    "spearman":
                        np.nan,
                }
            )

            continue

        rows.append(
            {
                "signal":
                    signal,

                "target":
                    target,

                "n":
                    len(
                        subset
                    ),

                "pearson":
                    float(
                        subset[
                            signal
                        ].corr(
                            subset[
                                target
                            ],
                            method="pearson",
                        )
                    ),

                "spearman":
                    float(
                        subset[
                            signal
                        ].corr(
                            subset[
                                target
                            ],
                            method="spearman",
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Report helpers
# ============================================================

def add_distribution(
    lines: list[str],
    df: pd.DataFrame,
    column: str,
) -> None:

    series = (
        pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if len(
        series
    ) == 0:

        return

    lines.append(
        f"{column}:"
    )

    lines.append(
        f"  n      = {len(series):,}"
    )

    lines.append(
        f"  mean   = {series.mean():+.8f}"
    )

    lines.append(
        f"  median = {series.median():+.8f}"
    )

    lines.append(
        f"  std    = {series.std(ddof=0):.8f}"
    )

    lines.append(
        f"  p90    = {series.quantile(0.90):+.8f}"
    )

    lines.append(
        f"  p95    = {series.quantile(0.95):+.8f}"
    )

    lines.append(
        f"  max    = {series.max():+.8f}"
    )

    lines.append()


def build_report(
    *,
    df: pd.DataFrame,
    correlation_tables: dict[str, pd.DataFrame],
    checkpoint_path: Path,
    queue_path: Path,
    micro_lr: float,
    grad_clip: float,
    policy_coef: float,
    value_coef: float,
    seed: int,
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
        "=" * 70
    )

    add(
        "ALBERTA - LOCAL ORACLE ANNOTATION RESPONSE"
    )

    add(
        "=" * 70
    )

    add()

    add(
        f"Checkpoint: {checkpoint_path}"
    )

    add(
        f"Oracle queue: {queue_path}"
    )

    add(
        f"Annotations: {len(df):,}"
    )

    add(
        f"Seed: {seed}"
    )

    add()

    # ========================================================
    # Configuration
    # ========================================================

    add(
        "CONFIGURATION"
    )

    add(
        "-" * 70
    )

    add(
        f"Micro LR: {micro_lr}"
    )

    add(
        f"Gradient clipping: {grad_clip}"
    )

    add(
        f"Oracle policy coefficient: {policy_coef}"
    )

    add(
        f"Oracle value coefficient: {value_coef}"
    )

    add(
        "BatchNorm running statistics: frozen"
    )

    add(
        "Optimizer: fresh Adam for each annotation"
    )

    add()

    add(
        "The Oracle objective is computed directly by "
        "training.train_al.compute_oracle_loss()."
    )

    add()

    add(
        "Because each diagnostic batch contains exactly one "
        "annotation, confidence × situation weighting cancels "
        "in the normalized Oracle loss."
    )

    add()

    # ========================================================
    # Local response
    # ========================================================

    add(
        "LOCAL RESPONSE"
    )

    add(
        "-" * 70
    )

    for column in (
        "delta_oracle_probability",
        "surprise_reduction",
        "delta_value",
        "value_error_reduction",
        "policy_kl_before_after",
        "gradient_norm_before_clip",
        "adam_step_norm",
    ):

        add_distribution(
            lines,
            df,
            column,
        )

    # ========================================================
    # Correlations
    # ========================================================

    add(
        "H / U / HU CORRELATIONS"
    )

    add(
        "-" * 70
    )

    for (
        target,
        table,
    ) in correlation_tables.items():

        add(
            f"Target: {target}"
        )

        add(
            table.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:+.6f}",
            )
        )

        add()

    # ========================================================
    # Interpretation
    # ========================================================

    add(
        "INTERPRETATION"
    )

    add(
        "-" * 70
    )

    add(
        "Each annotation starts from the exact same ActorCritic "
        "checkpoint and is therefore an independent local probe."
    )

    add()

    add(
        "gradient_norm_before_clip measures the magnitude of the "
        "Oracle gradient before diagnostic clipping."
    )

    add(
        "adam_step_norm measures the realized parameter "
        "displacement produced by one fresh-Adam step."
    )

    add(
        "Because Adam adaptively rescales gradient components, "
        "adam_step_norm must not be interpreted as a direct "
        "gradient-magnitude estimate."
    )

    add()

    add(
        "Policy probability, surprise, KL and value-error "
        "changes describe immediate local model susceptibility."
    )

    add(
        "They do not measure downstream playing-strength gain "
        "or long-horizon annotation utility."
    )

    add()

    add(
        "A correlation between H/U/HU and local response means "
        "that the signal predicts immediate response magnitude "
        "or sensitivity. It does not establish that the update "
        "points in a useful learning direction."
    )

    add()

    add(
        "Any interpretation involving U, HU, or cohorts selected "
        "from them must use the corrected league uncertainty "
        "estimator."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.micro_lr <= 0.0:

        raise ValueError(
            "--micro-lr must be strictly positive."
        )

    if args.grad_clip <= 0.0:

        raise ValueError(
            "--grad-clip must be strictly positive."
        )

    device = torch.device(
        args.device
    )

    seed_everything(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("ALBERTA - LOCAL ORACLE ANNOTATION RESPONSE")
    print("=" * 70)

    print(
        f"Device:     {device}"
    )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    print(
        f"Queue:      {args.queue}"
    )

    # ========================================================
    # Load
    # ========================================================

    base_model = load_model(
        args.checkpoint,
        device,
    )

    annotations = load_oracle_queue(
        args.queue
    )

    # ========================================================
    # Analyze
    # ========================================================

    results = []

    total = len(
        annotations
    )

    print()
    print("=" * 70)
    print("ANALYSING ANNOTATIONS")
    print("=" * 70)

    for (
        index,
        record,
    ) in enumerate(
        annotations,
        start=1,
    ):

        result = analyze_annotation(
            base_model=base_model,
            record=record,
            index=index,
            device=device,
            micro_lr=args.micro_lr,
            grad_clip=args.grad_clip,
            policy_coef=args.oracle_policy_coef,
            value_coef=args.oracle_value_coef,
        )

        results.append(
            result
        )

        if (
            index == 1
            or index % 25 == 0
            or index == total
        ):

            print(
                f"[{index:>4}/{total}] completed",
                flush=True,
            )

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # Correlations
    # ========================================================

    correlation_targets = (
        "policy_kl_before_after",
        "delta_oracle_probability",
        "surprise_reduction",
        "delta_value",
        "value_error_reduction",
        "gradient_norm_before_clip",
        "adam_step_norm",
    )

    correlation_tables = {
        target:
            correlation_table(
                df,
                target,
            )

        for target in correlation_targets
    }

    # ========================================================
    # Output
    # ========================================================

    output_stem = (
        f"{args.checkpoint.stem}"
        f"__{args.queue.stem}"
    )

    csv_path = (
        args.output_dir
        / f"{output_stem}.csv"
    )

    report_path = (
        args.output_dir
        / f"{output_stem}_report.txt"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    report = build_report(
        df=df,
        correlation_tables=correlation_tables,
        checkpoint_path=args.checkpoint,
        queue_path=args.queue,
        micro_lr=args.micro_lr,
        grad_clip=args.grad_clip,
        policy_coef=args.oracle_policy_coef,
        value_coef=args.oracle_value_coef,
        seed=args.seed,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Annotations: {len(df):,}"
    )

    for target in (
        "delta_oracle_probability",
        "surprise_reduction",
        "value_error_reduction",
        "policy_kl_before_after",
        "gradient_norm_before_clip",
        "adam_step_norm",
    ):

        series = (
            pd.to_numeric(
                df[
                    target
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if len(
            series
        ) == 0:

            continue

        print()
        print(
            target
        )

        print(
            f"  mean:   {series.mean():+.8f}"
        )

        print(
            f"  median: {series.median():+.8f}"
        )

        print(
            f"  std:    {series.std(ddof=0):.8f}"
        )

    print()
    print(
        f"CSV:    {csv_path}"
    )

    print(
        f"Report: {report_path}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()