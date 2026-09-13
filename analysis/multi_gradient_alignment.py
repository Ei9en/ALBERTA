#!/usr/bin/env python3

"""
ALBERTA - Multi-Reference Gradient Alignment Diagnostic
=======================================================

Study whether Oracle annotations point in directions compatible
with multiple independently sampled PPO surrogate gradients.

Instead of defining one global PPO reference direction,

    g_ref,

we construct K independent reference gradients from the same
on-policy replay buffer:

    G_pi = {g_pi^(1), ..., g_pi^(K)}
    G_V  = {g_V^(1),  ..., g_V^(K)}

For every complete Oracle annotation, we compute its alignment
with every reference direction and summarize the resulting
distribution.

Alignment summaries
-------------------

For cosine similarities D_1, ..., D_K:

    D_mean
        mean alignment over all references

    D_max
        maximum observed alignment

    D_top
        mean over the top fraction of alignments

    D_positive_mean
        mean over positive alignments only

    P_positive
        fraction of positive alignments

    D_std
        variability across reference directions

Scientific question
-------------------

Can H / U / HU predict either:

    1. the direction of an Oracle gradient relative to plausible
       PPO update directions;

or:

    2. a magnitude × direction quantity?

Important scope
---------------

This diagnostic computes gradients in:

    - the policy head for actor / Oracle-policy losses;
    - the value head for critic / Oracle-value losses.

It therefore measures head-space alignment, not the exact full
optimizer update across every shared parameter.

No optimizer.step() is performed.

PPO actor reference semantics
-----------------------------

The PPO actor reference reconstructs the canonical rollout/PPO
policy used by ALBERTA:

    legal RL logits
    + BC opening prior
    -> divide by rollout temperature
    -> legal softmax
    -> PPO clipped surrogate

The replay row must therefore contain:

    fen
    action
    legal_moves
    old_log_prob
    advantage
    return
    value
    ply

Critic reference semantics
--------------------------

The critic reference preserves ALBERTA's historical clipped-value
objective:

    V_clip =
        V_old + clip(V - V_old, -epsilon, +epsilon)

    L_V =
        MSE(V_clip, return)

This is intentionally preserved for reproducibility.

Oracle semantics
----------------

Oracle policy gradients use the legal-action cross-entropy implied
by:

    training.train_al.compute_oracle_loss()

Oracle value gradients use:

    MSE(V(s), R_oracle)

For a single annotation, confidence × situation weighting cancels
because the canonical Oracle loss normalizes by the sum of weights.

Uncertainty warning
-------------------

Any conclusion involving U, HU, or queues selected from them is
valid only when the queue was generated using the corrected league
uncertainty estimator that excludes untrained BC value heads.

Example
-------

python analysis/multi_gradient_alignment.py \
    --checkpoint checkpoints/rl_epoch/rl_epoch_10.pt \
    --oracle-queue checkpoints/queue/oracle_queue_1-10_random.jsonl \
    --replay checkpoints/buffer/replay_buffer_epoch_10.pkl \
    --bc-checkpoint checkpoints/bc_epoch/bc_epoch_7.pt \
    --reference-size 256 \
    --n-references 16 \
    --top-fraction 0.25 \
    --temperature 2.0 \
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path

import chess.variant
import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


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
from src.models.resnet import ChessResNet


# ============================================================
# Defaults
# ============================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_REPLAY = (
    PROJECT_ROOT
    / "checkpoints"
    / "buffer"
    / "replay_buffer_epoch_10.pkl"
)

DEFAULT_ORACLE_QUEUE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_random.jsonl"
)

DEFAULT_BC_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_epoch_7.pt"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "gradient_alignment"
    / "multi_gradient_alignment_results.jsonl"
)

DEFAULT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "gradient_alignment"
    / "multi_gradient_alignment_report.txt"
)

DEFAULT_SEED = 42

DEFAULT_REFERENCE_SIZE = 256
DEFAULT_N_REFERENCES = 16
DEFAULT_TOP_FRACTION = 0.25

DEFAULT_PPO_CLIP = 0.2
DEFAULT_TEMPERATURE = 2.0

DEFAULT_OPENING_PRIOR_PLIES = 6
DEFAULT_OPENING_PRIOR_STRENGTH = 1.0

EPS = 1e-12


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Measure multi-reference policy-head and value-head "
            "gradient alignment for ALBERTA Oracle annotations."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--oracle-queue",
        type=Path,
        default=DEFAULT_ORACLE_QUEUE,
    )

    parser.add_argument(
        "--replay",
        type=Path,
        default=DEFAULT_REPLAY,
    )

    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=DEFAULT_BC_CHECKPOINT,
    )

    parser.add_argument(
        "--reference-size",
        type=int,
        default=DEFAULT_REFERENCE_SIZE,
    )

    parser.add_argument(
        "--n-references",
        type=int,
        default=DEFAULT_N_REFERENCES,
    )

    parser.add_argument(
        "--top-fraction",
        type=float,
        default=DEFAULT_TOP_FRACTION,
    )

    parser.add_argument(
        "--ppo-clip",
        type=float,
        default=DEFAULT_PPO_CLIP,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
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
        "--max-annotations",
        type=int,
        default=None,
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
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
# Gradient utilities
# ============================================================

def zero_model_grads(
    model,
) -> None:

    model.zero_grad(
        set_to_none=True
    )


def flatten_parameter_grads(
    parameters,
) -> torch.Tensor:
    """
    Flatten gradients for a fixed parameter collection.

    Missing gradients are represented by zeros so every vector
    has an identical dimensionality.
    """

    chunks = []

    for parameter in parameters:

        if parameter.grad is None:

            chunks.append(
                torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
                .reshape(
                    -1
                )
                .cpu()
            )

        else:

            chunks.append(
                parameter.grad
                .detach()
                .reshape(
                    -1
                )
                .cpu()
            )

    if not chunks:

        return torch.empty(
            0,
            dtype=torch.float32,
        )

    return torch.cat(
        chunks
    )


def cosine_similarity_flat(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = EPS,
) -> float:

    if (
        a.numel() == 0
        or b.numel() == 0
    ):

        return float(
            "nan"
        )

    if a.shape != b.shape:

        raise ValueError(
            "Gradient vector shape mismatch: "
            f"{a.shape} != {b.shape}"
        )

    norm_a = torch.linalg.vector_norm(
        a
    )

    norm_b = torch.linalg.vector_norm(
        b
    )

    if (
        norm_a.item() < eps
        or norm_b.item() < eps
    ):

        return float(
            "nan"
        )

    return float(
        torch.dot(
            a,
            b,
        ).item()
        /
        (
            norm_a.item()
            * norm_b.item()
        )
    )


# ============================================================
# Model loading
# ============================================================

def load_actor_critic(
    checkpoint_path: Path,
    device: torch.device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            f"Checkpoint has no model_state_dict:\n"
            f"{checkpoint_path}"
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

    print()
    print("=" * 72)
    print("ACTOR-CRITIC")
    print("=" * 72)

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Epoch:      "
        f"{checkpoint.get('epoch', '?')}"
    )

    return model


def load_bc_policy(
    checkpoint_path: Path,
    device: torch.device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"BC checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            f"BC checkpoint has no model_state_dict:\n"
            f"{checkpoint_path}"
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
# Parameter groups
# ============================================================

def get_policy_parameters(
    model,
) -> list[torch.nn.Parameter]:

    return list(
        model.policy.parameters()
    )


def get_value_parameters(
    model,
) -> list[torch.nn.Parameter]:

    return list(
        model.value.parameters()
    )


# ============================================================
# Replay loading
# ============================================================

def normalize_replay_container(
    obj,
) -> list[dict]:

    if isinstance(
        obj,
        list,
    ):

        return obj

    if isinstance(
        obj,
        tuple,
    ):

        return list(
            obj
        )

    if isinstance(
        obj,
        dict,
    ):

        for key in (
            "buffer",
            "data",
            "transitions",
            "items",
            "replay",
        ):

            if key in obj:

                return normalize_replay_container(
                    obj[
                        key
                    ]
                )

        expected = {
            "fen",
            "action",
            "legal_moves",
            "old_log_prob",
            "advantage",
            "return",
            "value",
            "ply",
        }

        if expected.issubset(
            obj.keys()
        ):

            n = len(
                obj[
                    "fen"
                ]
            )

            rows = []

            for index in range(
                n
            ):

                row = {}

                for (
                    key,
                    values,
                ) in obj.items():

                    try:

                        row[
                            key
                        ] = values[
                            index
                        ]

                    except Exception:

                        pass

                rows.append(
                    row
                )

            return rows

    for attr in (
        "buffer",
        "data",
        "transitions",
    ):

        if hasattr(
            obj,
            attr,
        ):

            return normalize_replay_container(
                getattr(
                    obj,
                    attr,
                )
            )

    raise RuntimeError(
        "Unrecognized replay-buffer format."
    )


def load_replay(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Replay buffer not found:\n{path}"
        )

    print()
    print("=" * 72)
    print("PPO REPLAY")
    print("=" * 72)

    print(
        f"Path: {path}"
    )

    if path.suffix.lower() == ".jsonl":

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

                    row = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:

                    raise ValueError(
                        f"Invalid JSON at line "
                        f"{line_number}."
                    ) from exc

                rows.append(
                    row
                )

        return rows

    if path.suffix.lower() in {
        ".pkl",
        ".pickle",
    }:

        with path.open(
            "rb"
        ) as file:

            obj = pickle.load(
                file
            )

        return normalize_replay_container(
            obj
        )

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return normalize_replay_container(
        obj
    )


def normalize_legal_moves(
    legal_moves,
) -> list[str]:

    normalized = []

    for move in legal_moves:

        if isinstance(
            move,
            str,
        ):

            normalized.append(
                move
            )

        elif hasattr(
            move,
            "uci",
        ):

            normalized.append(
                move.uci()
            )

        else:

            raise ValueError(
                f"Unsupported legal move value: {move!r}"
            )

    return normalized


def filter_valid_replay_rows(
    rows: list[dict],
) -> list[dict]:

    required = (
        "fen",
        "action",
        "legal_moves",
        "old_log_prob",
        "advantage",
        "return",
        "value",
        "ply",
    )

    valid = []

    skipped = 0

    for row in rows:

        if not isinstance(
            row,
            dict,
        ):

            skipped += 1
            continue

        if not all(
            key in row
            for key in required
        ):

            skipped += 1
            continue

        try:

            board = chess.variant.AtomicBoard(
                row[
                    "fen"
                ]
            )

            legal_moves = normalize_legal_moves(
                row[
                    "legal_moves"
                ]
            )

            action = int(
                row[
                    "action"
                ]
            )

            old_log_prob = float(
                row[
                    "old_log_prob"
                ]
            )

            advantage = float(
                row[
                    "advantage"
                ]
            )

            ret = float(
                row[
                    "return"
                ]
            )

            old_value = float(
                row[
                    "value"
                ]
            )

            ply = int(
                row[
                    "ply"
                ]
            )

        except (
            TypeError,
            ValueError,
            KeyError,
        ):

            skipped += 1
            continue

        if not (
            np.isfinite(
                old_log_prob
            )
            and np.isfinite(
                advantage
            )
            and np.isfinite(
                ret
            )
            and np.isfinite(
                old_value
            )
        ):

            skipped += 1
            continue

        if action < 0 or action >= len(
            ACTIONS
        ):

            skipped += 1
            continue

        legal_indices = []

        invalid_legal = False

        for move in legal_moves:

            index = ACTION_TO_INDEX.get(
                move
            )

            if index is None:

                invalid_legal = True
                break

            legal_indices.append(
                index
            )

        if invalid_legal or not legal_indices:

            skipped += 1
            continue

        if action not in legal_indices:

            skipped += 1
            continue

        # Sanity-check against actual Atomic position.
        actual_legal = {
            move.uci()
            for move in board.legal_moves
        }

        if set(
            legal_moves
        ) != actual_legal:

            skipped += 1
            continue

        normalized = dict(
            row
        )

        normalized[
            "action"
        ] = action

        normalized[
            "legal_moves"
        ] = legal_moves

        normalized[
            "old_log_prob"
        ] = old_log_prob

        normalized[
            "advantage"
        ] = advantage

        normalized[
            "return"
        ] = ret

        normalized[
            "value"
        ] = old_value

        normalized[
            "ply"
        ] = ply

        valid.append(
            normalized
        )

    print(
        f"Valid replay transitions: "
        f"{len(valid):,}"
    )

    print(
        f"Skipped replay rows:      "
        f"{skipped:,}"
    )

    return valid


# ============================================================
# BC prior
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
# Canonical legal PPO policy
# ============================================================

def legal_log_prob_for_replay_action(
    *,
    rl_logits: torch.Tensor,
    bc_logits: torch.Tensor | None,
    legal_indices: list[int],
    action_index: int,
    ply: int,
    temperature: float,
    opening_prior_plies: int,
    opening_prior_strength_value: float,
) -> torch.Tensor:

    legal_index_tensor = torch.tensor(
        legal_indices,
        dtype=torch.long,
        device=rl_logits.device,
    )

    legal_logits = rl_logits[
        legal_index_tensor
    ]

    alpha = opening_prior_strength(
        ply,
        prior_plies=opening_prior_plies,
        prior_strength=opening_prior_strength_value,
    )

    if (
        alpha > 0.0
        and bc_logits is not None
    ):

        bc_legal_logits = bc_logits[
            legal_index_tensor
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

    sampling_logits = (
        legal_logits
        / temperature
    )

    legal_log_probs = F.log_softmax(
        sampling_logits,
        dim=0,
    )

    try:

        local_action_index = legal_indices.index(
            action_index
        )

    except ValueError as exc:

        raise RuntimeError(
            "Replay action is not present in legal action support."
        ) from exc

    return legal_log_probs[
        local_action_index
    ]


# ============================================================
# PPO reference gradient
# ============================================================

def compute_reference_gradient(
    *,
    model,
    bc_policy,
    rows: list[dict],
    device: torch.device,
    ppo_clip: float,
    temperature: float,
    opening_prior_plies: int,
    opening_prior_strength_value: float,
) -> dict:

    boards = [
        chess.variant.AtomicBoard(
            row[
                "fen"
            ]
        )
        for row
        in rows
    ]

    encoded = encode_boards(
        boards
    ).to(
        device
    )

    advantages = torch.tensor(
        [
            row[
                "advantage"
            ]
            for row
            in rows
        ],
        dtype=torch.float32,
        device=device,
    )

    old_log_probs = torch.tensor(
        [
            row[
                "old_log_prob"
            ]
            for row
            in rows
        ],
        dtype=torch.float32,
        device=device,
    )

    returns = torch.tensor(
        [
            row[
                "return"
            ]
            for row
            in rows
        ],
        dtype=torch.float32,
        device=device,
    )

    old_values = torch.tensor(
        [
            row[
                "value"
            ]
            for row
            in rows
        ],
        dtype=torch.float32,
        device=device,
    )

    # ========================================================
    # Advantage normalization
    # ========================================================

    if len(
        advantages
    ) > 1:

        advantage_std = advantages.std(
            unbiased=False
        )

        if advantage_std.item() > 1e-8:

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantage_std
                + 1e-8
            )

    # ========================================================
    # Forward
    # ========================================================

    zero_model_grads(
        model
    )

    policy_logits, values = model(
        encoded
    )

    # ========================================================
    # BC logits
    # ========================================================

    with torch.no_grad():

        bc_logits = bc_policy(
            encoded
        )

        if isinstance(
            bc_logits,
            tuple,
        ):

            bc_logits = bc_logits[
                0
            ]

    # ========================================================
    # Canonical replay-action log probabilities
    # ========================================================

    new_log_probs = []

    for index, row in enumerate(
        rows
    ):

        legal_indices = [
            ACTION_TO_INDEX[
                move
            ]
            for move
            in row[
                "legal_moves"
            ]
        ]

        log_prob = legal_log_prob_for_replay_action(
            rl_logits=policy_logits[
                index
            ],
            bc_logits=bc_logits[
                index
            ],
            legal_indices=legal_indices,
            action_index=row[
                "action"
            ],
            ply=row[
                "ply"
            ],
            temperature=temperature,
            opening_prior_plies=opening_prior_plies,
            opening_prior_strength_value=(
                opening_prior_strength_value
            ),
        )

        new_log_probs.append(
            log_prob
        )

    new_log_probs = torch.stack(
        new_log_probs
    )

    # ========================================================
    # PPO actor surrogate
    # ========================================================

    ratio = torch.exp(
        new_log_probs
        - old_log_probs
    )

    unclipped = (
        ratio
        * advantages
    )

    clipped = (
        torch.clamp(
            ratio,
            1.0 - ppo_clip,
            1.0 + ppo_clip,
        )
        * advantages
    )

    actor_loss = -torch.min(
        unclipped,
        clipped,
    ).mean()

    actor_loss.backward(
        retain_graph=True
    )

    g_policy = flatten_parameter_grads(
        get_policy_parameters(
            model
        )
    )

    # ========================================================
    # Historical ALBERTA critic objective
    # ========================================================

    zero_model_grads(
        model
    )

    _, values = model(
        encoded
    )

    predicted_values = values[
        :,
        0
    ]

    clipped_values = (
        old_values
        + (
            predicted_values
            - old_values
        ).clamp(
            -ppo_clip,
            ppo_clip,
        )
    )

    critic_loss = F.mse_loss(
        clipped_values,
        returns,
    )

    critic_loss.backward()

    g_value = flatten_parameter_grads(
        get_value_parameters(
            model
        )
    )

    zero_model_grads(
        model
    )

    return {
        "g_policy":
            g_policy,

        "g_value":
            g_value,

        "actor_loss":
            float(
                actor_loss.item()
            ),

        "critic_loss":
            float(
                critic_loss.item()
            ),

        "policy_norm":
            float(
                torch.linalg.vector_norm(
                    g_policy
                ).item()
            ),

        "value_norm":
            float(
                torch.linalg.vector_norm(
                    g_value
                ).item()
            ),
    }


# ============================================================
# Reference bank
# ============================================================

def build_reference_bank(
    *,
    model,
    bc_policy,
    replay_rows: list[dict],
    device: torch.device,
    reference_size: int,
    n_references: int,
    ppo_clip: float,
    temperature: float,
    opening_prior_plies: int,
    opening_prior_strength_value: float,
    seed: int,
) -> list[dict]:

    valid = filter_valid_replay_rows(
        replay_rows
    )

    if len(
        valid
    ) < reference_size:

        raise RuntimeError(
            f"Only {len(valid):,} valid replay transitions "
            f"for reference-size={reference_size}."
        )

    references = []

    print()
    print("=" * 72)
    print("BUILDING PPO REFERENCE BANK")
    print("=" * 72)

    for reference_index in range(
        n_references
    ):

        rng = random.Random(
            seed
            + reference_index
        )

        sampled_rows = rng.sample(
            valid,
            reference_size,
        )

        reference = compute_reference_gradient(
            model=model,
            bc_policy=bc_policy,
            rows=sampled_rows,
            device=device,
            ppo_clip=ppo_clip,
            temperature=temperature,
            opening_prior_plies=opening_prior_plies,
            opening_prior_strength_value=(
                opening_prior_strength_value
            ),
        )

        references.append(
            reference
        )

        print(
            f"[{reference_index + 1:02d}/"
            f"{n_references:02d}] "
            f"Lpi={reference['actor_loss']:+.6f} "
            f"LV={reference['critic_loss']:.6f} | "
            f"|gpi|={reference['policy_norm']:.6e} "
            f"|gV|={reference['value_norm']:.6e}"
        )

    return references


# ============================================================
# Oracle queue
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


def load_oracle_queue(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Oracle queue not found:\n{path}"
        )

    rows = []

    skipped = 0

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

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):

                skipped += 1
                continue

            if row.get(
                "status"
            ) == "discarded":

                skipped += 1
                continue

            fen = row.get(
                "fen"
            )

            oracle_move = row.get(
                "oracle_move"
            )

            reward = safe_float(
                row.get(
                    "reward"
                )
            )

            H = safe_float(
                row.get(
                    "H"
                )
            )

            U = safe_float(
                row.get(
                    "U"
                )
            )

            if (
                not isinstance(
                    fen,
                    str,
                )
                or not fen
                or oracle_move is None
                or not np.isfinite(
                    reward
                )
                or not np.isfinite(
                    H
                )
                or not np.isfinite(
                    U
                )
            ):

                skipped += 1
                continue

            if reward not in {
                -1.0,
                0.0,
                1.0,
            }:

                skipped += 1
                continue

            if oracle_move not in ACTION_TO_INDEX:

                skipped += 1
                continue

            board = chess.variant.AtomicBoard(
                fen
            )

            legal_moves = {
                move.uci()
                for move
                in board.legal_moves
            }

            if oracle_move not in legal_moves:

                skipped += 1
                continue

            normalized = dict(
                row
            )

            normalized[
                "fen"
            ] = fen

            normalized[
                "oracle_move"
            ] = oracle_move

            normalized[
                "reward"
            ] = reward

            normalized[
                "H"
            ] = H

            normalized[
                "U"
            ] = U

            normalized[
                "HU"
            ] = (
                H
                * U
            )

            rows.append(
                normalized
            )

    print()
    print("=" * 72)
    print("ORACLE QUEUE")
    print("=" * 72)

    print(
        f"Usable annotations: {len(rows):,}"
    )

    print(
        f"Skipped:            {skipped:,}"
    )

    return rows


# ============================================================
# Oracle gradients
# ============================================================

def compute_oracle_gradients(
    *,
    model,
    row: dict,
    device: torch.device,
) -> dict:
    """
    Compute single-annotation policy-head and value-head gradients.

    The policy branch uses legal-action cross-entropy, matching the
    canonical Oracle objective semantics.
    """

    board = chess.variant.AtomicBoard(
        row[
            "fen"
        ]
    )

    legal_moves = list(
        board.legal_moves
    )

    legal_indices = [
        ACTION_TO_INDEX[
            move.uci()
        ]
        for move
        in legal_moves
    ]

    oracle_action = ACTION_TO_INDEX[
        row[
            "oracle_move"
        ]
    ]

    try:

        oracle_local_index = legal_indices.index(
            oracle_action
        )

    except ValueError as exc:

        raise RuntimeError(
            "Oracle move is not legal."
        ) from exc

    encoded = encode_boards(
        [
            board
        ]
    ).to(
        device
    )

    # ========================================================
    # Policy Oracle gradient
    # ========================================================

    zero_model_grads(
        model
    )

    policy_logits, _ = model(
        encoded
    )

    legal_index_tensor = torch.tensor(
        legal_indices,
        dtype=torch.long,
        device=device,
    )

    legal_logits = policy_logits[
        0,
        legal_index_tensor,
    ]

    target = torch.tensor(
        [
            oracle_local_index
        ],
        dtype=torch.long,
        device=device,
    )

    policy_loss = F.cross_entropy(
        legal_logits.unsqueeze(
            0
        ),
        target,
    )

    policy_loss.backward()

    g_policy = flatten_parameter_grads(
        get_policy_parameters(
            model
        )
    )

    # ========================================================
    # Value Oracle gradient
    # ========================================================

    zero_model_grads(
        model
    )

    _, value = model(
        encoded
    )

    reward = torch.tensor(
        [
            float(
                row[
                    "reward"
                ]
            )
        ],
        dtype=torch.float32,
        device=device,
    )

    predicted_value = value[
        :,
        0
    ]

    value_loss = F.mse_loss(
        predicted_value,
        reward,
    )

    value_loss.backward()

    g_value = flatten_parameter_grads(
        get_value_parameters(
            model
        )
    )

    zero_model_grads(
        model
    )

    return {
        "g_policy":
            g_policy,

        "g_value":
            g_value,

        "M_policy":
            float(
                torch.linalg.vector_norm(
                    g_policy
                ).item()
            ),

        "M_value":
            float(
                torch.linalg.vector_norm(
                    g_value
                ).item()
            ),

        "oracle_policy_loss":
            float(
                policy_loss.item()
            ),

        "oracle_value_loss":
            float(
                value_loss.item()
            ),
    }


# ============================================================
# Multi-reference alignment summaries
# ============================================================

def summarize_alignments(
    values,
    top_fraction: float,
) -> dict:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if len(
        finite
    ) == 0:

        return {
            "mean":
                np.nan,

            "max":
                np.nan,

            "top":
                np.nan,

            "positive_mean":
                np.nan,

            "positive_fraction":
                np.nan,

            "std":
                np.nan,
        }

    n_top = max(
        1,
        int(
            math.ceil(
                len(
                    finite
                )
                * top_fraction
            )
        ),
    )

    sorted_values = np.sort(
        finite
    )

    top_values = sorted_values[
        -n_top:
    ]

    positive = finite[
        finite
        > 0.0
    ]

    return {
        "mean":
            float(
                finite.mean()
            ),

        "max":
            float(
                finite.max()
            ),

        "top":
            float(
                top_values.mean()
            ),

        "positive_mean":
            (
                float(
                    positive.mean()
                )
                if len(
                    positive
                )
                else 0.0
            ),

        "positive_fraction":
            float(
                (
                    finite
                    > 0.0
                ).mean()
            ),

        "std":
            float(
                finite.std(
                    ddof=0
                )
            ),
    }


def compute_multi_alignment(
    *,
    oracle_gradients: dict,
    references: list[dict],
    top_fraction: float,
) -> tuple[
    list[float],
    list[float],
    dict,
    dict,
]:

    policy_cosines = []

    value_cosines = []

    for reference in references:

        policy_cosines.append(
            cosine_similarity_flat(
                oracle_gradients[
                    "g_policy"
                ],
                reference[
                    "g_policy"
                ],
            )
        )

        value_cosines.append(
            cosine_similarity_flat(
                oracle_gradients[
                    "g_value"
                ],
                reference[
                    "g_value"
                ],
            )
        )

    return (
        policy_cosines,
        value_cosines,
        summarize_alignments(
            policy_cosines,
            top_fraction,
        ),
        summarize_alignments(
            value_cosines,
            top_fraction,
        ),
    )


# ============================================================
# Statistics
# ============================================================

def safe_corr(
    x,
    y,
    method: str,
) -> float:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    x = x[
        mask
    ]

    y = y[
        mask
    ]

    if len(
        x
    ) < 3:

        return np.nan

    if (
        np.std(
            x
        ) < 1e-12
        or np.std(
            y
        ) < 1e-12
    ):

        return np.nan

    if method == "pearson":

        return float(
            pearsonr(
                x,
                y,
            ).statistic
        )

    if method == "spearman":

        return float(
            spearmanr(
                x,
                y,
            ).statistic
        )

    raise ValueError(
        method
    )


def evaluate_direction_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict | None:

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    mask = (
        np.all(
            np.isfinite(
                X
            ),
            axis=1,
        )
        &
        np.isfinite(
            y
        )
    )

    X = X[
        mask
    ]

    y = y[
        mask
    ]

    n = len(
        y
    )

    if n < 10:

        return None

    n_splits = min(
        5,
        n,
    )

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    predictions = np.zeros_like(
        y,
        dtype=np.float64,
    )

    for (
        train_indices,
        test_indices,
    ) in kfold.split(
        X
    ):

        X_train = X[
            train_indices
        ]

        X_test = X[
            test_indices
        ]

        y_train = y[
            train_indices
        ]

        feature_mean = X_train.mean(
            axis=0
        )

        feature_std = X_train.std(
            axis=0,
            ddof=0,
        )

        feature_std[
            feature_std
            < 1e-12
        ] = 1.0

        X_train_scaled = (
            X_train
            - feature_mean
        ) / feature_std

        X_test_scaled = (
            X_test
            - feature_mean
        ) / feature_std

        estimator = Ridge(
            alpha=1.0
        )

        estimator.fit(
            X_train_scaled,
            y_train,
        )

        predictions[
            test_indices
        ] = estimator.predict(
            X_test_scaled
        )

    return {
        "n":
            n,

        "r2":
            float(
                r2_score(
                    y,
                    predictions,
                )
            ),

        "mae":
            float(
                mean_absolute_error(
                    y,
                    predictions,
                )
            ),

        "pearson":
            safe_corr(
                y,
                predictions,
                "pearson",
            ),

        "spearman":
            safe_corr(
                y,
                predictions,
                "spearman",
            ),
    }


# ============================================================
# Report
# ============================================================

def build_report(
    *,
    results: list[dict],
    checkpoint: Path,
    replay: Path,
    oracle_queue: Path,
    reference_size: int,
    n_references: int,
    top_fraction: float,
    temperature: float,
    ppo_clip: float,
    prediction_results: dict,
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
        "=" * 72
    )

    add(
        "ALBERTA - MULTI-REFERENCE GRADIENT ALIGNMENT"
    )

    add(
        "=" * 72
    )

    add()

    add(
        f"Checkpoint:   {checkpoint}"
    )

    add(
        f"Replay:       {replay}"
    )

    add(
        f"Oracle queue: {oracle_queue}"
    )

    add(
        f"Annotations:  {len(results):,}"
    )

    add(
        f"Reference size: {reference_size}"
    )

    add(
        f"References:     {n_references}"
    )

    add(
        f"Top fraction:   {top_fraction}"
    )

    add(
        f"Temperature:    {temperature}"
    )

    add(
        f"PPO clip:       {ppo_clip}"
    )

    add()

    add(
        "GRADIENT SPACE"
    )

    add(
        "-" * 72
    )

    add(
        "Actor alignment is measured in policy-head parameter "
        "space."
    )

    add(
        "Critic alignment is measured in value-head parameter "
        "space."
    )

    add(
        "The diagnostic is therefore not the exact complete "
        "optimizer-update vector."
    )

    add()

    add(
        "PREDICTABILITY"
    )

    add(
        "-" * 72
    )

    for (
        target,
        models,
    ) in prediction_results.items():

        add(
            target
        )

        for (
            model_name,
            metrics,
        ) in models.items():

            if metrics is None:

                add(
                    f"  {model_name}: insufficient samples"
                )

                continue

            add(
                f"  {model_name}: "
                f"N={metrics['n']}, "
                f"R2={metrics['r2']:+.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"Pearson={metrics['pearson']:+.6f}, "
                f"Spearman={metrics['spearman']:+.6f}"
            )

        add()

    add(
        "INTERPRETATION"
    )

    add(
        "-" * 72
    )

    add(
        "Large positive alignment means that an Oracle gradient "
        "points in a similar policy-head or value-head direction "
        "to at least part of the sampled PPO reference bank."
    )

    add()

    add(
        "Low predictability from H/U/HU means that these scalar "
        "uncertainty descriptors contain little information about "
        "the measured gradient direction under this diagnostic."
    )

    add()

    add(
        "This does not imply that uncertainty is generally useless "
        "in reinforcement learning."
    )

    add()

    add(
        "A large M * D quantity combines local Oracle gradient "
        "magnitude with directional compatibility, but remains "
        "a diagnostic proxy rather than a measured downstream "
        "performance gain."
    )

    add()

    add(
        "Any U/HU-dependent result requires queues generated with "
        "the corrected uncertainty estimator."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.reference_size <= 0:

        raise ValueError(
            "--reference-size must be positive."
        )

    if args.n_references <= 1:

        raise ValueError(
            "--n-references must be >= 2."
        )

    if not (
        0.0
        < args.top_fraction
        <= 1.0
    ):

        raise ValueError(
            "--top-fraction must be in (0, 1]."
        )

    if args.temperature <= 0.0:

        raise ValueError(
            "--temperature must be strictly positive."
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    # ========================================================
    # Models
    # ========================================================

    model = load_actor_critic(
        args.checkpoint,
        device,
    )

    bc_policy = load_bc_policy(
        args.bc_checkpoint,
        device,
    )

    # ========================================================
    # Replay / reference bank
    # ========================================================

    replay_rows = load_replay(
        args.replay
    )

    references = build_reference_bank(
        model=model,
        bc_policy=bc_policy,
        replay_rows=replay_rows,
        device=device,
        reference_size=args.reference_size,
        n_references=args.n_references,
        ppo_clip=args.ppo_clip,
        temperature=args.temperature,
        opening_prior_plies=args.opening_prior_plies,
        opening_prior_strength_value=(
            args.opening_prior_strength
        ),
        seed=args.seed,
    )

    # ========================================================
    # Oracle annotations
    # ========================================================

    oracle_rows = load_oracle_queue(
        args.oracle_queue
    )

    if args.max_annotations is not None:

        oracle_rows = oracle_rows[
            :args.max_annotations
        ]

    if not oracle_rows:

        raise RuntimeError(
            "No usable Oracle annotations."
        )

    print()
    print("=" * 72)
    print("MULTI-REFERENCE ORACLE ALIGNMENT")
    print("=" * 72)

    results = []

    total = len(
        oracle_rows
    )

    for (
        index,
        row,
    ) in enumerate(
        oracle_rows,
        start=1,
    ):

        try:

            oracle_gradients = compute_oracle_gradients(
                model=model,
                row=row,
                device=device,
            )

            (
                policy_cosines,
                value_cosines,
                policy_summary,
                value_summary,
            ) = compute_multi_alignment(
                oracle_gradients=oracle_gradients,
                references=references,
                top_fraction=args.top_fraction,
            )

        except Exception as exc:

            print(
                f"[{index}/{total}] SKIP: {exc}"
            )

            continue

        H = float(
            row[
                "H"
            ]
        )

        U = float(
            row[
                "U"
            ]
        )

        HU = (
            H
            * U
        )

        logU = math.log1p(
            U
            / 0.05
        )

        result = {
            "query_id":
                row.get(
                    "query_id"
                ),

            "fen":
                row[
                    "fen"
                ],

            "oracle_move":
                row[
                    "oracle_move"
                ],

            "reward":
                row[
                    "reward"
                ],

            "H":
                H,

            "U":
                U,

            "HU":
                HU,

            "logU":
                logU,

            "M_policy":
                oracle_gradients[
                    "M_policy"
                ],

            "M_value":
                oracle_gradients[
                    "M_value"
                ],

            "D_policy_mean":
                policy_summary[
                    "mean"
                ],

            "D_policy_max":
                policy_summary[
                    "max"
                ],

            "D_policy_top":
                policy_summary[
                    "top"
                ],

            "D_policy_positive_mean":
                policy_summary[
                    "positive_mean"
                ],

            "D_policy_positive_fraction":
                policy_summary[
                    "positive_fraction"
                ],

            "D_policy_std":
                policy_summary[
                    "std"
                ],

            "D_value_mean":
                value_summary[
                    "mean"
                ],

            "D_value_max":
                value_summary[
                    "max"
                ],

            "D_value_top":
                value_summary[
                    "top"
                ],

            "D_value_positive_mean":
                value_summary[
                    "positive_mean"
                ],

            "D_value_positive_fraction":
                value_summary[
                    "positive_fraction"
                ],

            "D_value_std":
                value_summary[
                    "std"
                ],

            "policy_cosines":
                policy_cosines,

            "value_cosines":
                value_cosines,
        }

        results.append(
            result
        )

        if (
            index == 1
            or index % 25 == 0
            or index == total
        ):

            print(
                f"[{index:4d}/{total:4d}] "
                f"pi_top={result['D_policy_top']:+.4f} "
                f"pi+={result['D_policy_positive_fraction']:.2f} | "
                f"V_top={result['D_value_top']:+.4f} "
                f"V+={result['D_value_positive_fraction']:.2f}"
            )

    if not results:

        raise RuntimeError(
            "No valid alignment result."
        )

    # ========================================================
    # Save raw results
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in results:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # Feature arrays
    # ========================================================

    H = np.asarray(
        [
            row[
                "H"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    U = np.asarray(
        [
            row[
                "U"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    HU = np.asarray(
        [
            row[
                "HU"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    logU = np.asarray(
        [
            row[
                "logU"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    X_raw = np.column_stack(
        [
            H,
            U,
            HU,
        ]
    )

    X_log = np.column_stack(
        [
            H,
            logU,
            H
            * logU,
        ]
    )

    # ========================================================
    # Targets
    # ========================================================

    target_names = (
        "D_policy_top",
        "D_policy_positive_mean",
        "D_policy_positive_fraction",
        "D_value_top",
        "D_value_positive_mean",
        "D_value_positive_fraction",
    )

    targets = {
        name:
            np.asarray(
                [
                    row[
                        name
                    ]
                    for row
                    in results
                ],
                dtype=np.float64,
            )

        for name
        in target_names
    }

    # ========================================================
    # Magnitude × direction
    # ========================================================

    M_policy = np.asarray(
        [
            row[
                "M_policy"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    M_value = np.asarray(
        [
            row[
                "M_value"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    targets[
        "M_policy_x_D_policy_top"
    ] = (
        M_policy
        * targets[
            "D_policy_top"
        ]
    )

    targets[
        "M_value_x_D_value_top"
    ] = (
        M_value
        * targets[
            "D_value_top"
        ]
    )

    # ========================================================
    # Correlations + OOF prediction
    # ========================================================

    print()
    print("=" * 72)
    print("PREDICTABILITY FROM H / U / HU")
    print("=" * 72)

    feature_dict = {
        "H":
            H,

        "U":
            U,

        "HU":
            HU,

        "logU":
            logU,
    }

    prediction_results = {}

    for (
        target_name,
        target,
    ) in targets.items():

        print()
        print(
            target_name
        )

        print(
            "-" * 72
        )

        for (
            feature_name,
            values,
        ) in feature_dict.items():

            print(
                f"{feature_name:6s} "
                f"Pearson="
                f"{safe_corr(values, target, 'pearson'):+.4f} "
                f"Spearman="
                f"{safe_corr(values, target, 'spearman'):+.4f}"
            )

        raw_metrics = evaluate_direction_model(
            X=X_raw,
            y=target,
            seed=args.seed,
        )

        log_metrics = evaluate_direction_model(
            X=X_log,
            y=target,
            seed=args.seed,
        )

        prediction_results[
            target_name
        ] = {
            "H+U+HU":
                raw_metrics,

            "H+logU+HlogU":
                log_metrics,
        }

        for (
            name,
            metrics,
        ) in prediction_results[
            target_name
        ].items():

            if metrics is None:

                print(
                    f"{name}: insufficient samples"
                )

                continue

            print(
                f"{name}: "
                f"OOF R2={metrics['r2']:+.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"rho={metrics['spearman']:+.6f}"
            )

    # ========================================================
    # Report
    # ========================================================

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_report(
        results=results,
        checkpoint=args.checkpoint,
        replay=args.replay,
        oracle_queue=args.oracle_queue,
        reference_size=args.reference_size,
        n_references=args.n_references,
        top_fraction=args.top_fraction,
        temperature=args.temperature,
        ppo_clip=args.ppo_clip,
        prediction_results=prediction_results,
    )

    args.report.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Raw results: {args.output}"
    )

    print(
        f"Report:      {args.report}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()