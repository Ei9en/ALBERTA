#!/usr/bin/env python3

"""
ALBERTA - Reference Gradient Stability Diagnostic
=================================================

Measure the stability of PPO reference-gradient directions across
independent samples from the same on-policy replay buffer.

For K independently sampled reference sets:

    g_pi^(1), ..., g_pi^(K)
    g_V^(1),  ..., g_V^(K)

we compute pairwise cosine similarities:

    cos(g_i, g_j)

Scientific question
-------------------

Is a PPO gradient estimated from one small replay subsample a stable
directional reference?

If independently sampled reference gradients have low pairwise
cosine similarity, then a single global reference direction is a
poorly identified object.

This matters for gradient-alignment acquisition diagnostics:

    D_i = cos(g_oracle_i, g_ref)

because instability of g_ref implies instability of D_i even before
asking whether H / U / HU predict it.

Gradient space
--------------

The diagnostic intentionally measures:

    - actor gradients in the policy head;
    - critic gradients in the value head.

It therefore measures head-space directional stability, not the
complete optimizer update over the shared representation.

PPO actor semantics
-------------------

The actor reference reproduces the canonical ALBERTA rollout/PPO
policy:

    legal RL logits
    + BC opening prior
    -> divide by rollout temperature
    -> legal softmax
    -> clipped PPO surrogate

The replay rows therefore require:

    fen
    action
    legal_moves
    old_log_prob
    advantage
    return
    value
    ply

Critic semantics
----------------

The critic reference preserves ALBERTA's historical clipped-value
objective:

    V_clip =
        V_old + clip(V - V_old, -epsilon, +epsilon)

    L_V =
        MSE(V_clip, return)

This non-standard critic clipping is preserved intentionally for
reproducibility.

Optional Oracle diagnostic
--------------------------

If an Oracle queue is supplied, each Oracle annotation gradient is
compared against every sampled PPO reference direction.

This measures how unstable:

    cos(g_oracle, g_ref)

is under different plausible choices of g_ref.

Oracle policy gradients use legal-action cross-entropy, consistent
with the canonical Oracle objective.

No optimizer.step().
No self-play.

Example
-------

python analysis/reference_gradient_stability.py \
    --checkpoint checkpoints/rl_epoch/rl_epoch_10.pt \
    --replay checkpoints/buffer/replay_buffer_epoch_10.pkl \
    --bc-checkpoint checkpoints/bc_epoch/bc_epoch_7.pt \
    --reference-size 256 \
    --n-references 10 \
    --oracle-queue checkpoints/queue/oracle_queue_1-10_random.jsonl \
    --max-annotations 20 \
    --temperature 2.0 \
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import chess.variant
import numpy as np
import torch
import torch.nn.functional as F


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
    / "reference_gradient_stability.json"
)

DEFAULT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "gradient_alignment"
    / "reference_gradient_stability_report.txt"
)

DEFAULT_SEED = 42

DEFAULT_REFERENCE_SIZE = 256
DEFAULT_N_REFERENCES = 10

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
            "Measure the stability of PPO policy-head and "
            "value-head reference-gradient directions."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
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
        "--oracle-queue",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-annotations",
        type=int,
        default=20,
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

    Missing gradients are represented explicitly by zeros so
    every gradient vector has exactly the same dimensionality.
    """

    chunks = []

    for parameter in parameters:

        if parameter.grad is None:

            gradient = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )

        else:

            gradient = parameter.grad.detach()

        chunks.append(
            gradient.reshape(
                -1
            ).cpu()
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

        return np.nan

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

        return np.nan

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
# Models
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
            "Checkpoint does not contain model_state_dict:\n"
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
            "BC checkpoint does not contain "
            "model_state_dict:\n"
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

    for attribute in (
        "buffer",
        "data",
        "transitions",
    ):

        if hasattr(
            obj,
            attribute,
        ):

            return normalize_replay_container(
                getattr(
                    obj,
                    attribute,
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
                f"Unsupported legal move: {move!r}"
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

        if not all(
            np.isfinite(
                value
            )
            for value in (
                old_log_prob,
                advantage,
                ret,
                old_value,
            )
        ):

            skipped += 1
            continue

        if (
            action < 0
            or action >= len(
                ACTIONS
            )
        ):

            skipped += 1
            continue

        legal_indices = []

        invalid_legal = False

        for move in legal_moves:

            action_index = ACTION_TO_INDEX.get(
                move
            )

            if action_index is None:

                invalid_legal = True
                break

            legal_indices.append(
                action_index
            )

        if (
            invalid_legal
            or not legal_indices
            or action not in legal_indices
        ):

            skipped += 1
            continue

        actual_legal_moves = {
            move.uci()
            for move
            in board.legal_moves
        }

        if set(
            legal_moves
        ) != actual_legal_moves:

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
            "Replay action not present in legal support."
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
    # Actor
    # ========================================================

    zero_model_grads(
        model
    )

    policy_logits, _ = model(
        encoded
    )

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

        new_log_probs.append(
            legal_log_prob_for_replay_action(
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
        )

    new_log_probs = torch.stack(
        new_log_probs
    )

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

    actor_loss.backward()

    g_policy = flatten_parameter_grads(
        get_policy_parameters(
            model
        )
    )

    # ========================================================
    # Critic
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
# Pairwise stability
# ============================================================

def pairwise_cosines(
    vectors: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:

    n = len(
        vectors
    )

    matrix = np.full(
        (
            n,
            n,
        ),
        np.nan,
        dtype=np.float64,
    )

    np.fill_diagonal(
        matrix,
        1.0,
    )

    values = []

    for i in range(
        n
    ):

        for j in range(
            i + 1,
            n,
        ):

            cosine = cosine_similarity_flat(
                vectors[
                    i
                ],
                vectors[
                    j
                ],
            )

            matrix[
                i,
                j,
            ] = cosine

            matrix[
                j,
                i,
            ] = cosine

            values.append(
                cosine
            )

    return (
        np.asarray(
            values,
            dtype=np.float64,
        ),
        matrix,
    )


def summarize_values(
    values: np.ndarray,
) -> dict:

    finite = values[
        np.isfinite(
            values
        )
    ]

    if len(
        finite
    ) == 0:

        return {
            "n":
                0,

            "mean":
                np.nan,

            "median":
                np.nan,

            "std":
                np.nan,

            "min":
                np.nan,

            "p10":
                np.nan,

            "p25":
                np.nan,

            "p75":
                np.nan,

            "p90":
                np.nan,

            "max":
                np.nan,

            "fraction_positive":
                np.nan,

            "fraction_gt_05":
                np.nan,

            "fraction_gt_08":
                np.nan,
        }

    return {
        "n":
            int(
                len(
                    finite
                )
            ),

        "mean":
            float(
                np.mean(
                    finite
                )
            ),

        "median":
            float(
                np.median(
                    finite
                )
            ),

        "std":
            float(
                np.std(
                    finite,
                    ddof=0,
                )
            ),

        "min":
            float(
                np.min(
                    finite
                )
            ),

        "p10":
            float(
                np.percentile(
                    finite,
                    10,
                )
            ),

        "p25":
            float(
                np.percentile(
                    finite,
                    25,
                )
            ),

        "p75":
            float(
                np.percentile(
                    finite,
                    75,
                )
            ),

        "p90":
            float(
                np.percentile(
                    finite,
                    90,
                )
            ),

        "max":
            float(
                np.max(
                    finite
                )
            ),

        "fraction_positive":
            float(
                np.mean(
                    finite
                    > 0.0
                )
            ),

        "fraction_gt_05":
            float(
                np.mean(
                    finite
                    > 0.5
                )
            ),

        "fraction_gt_08":
            float(
                np.mean(
                    finite
                    > 0.8
                )
            ),
    }


def print_pairwise_summary(
    label: str,
    summary: dict,
) -> None:

    print()
    print(
        label
    )

    print(
        "-" * 72
    )

    print(
        f"N pairs : {summary['n']}"
    )

    print(
        f"mean    : {summary['mean']:+.6f}"
    )

    print(
        f"median  : {summary['median']:+.6f}"
    )

    print(
        f"std     : {summary['std']:.6f}"
    )

    print(
        f"min     : {summary['min']:+.6f}"
    )

    print(
        f"p10     : {summary['p10']:+.6f}"
    )

    print(
        f"p25     : {summary['p25']:+.6f}"
    )

    print(
        f"p75     : {summary['p75']:+.6f}"
    )

    print(
        f"p90     : {summary['p90']:+.6f}"
    )

    print(
        f"max     : {summary['max']:+.6f}"
    )

    print(
        f"> 0     : "
        f"{summary['fraction_positive']:.2%}"
    )

    print(
        f"> 0.5   : "
        f"{summary['fraction_gt_05']:.2%}"
    )

    print(
        f"> 0.8   : "
        f"{summary['fraction_gt_08']:.2%}"
    )


# ============================================================
# Oracle loading
# ============================================================

def load_oracle_queue(
    path: Path,
    max_annotations: int | None,
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

            reward_raw = row.get(
                "reward"
            )

            if (
                not isinstance(
                    fen,
                    str,
                )
                or not fen
                or oracle_move is None
                or reward_raw is None
            ):

                skipped += 1
                continue

            try:

                reward = float(
                    reward_raw
                )

            except (
                TypeError,
                ValueError,
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

            rows.append(
                normalized
            )

            if (
                max_annotations is not None
                and len(
                    rows
                ) >= max_annotations
            ):

                break

    print()
    print("=" * 72)
    print("ORACLE QUEUE")
    print("=" * 72)

    print(
        f"Selected: {len(rows):,}"
    )

    print(
        f"Skipped:  {skipped:,}"
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

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
            "Oracle action is not legal."
        ) from exc

    encoded = encode_boards(
        [
            board
        ]
    ).to(
        device
    )

    # ========================================================
    # Policy
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
    # Value
    # ========================================================

    zero_model_grads(
        model
    )

    _, value = model(
        encoded
    )

    reward = torch.tensor(
        [
            row[
                "reward"
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    value_loss = F.mse_loss(
        value[
            :,
            0
        ],
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

    return (
        g_policy,
        g_value,
    )


# ============================================================
# Oracle alignment variability
# ============================================================

def finite_summary(
    values,
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

            "std":
                np.nan,

            "min":
                np.nan,

            "max":
                np.nan,
        }

    return {
        "mean":
            float(
                finite.mean()
            ),

        "std":
            float(
                finite.std(
                    ddof=0
                )
            ),

        "min":
            float(
                finite.min()
            ),

        "max":
            float(
                finite.max()
            ),
    }


def oracle_alignment_stability(
    *,
    model,
    oracle_rows: list[dict],
    references: list[dict],
    device: torch.device,
) -> list[dict]:

    print()
    print("=" * 72)
    print("ORACLE ALIGNMENT STABILITY")
    print("=" * 72)

    output = []

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

            (
                g_oracle_policy,
                g_oracle_value,
            ) = compute_oracle_gradients(
                model=model,
                row=row,
                device=device,
            )

        except Exception as exc:

            print(
                f"[{index}/{total}] SKIP: {exc}"
            )

            continue

        policy_values = [
            cosine_similarity_flat(
                g_oracle_policy,
                reference[
                    "g_policy"
                ],
            )
            for reference
            in references
        ]

        value_values = [
            cosine_similarity_flat(
                g_oracle_value,
                reference[
                    "g_value"
                ],
            )
            for reference
            in references
        ]

        policy_summary = finite_summary(
            policy_values
        )

        value_summary = finite_summary(
            value_values
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

            "D_policy_mean":
                policy_summary[
                    "mean"
                ],

            "D_policy_std":
                policy_summary[
                    "std"
                ],

            "D_policy_min":
                policy_summary[
                    "min"
                ],

            "D_policy_max":
                policy_summary[
                    "max"
                ],

            "D_value_mean":
                value_summary[
                    "mean"
                ],

            "D_value_std":
                value_summary[
                    "std"
                ],

            "D_value_min":
                value_summary[
                    "min"
                ],

            "D_value_max":
                value_summary[
                    "max"
                ],

            "D_policy_values":
                policy_values,

            "D_value_values":
                value_values,
        }

        output.append(
            result
        )

        print(
            f"[{index:3d}/{total:3d}] "
            f"Dpi={result['D_policy_mean']:+.4f} "
            f"+/- {result['D_policy_std']:.4f} | "
            f"DV={result['D_value_mean']:+.4f} "
            f"+/- {result['D_value_std']:.4f}"
        )

    return output


# ============================================================
# Safe coefficient of variation
# ============================================================

def coefficient_of_variation(
    values: np.ndarray,
) -> float:

    mean_value = float(
        np.mean(
            values
        )
    )

    if abs(
        mean_value
    ) < EPS:

        return np.nan

    return float(
        np.std(
            values,
            ddof=0,
        )
        / abs(
            mean_value
        )
    )


# ============================================================
# Report
# ============================================================

def build_report(
    *,
    checkpoint: Path,
    replay: Path,
    reference_size: int,
    n_references: int,
    ppo_clip: float,
    temperature: float,
    policy_summary: dict,
    value_summary: dict,
    policy_norms: np.ndarray,
    value_norms: np.ndarray,
    oracle_results: list[dict],
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
        "ALBERTA - REFERENCE GRADIENT STABILITY"
    )

    add(
        "=" * 72
    )

    add()

    add(
        f"Checkpoint: {checkpoint}"
    )

    add(
        f"Replay:     {replay}"
    )

    add(
        f"Reference size: {reference_size}"
    )

    add(
        f"References:     {n_references}"
    )

    add(
        f"PPO clip:       {ppo_clip}"
    )

    add(
        f"Temperature:    {temperature}"
    )

    add()

    # ========================================================
    # Main stability result
    # ========================================================

    add(
        "PAIRWISE REFERENCE STABILITY"
    )

    add(
        "-" * 72
    )

    for (
        label,
        summary,
    ) in (
        (
            "Policy head",
            policy_summary,
        ),
        (
            "Value head",
            value_summary,
        ),
    ):

        add(
            label
        )

        add(
            f"  pairs   = {summary['n']}"
        )

        add(
            f"  mean    = {summary['mean']:+.6f}"
        )

        add(
            f"  median  = {summary['median']:+.6f}"
        )

        add(
            f"  std     = {summary['std']:.6f}"
        )

        add(
            f"  min     = {summary['min']:+.6f}"
        )

        add(
            f"  max     = {summary['max']:+.6f}"
        )

        add(
            f"  > 0     = "
            f"{summary['fraction_positive']:.2%}"
        )

        add(
            f"  > 0.5   = "
            f"{summary['fraction_gt_05']:.2%}"
        )

        add(
            f"  > 0.8   = "
            f"{summary['fraction_gt_08']:.2%}"
        )

        add()

    # ========================================================
    # Norms
    # ========================================================

    add(
        "GRADIENT NORM VARIABILITY"
    )

    add(
        "-" * 72
    )

    add(
        f"Policy norm mean = "
        f"{policy_norms.mean():.6e}"
    )

    add(
        f"Policy norm std  = "
        f"{policy_norms.std(ddof=0):.6e}"
    )

    add(
        f"Policy norm CV   = "
        f"{coefficient_of_variation(policy_norms):.6f}"
    )

    add()

    add(
        f"Value norm mean  = "
        f"{value_norms.mean():.6e}"
    )

    add(
        f"Value norm std   = "
        f"{value_norms.std(ddof=0):.6e}"
    )

    add(
        f"Value norm CV    = "
        f"{coefficient_of_variation(value_norms):.6f}"
    )

    add()

    # ========================================================
    # Oracle
    # ========================================================

    if oracle_results:

        policy_stds = np.asarray(
            [
                row[
                    "D_policy_std"
                ]
                for row
                in oracle_results
            ],
            dtype=np.float64,
        )

        value_stds = np.asarray(
            [
                row[
                    "D_value_std"
                ]
                for row
                in oracle_results
            ],
            dtype=np.float64,
        )

        add(
            "ORACLE ALIGNMENT VARIABILITY"
        )

        add(
            "-" * 72
        )

        add(
            f"Annotations = "
            f"{len(oracle_results)}"
        )

        add(
            f"mean std(D_policy)   = "
            f"{np.nanmean(policy_stds):.6f}"
        )

        add(
            f"median std(D_policy) = "
            f"{np.nanmedian(policy_stds):.6f}"
        )

        add(
            f"mean std(D_value)    = "
            f"{np.nanmean(value_stds):.6f}"
        )

        add(
            f"median std(D_value)  = "
            f"{np.nanmedian(value_stds):.6f}"
        )

        add()

    # ========================================================
    # Interpretation
    # ========================================================

    add(
        "INTERPRETATION"
    )

    add(
        "-" * 72
    )

    add(
        "Pairwise cosine similarity measures whether independent "
        "replay subsamples produce similar PPO gradient directions."
    )

    add()

    add(
        "Values near +1 indicate a stable common direction. "
        "Values near 0 indicate largely orthogonal directions. "
        "Negative values indicate conflicting sampled directions."
    )

    add()

    add(
        "Low pairwise stability means that a single small-batch "
        "PPO reference gradient should not be interpreted as a "
        "well-defined global learning direction."
    )

    add()

    add(
        "The optional Oracle experiment shows the consequence: "
        "alignment assigned to the same annotation can change "
        "substantially depending on which PPO reference sample "
        "is used."
    )

    add()

    add(
        "This result is independent of the H/U uncertainty "
        "estimator and therefore does not depend on the historical "
        "BC-value-head contamination discovered elsewhere."
    )

    add()

    add(
        "The diagnostic is measured in policy-head and value-head "
        "parameter spaces rather than over the complete shared "
        "ActorCritic parameter vector."
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

    if args.n_references < 2:

        raise ValueError(
            "--n-references must be >= 2."
        )

    if args.temperature <= 0.0:

        raise ValueError(
            "--temperature must be strictly positive."
        )

    if args.ppo_clip <= 0.0:

        raise ValueError(
            "--ppo-clip must be strictly positive."
        )

    if (
        args.max_annotations is not None
        and args.max_annotations <= 0
    ):

        raise ValueError(
            "--max-annotations must be positive."
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
    # Replay
    # ========================================================

    replay_rows = load_replay(
        args.replay
    )

    replay_rows = filter_valid_replay_rows(
        replay_rows
    )

    if len(
        replay_rows
    ) < args.reference_size:

        raise RuntimeError(
            f"Only {len(replay_rows):,} valid replay rows, "
            f"but reference-size={args.reference_size}."
        )

    # ========================================================
    # Independent reference gradients
    # ========================================================

    print()
    print("=" * 72)
    print("BUILDING REFERENCE GRADIENTS")
    print("=" * 72)

    references = []

    for reference_index in range(
        args.n_references
    ):

        rng = random.Random(
            args.seed
            + reference_index
        )

        sampled_rows = rng.sample(
            replay_rows,
            args.reference_size,
        )

        reference = compute_reference_gradient(
            model=model,
            bc_policy=bc_policy,
            rows=sampled_rows,
            device=device,
            ppo_clip=args.ppo_clip,
            temperature=args.temperature,
            opening_prior_plies=args.opening_prior_plies,
            opening_prior_strength_value=(
                args.opening_prior_strength
            ),
        )

        references.append(
            reference
        )

        print(
            f"[{reference_index + 1:02d}/"
            f"{args.n_references:02d}] "
            f"Lpi={reference['actor_loss']:+.6f} "
            f"LV={reference['critic_loss']:.6f} | "
            f"|gpi|={reference['policy_norm']:.6e} "
            f"|gV|={reference['value_norm']:.6e}"
        )

    # ========================================================
    # Pairwise stability
    # ========================================================

    policy_vectors = [
        reference[
            "g_policy"
        ]
        for reference
        in references
    ]

    value_vectors = [
        reference[
            "g_value"
        ]
        for reference
        in references
    ]

    (
        policy_cosines,
        policy_matrix,
    ) = pairwise_cosines(
        policy_vectors
    )

    (
        value_cosines,
        value_matrix,
    ) = pairwise_cosines(
        value_vectors
    )

    policy_summary = summarize_values(
        policy_cosines
    )

    value_summary = summarize_values(
        value_cosines
    )

    print()
    print("=" * 72)
    print("REFERENCE GRADIENT STABILITY")
    print("=" * 72)

    print_pairwise_summary(
        "POLICY-HEAD REFERENCE COSINES",
        policy_summary,
    )

    print_pairwise_summary(
        "VALUE-HEAD REFERENCE COSINES",
        value_summary,
    )

    # ========================================================
    # Norm variability
    # ========================================================

    policy_norms = np.asarray(
        [
            reference[
                "policy_norm"
            ]
            for reference
            in references
        ],
        dtype=np.float64,
    )

    value_norms = np.asarray(
        [
            reference[
                "value_norm"
            ]
            for reference
            in references
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 72)
    print("GRADIENT NORM VARIABILITY")
    print("=" * 72)

    print(
        f"Policy mean: "
        f"{policy_norms.mean():.6e}"
    )

    print(
        f"Policy std:  "
        f"{policy_norms.std(ddof=0):.6e}"
    )

    print(
        f"Policy CV:   "
        f"{coefficient_of_variation(policy_norms):.6f}"
    )

    print()

    print(
        f"Value mean:  "
        f"{value_norms.mean():.6e}"
    )

    print(
        f"Value std:   "
        f"{value_norms.std(ddof=0):.6e}"
    )

    print(
        f"Value CV:    "
        f"{coefficient_of_variation(value_norms):.6f}"
    )

    # ========================================================
    # Optional Oracle alignment stability
    # ========================================================

    oracle_results = []

    if args.oracle_queue is not None:

        oracle_rows = load_oracle_queue(
            args.oracle_queue,
            args.max_annotations,
        )

        if oracle_rows:

            oracle_results = oracle_alignment_stability(
                model=model,
                oracle_rows=oracle_rows,
                references=references,
                device=device,
            )

    # ========================================================
    # Save JSON
    # ========================================================

    output = {
        "seed":
            args.seed,

        "checkpoint":
            str(
                args.checkpoint
            ),

        "replay":
            str(
                args.replay
            ),

        "bc_checkpoint":
            str(
                args.bc_checkpoint
            ),

        "reference_size":
            args.reference_size,

        "n_references":
            args.n_references,

        "ppo_clip":
            args.ppo_clip,

        "temperature":
            args.temperature,

        "opening_prior_plies":
            args.opening_prior_plies,

        "opening_prior_strength":
            args.opening_prior_strength,

        "gradient_space": {
            "policy":
                "policy_head",

            "value":
                "value_head",
        },

        "policy_summary":
            policy_summary,

        "value_summary":
            value_summary,

        "policy_pairwise_cosines":
            policy_cosines.tolist(),

        "value_pairwise_cosines":
            value_cosines.tolist(),

        "policy_cosine_matrix":
            policy_matrix.tolist(),

        "value_cosine_matrix":
            value_matrix.tolist(),

        "policy_norms":
            policy_norms.tolist(),

        "value_norms":
            value_norms.tolist(),

        "references": [
            {
                "actor_loss":
                    reference[
                        "actor_loss"
                    ],

                "critic_loss":
                    reference[
                        "critic_loss"
                    ],

                "policy_norm":
                    reference[
                        "policy_norm"
                    ],

                "value_norm":
                    reference[
                        "value_norm"
                    ],
            }

            for reference
            in references
        ],

        "oracle_results":
            oracle_results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Report
    # ========================================================

    report = build_report(
        checkpoint=args.checkpoint,
        replay=args.replay,
        reference_size=args.reference_size,
        n_references=args.n_references,
        ppo_clip=args.ppo_clip,
        temperature=args.temperature,
        policy_summary=policy_summary,
        value_summary=value_summary,
        policy_norms=policy_norms,
        value_norms=value_norms,
        oracle_results=oracle_results,
    )

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        f"JSON:   {args.output}"
    )

    print(
        f"Report: {args.report}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()