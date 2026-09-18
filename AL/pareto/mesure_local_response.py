#!/usr/bin/env python3

"""
ALBERTA - Dynamic Pareto Local Response
=======================================

Refresh the fixed manually annotated Pareto calibration probes under
the RL10 learner and measure their controlled local Oracle response.

Pipeline
--------

    fixed annotated calibration FENs
                |
                v
    reconstruct historical league RL1 ... RL9
                |
                v
    current learner RL10
                |
                v
    recompute H_10 / U_10 / HU_10
                |
                v
    apply one controlled Oracle-only optimizer step
                |
                v
    measure Delta_KL_10 and |Delta_V_10|
                |
                v
    calibration dataset for Pareto response surrogate

The resulting dataset maps:

    (H_t, U_t, H_t * U_t)
        ->
    (Delta_KL_t, Delta_V_t)

where:

    Delta_KL_t =
        KL(pi_before || pi_after)

and:

    Delta_V_t =
        |V_after - V_before|

Delta_V is therefore a response magnitude, not a signed value
improvement.

League reconstruction
---------------------

At the branch point t = 10:

    BC6
        protected opponent
        excluded from uncertainty

    RL1 ... RL9
        historical uncertainty contributors

    RL10
        current learner
        supplied separately to League.uncertainty_batch()

Thus U_10 is computed from exactly:

    RL1 ... RL9 + current RL10

and the current critic contributes exactly once.

Oracle loss
-----------

The Oracle objective is delegated directly to:

    training.train_al.compute_oracle_loss()

Optimizer
---------

The diagnostic restores the RL10 learner checkpoint AND its optimizer
state before every annotation.

Therefore every probe starts from the same:

    theta_10
    Adam state_10

and receives one Oracle-only step.

No PPO loss is included in this local probe.

Output
------

data/pareto/pareto_local_responses.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import chess.variant
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam


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

DEFAULT_EPOCH = 10

DEFAULT_ORACLE_QUEUE = (
    PROJECT_ROOT
    / "data"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
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

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "pareto"
    / "pareto_local_responses.jsonl"
)

DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42

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

DEFAULT_ORACLE_POLICY_COEF = (
    al.DEFAULT_ORACLE_POLICY_COEF
)

DEFAULT_ORACLE_VALUE_COEF = (
    al.DEFAULT_ORACLE_VALUE_COEF
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Refresh the fixed Pareto calibration probe at RL10 "
            "and measure local Oracle responses."
        )
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=DEFAULT_EPOCH,
    )

    parser.add_argument(
        "--oracle-queue",
        type=Path,
        default=DEFAULT_ORACLE_QUEUE,
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
        default=DEFAULT_OUTPUT,
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
        "--oracle-policy-coef",
        type=float,
        default=DEFAULT_ORACLE_POLICY_COEF,
    )

    parser.add_argument(
        "--oracle-value-coef",
        type=float,
        default=DEFAULT_ORACLE_VALUE_COEF,
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

    for module in model.modules():

        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()


# ============================================================
# ActorCritic loading
# ============================================================

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
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    return model


def load_current_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Current learner checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    for key in (
        "model_state_dict",
        "optimizer_state_dict",
    ):

        if key not in checkpoint:
            raise RuntimeError(
                f"Current checkpoint has no {key}:\n{path}"
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

    return checkpoint


# ============================================================
# BC loading
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
        num_actions=len(ACTIONS),
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
# Historical league reconstruction
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
    """
    Reconstruct the historical league immediately before the
    current RL10 learner.

    At t=10:

        BC6       protected, excluded from U
        RL1..RL9  historical U contributors
        RL10      NOT inserted here

    RL10 is supplied separately as current_model to
    League.uncertainty_batch(), so it contributes exactly once.
    """

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

    # --------------------------------------------------------
    # Protected BC6 anchor
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Historical RL snapshots: RL1 ... RL(t-1)
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

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
        len(league.uncertainty_names())
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
            "Current learner snapshot is already present in "
            "the historical league. This would duplicate the "
            "current critic in U."
        )

    return league


# ============================================================
# Human calibration annotations
# ============================================================

def load_annotations(
    path: Path,
) -> list[dict]:

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
                    f"Invalid JSON at line {line_number}."
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

            fen = record.get(
                "fen"
            )

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
                not isinstance(
                    fen,
                    str,
                )
                or not fen
                or oracle_move is None
                or confidence is None
                or situation is None
                or reward_raw is None
            ):
                skipped_incomplete += 1
                continue

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

            board = chess.variant.AtomicBoard(
                fen
            )

            legal_moves = {
                move.uci()
                for move
                in board.legal_moves
            }

            if oracle_move not in legal_moves:
                raise ValueError(
                    f"Illegal Oracle move at line "
                    f"{line_number}:\n"
                    f"FEN: {fen}\n"
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
                }
            )

    if not annotations:
        raise RuntimeError(
            "No complete Oracle annotations found."
        )

    print()
    print("=" * 72)
    print("CALIBRATION PROBE")
    print("=" * 72)

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
# Dynamic H
# ============================================================

@torch.no_grad()
def compute_entropy(
    *,
    current_model,
    bc_policy,
    board: chess.variant.AtomicBoard,
    device: torch.device,
    opening_prior_plies: int,
    opening_prior_strength: float,
) -> float:

    encoded = encode_boards(
        [
            board
        ]
    ).to(
        device
    )

    logits, _ = current_model(
        encoded
    )

    legal_moves = list(
        board.legal_moves
    )

    if not legal_moves:
        raise RuntimeError(
            "Probe position has no legal moves."
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
        0,
        legal_index_tensor,
    ]

    alpha = opening_prior_alpha(
        board,
        prior_plies=opening_prior_plies,
        prior_strength=opening_prior_strength,
    )

    if alpha > 0.0:

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

        bc_legal_logits = bc_logits[
            0,
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

    return float(
        entropy.item()
    )


# ============================================================
# Refresh H/U/HU
# ============================================================

@torch.no_grad()
def refresh_annotation_features(
    *,
    annotations: list[dict],
    current_model,
    bc_policy,
    league: League,
    device: torch.device,
    opening_prior_plies: int,
    opening_prior_strength: float,
) -> list[dict]:

    print()
    print("=" * 72)
    print("REFRESHING PARETO PROBE FEATURES")
    print("=" * 72)

    refreshed = []

    for index, annotation in enumerate(
        annotations,
        start=1,
    ):

        board = chess.variant.AtomicBoard(
            annotation[
                "fen"
            ]
        )

        H = compute_entropy(
            current_model=current_model,
            bc_policy=bc_policy,
            board=board,
            device=device,
            opening_prior_plies=opening_prior_plies,
            opening_prior_strength=(
                opening_prior_strength
            ),
        )

        encoded = encode_boards(
            [
                board
            ]
        ).to(
            device
        )

        U_tensor = league.uncertainty_batch(
            encoded,
            current_model=current_model,
        )

        U = float(
            U_tensor[
                0
            ].item()
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
                "Non-finite dynamic H/U."
            )

        updated = dict(
            annotation
        )

        updated[
            "H"
        ] = H

        updated[
            "U"
        ] = U

        updated[
            "HU"
        ] = (
            H
            * U
        )

        refreshed.append(
            updated
        )

        print(
            f"[{index:03d}/{len(annotations):03d}] "
            f"H={H:.6f} | "
            f"U={U:.6e} | "
            f"HU={H * U:.6e}"
        )

    return refreshed


# ============================================================
# Intrinsic policy/value evaluation
# ============================================================

@torch.no_grad()
def evaluate_position(
    *,
    model,
    fen: str,
    device: torch.device,
) -> dict:

    model.eval()

    board = chess.variant.AtomicBoard(
        fen
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
            "Probe position has no legal moves."
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
        0,
        legal_index_tensor,
    ]

    log_probs = F.log_softmax(
        legal_logits,
        dim=0,
    )

    probs = torch.exp(
        log_probs
    )

    return {
        "legal_indices":
            legal_indices,

        "probs":
            probs.detach().cpu(),

        "log_probs":
            log_probs.detach().cpu(),

        "value":
            float(
                values[
                    0,
                    0
                ].item()
            ),
    }


# ============================================================
# KL
# ============================================================

def compute_kl(
    before: dict,
    after: dict,
) -> float:

    if (
        before[
            "legal_indices"
        ]
        != after[
            "legal_indices"
        ]
    ):
        raise RuntimeError(
            "Legal support changed between evaluations."
        )

    kl = (
        before[
            "probs"
        ]
        * (
            before[
                "log_probs"
            ]
            - after[
                "log_probs"
            ]
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
# Controlled Oracle response
# ============================================================

def measure_annotation(
    *,
    model,
    optimizer,
    annotation: dict,
    device: torch.device,
    policy_coef: float,
    value_coef: float,
) -> dict:

    before = evaluate_position(
        model=model,
        fen=annotation[
            "fen"
        ],
        device=device,
    )

    model.train()

    freeze_batchnorm(
        model
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_info = al.compute_oracle_loss(
        model,
        [
            annotation
        ],
        device=device,
        policy_coef=policy_coef,
        value_coef=value_coef,
    )

    loss = loss_info[
        "loss"
    ]

    loss.backward()

    gradients = [
        parameter.grad
        .detach()
        .reshape(
            -1
        )
        for parameter
        in model.parameters()
        if parameter.grad is not None
    ]

    if not gradients:
        raise RuntimeError(
            "Oracle loss produced no gradients."
        )

    gradient_norm = torch.linalg.vector_norm(
        torch.cat(
            gradients
        )
    )

    optimizer.step()

    after = evaluate_position(
        model=model,
        fen=annotation[
            "fen"
        ],
        device=device,
    )

    delta_kl = compute_kl(
        before,
        after,
    )

    delta_v_signed = (
        after[
            "value"
        ]
        - before[
            "value"
        ]
    )

    delta_v = abs(
        delta_v_signed
    )

    confidence_weight = al.CONFIDENCE_WEIGHTS[
        annotation[
            "confidence"
        ]
    ]

    situation_weight = al.SITUATION_WEIGHTS[
        annotation[
            "situation"
        ]
    ]

    return {
        "delta_kl":
            delta_kl,

        "delta_v":
            delta_v,

        "delta_v_signed":
            delta_v_signed,

        "value_before":
            before[
                "value"
            ],

        "value_after":
            after[
                "value"
            ],

        "oracle_loss":
            float(
                loss.detach().item()
            ),

        "gradient_norm":
            float(
                gradient_norm.item()
            ),

        "confidence_weight":
            float(
                confidence_weight
            ),

        "situation_weight":
            float(
                situation_weight
            ),

        "oracle_weight":
            float(
                confidence_weight
                * situation_weight
            ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.epoch < 1:
        raise ValueError(
            "--epoch must be >= 1."
        )

    if args.league_max_agents <= 0:
        raise ValueError(
            "--league-max-agents must be positive."
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    checkpoint_path = args.checkpoint

    print()
    print("=" * 72)
    print("ALBERTA - DYNAMIC PARETO LOCAL RESPONSE")
    print("=" * 72)

    print(
        f"Epoch:                 {args.epoch}"
    )

    print(
        f"Checkpoint:            {checkpoint_path}"
    )

    print(
        f"Probe queue:           {args.oracle_queue}"
    )

    print(
        f"Baseline league:       "
        f"{args.baseline_league_dir}"
    )

    print(
        f"League max agents:     "
        f"{args.league_max_agents}"
    )

    print(
        f"Channels:              "
        f"{args.channels}"
    )

    print(
        f"Blocks:                "
        f"{args.blocks}"
    )

    print(
        f"Device:                {device}"
    )

    # ========================================================
    # Fixed human annotations
    # ========================================================

    annotations = load_annotations(
        args.oracle_queue
    )

    # ========================================================
    # Current learner state
    # ========================================================

    checkpoint = load_current_checkpoint(
        checkpoint_path,
        device,
    )

    base_model_state = copy.deepcopy(
        checkpoint[
            "model_state_dict"
        ]
    )

    base_optimizer_state = copy.deepcopy(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    # ========================================================
    # Current learner used for H/U
    # ========================================================

    current_model = al.build_actor_critic(
        args,
        device,
    )

    current_model.load_state_dict(
        base_model_state,
        strict=True,
    )

    current_model.eval()

    # ========================================================
    # BC6 opening prior
    # ========================================================

    bc_policy = load_bc_policy(
        epoch=DEFAULT_BC_PRIOR_EPOCH,
        bc_dir=args.bc_dir,
        device=device,
    )

    # ========================================================
    # Historical league
    # ========================================================

    league = load_current_league(
        current_epoch=args.epoch,
        baseline_league_dir=(
            args.baseline_league_dir
        ),
        bc_dir=args.bc_dir,
        args=args,
        device=device,
        max_agents=args.league_max_agents,
    )

    # ========================================================
    # Refresh learner-dependent features
    # ========================================================

    annotations = refresh_annotation_features(
        annotations=annotations,
        current_model=current_model,
        bc_policy=bc_policy,
        league=league,
        device=device,
        opening_prior_plies=(
            args.opening_prior_plies
        ),
        opening_prior_strength=(
            args.opening_prior_strength
        ),
    )

    # ========================================================
    # Probe model
    # ========================================================

    model = al.build_actor_critic(
        args,
        device,
    )

    # ========================================================
    # Measure independently
    # ========================================================

    results = []

    print()
    print("=" * 72)
    print("MEASURING LOCAL ORACLE RESPONSES")
    print("=" * 72)

    for index, annotation in enumerate(
        annotations,
        start=1,
    ):

        # ----------------------------------------------------
        # Exact common learner state
        # ----------------------------------------------------

        model.load_state_dict(
            base_model_state,
            strict=True,
        )

        # ----------------------------------------------------
        # Exact common optimizer state
        # ----------------------------------------------------

        optimizer = Adam(
            model.parameters(),
            lr=rl.DEFAULT_LR,
        )

        optimizer.load_state_dict(
            copy.deepcopy(
                base_optimizer_state
            )
        )

        actual_lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        response = measure_annotation(
            model=model,
            optimizer=optimizer,
            annotation=annotation,
            device=device,
            policy_coef=args.oracle_policy_coef,
            value_coef=args.oracle_value_coef,
        )

        result = {
            "query_id":
                annotation.get(
                    "query_id"
                ),

            "fen":
                annotation[
                    "fen"
                ],

            "H":
                annotation[
                    "H"
                ],

            "U":
                annotation[
                    "U"
                ],

            "HU":
                annotation[
                    "HU"
                ],

            "oracle_move":
                annotation[
                    "oracle_move"
                ],

            "oracle_confidence":
                annotation[
                    "confidence"
                ],

            "oracle_situation":
                annotation[
                    "situation"
                ],

            "reward":
                annotation[
                    "reward"
                ],

            "learner_epoch":
                args.epoch,

            "optimizer_lr":
                actual_lr,

            **response,
        }

        results.append(
            result
        )

        print(
            f"[{index:03d}/{len(annotations):03d}] "
            f"H={annotation['H']:.4f} | "
            f"U={annotation['U']:.6e} | "
            f"KL={response['delta_kl']:.6e} | "
            f"|dV|={response['delta_v']:.6e} | "
            f"|g|={response['gradient_norm']:.6e}"
        )

    # ========================================================
    # Save
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # Summary
    # ========================================================

    H_values = np.asarray(
        [
            row[
                "H"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    U_values = np.asarray(
        [
            row[
                "U"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    KL_values = np.asarray(
        [
            row[
                "delta_kl"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    DV_values = np.asarray(
        [
            row[
                "delta_v"
            ]
            for row
            in results
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 72)
    print("DYNAMIC LOCAL RESPONSE COMPLETE")
    print("=" * 72)

    print(
        f"Positions: {len(results):,}"
    )

    print()

    print(
        f"H:    mean={H_values.mean():.6e} "
        f"median={np.median(H_values):.6e}"
    )

    print(
        f"U:    mean={U_values.mean():.6e} "
        f"median={np.median(U_values):.6e}"
    )

    print(
        f"KL:   mean={KL_values.mean():.6e} "
        f"median={np.median(KL_values):.6e}"
    )

    print(
        f"|dV|: mean={DV_values.mean():.6e} "
        f"median={np.median(DV_values):.6e}"
    )

    print()

    print(
        f"Saved to:\n  {args.output}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()