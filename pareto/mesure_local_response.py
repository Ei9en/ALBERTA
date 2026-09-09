#!/usr/bin/env python3

"""
ALBERTA - Measure Local Oracle Response
=======================================

For each manually annotated Pareto probe position:

    1. restore the exact same RL10 model state
    2. restore the exact same optimizer state
    3. measure policy/value before Oracle update
    4. apply one controlled Oracle gradient step
    5. measure policy/value after Oracle update
    6. compute:

           Delta_KL
           Delta_V

    7. restore RL10 before the next annotation

The resulting dataset maps:

    (H, U, HU, score, I_norm)
        ->
    (Delta_KL, Delta_V)

Output
------

    pareto/pareto_local_responses.jsonl
"""

from __future__ import annotations

import copy
import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports
# ============================================================

import training.train_rl_league_colab as rl

from src.encoding import encode_boards

from src.actions_space import (
    ACTIONS,
    ACTION_TO_INDEX,
)

from src.models.resnet import ChessResNet

from src.models.actor_critic import ActorCritic


# ============================================================
# Paths
# ============================================================

ORACLE_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_pareto_probe.jsonl"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "pareto"
    / "pareto_local_responses.jsonl"
)


# ============================================================
# Device
# ============================================================

DEVICE = "cpu"


# ============================================================
# Probe configuration
# ============================================================

N_UPDATE_STEPS = 1


# ============================================================
# Oracle loss coefficients
#
# Keep identical to current AL experiment.
# ============================================================

ORACLE_POLICY_COEF = 0.05

ORACLE_VALUE_COEF = 0.5


# ============================================================
# Annotation weights
# ============================================================

CONFIDENCE_WEIGHTS = {
    "low": 0.50,
    "medium": 0.75,
    "high": 0.99,
}


CRITICALITY_WEIGHTS = {
    "critical": 1.00,
    "non_critical": 0.50,
    "outcome_independent": 0.25,
}


# ============================================================
# Load annotations
# ============================================================

def load_annotations():

    if not ORACLE_QUEUE_PATH.exists():

        raise FileNotFoundError(
            f"Oracle queue not found:\n"
            f"{ORACLE_QUEUE_PATH}"
        )

    annotations = []

    with open(
        ORACLE_QUEUE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            if record.get("status") != "answered":
                continue

            oracle_move = record.get(
                "oracle_move"
            )

            reward = record.get(
                "reward"
            )

            if oracle_move is None:
                continue

            if reward is None:
                continue

            confidence = record.get(
                "oracle_confidence",
                "medium",
            )

            criticality = record.get(
                "oracle_situation",
                "non_critical",
            )

            if confidence not in CONFIDENCE_WEIGHTS:

                raise ValueError(
                    f"Invalid oracle_confidence "
                    f"at line {line_number}: "
                    f"{confidence}"
                )

            if criticality not in CRITICALITY_WEIGHTS:

                raise ValueError(
                    f"Invalid oracle_situation "
                    f"at line {line_number}: "
                    f"{criticality}"
                )

            if oracle_move not in ACTION_TO_INDEX:

                raise ValueError(
                    f"Unknown oracle move at line "
                    f"{line_number}: "
                    f"{oracle_move}"
                )

            reward = float(
                reward
            )

            if reward not in (
                -1.0,
                0.0,
                1.0,
            ):

                raise ValueError(
                    f"Invalid reward at line "
                    f"{line_number}: "
                    f"{reward}"
                )

            annotations.append(
                {
                    "query_id":
                        record["query_id"],

                    "fen":
                        record["fen"],

                    "H":
                        float(record["H"]),

                    "U":
                        float(record["U"]),

                    "HU":
                        float(record["HU"]),

                    "score":
                        float(record["score"]),

                    "I_norm":
                        float(record["I_norm"]),

                    "oracle_move":
                        oracle_move,

                    "confidence":
                        confidence,

                    "criticality":
                        criticality,

                    "reward":
                        reward,
                }
            )

    if not annotations:

        raise RuntimeError(
            "No answered annotations found."
        )

    return annotations


# ============================================================
# Build model
# ============================================================

def create_model():

    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        base_model
    ).to(DEVICE)

    return model


# ============================================================
# Load base checkpoint
# ============================================================

def load_base_state():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint has no model_state_dict."
        )

    if "optimizer_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint has no optimizer_state_dict."
        )

    return checkpoint


# ============================================================
# Legal mask
# ============================================================

def get_legal_indices(
    board,
):

    legal_indices = []

    for move in board.legal_moves:

        move_uci = move.uci()

        if move_uci not in ACTION_TO_INDEX:

            raise ValueError(
                f"Legal Atomic move "
                f"{move_uci} is not present "
                f"in ACTION_TO_INDEX."
            )

        legal_indices.append(
            ACTION_TO_INDEX[
                move_uci
            ]
        )

    if not legal_indices:

        raise RuntimeError(
            f"No legal moves for position:\n"
            f"{board.fen()}"
        )

    return legal_indices


# ============================================================
# Evaluate one position
# ============================================================

@torch.no_grad()
def evaluate_position(
    model,
    fen,
):

    model.eval()

    board = chess.variant.AtomicBoard(
        fen
    )

    boards = encode_boards(
        [board]
    ).to(DEVICE)

    logits, values = model(
        boards
    )

    logits = logits[0]

    legal_indices = get_legal_indices(
        board
    )

    legal_logits = logits[
        legal_indices
    ]

    log_probs = F.log_softmax(
        legal_logits,
        dim=0,
    )

    probs = torch.exp(
        log_probs
    )

    value = float(
        values[
            0,
            0
        ].item()
    )

    return {
        "legal_indices":
            legal_indices,

        "probs":
            probs.cpu(),

        "log_probs":
            log_probs.cpu(),

        "value":
            value,
    }


# ============================================================
# Oracle loss for one annotation
# ============================================================

def compute_single_oracle_loss(
    model,
    annotation,
):

    board = chess.variant.AtomicBoard(
        annotation["fen"]
    )

    boards = encode_boards(
        [board]
    ).to(DEVICE)

    logits, values = model(
        boards
    )

    # --------------------------------------------------------
    # Legal mask
    # --------------------------------------------------------

    legal_indices = get_legal_indices(
        board
    )

    legal_mask = torch.zeros(
        logits.shape[1],
        dtype=torch.bool,
        device=DEVICE,
    )

    legal_mask[
        legal_indices
    ] = True

    masked_logits = logits.clone()

    masked_logits[
        0
    ] = masked_logits[
        0
    ].masked_fill(
        ~legal_mask,
        float("-inf"),
    )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    oracle_action = (
        ACTION_TO_INDEX[
            annotation["oracle_move"]
        ]
    )

    log_probs = F.log_softmax(
        masked_logits,
        dim=1,
    )

    policy_loss = (
        -log_probs[
            0,
            oracle_action
        ]
    )

    # --------------------------------------------------------
    # Value
    # --------------------------------------------------------

    reward = torch.tensor(
        annotation["reward"],
        dtype=torch.float32,
        device=DEVICE,
    )

    predicted_value = (
        values[
            0,
            0
        ]
    )

    value_loss = F.mse_loss(
        predicted_value,
        reward,
    )

    # --------------------------------------------------------
    # Annotation weighting
    # --------------------------------------------------------

    confidence_weight = (
        CONFIDENCE_WEIGHTS[
            annotation["confidence"]
        ]
    )

    criticality_weight = (
        CRITICALITY_WEIGHTS[
            annotation["criticality"]
        ]
    )

    weight = (
        confidence_weight
        *
        criticality_weight
    )

    weighted_policy_loss = (
        weight
        *
        policy_loss
    )

    weighted_value_loss = (
        weight
        *
        value_loss
    )

    total_loss = (
        ORACLE_POLICY_COEF * policy_loss
        +
        ORACLE_VALUE_COEF * value_loss
    )

    return {
        "loss":
            total_loss,

        "policy_loss":
            policy_loss,

        "value_loss":
            value_loss,

        "weight":
            weight,
    }


# ============================================================
# KL divergence
# ============================================================

def compute_kl(
    before,
    after,
):

    if (
        before["legal_indices"]
        !=
        after["legal_indices"]
    ):

        raise RuntimeError(
            "Legal action space changed "
            "between evaluations."
        )

    p = before[
        "probs"
    ]

    log_p = before[
        "log_probs"
    ]

    log_q = after[
        "log_probs"
    ]

    kl = torch.sum(
        p
        *
        (
            log_p
            -
            log_q
        )
    )

    return float(
        kl.item()
    )


# ============================================================
# Measure one annotation
# ============================================================

def measure_annotation(
    model,
    optimizer,
    annotation,
):

    # --------------------------------------------------------
    # Before
    # --------------------------------------------------------

    before = evaluate_position(
        model,
        annotation["fen"],
    )

    # --------------------------------------------------------
    # Controlled Oracle update
    # --------------------------------------------------------

    model.train()

    loss_info = None

    for _ in range(
        N_UPDATE_STEPS
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_info = (
            compute_single_oracle_loss(
                model,
                annotation,
            )
        )

        loss_info[
            "loss"
        ].backward()

        optimizer.step()

    # --------------------------------------------------------
    # After
    # --------------------------------------------------------

    after = evaluate_position(
        model,
        annotation["fen"],
    )

    # --------------------------------------------------------
    # Responses
    # --------------------------------------------------------

    delta_kl = compute_kl(
        before,
        after,
    )

    delta_v_signed = (
        after["value"]
        -
        before["value"]
    )

    delta_v = abs(
        delta_v_signed
    )

    return {
        "delta_kl":
            delta_kl,

        "delta_v":
            delta_v,

        "delta_v_signed":
            delta_v_signed,

        "value_before":
            before["value"],

        "value_after":
            after["value"],

        "oracle_loss":
            float(
                loss_info[
                    "loss"
                ].detach().item()
            ),

        "oracle_policy_loss":
            float(
                loss_info[
                    "policy_loss"
                ].detach().item()
            ),

        "oracle_value_loss":
            float(
                loss_info[
                    "value_loss"
                ].detach().item()
            ),

        "oracle_weight":
            float(
                loss_info[
                    "weight"
                ]
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "ALBERTA - LOCAL ORACLE RESPONSE"
    )
    print("=" * 70)

    print()

    print(
        f"Device            : {DEVICE}"
    )

    print(
        f"Checkpoint        : {CHECKPOINT_PATH}"
    )

    print(
        f"Oracle queue      : {ORACLE_QUEUE_PATH}"
    )

    print(
        f"Update steps      : {N_UPDATE_STEPS}"
    )

    print(
        f"Learning rate     : {rl.LR}"
    )

    print(
        f"Policy coef       : {ORACLE_POLICY_COEF}"
    )

    print(
        f"Value coef        : {ORACLE_VALUE_COEF}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    annotations = load_annotations()

    print()

    print(
        f"Answered annotations: "
        f"{len(annotations)}"
    )

    # --------------------------------------------------------
    # Frozen RL10 starting state
    # --------------------------------------------------------

    checkpoint = load_base_state()

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

    # --------------------------------------------------------
    # Model + optimizer
    # --------------------------------------------------------

    model = create_model()

    optimizer = Adam(
        model.parameters(),
        lr=rl.LR,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # --------------------------------------------------------
    # Probe loop
    # --------------------------------------------------------

    for i, annotation in enumerate(
        annotations,
        start=1,
    ):

        # ====================================================
        # Exact reset
        # ====================================================

        model.load_state_dict(
            base_model_state
        )

        optimizer.load_state_dict(
            base_optimizer_state
        )

        # ====================================================
        # Measure
        # ====================================================

        response = measure_annotation(
            model,
            optimizer,
            annotation,
        )

        result = {

            "query_id":
                annotation[
                    "query_id"
                ],

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

            "score":
                annotation[
                    "score"
                ],

            "I_norm":
                annotation[
                    "I_norm"
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
                    "criticality"
                ],

            "reward":
                annotation[
                    "reward"
                ],

            **response,
        }

        results.append(
            result
        )

        print(
            f"[{i:03d}/{len(annotations):03d}] "
            f"I={annotation['I_norm']:.4f} "
            f"| KL={response['delta_kl']:.6e} "
            f"| dV={response['delta_v']:.6e} "
            f"| V "
            f"{response['value_before']:+.4f}"
            f" -> "
            f"{response['value_after']:+.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result
                )
                +
                "\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    delta_kl = np.array(
        [
            x["delta_kl"]
            for x in results
        ],
        dtype=np.float64,
    )

    delta_v = np.array(
        [
            x["delta_v"]
            for x in results
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 70)
    print(
        "LOCAL RESPONSE COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        f"Positions : "
        f"{len(results)}"
    )

    print()

    print(
        "Delta KL"
    )

    print(
        f"  mean   : "
        f"{delta_kl.mean():.6e}"
    )

    print(
        f"  median : "
        f"{np.median(delta_kl):.6e}"
    )

    print(
        f"  min    : "
        f"{delta_kl.min():.6e}"
    )

    print(
        f"  max    : "
        f"{delta_kl.max():.6e}"
    )

    print()

    print(
        "Delta V"
    )

    print(
        f"  mean   : "
        f"{delta_v.mean():.6e}"
    )

    print(
        f"  median : "
        f"{np.median(delta_v):.6e}"
    )

    print(
        f"  min    : "
        f"{delta_v.min():.6e}"
    )

    print(
        f"  max    : "
        f"{delta_v.max():.6e}"
    )

    print()

    print(
        f"Saved to:\n"
        f"  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()