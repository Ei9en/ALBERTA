#!/usr/bin/env python3

"""
ALBERTA - Dynamic Local Oracle Response
=======================================

For each manually annotated Pareto calibration probe:

    1. load the current learner checkpoint
    2. load the current historical league
    3. load the BC opening-prior model
    4. recompute dynamically:

           H_t
           U_t
           HU_t = H_t * U_t

    5. restore the exact current learner state
    6. restore the exact current optimizer state
    7. measure policy/value before Oracle update
    8. apply one controlled Oracle gradient step
    9. measure policy/value after Oracle update
   10. compute:

           Delta_KL_t
           Delta_V_t

   11. restore the learner before the next annotation

The resulting calibration dataset maps:

    (H_t, U_t, HU_t)
        ->
    (Delta_KL_t, Delta_V_t)

The 200 FENs and their human Oracle annotations stay fixed.
The learner-dependent quantities are refreshed at every
dynamic acquisition generation.

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
# Dynamic experiment configuration
#
# Change CURRENT_EPOCH / CHECKPOINT_PATH for each dynamic step.
# ============================================================

CURRENT_EPOCH = 10


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
    / "dynamic_pareto_epoch"
    / f"al_epoch_{CURRENT_EPOCH}.pt"
)

LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "dynamic_pareto_league"
)

# ------------------------------------------------------------
# BC checkpoints
#
# BC6 + BC7 are persistent historical league anchors.
#
# BC7 is also used as the opening policy prior.
# ------------------------------------------------------------

BC6_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_epoch_6.pt"
)

BC7_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_epoch_7.pt"
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
# League configuration
#
# ALBERTA keeps:
#
#     BC6
#     BC7
#     last 10 historical learner snapshots
#
# ============================================================

LEAGUE_HISTORY = 10


# ============================================================
# Opening prior configuration
#
# Try to reuse training values when available.
#
# Historical ALBERTA configuration:
#
#   plies = 6
#
# Default strength = 1.0 if not exposed by training module.
# ============================================================

OPENING_PRIOR_PLIES = getattr(
    rl,
    "OPENING_PRIOR_PLIES",
    6,
)

OPENING_PRIOR_STRENGTH = getattr(
    rl,
    "OPENING_PRIOR_STRENGTH",
    1.0,
)


# ============================================================
# Oracle loss coefficients
#
# Must match AL training.
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
# Build ActorCritic model
# ============================================================

def create_model():

    base_model = ChessResNet(
        num_actions=len(
            ACTIONS
        ),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        base_model
    ).to(
        DEVICE
    )

    return model


# ============================================================
# Load generic ActorCritic checkpoint
# ============================================================

def load_actor_critic_from_path(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            f"No model_state_dict in:\n"
            f"{path}"
        )

    model = create_model()

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# ============================================================
# Load current learner checkpoint
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
# Load BC opening-prior model
# ============================================================

def load_bc_model():

    print()
    print(
        "Loading BC opening-prior model..."
    )

    print(
        f"  {BC7_CHECKPOINT_PATH}"
    )

    model = load_actor_critic_from_path(
        BC7_CHECKPOINT_PATH
    )

    return model


# ============================================================
# Load current historical league
#
# Composition:
#
#     BC6
#     BC7
#     up to last 10 historical learner snapshots
#
# Current learner itself is NOT included here because it is
# explicitly added during uncertainty computation.
# ============================================================

def load_current_league():

    print()
    print("=" * 70)
    print("LOADING HISTORICAL LEAGUE")
    print("=" * 70)
    print()

    models = []

    # ========================================================
    # Persistent BC anchors
    # ========================================================

    for name, path in (
        (
            "BC6",
            BC6_CHECKPOINT_PATH,
        ),
        (
            "BC7",
            BC7_CHECKPOINT_PATH,
        ),
    ):

        model = load_actor_critic_from_path(
            path
        )

        models.append(
            model
        )

        print(
            f"Loaded {name}"
        )


    # ========================================================
    # Historical learner snapshots
    # ========================================================

    start_epoch = max(
        1,
        CURRENT_EPOCH
        -
        LEAGUE_HISTORY
        +
        1,
    )

    loaded_snapshots = 0

    for epoch in range(
        start_epoch,
        CURRENT_EPOCH + 1,
    ):

        path = (
            LEAGUE_DIR
            /
            f"league_epoch_{epoch:03d}.pt"
        )

        if not path.exists():

            print(
                f"WARNING: missing league snapshot: "
                f"{path}"
            )

            continue

        model = load_actor_critic_from_path(
            path
        )

        models.append(
            model
        )

        loaded_snapshots += 1

        print(
            f"Loaded league_epoch_{epoch:03d}"
        )


    print()

    print(
        f"Persistent BC agents : 2"
    )

    print(
        f"Historical snapshots : "
        f"{loaded_snapshots}"
    )

    print(
        f"League models        : "
        f"{len(models)}"
    )

    return models


# ============================================================
# Load human probe annotations
#
# H/U/HU stored in the historical queue are intentionally
# ignored. They are recomputed dynamically later.
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

            # ------------------------------------------------
            # Answered probes only
            # ------------------------------------------------

            if record.get(
                "status"
            ) != "answered":

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


            # ------------------------------------------------
            # Annotation metadata
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Oracle action
            # ------------------------------------------------

            if oracle_move not in ACTION_TO_INDEX:

                raise ValueError(
                    f"Unknown oracle move at line "
                    f"{line_number}: "
                    f"{oracle_move}"
                )


            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

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


            # ------------------------------------------------
            # IMPORTANT
            #
            # Historical H/U/HU are NOT copied.
            #
            # Only fixed human information is retained.
            # ------------------------------------------------

            annotations.append(
                {
                    "query_id":
                        record[
                            "query_id"
                        ],

                    "fen":
                        record[
                            "fen"
                        ],

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
# Legal action indices
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
# Opening-prior coefficient
#
# Same linear schedule used during self-play.
# ============================================================

def opening_prior_alpha(
    board,
):

    ply = board.ply()

    if (
        OPENING_PRIOR_PLIES <= 0
        or
        OPENING_PRIOR_STRENGTH <= 0
        or
        ply >= OPENING_PRIOR_PLIES
    ):

        return 0.0


    return (
        OPENING_PRIOR_STRENGTH
        *
        (
            1.0
            -
            (
                ply
                /
                OPENING_PRIOR_PLIES
            )
        )
    )


# ============================================================
# Compute H
#
# H = entropy of the current guided policy over legal actions.
#
# BC opening prior is added BEFORE entropy.
# Temperature is NOT applied.
# ============================================================

@torch.no_grad()
def compute_entropy(
    current_model,
    bc_model,
    board,
):

    current_model.eval()
    bc_model.eval()

    x = encode_boards(
        [board]
    ).to(
        DEVICE
    )


    # ========================================================
    # Current learner policy
    # ========================================================

    logits, _ = current_model(
        x
    )

    legal_indices = get_legal_indices(
        board
    )

    legal_logits = logits[
        0,
        legal_indices
    ]


    # ========================================================
    # Optional BC opening prior
    # ========================================================

    alpha = opening_prior_alpha(
        board
    )

    if alpha > 0.0:

        bc_logits, _ = bc_model(
            x
        )

        bc_legal_logits = bc_logits[
            0,
            legal_indices
        ]

        bc_log_probs = F.log_softmax(
            bc_legal_logits,
            dim=0,
        )

        legal_logits = (
            legal_logits
            +
            alpha
            *
            bc_log_probs
        )


    # ========================================================
    # Intrinsic policy entropy
    # ========================================================

    log_probs = F.log_softmax(
        legal_logits,
        dim=0,
    )

    probs = torch.exp(
        log_probs
    )

    entropy = -torch.sum(
        probs
        *
        log_probs
    )

    return float(
        entropy.item()
    )


# ============================================================
# Compute U
#
# Variance of value predictions across:
#
#     historical league agents
#     +
#     current learner
#
# Same definition as:
#
#     torch.var(..., unbiased=False)
# ============================================================

@torch.no_grad()
def compute_uncertainty(
    current_model,
    league_models,
    board,
):

    x = encode_boards(
        [board]
    ).to(
        DEVICE
    )

    values = []


    # ========================================================
    # Historical league
    # ========================================================

    for model in league_models:

        model.eval()

        _, value = model(
            x
        )

        values.append(
            float(
                value[
                    0,
                    0
                ].item()
            )
        )


    # ========================================================
    # Current learner
    # ========================================================

    current_model.eval()

    _, current_value = current_model(
        x
    )

    values.append(
        float(
            current_value[
                0,
                0
            ].item()
        )
    )


    # ========================================================
    # Population variance
    # ========================================================

    values_tensor = torch.tensor(
        values,
        dtype=torch.float32,
    )

    uncertainty = torch.var(
        values_tensor,
        unbiased=False,
    )

    return float(
        uncertainty.item()
    )


# ============================================================
# Refresh H / U / HU for all fixed probes
# ============================================================

def refresh_annotation_features(
    annotations,
    current_model,
    bc_model,
    league_models,
):

    print()
    print("=" * 70)
    print("REFRESHING PARETO PROBE FEATURES")
    print("=" * 70)
    print()

    refreshed = []


    for i, annotation in enumerate(
        annotations,
        start=1,
    ):

        board = chess.variant.AtomicBoard(
            annotation[
                "fen"
            ]
        )


        # ====================================================
        # Dynamic features
        # ====================================================

        H = compute_entropy(
            current_model,
            bc_model,
            board,
        )

        U = compute_uncertainty(
            current_model,
            league_models,
            board,
        )

        HU = (
            H
            *
            U
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
        ] = HU


        refreshed.append(
            updated
        )


        print(
            f"[{i:03d}/{len(annotations):03d}] "
            f"H={H:.6f} "
            f"| U={U:.6e} "
            f"| HU={HU:.6e}"
        )


    return refreshed


# ============================================================
# Evaluate policy + value on one position
#
# NOTE:
#
# Delta_KL measures the raw learner policy response.
# The BC opening prior is NOT applied here.
#
# This matches the original local-response experiment:
#
#     learner policy before Oracle update
#         versus
#     learner policy after Oracle update
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
    ).to(
        DEVICE
    )

    logits, values = model(
        boards
    )

    logits = logits[
        0
    ]

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
# Oracle loss for one calibration annotation
# ============================================================

def compute_single_oracle_loss(
    model,
    annotation,
):

    board = chess.variant.AtomicBoard(
        annotation[
            "fen"
        ]
    )

    boards = encode_boards(
        [board]
    ).to(
        DEVICE
    )

    logits, values = model(
        boards
    )


    # ========================================================
    # Legal action mask
    # ========================================================

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


    # ========================================================
    # Policy loss
    # ========================================================

    oracle_action = ACTION_TO_INDEX[
        annotation[
            "oracle_move"
        ]
    ]


    if oracle_action not in legal_indices:

        raise ValueError(
            "Oracle move is not legal for position:\n"
            f"{annotation['fen']}\n"
            f"move={annotation['oracle_move']}"
        )


    log_probs = F.log_softmax(
        masked_logits,
        dim=1,
    )

    policy_loss = -log_probs[
        0,
        oracle_action
    ]


    # ========================================================
    # Value loss
    # ========================================================

    reward = torch.tensor(
        annotation[
            "reward"
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    predicted_value = values[
        0,
        0
    ]

    value_loss = F.mse_loss(
        predicted_value,
        reward,
    )


    # ========================================================
    # Human annotation weighting
    # ========================================================

    confidence_weight = CONFIDENCE_WEIGHTS[
        annotation[
            "confidence"
        ]
    ]

    criticality_weight = CRITICALITY_WEIGHTS[
        annotation[
            "criticality"
        ]
    ]

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


    # ========================================================
    # Total Oracle loss
    #
    # Weight ACTUALLY affects the gradient.
    # ========================================================

    total_loss = (
        ORACLE_POLICY_COEF
        *
        weighted_policy_loss
        +
        ORACLE_VALUE_COEF
        *
        weighted_value_loss
    )


    return {
        "loss":
            total_loss,

        "policy_loss":
            policy_loss,

        "value_loss":
            value_loss,

        "weighted_policy_loss":
            weighted_policy_loss,

        "weighted_value_loss":
            weighted_value_loss,

        "confidence_weight":
            confidence_weight,

        "criticality_weight":
            criticality_weight,

        "weight":
            weight,
    }


# ============================================================
# KL divergence
#
# KL(
#     policy_before
#     ||
#     policy_after
# )
#
# Restricted to legal actions.
# ============================================================

def compute_kl(
    before,
    after,
):

    if (
        before[
            "legal_indices"
        ]
        !=
        after[
            "legal_indices"
        ]
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

    # ========================================================
    # Before Oracle update
    # ========================================================

    before = evaluate_position(
        model,
        annotation[
            "fen"
        ],
    )


    # ========================================================
    # Controlled Oracle update
    # ========================================================

    model.train()

    loss_info = None


    for _ in range(
        N_UPDATE_STEPS
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_info = compute_single_oracle_loss(
            model,
            annotation,
        )

        loss_info[
            "loss"
        ].backward()

        optimizer.step()


    # ========================================================
    # After Oracle update
    # ========================================================

    after = evaluate_position(
        model,
        annotation[
            "fen"
        ],
    )


    # ========================================================
    # Actor response
    # ========================================================

    delta_kl = compute_kl(
        before,
        after,
    )


    # ========================================================
    # Critic response
    # ========================================================

    delta_v_signed = (
        after[
            "value"
        ]
        -
        before[
            "value"
        ]
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
            before[
                "value"
            ],

        "value_after":
            after[
                "value"
            ],

        "oracle_loss":
            float(
                loss_info[
                    "loss"
                ]
                .detach()
                .item()
            ),

        "oracle_policy_loss":
            float(
                loss_info[
                    "policy_loss"
                ]
                .detach()
                .item()
            ),

        "oracle_value_loss":
            float(
                loss_info[
                    "value_loss"
                ]
                .detach()
                .item()
            ),

        "weighted_oracle_policy_loss":
            float(
                loss_info[
                    "weighted_policy_loss"
                ]
                .detach()
                .item()
            ),

        "weighted_oracle_value_loss":
            float(
                loss_info[
                    "weighted_value_loss"
                ]
                .detach()
                .item()
            ),

        "confidence_weight":
            float(
                loss_info[
                    "confidence_weight"
                ]
            ),

        "criticality_weight":
            float(
                loss_info[
                    "criticality_weight"
                ]
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

    print()
    print("=" * 70)
    print("ALBERTA - DYNAMIC LOCAL ORACLE RESPONSE")
    print("=" * 70)
    print()

    print(
        f"Device                 : "
        f"{DEVICE}"
    )

    print(
        f"Current epoch          : "
        f"{CURRENT_EPOCH}"
    )

    print(
        f"Checkpoint             : "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Probe queue            : "
        f"{ORACLE_QUEUE_PATH}"
    )

    print(
        f"League directory       : "
        f"{LEAGUE_DIR}"
    )

    print(
        f"Historical snapshots   : "
        f"{LEAGUE_HISTORY}"
    )

    print(
        f"Opening prior plies    : "
        f"{OPENING_PRIOR_PLIES}"
    )

    print(
        f"Opening prior strength : "
        f"{OPENING_PRIOR_STRENGTH}"
    )

    print(
        f"Update steps           : "
        f"{N_UPDATE_STEPS}"
    )

    print(
        f"Configured RL LR       : "
        f"{rl.LR}"
    )

    print(
        f"Oracle policy coef     : "
        f"{ORACLE_POLICY_COEF}"
    )

    print(
        f"Oracle value coef      : "
        f"{ORACLE_VALUE_COEF}"
    )


    # ========================================================
    # Fixed human probe annotations
    # ========================================================

    annotations = load_annotations()

    print()

    print(
        f"Answered calibration probes: "
        f"{len(annotations)}"
    )


    # ========================================================
    # Current learner checkpoint
    # ========================================================

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


    # ========================================================
    # Current learner used for feature refresh
    # ========================================================

    current_model = create_model()

    current_model.load_state_dict(
        base_model_state
    )

    current_model.eval()


    # ========================================================
    # BC7 opening prior
    # ========================================================

    bc_model = load_bc_model()


    # ========================================================
    # Current historical league
    # ========================================================

    league_models = load_current_league()


    # ========================================================
    # Refresh H / U / HU
    # ========================================================

    annotations = refresh_annotation_features(
        annotations,
        current_model,
        bc_model,
        league_models,
    )


    # ========================================================
    # Model + optimizer used for controlled response probes
    # ========================================================

    model = create_model()

    optimizer = Adam(
        model.parameters(),
        lr=rl.LR,
    )


    # ========================================================
    # Important optimizer diagnostic
    #
    # optimizer.load_state_dict() overwrites optimizer
    # hyperparameters such as LR.
    # ========================================================

    model.load_state_dict(
        base_model_state
    )

    optimizer.load_state_dict(
        base_optimizer_state
    )


    actual_lr = optimizer.param_groups[
        0
    ][
        "lr"
    ]

    print()
    print(
        f"Actual checkpoint optimizer LR: "
        f"{actual_lr}"
    )


    # ========================================================
    # Output
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []


    # ========================================================
    # Local-response probe loop
    # ========================================================

    print()
    print("=" * 70)
    print("MEASURING LOCAL ORACLE RESPONSE")
    print("=" * 70)
    print()


    for (
        i,
        annotation,
    ) in enumerate(
        annotations,
        start=1,
    ):


        # ====================================================
        # Exact learner + optimizer reset
        # ====================================================

        model.load_state_dict(
            base_model_state
        )

        optimizer.load_state_dict(
            base_optimizer_state
        )


        # ====================================================
        # Controlled response measurement
        # ====================================================

        response = measure_annotation(
            model,
            optimizer,
            annotation,
        )


        # ====================================================
        # Final calibration row
        #
        # Dynamic:
        #
        #   H_t
        #   U_t
        #   HU_t
        #   Delta_KL_t
        #   Delta_V_t
        #
        # Fixed:
        #
        #   FEN
        #   Oracle move
        #   reward
        #   confidence
        #   criticality
        # ====================================================

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

            "learner_epoch":
                CURRENT_EPOCH,

            **response,
        }


        results.append(
            result
        )


        print(
            f"[{i:03d}/{len(annotations):03d}] "
            f"H={annotation['H']:.4f} "
            f"| U={annotation['U']:.6e} "
            f"| KL={response['delta_kl']:.6e} "
            f"| dV={response['delta_v']:.6e} "
            f"| w={response['oracle_weight']:.3f} "
            f"| V "
            f"{response['value_before']:+.4f}"
            f" -> "
            f"{response['value_after']:+.4f}"
        )


    # ========================================================
    # Save JSONL
    # ========================================================

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


    # ========================================================
    # Summary arrays
    # ========================================================

    H_values = np.asarray(
        [
            x[
                "H"
            ]
            for x in results
        ],
        dtype=np.float64,
    )

    U_values = np.asarray(
        [
            x[
                "U"
            ]
            for x in results
        ],
        dtype=np.float64,
    )

    HU_values = np.asarray(
        [
            x[
                "HU"
            ]
            for x in results
        ],
        dtype=np.float64,
    )

    delta_kl = np.asarray(
        [
            x[
                "delta_kl"
            ]
            for x in results
        ],
        dtype=np.float64,
    )

    delta_v = np.asarray(
        [
            x[
                "delta_v"
            ]
            for x in results
        ],
        dtype=np.float64,
    )

    weights = np.asarray(
        [
            x[
                "oracle_weight"
            ]
            for x in results
        ],
        dtype=np.float64,
    )


    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("DYNAMIC LOCAL RESPONSE COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Positions : "
        f"{len(results)}"
    )


    print()
    print(
        "Refreshed H"
    )

    print(
        f"  mean   : "
        f"{H_values.mean():.6e}"
    )

    print(
        f"  median : "
        f"{np.median(H_values):.6e}"
    )

    print(
        f"  min    : "
        f"{H_values.min():.6e}"
    )

    print(
        f"  max    : "
        f"{H_values.max():.6e}"
    )


    print()
    print(
        "Refreshed U"
    )

    print(
        f"  mean   : "
        f"{U_values.mean():.6e}"
    )

    print(
        f"  median : "
        f"{np.median(U_values):.6e}"
    )

    print(
        f"  min    : "
        f"{U_values.min():.6e}"
    )

    print(
        f"  max    : "
        f"{U_values.max():.6e}"
    )


    print()
    print(
        "Refreshed HU"
    )

    print(
        f"  mean   : "
        f"{HU_values.mean():.6e}"
    )

    print(
        f"  median : "
        f"{np.median(HU_values):.6e}"
    )

    print(
        f"  min    : "
        f"{HU_values.min():.6e}"
    )

    print(
        f"  max    : "
        f"{HU_values.max():.6e}"
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
        "Oracle weights"
    )

    print(
        f"  mean   : "
        f"{weights.mean():.6f}"
    )

    print(
        f"  median : "
        f"{np.median(weights):.6f}"
    )

    print(
        f"  min    : "
        f"{weights.min():.6f}"
    )

    print(
        f"  max    : "
        f"{weights.max():.6f}"
    )


    print()

    print(
        f"Saved to:\n"
        f"  {OUTPUT_PATH}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()