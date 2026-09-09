# ============================================================
# local_annotation_response.py
# ============================================================
#
# Mesure la réponse locale du modèle RL10 à une annotation Oracle
# individuelle, sans entraîner un nouvel agent RL.
#
# Pour chaque annotation :
#
#     theta_RL10
#         |
#         +--> Oracle update(s)
#                 |
#                 +--> delta P(oracle)
#                 +--> delta KL
#                 +--> delta V
#
# IMPORTANT :
# Chaque annotation repart exactement du même RL10.
#
# La loss Oracle est STRICTEMENT celle de train_al.py :
#
#     L = 0.05 * L_policy + 0.5 * L_value
#
# avec :
#
#     L_policy = weighted NLL de l'oracle_move
#     L_value  = weighted MSE(V(s), reward)
#
# Les poids sont :
#
#     weight = confidence_weight * criticality_weight
#
# ============================================================


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import copy
import json
import math

import chess
import chess.variant

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch.optim import Adam

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.encoding import encode_boards
from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path("/Users/tom/Desktop/Atomic")

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RL checkpoint
# ============================================================

START_EPOCH = 10


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / f"rl_epoch_{START_EPOCH}.pt"
)


# ============================================================
# Oracle queue
# ============================================================

ORACLE_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_middle.jsonl"
)


# ============================================================
# Oracle loss
# ============================================================

ORACLE_POLICY_COEF = 0.05

ORACLE_VALUE_COEF = 0.5


# ============================================================
# Micro-update
# ============================================================
#
# On reproduit ici une seule optimisation Oracle locale.
#
# Ce n'est PAS un nouvel entraînement RL.
#
# L'objectif est uniquement de mesurer :
#
#     "quelle réponse cette position provoque-t-elle
#      chez le modèle actuel ?"
#
# ============================================================

MICRO_LR = 1e-4

GRAD_CLIP = 1.0


# ============================================================
# Oracle supervision weights
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
# Outputs
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "annotation_response"
)


OUTPUT_CSV = (
    OUTPUT_DIR
    / "local_annotation_response.csv"
)


OUTPUT_REPORT = (
    OUTPUT_DIR
    / "local_annotation_response_report.txt"
)


# ============================================================
# Utility
# ============================================================

def zero_connected_loss(model):
    """
    Return zero connected to the model computation graph.
    """

    return sum(
        (
            parameter.sum()
            * 0.0
        )
        for parameter
        in model.parameters()
    )


# ============================================================
# Load Oracle queue
# ============================================================

def load_oracle_queue(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Oracle queue not found:\n{path}"
        )

    annotations = []

    with open(
        path,
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

            record = json.loads(line)

            # ------------------------------------------------
            # Ignore unanswered entries
            # ------------------------------------------------

            if (
                "oracle_move" not in record
                or "reward" not in record
            ):
                continue

            fen = record["fen"]

            oracle_move = record[
                "oracle_move"
            ]

            reward = float(
                record["reward"]
            )

            confidence = record.get(
                "confidence",
                "medium",
            )

            criticality = record.get(
                "criticality",
                "non_critical",
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if reward not in (
                -1.0,
                0.0,
                1.0,
            ):

                raise ValueError(
                    f"Invalid reward at line "
                    f"{line_number}: {reward}"
                )

            if confidence not in (
                CONFIDENCE_WEIGHTS
            ):

                raise ValueError(
                    f"Invalid confidence at line "
                    f"{line_number}: {confidence}"
                )

            if criticality not in (
                CRITICALITY_WEIGHTS
            ):

                raise ValueError(
                    f"Invalid criticality at line "
                    f"{line_number}: {criticality}"
                )

            if oracle_move not in (
                ACTION_TO_INDEX
            ):

                raise ValueError(
                    f"Unknown Oracle move at line "
                    f"{line_number}: "
                    f"{oracle_move}"
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

                    "criticality":
                        criticality,

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
                }
            )

    if not annotations:

        raise RuntimeError(
            "No answered Oracle annotations found."
        )

    print()
    print(
        "======================================"
    )

    print(
        "ORACLE QUEUE"
    )

    print(
        "======================================"
    )

    print(
        f"Path:        {path}"
    )

    print(
        f"Annotations: {len(annotations)}"
    )

    reward_counts = {
        -1.0: 0,
        0.0: 0,
        1.0: 0,
    }

    for record in annotations:

        reward_counts[
            record["reward"]
        ] += 1

    print(
        f"Losses:      "
        f"{reward_counts[-1.0]}"
    )

    print(
        f"Draws:       "
        f"{reward_counts[0.0]}"
    )

    print(
        f"Wins:        "
        f"{reward_counts[1.0]}"
    )

    print(
        "======================================"
    )

    return annotations


# ============================================================
# Load RL10
# ============================================================

def load_rl_model():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"RL checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    print()
    print(
        "======================================"
    )

    print(
        "RL STARTING CHECKPOINT"
    )

    print(
        "======================================"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # EXACT model architecture from train_al.py
    # --------------------------------------------------------

    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        base_model
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Loaded RL epoch "
        f"{checkpoint.get('epoch', '?')}."
    )

    return model


# ============================================================
# Build single-item Oracle batch
# ============================================================

def prepare_oracle_record(record):

    # --------------------------------------------------------
    # Atomic board
    # --------------------------------------------------------

    board = chess.variant.AtomicBoard(
        record["fen"]
    )

    # --------------------------------------------------------
    # Oracle action
    # --------------------------------------------------------

    oracle_action = ACTION_TO_INDEX[
        record["oracle_move"]
    ]

    # --------------------------------------------------------
    # Supervision weight
    # --------------------------------------------------------

    confidence = record.get(
        "confidence",
        "medium",
    )

    criticality = record.get(
        "criticality",
        "non_critical",
    )

    confidence_weight = (
        CONFIDENCE_WEIGHTS[
            confidence
        ]
    )

    criticality_weight = (
        CRITICALITY_WEIGHTS[
            criticality
        ]
    )

    weight = (
        confidence_weight
        * criticality_weight
    )

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    reward = float(
        record["reward"]
    )

    return (
        board,
        oracle_action,
        weight,
        reward,
    )


# ============================================================
# Forward pass with exact legal masking
# ============================================================

def oracle_forward(
    model,
    record,
):
    """
    Forward pass reproducing the relevant part of
    compute_oracle_loss() from train_al.py.
    """

    (
        board,
        oracle_action,
        weight,
        reward,
    ) = prepare_oracle_record(
        record
    )

    # --------------------------------------------------------
    # Encode board
    # --------------------------------------------------------

    boards = encode_boards(
        [board]
    ).to(DEVICE)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, values = model(
        boards
    )

    # --------------------------------------------------------
    # Legal move masking
    # --------------------------------------------------------

    masked_logits = logits.clone()

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

    legal_mask = torch.zeros(
        logits.shape[1],
        dtype=torch.bool,
        device=DEVICE,
    )

    legal_mask[
        legal_indices
    ] = True

    masked_logits[0] = (
        masked_logits[0]
        .masked_fill(
            ~legal_mask,
            float("-inf"),
        )
    )

    # --------------------------------------------------------
    # Log probabilities
    # --------------------------------------------------------

    log_probs = F.log_softmax(
        masked_logits,
        dim=1,
    )

    oracle_log_prob = (
        log_probs[
            0,
            oracle_action,
        ]
    )

    oracle_probability = (
        oracle_log_prob.exp()
    )

    # --------------------------------------------------------
    # Policy loss
    # --------------------------------------------------------

    policy_loss = (
        -oracle_log_prob
    )

    # --------------------------------------------------------
    # Value
    # --------------------------------------------------------

    predicted_value = (
        values[
            0,
            0,
        ]
    )

    reward_tensor = torch.tensor(
        reward,
        dtype=torch.float32,
        device=DEVICE,
    )

    value_loss = F.mse_loss(
        predicted_value,
        reward_tensor,
        reduction="mean",
    )

    # --------------------------------------------------------
    # Exact single-sample Oracle loss
    #
    # The weight cancels for a single sample in train_al.py:
    #
    #   weighted_loss / weights.sum()
    #
    # = loss
    #
    # Therefore confidence and criticality do not alter the
    # numerical loss for a batch containing exactly one item.
    #
    # --------------------------------------------------------

    oracle_loss = (
        ORACLE_POLICY_COEF
        * policy_loss
        +
        ORACLE_VALUE_COEF
        * value_loss
    )

    return {
        "board":
            board,

        "oracle_action":
            oracle_action,

        "weight":
            weight,

        "reward":
            reward,

        "logits":
            logits,

        "masked_logits":
            masked_logits,

        "log_probs":
            log_probs,

        "oracle_log_prob":
            oracle_log_prob,

        "oracle_probability":
            oracle_probability,

        "policy_loss":
            policy_loss,

        "predicted_value":
            predicted_value,

        "value_loss":
            value_loss,

        "oracle_loss":
            oracle_loss,
    }


# ============================================================
# Policy KL
# ============================================================

# ============================================================
# Policy KL
# ============================================================

def policy_kl_before_after(
    before,
    after,
):
    """
    KL(pi_before || pi_after)

    The distributions are evaluated only over legal actions.

    Illegal actions have -inf log-probabilities because of the
    legal-action masking. They must therefore be excluded from
    the KL computation to avoid 0 * (-inf) = NaN.
    """

    before_log_probs = (
        before["log_probs"]
    )

    after_log_probs = (
        after["log_probs"]
    )

    # --------------------------------------------------------
    # Legal actions
    # --------------------------------------------------------
    #
    # masked_logits contains finite values for legal actions
    # and -inf for illegal actions.
    #
    # The same board is used before and after, so the legal
    # action support is identical.
    # --------------------------------------------------------

    legal_mask = torch.isfinite(
        before["masked_logits"][0]
    )

    before_log_probs = (
        before_log_probs[0, legal_mask]
    )

    after_log_probs = (
        after_log_probs[0, legal_mask]
    )

    # --------------------------------------------------------
    # Numerical sanity checks
    # --------------------------------------------------------

    if not torch.isfinite(
        before_log_probs
    ).all():

        raise RuntimeError(
            "Non-finite log-probabilities "
            "in BEFORE legal-action distribution."
        )

    if not torch.isfinite(
        after_log_probs
    ).all():

        raise RuntimeError(
            "Non-finite log-probabilities "
            "in AFTER legal-action distribution."
        )

    # --------------------------------------------------------
    # KL(pi_before || pi_after)
    # --------------------------------------------------------
    #
    # p * (log p - log q)
    #
    # exp(log p) is numerically preferable to recomputing
    # probabilities from the full masked distribution.
    # --------------------------------------------------------

    before_probs = (
        before_log_probs.exp()
    )

    kl_terms = (
        before_probs
        * (
            before_log_probs
            - after_log_probs
        )
    )

    kl = kl_terms.sum()

    if not torch.isfinite(kl):

        raise RuntimeError(
            "Non-finite KL divergence."
        )

    return float(
        kl.item()
    )


# ============================================================
# Parameter update norm
# ============================================================

def parameter_update_norm(
    before_state,
    after_model,
):
    """
    L2 norm of theta_after - theta_before.
    """

    squared_sum = 0.0

    for (
        name,
        parameter,
    ) in after_model.named_parameters():

        if name not in before_state:
            continue

        before = before_state[
            name
        ].to(
            parameter.device
        )

        delta = (
            parameter.detach()
            - before
        )

        squared_sum += (
            delta
            .pow(2)
            .sum()
            .item()
        )

    return math.sqrt(
        squared_sum
    )


# ============================================================
# Analyze one annotation
# ============================================================

def analyze_annotation(
    base_model,
    record,
    index,
):
    """
    Analyze one annotation independently.

    Every call starts from the exact same RL10 state.
    """

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    model = copy.deepcopy(
        base_model
    ).to(DEVICE)

    model.train()

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    with torch.no_grad():

        before = oracle_forward(
            model,
            record,
        )

    # --------------------------------------------------------
    # Save exact starting parameters
    # --------------------------------------------------------

    before_state = {
        name:
            parameter.detach()
            .clone()
        for (
            name,
            parameter
        ) in model.named_parameters()
    }

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=MICRO_LR,
    )

    # --------------------------------------------------------
    # Oracle update
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_data = oracle_forward(
        model,
        record,
    )

    loss = (
        loss_data["oracle_loss"]
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    optimizer.step()

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        after = oracle_forward(
            model,
            record,
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    oracle_probability_before = float(
        before[
            "oracle_probability"
        ].item()
    )

    oracle_probability_after = float(
        after[
            "oracle_probability"
        ].item()
    )

    surprise_before = float(
        -before[
            "oracle_log_prob"
        ].item()
    )

    surprise_after = float(
        -after[
            "oracle_log_prob"
        ].item()
    )

    delta_probability = (
        oracle_probability_after
        - oracle_probability_before
    )

    delta_surprise = (
        surprise_after
        - surprise_before
    )

    value_before = float(
        before[
            "predicted_value"
        ].item()
    )

    value_after = float(
        after[
            "predicted_value"
        ].item()
    )

    delta_value = (
        value_after
        - value_before
    )

    kl = policy_kl_before_after(
        before,
        after,
    )

    update_norm = (
        parameter_update_norm(
            before_state,
            model,
        )
    )

    return {
        "index":
            index,

        "query_id":
            record.get(
                "query_id"
            ),

        "fen":
            record["fen"],

        "oracle_move":
            record["oracle_move"],

        "confidence":
            record["confidence"],

        "criticality":
            record["criticality"],

        "reward":
            record["reward"],

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

        "supervision_weight":
            before["weight"],

        "oracle_probability_before":
            oracle_probability_before,

        "oracle_probability_after":
            oracle_probability_after,

        "delta_oracle_probability":
            delta_probability,

        "surprise_before":
            surprise_before,

        "surprise_after":
            surprise_after,

        "delta_surprise":
            delta_surprise,

        "value_before":
            value_before,

        "value_after":
            value_after,

        "delta_value":
            delta_value,

        "policy_loss_before":
            float(
                before[
                    "policy_loss"
                ].item()
            ),

        "value_loss_before":
            float(
                before[
                    "value_loss"
                ].item()
            ),

        "oracle_loss_before":
            float(
                before[
                    "oracle_loss"
                ].item()
            ),

        "policy_loss_after":
            float(
                after[
                    "policy_loss"
                ].item()
            ),

        "value_loss_after":
            float(
                after[
                    "value_loss"
                ].item()
            ),

        "oracle_loss_after":
            float(
                after[
                    "oracle_loss"
                ].item()
            ),

        "delta_kl":
            kl,

        "parameter_update_norm":
            update_norm,
    }


# ============================================================
# Correlation helper
# ============================================================

def correlation_table(
    df,
    target,
):
    """
    Pearson + Spearman correlations between H/U/HU
    and a response variable.
    """

    rows = []

    for signal in [
        "H",
        "U",
        "HU",
    ]:

        subset = df[
            [
                signal,
                target,
            ]
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

        if len(subset) < 3:

            rows.append(
                {
                    "signal":
                        signal,

                    "target":
                        target,

                    "n":
                        len(subset),

                    "pearson":
                        np.nan,

                    "spearman":
                        np.nan,
                }
            )

            continue

        pearson = (
            subset[
                signal
            ].corr(
                subset[
                    target
                ],
                method="pearson",
            )
        )

        spearman = (
            subset[
                signal
            ].corr(
                subset[
                    target
                ],
                method="spearman",
            )
        )

        rows.append(
            {
                "signal":
                    signal,

                "target":
                    target,

                "n":
                    len(subset),

                "pearson":
                    float(pearson),

                "spearman":
                    float(spearman),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Report
# ============================================================

def build_report(
    df,
    correlation_tables,
):
    lines = []

    def add(text=""):
        lines.append(
            str(text)
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
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    add(
        f"Oracle queue: "
        f"{ORACLE_QUEUE_PATH}"
    )

    add(
        f"Annotations analysed: "
        f"{len(df)}"
    )

    add()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    add(
        "CONFIGURATION"
    )

    add(
        "-" * 70
    )

    add(
        f"Micro LR: "
        f"{MICRO_LR}"
    )

    add(
        f"Gradient clipping: "
        f"{GRAD_CLIP}"
    )

    add(
        f"Oracle policy coefficient: "
        f"{ORACLE_POLICY_COEF}"
    )

    add(
        f"Oracle value coefficient: "
        f"{ORACLE_VALUE_COEF}"
    )

    add()

    # --------------------------------------------------------
    # Response statistics
    # --------------------------------------------------------

    add(
        "LOCAL RESPONSE"
    )

    add(
        "-" * 70
    )

    for column in [
        "delta_oracle_probability",
        "delta_surprise",
        "delta_value",
        "delta_kl",
        "parameter_update_norm",
    ]:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            continue

        add(
            f"{column}:"
        )

        add(
            f"  mean   = "
            f"{series.mean():+.8f}"
        )

        add(
            f"  median = "
            f"{series.median():+.8f}"
        )

        add(
            f"  std    = "
            f"{series.std():.8f}"
        )

        add(
            f"  p90    = "
            f"{series.quantile(.90):+.8f}"
        )

        add(
            f"  p95    = "
            f"{series.quantile(.95):+.8f}"
        )

        add(
            f"  max    = "
            f"{series.max():+.8f}"
        )

        add()

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    add(
        "CORRELATIONS"
    )

    add(
        "-" * 70
    )

    for target, table in (
        correlation_tables.items()
    ):

        add(
            f"Target: {target}"
        )

        add(
            table.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

        add()

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    add(
        "TOP RESPONSE POSITIONS"
    )

    add(
        "-" * 70
    )

    for target in [
        "delta_kl",
        "delta_oracle_probability",
        "delta_value",
    ]:

        add(
            f"Top 20 by {target}:"
        )

        columns = [
            "index",
            "H",
            "U",
            "HU",
            "reward",
            "confidence",
            "criticality",
            "oracle_probability_before",
            "delta_oracle_probability",
            "delta_value",
            "delta_kl",
        ]

        existing = [
            c
            for c in columns
            if c in df.columns
        ]

        add(
            df.sort_values(
                target,
                ascending=False,
            )
            .head(20)[existing]
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

        add()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    add(
        "INTERPRETATION"
    )

    add(
        "-" * 70
    )

    add(
        "This experiment measures the local response "
        "of the RL10 policy/value model to individual "
        "Oracle annotations."
    )

    add(
        "Each annotation is evaluated independently "
        "from the same RL10 starting parameters."
    )

    add(
        "The experiment therefore does not measure "
        "downstream RL performance."
    )

    add(
        "It measures whether the immediate susceptibility "
        "of the model to an annotation varies across "
        "positions."
    )

    add(
        "A strong correlation between H/U/HU and local "
        "response would indicate that the uncertainty "
        "signals may predict immediate annotation "
        "sensitivity."
    )

    add(
        "A weak correlation would indicate that the "
        "uncertainty signals do not explain much of the "
        "heterogeneity in local model response."
    )

    add()

    add(
        "IMPORTANT LIMITATION"
    )

    add(
        "The micro-update is an operational diagnostic "
        "and is not identical to a complete PPO + Oracle "
        "training step."
    )

    add(
        "The result should therefore be interpreted as "
        "evidence about local model response, not as a "
        "direct estimate of downstream annotation value."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "============================================================"
    )

    print(
        "ALBERTA - LOCAL ORACLE ANNOTATION RESPONSE"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Determinism
    # --------------------------------------------------------

    torch.manual_seed(0)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            0
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    base_model = load_rl_model()

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    annotations = load_oracle_queue(
        ORACLE_QUEUE_PATH
    )

    # --------------------------------------------------------
    # Analyze annotations
    # --------------------------------------------------------

    results = []

    total = len(
        annotations
    )

    print()
    print(
        "ANALYSING ANNOTATIONS"
    )

    print(
        "======================================"
    )

    for index, record in enumerate(
        annotations,
        start=1,
    ):

        result = analyze_annotation(
            base_model,
            record,
            index,
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
                f"[{index:>3}/{total}] "
                f"completed",
                flush=True,
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(
        f"CSV saved:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_tables = {}

    for target in [
        "delta_kl",
        "delta_oracle_probability",
        "delta_value",
        "parameter_update_norm",
    ]:

        correlation_tables[
            target
        ] = correlation_table(
            df,
            target,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        df,
        correlation_tables,
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            report
        )

    print()
    print(
        f"Report saved:"
    )

    print(
        f"  {OUTPUT_REPORT}"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )

    print(
        "SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        f"Annotations: "
        f"{len(df)}"
    )

    for target in [
        "delta_kl",
        "delta_oracle_probability",
        "delta_value",
    ]:

        series = pd.to_numeric(
            df[target],
            errors="coerce",
        ).dropna()

        if len(series):

            print()
            print(
                target
            )

            print(
                f"  mean:   "
                f"{series.mean():+.8f}"
            )

            print(
                f"  median: "
                f"{series.median():+.8f}"
            )

            print(
                f"  std:    "
                f"{series.std():.8f}"
            )

    print()
    print(
        "============================================================"
    )

    print(
        "DONE"
    )

    print(
        "============================================================"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()