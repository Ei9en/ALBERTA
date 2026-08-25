# Atomic/tests/sanity_check_I.py

from pathlib import Path
import sys
import json

import chess
import chess.variant
import numpy as np
import torch


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.actions_space import ACTION_TO_INDEX


# ============================================================
# Paths
# ============================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "sanity_check_I.json"
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 10
BOTTOM_K = 10

MIN_LEGAL_MOVES = 2

DEVICE = "cpu"


# ============================================================
# Active Learning weights
# ============================================================

W_H = 0.144
W_U = 0.527
W_HU = 0.329


# ============================================================
# Percentile rank
# ============================================================

def percentile_rank(values):
    """
    Empirical percentile rank in [0, 1].

    Ties receive the same mid-rank.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n < 2:
        raise ValueError(
            "At least two values are required."
        )

    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[order]

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        rank = (
            (start + end - 1)
            / 2.0
        )

        ranks[
            order[start:end]
        ] = rank / (n - 1)

        start = end

    return ranks


# ============================================================
# Side-aware normalization
# ============================================================

def normalize_side_aware(
    values,
    white_mask,
    black_mask,
):

    normalized = np.zeros_like(
        values,
        dtype=np.float64,
    )

    normalized[
        white_mask
    ] = percentile_rank(
        values[
            white_mask
        ]
    )

    normalized[
        black_mask
    ] = percentile_rank(
        values[
            black_mask
        ]
    )

    return normalized


# ============================================================
# Load RL model
# ============================================================

def load_model(checkpoint_path):

    model = ChessResNet(
        num_actions=len(ACTION_TO_INDEX),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        model
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# Policy extraction
# ============================================================

@torch.no_grad()
def get_policy(
    model,
    board,
):

    from src.encoding import encode_board

    # --------------------------------------------------------
    # Encode board
    # --------------------------------------------------------

    x = encode_board(
        board
    )

    x = x.unsqueeze(0).to(DEVICE)

    policy_logits, value = model(x)

    logits = policy_logits[0]


    # --------------------------------------------------------
    # Legal moves
    # --------------------------------------------------------

    legal_moves = list(
        board.legal_moves
    )

    if len(legal_moves) < MIN_LEGAL_MOVES:

        raise ValueError(
            "Position has fewer than "
            f"{MIN_LEGAL_MOVES} legal moves."
        )


    # --------------------------------------------------------
    # Extract legal logits
    # --------------------------------------------------------

    move_entries = []

    legal_indices = []

    for move in legal_moves:

        uci = move.uci()

        action_index = ACTION_TO_INDEX[
            uci
        ]

        legal_indices.append(
            action_index
        )

        move_entries.append(
            (
                uci,
                action_index,
            )
        )


    legal_indices_tensor = torch.tensor(
        legal_indices,
        dtype=torch.long,
        device=DEVICE,
    )

    legal_logits = logits[
        legal_indices_tensor
    ]


    # --------------------------------------------------------
    # Intrinsic policy
    #
    # IMPORTANT:
    # No self-play temperature here.
    # Temperature = 1.
    # --------------------------------------------------------

    log_probs = torch.log_softmax(
        legal_logits,
        dim=0,
    )

    probs = torch.exp(
        log_probs
    )


    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = -(
        probs * log_probs
    ).sum().item()


    # --------------------------------------------------------
    # JSON-friendly policy
    # --------------------------------------------------------

    policy = []

    for (
        (uci, action_index),
        probability,
        log_probability,
    ) in zip(
        move_entries,
        probs.tolist(),
        log_probs.tolist(),
    ):

        policy.append(
            {
                "move": uci,
                "action_index": action_index,
                "probability": probability,
                "log_probability": log_probability,
            }
        )


    # --------------------------------------------------------
    # Highest probability first
    # --------------------------------------------------------

    policy.sort(
        key=lambda x: x["probability"],
        reverse=True,
    )


    top_move = policy[0]


    return {
        "value": value.item(),
        "entropy": entropy,
        "num_legal_moves": len(legal_moves),
        "top_move": top_move,
        "policy": policy,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "ALBERTA - I EXTREME POLICY SANITY CHECK"
    )
    print("=" * 70)


    # ========================================================
    # Load data
    # ========================================================

    print()
    print(
        "Loading uncertainty statistics"
    )
    print("-" * 70)

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    print(
        f"Records : {len(data):,}"
    )


    # ========================================================
    # Extract signals
    # ========================================================

    fens = np.array(
        [
            record["fen"]
            for record in data
        ],
        dtype=object,
    )

    H = np.array(
        [
            record["H"]
            for record in data
        ],
        dtype=np.float64,
    )

    U = np.array(
        [
            record["U"]
            for record in data
        ],
        dtype=np.float64,
    )

    HU = np.array(
        [
            record["HU"]
            for record in data
        ],
        dtype=np.float64,
    )


    # ========================================================
    # Side to move
    # ========================================================

    sides = np.array(
        [
            fen.split()[1]
            for fen in fens
        ],
        dtype="<U1",
    )

    white_mask = (
        sides == "w"
    )

    black_mask = (
        sides == "b"
    )


    # ========================================================
    # Side-aware normalization
    # ========================================================

    print()
    print(
        "Computing side-aware I score"
    )
    print("-" * 70)

    H_norm = normalize_side_aware(
        H,
        white_mask,
        black_mask,
    )

    U_norm = normalize_side_aware(
        U,
        white_mask,
        black_mask,
    )

    HU_norm = normalize_side_aware(
        HU,
        white_mask,
        black_mask,
    )


    # ========================================================
    # Weighted composite
    # ========================================================

    I_raw = (
        W_H * H_norm
        +
        W_U * U_norm
        +
        W_HU * HU_norm
    )


    # ========================================================
    # Final side-aware I
    # ========================================================

    I = np.zeros_like(
        I_raw,
        dtype=np.float64,
    )

    I[white_mask] = percentile_rank(
        I_raw[white_mask]
    )

    I[black_mask] = percentile_rank(
        I_raw[black_mask]
    )


    # ========================================================
    # Efficient extreme selection
    # ========================================================
    #
    # DO NOT count legal moves for all 1.1M positions.
    #
    # We sort I once, then walk inward from each extreme
    # until enough positions with >= 2 legal moves are found.
    # ========================================================

    print()
    print(
        "Selecting extreme positions"
    )
    print("-" * 70)

    sorted_indices = np.argsort(
        I
    )

    bottom_candidates = sorted_indices

    top_candidates = sorted_indices[::-1]


    # --------------------------------------------------------
    # Bottom positions
    # --------------------------------------------------------

    bottom_indices = []

    bottom_checked = 0

    for index in bottom_candidates:

        board = chess.variant.AtomicBoard(
            fens[index]
        )

        legal_count = (
            board.legal_moves.count()
        )

        bottom_checked += 1

        if legal_count >= MIN_LEGAL_MOVES:

            bottom_indices.append(
                index
            )

            if len(bottom_indices) >= BOTTOM_K:

                break


    # --------------------------------------------------------
    # Top positions
    # --------------------------------------------------------

    top_indices = []

    top_checked = 0

    for index in top_candidates:

        board = chess.variant.AtomicBoard(
            fens[index]
        )

        legal_count = (
            board.legal_moves.count()
        )

        top_checked += 1

        if legal_count >= MIN_LEGAL_MOVES:

            top_indices.append(
                index
            )

            if len(top_indices) >= TOP_K:

                break


    # ========================================================
    # Safety checks
    # ========================================================

    if len(bottom_indices) < BOTTOM_K:

        raise RuntimeError(
            "Could not find enough valid "
            "bottom-I positions."
        )

    if len(top_indices) < TOP_K:

        raise RuntimeError(
            "Could not find enough valid "
            "top-I positions."
        )


    # ========================================================
    # Convert to numpy arrays
    # ========================================================

    bottom_indices = np.array(
        bottom_indices,
        dtype=np.int64,
    )

    top_indices = np.array(
        top_indices,
        dtype=np.int64,
    )


    # ========================================================
    # Legal move counts for selected positions only
    # ========================================================

    legal_move_counts = {}

    selected_indices = np.concatenate(
        [
            top_indices,
            bottom_indices,
        ]
    )

    for index in selected_indices:

        board = chess.variant.AtomicBoard(
            fens[index]
        )

        legal_move_counts[
            int(index)
        ] = board.legal_moves.count()


    # ========================================================
    # Diagnostics
    # ========================================================

    print(
        f"Minimum legal moves : "
        f"{MIN_LEGAL_MOVES}"
    )

    print()

    print(
        f"Bottom positions checked : "
        f"{bottom_checked:,}"
    )

    print(
        f"Top positions checked    : "
        f"{top_checked:,}"
    )

    print()

    print(
        f"Bottom positions kept : "
        f"{len(bottom_indices)}"
    )

    print(
        f"Top positions kept    : "
        f"{len(top_indices)}"
    )


    # ========================================================
    # Load RL milestones
    # ========================================================

    print()
    print(
        "Loading RL milestones"
    )
    print("-" * 70)

    checkpoints = {

        "RL10":
            CHECKPOINT_DIR
            / "rl_epoch_10.pt",

        "RL20":
            CHECKPOINT_DIR
            / "rl_epoch_20.pt",

        "RL30":
            CHECKPOINT_DIR
            / "rl_epoch_30.pt",

    }

    models = {}

    for name, path in checkpoints.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing checkpoint: {path}"
            )

        print(
            f"Loading {name}: {path}"
        )

        models[name] = load_model(
            path
        )


    # ========================================================
    # Output structure
    # ========================================================

    result = {

        "metadata": {

            "data_file":
                str(DATA_FILE),

            "num_records":
                len(data),

            "min_legal_moves":
                MIN_LEGAL_MOVES,

            "top_k":
                TOP_K,

            "bottom_k":
                BOTTOM_K,

            "weights": {

                "H": W_H,

                "U": W_U,

                "HU": W_HU,

            },

            "normalization":
                "side-aware empirical percentile rank",

            "final_I":
                "side-aware empirical percentile rank of weighted composite",

            "policy":
                "intrinsic legal-move softmax, temperature=1.0",

        },

        "top_I": [],

        "bottom_I": [],

    }


    # ========================================================
    # Evaluate one position
    # ========================================================

    def evaluate_position(
        index,
        rank,
    ):

        fen = fens[index]

        board = chess.variant.AtomicBoard(
            fen
        )

        entry = {

            "rank":
                rank,

            "index":
                int(index),

            "fen":
                fen,

            "side_to_move":
                (
                    "white"
                    if sides[index] == "w"
                    else "black"
                ),

            "I":
                float(I[index]),

            "I_raw":
                float(I_raw[index]),

            "H":
                float(H[index]),

            "U":
                float(U[index]),

            "HU":
                float(HU[index]),

            "H_normalized":
                float(H_norm[index]),

            "U_normalized":
                float(U_norm[index]),

            "HU_normalized":
                float(HU_norm[index]),

            "legal_moves":
                int(
                    legal_move_counts[
                        int(index)
                    ]
                ),

            "policies":
                {},

        }


        # ----------------------------------------------------
        # Evaluate RL10 / RL20 / RL30
        # ----------------------------------------------------

        for name, model in models.items():

            entry[
                "policies"
            ][name] = get_policy(
                model,
                board,
            )


        return entry


    # ========================================================
    # Top I
    # ========================================================

    print()
    print(
        "Evaluating TOP-I positions"
    )
    print("-" * 70)

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        print(
            f"Top I #{rank:02d} | "
            f"I={I[index]:.9f} | "
            f"legal={legal_move_counts[int(index)]}"
        )

        result[
            "top_I"
        ].append(
            evaluate_position(
                index,
                rank,
            )
        )


    # ========================================================
    # Bottom I
    # ========================================================

    print()
    print(
        "Evaluating BOTTOM-I positions"
    )
    print("-" * 70)

    for rank, index in enumerate(
        bottom_indices,
        start=1,
    ):

        print(
            f"Bottom I #{rank:02d} | "
            f"I={I[index]:.9f} | "
            f"legal={legal_move_counts[int(index)]}"
        )

        result[
            "bottom_I"
        ].append(
            evaluate_position(
                index,
                rank,
            )
        )


    # ========================================================
    # Save JSON
    # ========================================================

    print()
    print(
        "Saving JSON"
    )
    print("-" * 70)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SANITY CHECK COMPLETED"
    )
    print("=" * 70)

    print(
        f"Top positions    : "
        f"{len(top_indices)}"
    )

    print(
        f"Bottom positions : "
        f"{len(bottom_indices)}"
    )

    print(
        f"Legal move filter: "
        f">= {MIN_LEGAL_MOVES}"
    )

    print()

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()