from pathlib import Path
import sys
import json
import random
import math

import chess
import chess.variant
import torch
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.encoding import encode_board
from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.actions_space import ACTIONS, ACTION_TO_INDEX


# ============================================================
# CONFIGURATION
# ============================================================

UNCERTAINTY_PATH = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats_1-100.json"
)

BC_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_v3_epoch_5.pt"
)

RL_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
)

RL_EPOCHS = list(
    range(10, 101, 10)
)

# ============================================================
# TOTAL NUMBER OF POSITIONS
#
# 1100 total
# 11 strata
# => 100 positions / stratum
# ============================================================

N = 1100

SEED = 42

PRINT_PROGRESS = True


# ============================================================
# PLY STRATA
# ============================================================

STRATA = [
    ("1-5",   1,  5),
    ("6-10",  6, 10),
    ("11-15", 11, 15),
    ("16-20", 16, 20),
    ("21-25", 21, 25),
    ("26-30", 26, 30),
    ("31-35", 31, 35),
    ("36-40", 36, 40),
    ("41-45", 41, 45),
    ("46-50", 46, 50),
    ("51+",   51, None),
]


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = random.Random(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# LOAD BC
# ============================================================

print("=" * 70)
print("LOADING BC MODEL")
print("=" * 70)

print()
print(
    f"Loading BC checkpoint:"
)
print(
    BC_CHECKPOINT
)


bc_model = ChessResNet(
    num_actions=len(ACTIONS),
    channels=32,
    blocks=4,
).to(device)


bc_checkpoint = torch.load(
    BC_CHECKPOINT,
    map_location=device,
)


bc_model.load_state_dict(
    bc_checkpoint["model_state_dict"]
)

bc_model.eval()


print("BC loaded.")


# ============================================================
# PLY EXTRACTION
# ============================================================

def get_ply(record):
    """
    Extract the ply from a FEN.

    White to move:
        ply = 2 * (fullmove - 1)

    Black to move:
        ply = 2 * (fullmove - 1) + 1
    """

    fen = record["fen"]

    board = chess.variant.AtomicBoard(
        fen
    )

    fullmove = board.fullmove_number

    if board.turn == chess.WHITE:
        return 2 * (fullmove - 1)

    return 2 * (fullmove - 1) + 1


# ============================================================
# STRATUM IDENTIFICATION
# ============================================================

def get_stratum(ply):

    for name, low, high in STRATA:

        if high is None:

            if ply >= low:
                return name

        else:

            if low <= ply <= high:
                return name

    return None


# ============================================================
# LOAD UNCERTAINTY DATA
# ============================================================

print()
print("=" * 70)
print("LOADING UNCERTAINTY DATA")
print("=" * 70)

print()
print(
    f"Loading:"
)
print(
    UNCERTAINTY_PATH
)

with open(
    UNCERTAINTY_PATH,
    "r",
) as f:

    data = json.load(f)


print(
    f"Total records: {len(data):,}"
)


# ============================================================
# TARGET SAMPLE SIZE
# ============================================================

NUM_STRATA = len(STRATA)

base_n = N // NUM_STRATA
remainder = N % NUM_STRATA


TARGETS = {}

for i, (name, _, _) in enumerate(STRATA):

    target = base_n

    if i < remainder:
        target += 1

    TARGETS[name] = target


# ============================================================
# STRATIFIED RESERVOIR SAMPLING
# ============================================================
#
# We sample ONCE.
#
# The exact same positions will then be evaluated by:
#
# RL10
# RL20
# RL30
# ...
# RL100
#
# This makes the comparison paired across checkpoints.
#
# ============================================================

print()
print("=" * 70)
print("STRATIFIED SAMPLING")
print("=" * 70)


samples = {
    name: []
    for name, _, _ in STRATA
}


stratum_counts = {
    name: 0
    for name, _, _ in STRATA
}


for record in data:

    ply = get_ply(record)

    stratum = get_stratum(ply)

    if stratum is None:
        continue

    stratum_counts[stratum] += 1

    target = TARGETS[stratum]

    reservoir = samples[stratum]

    current_count = stratum_counts[stratum]

    # Reservoir not full yet
    if len(reservoir) < target:

        reservoir.append(record)

    else:

        j = rng.randrange(
            current_count
        )

        if j < target:

            reservoir[j] = record


# ============================================================
# DISPLAY SAMPLING
# ============================================================

print()

for name, _, _ in STRATA:

    available = stratum_counts[name]
    sampled = len(samples[name])
    target = TARGETS[name]

    print(
        f"{name:>7} : "
        f"{available:>10,} available -> "
        f"{sampled:>7,} sampled"
    )


total_sampled = sum(
    len(v)
    for v in samples.values()
)


print()
print(
    f"Total sampled: {total_sampled:,}"
)


# ============================================================
# BC DISTRIBUTIONS
# ============================================================
#
# IMPORTANT:
#
# BC never changes, so we compute its distribution ONCE
# for every sampled position.
#
# This avoids running the BC model 10 times.
#
# ============================================================

print()
print("=" * 70)
print("COMPUTING BC DISTRIBUTIONS")
print("=" * 70)


bc_distributions = {}


bc_position_id = 0


for name, _, _ in STRATA:

    print(
        f"BC -> {name}"
    )

    for record in samples[name]:

        fen = record["fen"]

        board = chess.variant.AtomicBoard(
            fen
        )

        x = encode_board(board)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():

            logits = bc_model(x)[0]

        legal_moves = list(
            board.legal_moves
        )

        legal_uci = [
            move.uci()
            for move in legal_moves
        ]

        legal_indices = [
            ACTION_TO_INDEX[uci]
            for uci in legal_uci
        ]

        legal_logits = logits[
            legal_indices
        ]

        log_probs = torch.log_softmax(
            legal_logits,
            dim=0,
        )

        probs = torch.exp(
            log_probs
        )

        bc_distributions[
            bc_position_id
        ] = {
            uci: prob.item()
            for uci, prob in zip(
                legal_uci,
                probs,
            )
        }

        bc_position_id += 1


print(
    f"BC distributions computed: "
    f"{len(bc_distributions):,}"
)


# ============================================================
# BUILD FLAT POSITION LIST
# ============================================================
#
# Each position gets a stable integer ID.
#
# The same IDs are used for every RL checkpoint.
#
# ============================================================

positions = []

position_id = 0


for name, _, _ in STRATA:

    for record in samples[name]:

        positions.append(
            {
                "id": position_id,
                "stratum": name,
                "record": record,
            }
        )

        position_id += 1


# ============================================================
# KL DIVERGENCE
# ============================================================

def kl_divergence(
    p,
    q,
    epsilon=1e-12,
):
    """
    D_KL(P || Q)

        = sum_a P(a) log(P(a) / Q(a))

    Here:

        P = RL policy
        Q = BC policy

    Therefore:

        D_KL(pi_RL || pi_BC)

    The distributions are defined over legal moves only.
    """

    kl = 0.0

    for action, p_prob in p.items():

        q_prob = q.get(
            action,
            0.0,
        )

        p_prob = max(
            p_prob,
            epsilon,
        )

        q_prob = max(
            q_prob,
            epsilon,
        )

        kl += (
            p_prob
            * math.log(
                p_prob / q_prob
            )
        )

    return kl


# ============================================================
# RL MODEL CREATION
# ============================================================

def create_rl_model():

    backbone = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        backbone
    ).to(device)

    return model


# ============================================================
# GET RL DISTRIBUTION
# ============================================================

@torch.no_grad()
def get_rl_distribution(
    model,
    board,
):
    """
    Intrinsic RL policy over legal moves.

    No temperature.
    """

    x = encode_board(board)
    x = x.unsqueeze(0).to(device)

    policy, _ = model(x)

    logits = policy[0]

    legal_moves = list(
        board.legal_moves
    )

    legal_uci = [
        move.uci()
        for move in legal_moves
    ]

    legal_indices = [
        ACTION_TO_INDEX[uci]
        for uci in legal_uci
    ]

    legal_logits = logits[
        legal_indices
    ]

    log_probs = torch.log_softmax(
        legal_logits,
        dim=0,
    )

    probs = torch.exp(
        log_probs
    )

    return {
        uci: prob.item()
        for uci, prob in zip(
            legal_uci,
            probs,
        )
    }


# ============================================================
# RESULTS
# ============================================================

all_results = {}


# ============================================================
# PROCESS EACH RL CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("RL / BC KL ANALYSIS")
print("=" * 70)


for epoch in RL_EPOCHS:

    rl_checkpoint_path = (
        RL_CHECKPOINT_DIR
        / f"rl_epoch_{epoch}.pt"
    )

    print()
    print("=" * 70)
    print(
        f"RL EPOCH {epoch}"
    )
    print("=" * 70)

    if not rl_checkpoint_path.exists():

        print(
            f"WARNING: checkpoint not found:"
        )

        print(
            rl_checkpoint_path
        )

        continue


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    rl_model = create_rl_model()

    checkpoint = torch.load(
        rl_checkpoint_path,
        map_location=device,
    )

    rl_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    rl_model.eval()


    print(
        f"Loaded: {rl_checkpoint_path}"
    )


    # --------------------------------------------------------
    # KL results
    # --------------------------------------------------------

    checkpoint_results = {
        name: []
        for name, _, _ in STRATA
    }


    for index, position in enumerate(
        positions
    ):

        position_id = position["id"]
        stratum = position["stratum"]
        record = position["record"]

        fen = record["fen"]

        board = chess.variant.AtomicBoard(
            fen
        )

        # ----------------------------------------------------
        # RL distribution
        # ----------------------------------------------------

        rl_probs = get_rl_distribution(
            rl_model,
            board,
        )

        # ----------------------------------------------------
        # BC distribution
        # ----------------------------------------------------

        bc_probs = bc_distributions[
            position_id
        ]

        # ----------------------------------------------------
        # D_KL(RL || BC)
        # ----------------------------------------------------

        kl = kl_divergence(
            rl_probs,
            bc_probs,
        )

        checkpoint_results[
            stratum
        ].append(kl)


        if (
            PRINT_PROGRESS
            and (index + 1) % 500 == 0
        ):

            print(
                f"  {index + 1:,} / "
                f"{total_sampled:,}"
            )


    all_results[
        f"RL{epoch}"
    ] = checkpoint_results


    # --------------------------------------------------------
    # Free checkpoint before next one
    # --------------------------------------------------------

    del rl_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# STATISTICS FUNCTION
# ============================================================

def compute_statistics(values):

    values = np.array(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return None

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


# ============================================================
# PRINT ONE TABLE PER CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("KL DISTRIBUTIONS BY PLY")
print("=" * 70)


for epoch in RL_EPOCHS:

    key = f"RL{epoch}"

    if key not in all_results:
        continue


    print()
    print(
        "=" * 70
    )

    print(
        f"{key}: D_KL(RL || BC)"
    )

    print(
        "=" * 70
    )

    header = (
        f"{'PLY':>7} "
        f"{'N':>7} "
        f"{'Mean':>12} "
        f"{'Median':>12} "
        f"{'Std':>12} "
        f"{'P10':>12} "
        f"{'P25':>12} "
        f"{'P75':>12} "
        f"{'P90':>12} "
        f"{'P95':>12} "
        f"{'P99':>12} "
        f"{'Max':>12}"
    )

    print()
    print(header)
    print("-" * len(header))


    for name, _, _ in STRATA:

        values = all_results[
            key
        ][name]

        stats = compute_statistics(
            values
        )

        if stats is None:
            continue


        print(
            f"{name:>7} "
            f"{stats['n']:>7} "
            f"{stats['mean']:>12.6f} "
            f"{stats['median']:>12.6f} "
            f"{stats['std']:>12.6f} "
            f"{stats['p10']:>12.6f} "
            f"{stats['p25']:>12.6f} "
            f"{stats['p75']:>12.6f} "
            f"{stats['p90']:>12.6f} "
            f"{stats['p95']:>12.6f} "
            f"{stats['p99']:>12.6f} "
            f"{stats['max']:>12.6f}"
        )


# ============================================================
# SUMMARY TABLE
# ============================================================
#
# Mean KL uniquement.
#
# Cela permet de voir rapidement l'évolution de la distance
# RL -> BC avec l'entraînement.
#
# ============================================================

print()
print("=" * 70)
print("MEAN KL SUMMARY")
print("=" * 70)

print()

header = (
    f"{'PLY':>7}"
)

for epoch in RL_EPOCHS:

    header += (
        f" {'RL' + str(epoch):>12}"
    )

print(header)
print("-" * len(header))


for name, _, _ in STRATA:

    row = f"{name:>7}"

    for epoch in RL_EPOCHS:

        key = f"RL{epoch}"

        if key not in all_results:

            row += f" {'N/A':>12}"
            continue


        values = all_results[
            key
        ][name]

        if len(values) == 0:

            row += f" {'N/A':>12}"

        else:

            mean_kl = np.mean(
                values
            )

            row += (
                f" {mean_kl:>12.6f}"
            )


    print(row)


# ============================================================
# GLOBAL MEAN KL
# ============================================================

print()
print("=" * 70)
print("GLOBAL MEAN KL")
print("=" * 70)

print()

for epoch in RL_EPOCHS:

    key = f"RL{epoch}"

    if key not in all_results:
        continue

    all_values = []

    for name, _, _ in STRATA:

        all_values.extend(
            all_results[key][name]
        )

    if len(all_values) == 0:
        continue

    print(
        f"{key:>6} : "
        f"mean KL = "
        f"{np.mean(all_values):.6f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_bc_kl_by_ply_all_checkpoints.json"
)


output = {
    "N_requested": N,
    "N_sampled": total_sampled,

    "positions_per_stratum": {
        name: len(samples[name])
        for name, _, _ in STRATA
    },

    "seed": SEED,

    "bc_checkpoint": str(
        BC_CHECKPOINT
    ),

    "rl_checkpoints": [
        f"rl_epoch_{epoch}.pt"
        for epoch in RL_EPOCHS
        if f"RL{epoch}" in all_results
    ],

    "definition": {
        "P": "RL policy",
        "Q": "BC policy",
        "KL": "D_KL(RL || BC)",
        "temperature": None,
        "support": "legal moves only",
    },

    "results": {},
}


# Convert NumPy values to JSON-safe floats

for epoch in RL_EPOCHS:

    key = f"RL{epoch}"

    if key not in all_results:
        continue

    output["results"][key] = {}

    for name, _, _ in STRATA:

        values = all_results[
            key
        ][name]

        stats = compute_statistics(
            values
        )

        output["results"][key][name] = stats


with open(
    OUTPUT_PATH,
    "w",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print(
    f"Results saved to:"
)

print(
    OUTPUT_PATH
)

print()
print(
    f"Positions per checkpoint: "
    f"{total_sampled:,}"
)

print(
    f"Checkpoints analyzed: "
    f"{len(all_results)}"
)

print("=" * 70)