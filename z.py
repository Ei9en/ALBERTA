from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# IMPORTS
# ============================================================

import chess
import torch

from src.models.resnet import ChessResNet
from src.actions_space import (
    ACTIONS,
    ACTION_TO_INDEX,
)

from lichess_bot.atomic_engine.rl_bot import RLBot


# ============================================================
# CONFIG
# ============================================================

CHECKPOINTS = {

    "BC5": (
        PROJECT_ROOT
        / "checkpoints"
        / "bc_epoch"
        / "bc_v3_epoch_5.pt"
    ),

    "RL20": (
        PROJECT_ROOT
        / "checkpoints"
        / "rl_epoch"
        / "rl_epoch_40.pt"
    ),

}


# ============================================================
# BC
# ============================================================

def load_bc(
    checkpoint,
):

    device = torch.device(
        "cpu"
    )

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    ).to(device)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint_data[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        device,
    )


@torch.no_grad()
def evaluate_bc(
    model,
    device,
    board,
):

    from src.encoding import encode_fen

    x = encode_fen(
        board.fen()
    )

    x = x.unsqueeze(
        0
    ).to(device)

    logits = model(x)[0]

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

    entropy = -(
        probs
        * log_probs
    ).sum().item()

    return {
        "moves": legal_uci,
        "probs": probs.cpu().tolist(),
        "entropy": entropy,
    }


# ============================================================
# RL
# ============================================================

def load_rl(
    checkpoint,
):

    return RLBot(
        checkpoint=checkpoint,

        # ----------------------------------------------------
        # Évaluation déterministe.
        #
        # On ne veut pas mesurer l'exploration du self-play,
        # mais la politique apprise par le modèle.
        # ----------------------------------------------------

        temperature=1.0,
        deterministic=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    board = chess.Board()

    print(
        "=" * 120
    )

    print(
        "INITIAL POSITION — BC5 vs RL10"
    )

    print(
        "=" * 120
    )

    print()


    # ========================================================
    # CHECKPOINTS
    # ========================================================

    print(
        "Checking checkpoints..."
    )

    print()


    available_checkpoints = {}

    for name, checkpoint in CHECKPOINTS.items():

        if not checkpoint.exists():

            print(
                f"[SKIP] {name}: checkpoint not found"
            )

            print(
                f"       {checkpoint}"
            )

            continue


        print(
            f"[OK]   {name}: {checkpoint}"
        )

        available_checkpoints[
            name
        ] = checkpoint


    print()


    if not available_checkpoints:

        print(
            "ERROR: no checkpoint found."
        )

        return


    # ========================================================
    # LOAD MODELS
    # ========================================================

    print(
        "Loading models..."
    )

    print()


    bc_model = None
    bc_device = None

    rl_bots = {}


    # --------------------------------------------------------
    # BC
    # --------------------------------------------------------

    if "BC5" in available_checkpoints:

        print(
            "Loading BC5..."
        )

        bc_model, bc_device = load_bc(
            available_checkpoints[
                "BC5"
            ]
        )

        print(
            "BC5 loaded."
        )

        print()


    # --------------------------------------------------------
    # RL
    # --------------------------------------------------------

    for name, checkpoint in (
        available_checkpoints.items()
    ):

        if name == "BC5":
            continue


        print(
            f"Loading {name}..."
        )


        rl_bots[name] = load_rl(
            checkpoint
        )


        print(
            f"{name} loaded."
        )

        print()


    # ========================================================
    # EVALUATE POLICIES
    # ========================================================

    distributions = {}


    # --------------------------------------------------------
    # BC
    # --------------------------------------------------------

    if bc_model is not None:

        distributions[
            "BC5"
        ] = evaluate_bc(
            bc_model,
            bc_device,
            board,
        )


    # --------------------------------------------------------
    # RL
    # --------------------------------------------------------

    for name, bot in rl_bots.items():

        result = bot.evaluate_policy(
            board
        )


        distributions[
            name
        ] = {

            "moves":
                result[
                    "moves"
                ],

            "probs":
                result[
                    "probs"
                ],

            "entropy":
                result[
                    "entropy"
                ],
        }


    # ========================================================
    # CHECK
    # ========================================================

    if not distributions:

        print(
            "ERROR: no policy could be evaluated."
        )

        return


    # ========================================================
    # AVAILABLE MODELS
    # ========================================================

    models = list(
        distributions.keys()
    )


    # ========================================================
    # UNION OF LEGAL MOVES
    # ========================================================

    all_moves = set()


    for distribution in (
        distributions.values()
    ):

        all_moves.update(
            distribution[
                "moves"
            ]
        )


    # ========================================================
    # REFERENCE ORDER
    #
    # Priorité à BC5 si disponible.
    # Sinon RL10.
    # ========================================================

    reference = {}


    if "BC5" in distributions:

        reference = dict(
            zip(
                distributions[
                    "BC5"
                ][
                    "moves"
                ],

                distributions[
                    "BC5"
                ][
                    "probs"
                ],
            )
        )

    else:

        first_model = models[0]

        reference = dict(
            zip(
                distributions[
                    first_model
                ][
                    "moves"
                ],

                distributions[
                    first_model
                ][
                    "probs"
                ],
            )
        )


    moves = sorted(
        all_moves,
        key=lambda move:
            reference.get(
                move,
                0.0,
            ),
        reverse=True,
    )


    # ========================================================
    # PROBABILITY MAPS
    # ========================================================

    probability_maps = {}


    for name, distribution in (
        distributions.items()
    ):

        probability_maps[
            name
        ] = dict(
            zip(
                distribution[
                    "moves"
                ],

                distribution[
                    "probs"
                ],
            )
        )


    # ========================================================
    # TABLE
    # ========================================================

    print(
        "=" * 120
    )

    print(
        "POLICY DISTRIBUTIONS"
    )

    print(
        "=" * 120
    )

    print()


    print(
        f"{'Rank':>4}  "
        f"{'Move':<8}",
        end="",
    )


    for name in models:

        print(
            f"{name:>15}",
            end="",
        )


    print()


    print(
        "-" * 120
    )


    for rank, move in enumerate(
        moves,
        start=1,
    ):

        print(
            f"{rank:>4}  "
            f"{move:<8}",
            end="",
        )


        for name in models:

            probability = (
                probability_maps[
                    name
                ].get(
                    move,
                    0.0,
                )
            )


            print(
                f"{probability * 100:>14.4f}%",
                end="",
            )


        print()


    # ========================================================
    # ENTROPY
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "INTRINSIC POLICY ENTROPY"
    )

    print(
        "=" * 120
    )

    print()


    for name in models:

        entropy = (
            distributions[
                name
            ][
                "entropy"
            ]
        )


        print(
            f"{name:<12} : "
            f"{entropy:.6f}"
        )


    # ========================================================
    # TOP MOVE
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "TOP MOVE"
    )

    print(
        "=" * 120
    )

    print()


    for name in models:

        probs = probability_maps[
            name
        ]


        best_move = max(
            probs,
            key=probs.get,
        )


        best_probability = (
            probs[
                best_move
            ]
        )


        print(
            f"{name:<12} : "
            f"{best_move:<8} "
            f"{best_probability * 100:.4f}%"
        )


    # ========================================================
    # FOCUS — INITIAL MOVE
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "FOCUS — INITIAL MOVE"
    )

    print(
        "=" * 120
    )

    print()


    print(
        f"{'Model':<15}"
        f"{'Nf3':>12}"
        f"{'e4':>12}"
        f"{'d4':>12}"
        f"{'f3':>12}"
    )


    print(
        "-" * 65
    )


    for name in models:

        probs = probability_maps[
            name
        ]


        print(
            f"{name:<15}"
            f"{probs.get('g1f3', 0.0) * 100:>11.4f}%"
            f"{probs.get('e2e4', 0.0) * 100:>11.4f}%"
            f"{probs.get('d2d4', 0.0) * 100:>11.4f}%"
            f"{probs.get('f2f3', 0.0) * 100:>11.4f}%"
        )


    print()


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "=" * 120
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 120
    )

    print()


    print(
        f"Position: "
        f"{board.fen()}"
    )

    print()


    for name in models:

        probs = probability_maps[
            name
        ]


        best_move = max(
            probs,
            key=probs.get,
        )


        print(
            f"{name:<12} "
            f"best move = "
            f"{best_move:<8} "
            f"probability = "
            f"{probs[best_move] * 100:.4f}% "
            f"entropy = "
            f"{distributions[name]['entropy']:.6f}"
        )


    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()