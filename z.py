from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch

from src.models.resnet import ChessResNet
from src.actions_space import ACTIONS, ACTION_TO_INDEX

from lichess_bot.atomic_engine.rl_bot import RLBot


# ============================================================
# CONFIG
# ============================================================

CHECKPOINTS = {
    "BC": (
        PROJECT_ROOT
        / "checkpoints"
        / "bc_epoch"
        / "bc_v3_epoch_5.pt"
    ),

    "RL10": (
        PROJECT_ROOT
        / "checkpoints"
        / "rl_epoch"
        / "rl_epoch_10.pt"
    ),

    "RL20 pure": (
        PROJECT_ROOT
        / "checkpoints"
        / "rl_epoch"
        / "rl_epoch_20.pt"
    ),

    # À ADAPTER AU NOM EXACT DU CHECKPOINT POST-AL
    "RL20 AL": (
        PROJECT_ROOT
        / "checkpoints"
        / "al_epoch"
        / "al1_epoch_20.pt"
    ),

    "RL30": (
        PROJECT_ROOT
        / "checkpoints"
        / "rl_epoch"
        / "rl_epoch_30.pt"
    ),
}


# ============================================================
# BC
# ============================================================

def load_bc(checkpoint):

    device = torch.device("cpu")

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
        checkpoint_data["model_state_dict"]
    )

    model.eval()

    return model, device


@torch.no_grad()
def evaluate_bc(model, device, board):

    from src.encoding import encode_fen

    x = encode_fen(
        board.fen()
    )

    x = x.unsqueeze(0).to(device)

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
        probs * log_probs
    ).sum().item()

    return {
        "moves": legal_uci,
        "probs": probs.cpu().tolist(),
        "entropy": entropy,
    }


# ============================================================
# RL
# ============================================================

def load_rl(checkpoint):

    return RLBot(
        checkpoint=checkpoint,
        temperature=2.0,
        deterministic=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    board = chess.Board()

    print("=" * 120)
    print("INITIAL POSITION — POLICY COMPARISON")
    print("=" * 120)
    print()

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    print("Loading models...")
    print()

    bc_model, bc_device = load_bc(
        CHECKPOINTS["BC"]
    )

    rl_bots = {}

    for name, checkpoint in CHECKPOINTS.items():

        if name == "BC":
            continue

        print(
            f"Loading {name}: {checkpoint}"
        )

        rl_bots[name] = load_rl(
            checkpoint
        )

    print()

    # --------------------------------------------------------
    # Évaluation
    # --------------------------------------------------------

    distributions = {}

    distributions["BC"] = evaluate_bc(
        bc_model,
        bc_device,
        board,
    )

    for name, bot in rl_bots.items():

        result = bot.evaluate_policy(
            board
        )

        distributions[name] = {
            "moves": result["moves"],
            "probs": result["probs"],
            "entropy": result["entropy"],
        }

    # --------------------------------------------------------
    # Liste des coups
    #
    # Union des coups observés dans toutes les policies.
    # --------------------------------------------------------

    all_moves = set()

    for distribution in distributions.values():

        all_moves.update(
            distribution["moves"]
        )

    # Tri selon BC puis RL20 AL
    # pour garder les coups intéressants en haut.

    reference = {}

    for name in ["BC", "RL20 AL"]:

        if name not in distributions:
            continue

        reference = dict(
            zip(
                distributions[name]["moves"],
                distributions[name]["probs"],
            )
        )

        break

    moves = sorted(
        all_moves,
        key=lambda move:
            reference.get(move, 0.0),
        reverse=True,
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    print("=" * 120)
    print("POLICY DISTRIBUTIONS")
    print("=" * 120)
    print()

    print(
        f"{'Rank':>4}  "
        f"{'Move':<6}",
        end="",
    )

    for name in CHECKPOINTS:

        print(
            f"{name:>14}",
            end="",
        )

    print()

    print("-" * 120)

    probability_maps = {}

    for name, distribution in distributions.items():

        probability_maps[name] = dict(
            zip(
                distribution["moves"],
                distribution["probs"],
            )
        )

    for rank, move in enumerate(
        moves,
        start=1,
    ):

        print(
            f"{rank:>4}  "
            f"{move:<6}",
            end="",
        )

        for name in CHECKPOINTS:

            probability = probability_maps[
                name
            ].get(
                move,
                0.0,
            )

            print(
                f"{probability * 100:>13.4f}%",
                end="",
            )

        print()

    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("INTRINSIC POLICY ENTROPY")
    print("=" * 120)
    print()

    for name in CHECKPOINTS:

        entropy = distributions[
            name
        ]["entropy"]

        print(
            f"{name:<12} : {entropy:.4f}"
        )

    print()

    # --------------------------------------------------------
    # FOCUS ON E4 / NF3
    # --------------------------------------------------------

    print("=" * 120)
    print("FOCUS — INITIAL MOVE")
    print("=" * 120)
    print()

    print(
        f"{'Model':<15}"
        f"{'Nf3':>12}"
        f"{'e4':>12}"
        f"{'d4':>12}"
        f"{'f3':>12}"
    )

    print("-" * 65)

    for name in CHECKPOINTS:

        probs = probability_maps[name]

        print(
            f"{name:<15}"
            f"{probs.get('g1f3', 0.0) * 100:>11.4f}%"
            f"{probs.get('e2e4', 0.0) * 100:>11.4f}%"
            f"{probs.get('d2d4', 0.0) * 100:>11.4f}%"
            f"{probs.get('f2f3', 0.0) * 100:>11.4f}%"
        )

    print()


if __name__ == "__main__":
    main()