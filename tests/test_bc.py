from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import chess.variant

from src.models.resnet import ChessResNet
from src.encoding import encode_fen
from src.actions_space import ACTIONS, INDEX_TO_ACTION


CHECKPOINT = "checkpoints/bc_epoch_0.pt"


def main():

    device = "cpu"

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    board = chess.variant.AtomicBoard()

    print()
    print("=== BC BOT TEST ===")
    print()

    for ply in range(20):

        print(board)
        print()

        x = encode_fen(
            board.fen()
        ).unsqueeze(0)

        with torch.no_grad():

            logits = model(x)[0]

        #
        # Mask illegal moves
        #

        legal = {
            move.uci()
            for move in board.legal_moves
        }

        for idx, uci in INDEX_TO_ACTION.items():

            if uci not in legal:
                logits[idx] = -float("inf")

        probs = torch.softmax(
            logits,
            dim=0,
        )

        values, indices = torch.topk(
            probs,
            k=min(5, len(legal))
        )

        print("Top 5 predictions:")

        for p, idx in zip(values, indices):

            print(
                f"{INDEX_TO_ACTION[idx.item()]:8s}"
                f"{100*p.item():6.2f}%"
            )

        best = indices[0].item()

        move = chess.Move.from_uci(
            INDEX_TO_ACTION[best]
        )

        print()
        print("Chosen :", move)
        print("-" * 40)

        board.push(move)

        if board.is_game_over():

            print()
            print("Game over.")
            print(board.result())

            break


if __name__ == "__main__":
    main()