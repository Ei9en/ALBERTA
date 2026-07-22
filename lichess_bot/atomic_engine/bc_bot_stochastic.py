from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch

from src.models.resnet import ChessResNet
from src.encoding import encode_fen
from src.actions_space import ACTIONS, INDEX_TO_ACTION


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_v2_5_epoch_7.pt"
)


class BCBotStochastic:

    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        temperature=0.75,
    ):

        self.device = torch.device("cpu")

        self.temperature = temperature

        self.model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=64,
            blocks=4,
        ).to(self.device)


        checkpoint_data = torch.load(
            checkpoint,
            map_location=self.device,
        )


        self.model.load_state_dict(
            checkpoint_data["model_state_dict"]
        )

        self.model.eval()


        print(
            f"Loaded checkpoint: {checkpoint}"
        )


    @torch.no_grad()
    def choose_move(
        self,
        board: chess.Board,
    ):

        x = encode_fen(
            board.fen()
        )

        x = x.unsqueeze(0).to(self.device)


        logits = self.model(x)[0]


        legal_moves = {
            move.uci(): move
            for move in board.legal_moves
        }


        logits = logits.clone()


        for idx, uci in INDEX_TO_ACTION.items():

            if uci not in legal_moves:

                logits[idx] = float("-inf")


        #
        # Sampling avec température
        #

        probs = torch.softmax(
            logits / self.temperature,
            dim=0,
        )


        action = torch.multinomial(
            probs,
            1,
        ).item()


        move_uci = INDEX_TO_ACTION[action]

        return {
            "action": action,
            "move": legal_moves[move_uci],
            "value": 0.0,
            "entropy": (
                -(probs * torch.log(probs + 1e-8))
                .sum()
                .item()
            ),
        }