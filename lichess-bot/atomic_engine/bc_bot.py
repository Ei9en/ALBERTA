from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch

from src.models.resnet import ChessResNet
from src.encoding import encode_fen
from src.actions_space import ACTIONS, INDEX_TO_ACTION


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bc_epoch_0.pt"


class BCBot:

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):

        self.device = torch.device("cpu")

        self.model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=64,
            blocks=4,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print(f"Loaded checkpoint: {DEFAULT_CHECKPOINT}")

    @torch.no_grad()
    def choose_move(self, board: chess.Board):

        x = encode_fen(board.fen())

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

        best_idx = torch.argmax(logits).item()

        best_uci = INDEX_TO_ACTION[best_idx]

        return legal_moves[best_uci]