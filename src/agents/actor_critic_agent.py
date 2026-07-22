# Actor-critic Agent.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch

from src.encoding import encode_fen
from src.actions_space import INDEX_TO_ACTION


class ActorCriticAgent:

    def __init__(
        self,
        model,
        device="cpu",
        deterministic=False,
        temperature=0.75,
    ):

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.deterministic = deterministic

        self.temperature = temperature

        self.model.eval()


    @torch.no_grad()
    def choose_move(
        self,
        board: chess.Board,
    ):

        x = encode_fen(board.fen())

        x = x.unsqueeze(0).to(self.device)

        policy, value = self.model(x)

        logits = policy[0].clone()

        #
        # Masquage des coups illégaux
        #
        legal_moves = {
            move.uci(): move
            for move in board.legal_moves
        }

        for idx, uci in INDEX_TO_ACTION.items():

            if uci not in legal_moves:

                logits[idx] = float("-inf")

        # ==========================
        # Entropie (avant température)
        # ==========================

        probs = torch.softmax(logits, dim=0)

        log_probs = torch.log(probs + 1e-8)

        entropy = -(probs * log_probs).sum().item()

        #
        # Choix du coup
        #
        if self.deterministic:

            action = torch.argmax(logits).item()

        else:

            logits = logits - logits.max()

            logits = logits / self.temperature

            probs = torch.softmax(
                logits,
                dim=0,
            )

            action = torch.multinomial(
                probs,
                1,
            ).item()

        move_uci = INDEX_TO_ACTION[action]

        return {
            "move": legal_moves[move_uci],
            "action": action,
            "value": value.item(),
            "entropy": entropy,
            "fen": board.fen(),
        }