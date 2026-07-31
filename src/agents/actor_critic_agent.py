from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch

from src.encoding import encode_board, encode_boards
from src.actions_space import (
    ACTION_TO_INDEX,
    INDEX_TO_ACTION,
)


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

        x = encode_board(board)

        x = x.unsqueeze(0).to(self.device)

        policy, value = self.model(x)

        logits = policy[0]

        #
        # Coups légaux
        #
        legal_moves = {
            move.uci(): move
            for move in board.legal_moves
        }

        legal_indices = [
            ACTION_TO_INDEX[uci]
            for uci in legal_moves
        ]

        #
        # On ne conserve que les logits légaux.
        #
        legal_logits = logits[
            legal_indices
        ]

        #
        # Entropie AVANT température
        #
        probs = torch.softmax(
            legal_logits,
            dim=0,
        )

        log_probs = torch.log(
            probs + 1e-8
        )

        entropy = -(
            probs * log_probs
        ).sum().item()

        #
        # Choix du coup
        #
        if self.deterministic:

            position = torch.argmax(
                legal_logits
            ).item()

        else:

            sampling_logits = (
                legal_logits
                - legal_logits.max()
            )

            sampling_logits = (
                sampling_logits
                / self.temperature
            )

            probs = torch.softmax(
                sampling_logits,
                dim=0,
            )

            position = torch.multinomial(
                probs,
                1,
            ).item()

        #
        # Conversion position légale -> action globale
        #
        action = legal_indices[position]

        move_uci = INDEX_TO_ACTION[action]

        return {
            "move": legal_moves[move_uci],
            "action": action,
            "value": value.item(),
            "entropy": entropy,
            "fen": board.fen(),
        }


    @torch.no_grad()
    def choose_moves(
        self,
        boards,
    ):

        if len(boards) == 0:
            return []

        #
        # Encodage batch
        #
        x = encode_boards(
            boards
        ).to(self.device)

        #
        # Un seul forward
        #
        policies, values = self.model(x)

        results = []

        for i, board in enumerate(boards):

            logits = policies[i]

            #
            # Coups légaux
            #
            legal_moves = {
                move.uci(): move
                for move in board.legal_moves
            }

            legal_indices = [
                ACTION_TO_INDEX[uci]
                for uci in legal_moves
            ]

            #
            # On ne conserve que les logits légaux.
            #
            legal_logits = logits[
                legal_indices
            ]

            #
            # Entropie AVANT température
            #
            probs = torch.softmax(
                legal_logits,
                dim=0,
            )

            log_probs = torch.log(
                probs + 1e-8
            )

            entropy = -(
                probs * log_probs
            ).sum().item()

            #
            # Choix du coup
            #
            if self.deterministic:

                position = torch.argmax(
                    legal_logits
                ).item()

            else:

                sampling_logits = (
                    legal_logits
                    - legal_logits.max()
                )

                sampling_logits = (
                    sampling_logits
                    / self.temperature
                )

                probs = torch.softmax(
                    sampling_logits,
                    dim=0,
                )

                position = torch.multinomial(
                    probs,
                    1,
                ).item()

            #
            # Position dans legal_indices
            # -> index global de l'action
            #
            action = legal_indices[position]

            move_uci = INDEX_TO_ACTION[action]

            results.append(
                {
                    "move": legal_moves[move_uci],
                    "action": action,
                    "value": values[i].item(),
                    "entropy": entropy,
                    "fen": board.fen(),
                }
            )

        return results