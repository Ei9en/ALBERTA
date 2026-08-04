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
        # ========================================================
        # Encodage batch
        # ========================================================
        #

        x = encode_boards(
            boards
        ).to(self.device)

        #
        # ========================================================
        # Forward batch
        # ========================================================
        #

        policies, values = self.model(x)

        batch_size = len(boards)
        action_size = policies.shape[1]

        #
        # ========================================================
        # Coups légaux
        # ========================================================
        #
        # On construit les indices légaux pour chaque position.
        #
        # Cette partie reste nécessairement dépendante de chaque
        # board, puisque l'ensemble des coups légaux varie.
        #

        legal_indices = []
        legal_moves = []

        for board in boards:

            board_legal_moves = list(
                board.legal_moves
            )

            board_legal_moves_dict = {
                move.uci(): move
                for move in board_legal_moves
            }

            indices = [
                ACTION_TO_INDEX[uci]
                for uci in board_legal_moves_dict
            ]

            legal_indices.append(
                indices
            )

            legal_moves.append(
                board_legal_moves_dict
            )

        #
        # ========================================================
        # Masque légal batché
        # ========================================================
        #

        legal_mask = torch.zeros(
            (
                batch_size,
                action_size,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        for i, indices in enumerate(
            legal_indices
        ):

            legal_mask[
                i,
                indices,
            ] = True

        #
        # ========================================================
        # Logits illégaux -> -inf
        # ========================================================
        #

        masked_logits = policies.masked_fill(
            ~legal_mask,
            float("-inf"),
        )

        #
        # ========================================================
        # Entropie AVANT température
        # ========================================================
        #

        probs = torch.softmax(
            masked_logits,
            dim=1,
        )

        log_probs = torch.log(
            probs + 1e-8
        )

        entropy = -(
            probs * log_probs
        ).sum(dim=1)

        #
        # ========================================================
        # Choix des coups
        # ========================================================
        #

        if self.deterministic:

            actions = torch.argmax(
                masked_logits,
                dim=1,
            )

        else:

            sampling_logits = (
                masked_logits
                - masked_logits.max(
                    dim=1,
                    keepdim=True,
                ).values
            )

            sampling_logits = (
                sampling_logits
                / self.temperature
            )

            sampling_probs = torch.softmax(
                sampling_logits,
                dim=1,
            )

            actions = torch.multinomial(
                sampling_probs,
                1,
            ).squeeze(1)

        #
        # ========================================================
        # Conversion action globale -> Move
        # ========================================================
        #

        results = []

        for i in range(batch_size):

            action = actions[i].item()

            move_uci = INDEX_TO_ACTION[
                action
            ]

            results.append(
                {
                    "move":
                        legal_moves[i][move_uci],

                    "action":
                        action,

                    "value":
                        values[i].item(),

                    "entropy":
                        entropy[i].item(),

                    "fen":
                        boards[i].fen(),
                }
            )

        return results