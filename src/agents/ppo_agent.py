from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch

from src.encoding import encode_board, encode_boards
from src.actions_space import ACTION_TO_INDEX


class PPOAgent:

    def __init__(
        self,
        model,
        device="cpu",
        deterministic=False,
        temperature=1.5,
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


        # ====================================================
        # Coups légaux
        # ====================================================

        legal_moves = {
            move.uci(): move
            for move in board.legal_moves
        }

        legal_indices = [
            ACTION_TO_INDEX[uci]
            for uci in legal_moves
        ]


        # ====================================================
        # Logits légaux
        # ====================================================

        legal_logits = logits[
            legal_indices
        ]


        # ====================================================
        # Distribution comportementale
        #
        # C'est cette distribution qui est réellement utilisée
        # pour le sampling.
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0
        ):

            sampling_log_probs = torch.log_softmax(
                legal_logits,
                dim=0,
            )

            sampling_probs = torch.exp(
                sampling_log_probs
            )

            position = torch.argmax(
                legal_logits
            ).item()

        else:

            sampling_logits = (
                legal_logits
                / self.temperature
            )

            sampling_log_probs = torch.log_softmax(
                sampling_logits,
                dim=0,
            )

            sampling_probs = torch.exp(
                sampling_log_probs
            )

            position = torch.multinomial(
                sampling_probs,
                1,
            ).item()


        # ====================================================
        # Entropie
        #
        # IMPORTANT :
        # entropie de la distribution effectivement utilisée.
        # Donc après température.
        # ====================================================

        entropy = -(
            sampling_probs
            * sampling_log_probs
        ).sum().item()


        # ====================================================
        # Log probability du coup choisi
        #
        # Même distribution que celle utilisée pour sampler.
        # ====================================================

        chosen_log_prob = (
            sampling_log_probs[position]
        )


        # ====================================================
        # Conversion position légale -> action globale
        # ====================================================

        action = legal_indices[position]

        move_uci = list(
            legal_moves.keys()
        )[position]


        return {

            "move":
                legal_moves[move_uci],

            "action":
                action,

            "value":
                value.item(),

            "entropy":
                entropy,

            "log_prob":
                chosen_log_prob.item(),

            "fen":
                board.fen(),
        }


    @torch.no_grad()
    def choose_moves(
        self,
        boards,
    ):

        if len(boards) == 0:
            return []


        batch_size = len(boards)


        # ====================================================
        # Encodage batch
        # ====================================================

        x = encode_boards(
            boards
        ).to(self.device)


        # ====================================================
        # Forward
        # ====================================================

        policies, values = self.model(x)


        # ====================================================
        # Extraction des coups légaux
        # ====================================================

        legal_indices = []

        legal_moves = []

        max_legal_moves = 0


        for board in boards:

            board_legal_moves = list(
                board.legal_moves
            )

            board_indices = []

            board_moves = {}


            for move in board_legal_moves:

                uci = move.uci()

                action = ACTION_TO_INDEX[
                    uci
                ]

                board_indices.append(
                    action
                )

                board_moves[action] = move


            legal_indices.append(
                board_indices
            )

            legal_moves.append(
                board_moves
            )

            max_legal_moves = max(
                max_legal_moves,
                len(board_indices),
            )


        # ====================================================
        # Tensor des coups légaux
        # ====================================================

        legal_index_tensor = torch.zeros(
            (
                batch_size,
                max_legal_moves,
            ),
            dtype=torch.long,
            device=self.device,
        )


        legal_mask = torch.zeros(
            (
                batch_size,
                max_legal_moves,
            ),
            dtype=torch.bool,
            device=self.device,
        )


        for i, indices in enumerate(
            legal_indices
        ):

            n = len(indices)


            legal_index_tensor[
                i,
                :n,
            ] = torch.tensor(
                indices,
                dtype=torch.long,
                device=self.device,
            )


            legal_mask[
                i,
                :n,
            ] = True


        # ====================================================
        # Logits légaux
        # ====================================================

        legal_logits = policies.gather(
            1,
            legal_index_tensor,
        )


        legal_logits = legal_logits.masked_fill(
            ~legal_mask,
            float("-inf"),
        )


        # ====================================================
        # Distribution comportementale
        #
        # Température appliquée AVANT softmax.
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0
        ):

            sampling_logits = (
                legal_logits
            )

        else:

            sampling_logits = (
                legal_logits
                / self.temperature
            )


        sampling_log_probs = torch.log_softmax(
            sampling_logits,
            dim=1,
        )


        sampling_probs = torch.exp(
            sampling_log_probs
        )


        # ====================================================
        # Entropie
        #
        # Distribution effectivement utilisée.
        # ====================================================

        entropy = -(
            sampling_probs
            * sampling_log_probs
        ).sum(dim=1)


        # ====================================================
        # Choix des actions
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0
        ):

            positions = torch.argmax(
                legal_logits,
                dim=1,
            )

        else:

            positions = torch.multinomial(
                sampling_probs,
                1,
            ).squeeze(1)


        # ====================================================
        # Conversion
        # ====================================================

        results = []


        for i in range(batch_size):

            position = positions[i].item()


            action = legal_index_tensor[
                i,
                position,
            ].item()


            move = legal_moves[i][action]


            results.append(
                {

                    "move":
                        move,

                    "action":
                        action,

                    "value":
                        values[i].item(),

                    "entropy":
                        entropy[i].item(),

                    "log_prob":
                        sampling_log_probs[
                            i,
                            position,
                        ].item(),

                    "fen":
                        boards[i].fen(),
                }
            )


        return results