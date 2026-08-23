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
        # Logits légaux uniquement
        # ====================================================

        legal_logits = logits[
            legal_indices
        ]


        # ====================================================
        # Entropie intrinsèque de la policy
        #
        # IMPORTANT :
        # calculée AVANT la température de self-play.
        #
        # Il n'y a ici que des coups légaux, donc aucun
        # -inf provenant d'un masque.
        # ====================================================

        entropy_log_probs = torch.log_softmax(
            legal_logits,
            dim=0,
        )

        entropy_probs = torch.exp(
            entropy_log_probs
        )

        entropy = -(
            entropy_probs
            * entropy_log_probs
        ).sum().item()


        # ====================================================
        # Distribution utilisée pour choisir le coup
        #
        # La température ne modifie donc PAS H.
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0
        ):

            position = torch.argmax(
                legal_logits
            ).item()

            chosen_log_probs = torch.log_softmax(
                legal_logits,
                dim=0,
            )

            chosen_log_prob = (
                chosen_log_probs[position]
            )


        else:

            sampling_logits = (
                legal_logits
                /
                self.temperature
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
        # Entropie intrinsèque de la policy
        #
        # AVANT température.
        #
        # Pour les cases padding :
        #
        #     probability = 0
        #     log_probability = 0
        #
        # On évite donc :
        #
        #     0 * (-inf) = NaN
        # ====================================================

        entropy_log_probs = torch.log_softmax(
            legal_logits,
            dim=1,
        )

        entropy_probs = torch.exp(
            entropy_log_probs
        )

        entropy_log_probs_safe = (
            entropy_log_probs.masked_fill(
                ~legal_mask,
                0.0,
            )
        )

        entropy = -(
            entropy_probs
            * entropy_log_probs_safe
        ).sum(
            dim=1
        )


        # ====================================================
        # Distribution utilisée pour le self-play
        #
        # Cette distribution reçoit la température.
        # ====================================================

        sampling_log_probs = None

        sampling_probs = None


        if not (
            self.deterministic
            or self.temperature <= 0
        ):

            sampling_logits = (
                legal_logits
                /
                self.temperature
            )

            sampling_log_probs = torch.log_softmax(
                sampling_logits,
                dim=1,
            )

            sampling_probs = torch.exp(
                sampling_log_probs
            )


        # ====================================================
        # Choix de l'action
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0
        ):

            positions = torch.argmax(
                legal_logits,
                dim=1,
            )

            chosen_log_probs = torch.log_softmax(
                legal_logits,
                dim=1,
            )


        else:

            positions = torch.multinomial(
                sampling_probs,
                1,
            ).squeeze(1)

            chosen_log_probs = (
                sampling_log_probs
            )


        # ====================================================
        # Conversion
        # ====================================================

        results = []


        for i in range(
            batch_size
        ):

            position = positions[
                i
            ].item()


            action = legal_index_tensor[
                i,
                position,
            ].item()


            move = legal_moves[
                i
            ][
                action
            ]


            results.append(
                {

                    "move":
                        move,

                    "action":
                        action,

                    "value":
                        values[
                            i
                        ].item(),

                    "entropy":
                        entropy[
                            i
                        ].item(),

                    "log_prob":
                        chosen_log_probs[
                            i,
                            position,
                        ].item(),

                    "fen":
                        boards[
                            i
                        ].fen(),
                }
            )


        return results