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
        bc_model=None,
        opening_prior_strength=1.0,
        opening_prior_plies=6,
    ):

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.bc_model = (
            bc_model.to(self.device)
            if bc_model is not None
            else None
        )

        if self.bc_model is not None:
            self.bc_model.eval()

        self.deterministic = deterministic

        self.temperature = temperature

        self.opening_prior_strength = (
            opening_prior_strength
        )

        self.opening_prior_plies = (
            opening_prior_plies
        )

        self.model.eval()


    # ========================================================
    # BC opening prior strength
    # ========================================================

    def _opening_prior_strength(
        self,
        ply,
    ):

        if (
            self.bc_model is None
            or self.opening_prior_plies <= 0
        ):

            return 0.0


        if ply >= self.opening_prior_plies:

            return 0.0


        # ====================================================
        # Décroissance linéaire
        #
        # ply 0 -> 100 %
        # ply 1 ->  83 %
        # ply 2 ->  67 %
        # ply 3 ->  50 %
        # ply 4 ->  33 %
        # ply 5 ->  17 %
        # ply 6 ->   0 %
        # ====================================================

        progress = (
            ply
            / self.opening_prior_plies
        )


        return (
            self.opening_prior_strength
            * (1.0 - progress)
        )


    # ========================================================
    # BC prior — single position
    # ========================================================

    @torch.no_grad()
    def _apply_bc_prior_single(
        self,
        board,
        legal_indices,
        legal_logits,
    ):

        alpha = (
            self._opening_prior_strength(
                board.ply()
            )
        )


        if alpha <= 0.0:

            return legal_logits


        if self.bc_model is None:

            return legal_logits


        x = encode_board(
            board
        )

        x = x.unsqueeze(0).to(
            self.device
        )


        bc_policy, _ = self.bc_model(
            x
        )


        bc_logits = bc_policy[0]


        bc_legal_logits = bc_logits[
            legal_indices
        ]


        bc_log_probs = torch.log_softmax(
            bc_legal_logits,
            dim=0,
        )


        # ====================================================
        # BC comme prior
        #
        # RL logits + alpha * log P_BC
        #
        # Le BC influence la préférence mais n'interdit
        # jamais un coup.
        # ====================================================

        guided_logits = (
            legal_logits
            + alpha
            * bc_log_probs
        )


        return guided_logits


    # ========================================================
    # Single move
    # ========================================================

    @torch.no_grad()
    def choose_move(
        self,
        board: chess.Board,
    ):

        x = encode_board(
            board
        )

        x = x.unsqueeze(0).to(
            self.device
        )


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
        # BC opening prior
        #
        # Le prior est appliqué AVANT la température.
        # ====================================================

        legal_logits = (
            self._apply_bc_prior_single(
                board,
                legal_indices,
                legal_logits,
            )
        )


        # ====================================================
        # Entropie de la policy guidée
        #
        # AVANT température.
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
        # Température appliquée APRÈS le BC prior.
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0
        ):

            position = torch.argmax(
                legal_logits
            ).item()


            chosen_log_probs = (
                torch.log_softmax(
                    legal_logits,
                    dim=0,
                )
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


            sampling_log_probs = (
                torch.log_softmax(
                    sampling_logits,
                    dim=0,
                )
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

        action = legal_indices[
            position
        ]


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

            "ply":
                board.ply(),
        }


    # ========================================================
    # Batch moves
    # ========================================================

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
        # Forward RL
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


                board_moves[action] = (
                    move
                )


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


        legal_logits = (
            legal_logits.masked_fill(
                ~legal_mask,
                float("-inf"),
            )
        )


        # ====================================================
        # BC opening prior
        #
        # Seulement les positions encore dans la fenêtre
        # d'ouverture.
        # ====================================================

        if self.bc_model is not None:

            active_indices = [
                i
                for i, board in enumerate(boards)
                if self._opening_prior_strength(
                    board.ply()
                ) > 0.0
            ]


            if active_indices:

                active_boards = [
                    boards[i]
                    for i in active_indices
                ]


                active_x = encode_boards(
                    active_boards
                ).to(self.device)


                bc_policies, _ = (
                    self.bc_model(
                        active_x
                    )
                )


                for j, i in enumerate(
                    active_indices
                ):

                    n = len(
                        legal_indices[i]
                    )


                    if n == 0:

                        continue


                    indices = (
                        legal_index_tensor[
                            i,
                            :n,
                        ]
                    )


                    bc_legal_logits = (
                        bc_policies[j][
                            indices
                        ]
                    )


                    bc_log_probs = (
                        torch.log_softmax(
                            bc_legal_logits,
                            dim=0,
                        )
                    )


                    alpha = (
                        self._opening_prior_strength(
                            boards[i].ply()
                        )
                    )


                    legal_logits[
                        i,
                        :n,
                    ] = (
                        legal_logits[
                            i,
                            :n,
                        ]
                        +
                        alpha
                        * bc_log_probs
                    )


        # ====================================================
        # Entropie intrinsèque de la policy guidée
        #
        # AVANT température.
        #
        # Padding :
        # probability = 0
        # log_probability = 0
        # ====================================================

        entropy_log_probs = (
            torch.log_softmax(
                legal_logits,
                dim=1,
            )
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
        # Température appliquée APRÈS le prior.
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


            sampling_log_probs = (
                torch.log_softmax(
                    sampling_logits,
                    dim=1,
                )
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


            chosen_log_probs = (
                torch.log_softmax(
                    legal_logits,
                    dim=1,
                )
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


            action = (
                legal_index_tensor[
                    i,
                    position,
                ].item()
            )


            move = (
                legal_moves[i][
                    action
                ]
            )


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

                    "ply":
                        boards[
                            i
                        ].ply(),
                }
            )


        return results