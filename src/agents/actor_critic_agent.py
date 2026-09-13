from __future__ import annotations

from collections.abc import Sequence

import chess
import chess.variant
import torch
from torch import nn

from src.actions_space import ACTION_TO_INDEX
from src.encoding import encode_board, encode_boards


class ActorCriticAgent:
    """
    Inference wrapper for an ALBERTA ActorCritic model.

    Supports:
        - deterministic greedy move selection;
        - stochastic temperature-based sampling;
        - single-position inference;
        - batched inference.

    Unlike PPOAgent, this agent does not apply a BC opening prior.

    Random sampling uses PyTorch's global RNG. The calling script
    is responsible for setting the random seed when reproducible
    stochastic evaluation is required.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        temperature: float = 0.75,
    ) -> None:

        if temperature < 0.0:
            raise ValueError(
                "temperature cannot be negative."
            )

        self.device = torch.device(
            device
        )

        self.model = model.to(
            self.device
        )

        self.deterministic = (
            deterministic
        )

        self.temperature = (
            temperature
        )

        self.model.eval()


    # ========================================================
    # Single move
    # ========================================================

    @torch.no_grad()
    def choose_move(
        self,
        board: chess.variant.AtomicBoard,
    ) -> dict:
        """
        Select one legal move for a single Atomic Chess state.

        Entropy is computed from the legal policy before
        temperature scaling.
        """

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:
            raise RuntimeError(
                "choose_move() called on a position "
                "without legal moves."
            )

        # ----------------------------------------------------
        # Encode position
        # ----------------------------------------------------

        x = encode_board(
            board
        ).unsqueeze(0).to(
            self.device
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        policy, value = self.model(
            x
        )

        logits = (
            policy[0]
        )

        # ----------------------------------------------------
        # Legal actions
        # ----------------------------------------------------

        legal_indices = [
            ACTION_TO_INDEX[
                move.uci()
            ]
            for move in legal_moves
        ]

        legal_logits = (
            logits[
                legal_indices
            ]
        )

        # ----------------------------------------------------
        # Entropy before temperature scaling
        # ----------------------------------------------------

        log_probs = torch.log_softmax(
            legal_logits,
            dim=0,
        )

        probs = torch.exp(
            log_probs
        )

        entropy = -(
            probs
            * log_probs
        ).sum().item()

        # ----------------------------------------------------
        # Move selection
        # ----------------------------------------------------

        if (
            self.deterministic
            or self.temperature <= 0.0
        ):

            position = (
                torch.argmax(
                    legal_logits
                ).item()
            )

        else:

            sampling_logits = (
                legal_logits
                / self.temperature
            )

            sampling_probs = (
                torch.softmax(
                    sampling_logits,
                    dim=0,
                )
            )

            position = (
                torch.multinomial(
                    sampling_probs,
                    num_samples=1,
                ).item()
            )

        # ----------------------------------------------------
        # Legal-position index -> global action / move
        # ----------------------------------------------------

        action = (
            legal_indices[
                position
            ]
        )

        move = (
            legal_moves[
                position
            ]
        )

        return {
            "move":
                move,

            "action":
                action,

            "value":
                value.item(),

            "entropy":
                entropy,

            "fen":
                board.fen(),
        }


    # ========================================================
    # Batched moves
    # ========================================================

    @torch.no_grad()
    def choose_moves(
        self,
        boards: Sequence[
            chess.variant.AtomicBoard
        ],
    ) -> list[dict]:
        """
        Select one legal move for each board in a batch.

        The policy semantics are identical to choose_move(), while
        neural-network inference is batched.
        """

        if len(boards) == 0:
            return []

        batch_size = len(
            boards
        )

        # ====================================================
        # Legal actions
        # ====================================================

        board_legal_moves: list[
            list[chess.Move]
        ] = []

        legal_indices: list[
            list[int]
        ] = []

        for board in boards:

            moves = list(
                board.legal_moves
            )

            if not moves:
                raise RuntimeError(
                    "choose_moves() received a position "
                    "without legal moves."
                )

            indices = [
                ACTION_TO_INDEX[
                    move.uci()
                ]
                for move in moves
            ]

            board_legal_moves.append(
                moves
            )

            legal_indices.append(
                indices
            )

        max_legal_moves = max(
            len(indices)
            for indices in legal_indices
        )

        # ====================================================
        # Batched forward pass
        # ====================================================

        x = encode_boards(
            boards
        ).to(
            self.device
        )

        policies, values = self.model(
            x
        )

        # ====================================================
        # Padded legal-action representation
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

            n = len(
                indices
            )

            legal_index_tensor[
                i,
                :n,
            ] = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=self.device,
            )

            legal_mask[
                i,
                :n,
            ] = True

        # ====================================================
        # Legal logits only
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
        # Entropy before temperature scaling
        #
        # Padding log-probabilities are replaced by zero before
        # multiplication to avoid:
        #
        #     0 * (-inf) -> NaN
        # ====================================================

        log_probs = torch.log_softmax(
            legal_logits,
            dim=1,
        )

        probs = torch.exp(
            log_probs
        )

        safe_log_probs = (
            log_probs.masked_fill(
                ~legal_mask,
                0.0,
            )
        )

        entropy = -(
            probs
            * safe_log_probs
        ).sum(
            dim=1
        )

        # ====================================================
        # Move selection
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0.0
        ):

            positions = torch.argmax(
                legal_logits,
                dim=1,
            )

        else:

            sampling_logits = (
                legal_logits
                / self.temperature
            )

            sampling_probs = (
                torch.softmax(
                    sampling_logits,
                    dim=1,
                )
            )

            positions = (
                torch.multinomial(
                    sampling_probs,
                    num_samples=1,
                ).squeeze(
                    1
                )
            )

        # ====================================================
        # Convert legal position -> global action -> Move
        # ====================================================

        results = []

        for i in range(
            batch_size
        ):

            position = (
                positions[
                    i
                ].item()
            )

            action = (
                legal_index_tensor[
                    i,
                    position,
                ].item()
            )

            move = (
                board_legal_moves[
                    i
                ][
                    position
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

                    "fen":
                        boards[
                            i
                        ].fen(),
                }
            )

        return results