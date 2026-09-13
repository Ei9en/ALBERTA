from __future__ import annotations

from collections.abc import Sequence

import chess
import chess.variant
import torch
from torch import nn

from src.actions_space import ACTION_TO_INDEX
from src.encoding import encode_board, encode_boards


class PPOAgent:
    """
    Actor-critic inference agent used during PPO self-play.

    Supports:
        - deterministic greedy move selection;
        - stochastic temperature-based sampling;
        - optional BC-guided opening prior;
        - single-position inference;
        - batched inference.

    Random sampling uses PyTorch's global RNG. Reproducibility
    is controlled by the caller, typically training/train_rl.py.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        temperature: float = 0.75,
        bc_model: nn.Module | None = None,
        opening_prior_strength: float = 1.0,
        opening_prior_plies: int = 6,
    ) -> None:

        if temperature < 0.0:
            raise ValueError(
                "temperature cannot be negative."
            )

        if opening_prior_strength < 0.0:
            raise ValueError(
                "opening_prior_strength cannot be negative."
            )

        if opening_prior_plies < 0:
            raise ValueError(
                "opening_prior_plies cannot be negative."
            )

        self.device = torch.device(
            device
        )

        self.model = model.to(
            self.device
        )

        self.bc_model = (
            bc_model.to(self.device)
            if bc_model is not None
            else None
        )

        self.deterministic = (
            deterministic
        )

        self.temperature = (
            temperature
        )

        self.opening_prior_strength = (
            opening_prior_strength
        )

        self.opening_prior_plies = (
            opening_prior_plies
        )

        self.model.eval()

        if self.bc_model is not None:
            self.bc_model.eval()


    # ========================================================
    # BC opening prior strength
    # ========================================================

    def _opening_prior_strength(
        self,
        ply: int,
    ) -> float:
        """
        Return the BC-prior coefficient at the requested ply.

        The prior decays linearly:

            ply 0 -> 100 %
            ...
            ply opening_prior_plies -> 0 %
        """

        if (
            self.bc_model is None
            or self.opening_prior_plies <= 0
        ):
            return 0.0

        if ply >= self.opening_prior_plies:
            return 0.0

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
        board: chess.variant.AtomicBoard,
        legal_indices: list[int],
        legal_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the BC opening prior to legal RL logits.

        The guided policy is defined by:

            z_guided
                = z_RL
                + alpha * log(pi_BC)

        The BC prior biases legal-move preferences but never
        makes a legal move impossible.

        bc_model is expected to follow the ActorCritic API:

            policy_logits, value = bc_model(x)
        """

        alpha = (
            self._opening_prior_strength(
                board.ply()
            )
        )

        if (
            alpha <= 0.0
            or self.bc_model is None
        ):
            return legal_logits

        x = encode_board(
            board
        ).unsqueeze(0).to(
            self.device
        )

        bc_policy, _ = self.bc_model(
            x
        )

        bc_logits = (
            bc_policy[0]
        )

        bc_legal_logits = (
            bc_logits[
                legal_indices
            ]
        )

        bc_log_probs = (
            torch.log_softmax(
                bc_legal_logits,
                dim=0,
            )
        )

        return (
            legal_logits
            + alpha
            * bc_log_probs
        )


    # ========================================================
    # Single move
    # ========================================================

    @torch.no_grad()
    def choose_move(
        self,
        board: chess.variant.AtomicBoard,
    ) -> dict:
        """
        Select one move for a single Atomic Chess position.

        Entropy is computed from the BC-guided legal policy
        before temperature scaling.

        In stochastic mode, log_prob corresponds to the actual
        temperature-scaled sampling distribution.
        """

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:
            raise RuntimeError(
                "choose_move() called on a position "
                "without legal moves."
            )

        x = encode_board(
            board
        ).unsqueeze(0).to(
            self.device
        )

        policy, value = self.model(
            x
        )

        logits = (
            policy[0]
        )

        # ----------------------------------------------------
        # Legal action indices
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
        # BC opening prior
        #
        # Applied before temperature scaling.
        # ----------------------------------------------------

        legal_logits = (
            self._apply_bc_prior_single(
                board=board,
                legal_indices=legal_indices,
                legal_logits=legal_logits,
            )
        )

        # ----------------------------------------------------
        # Entropy of guided policy before temperature
        # ----------------------------------------------------

        entropy_log_probs = (
            torch.log_softmax(
                legal_logits,
                dim=0,
            )
        )

        entropy_probs = torch.exp(
            entropy_log_probs
        )

        entropy = -(
            entropy_probs
            * entropy_log_probs
        ).sum().item()

        # ----------------------------------------------------
        # Move-selection distribution
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

            chosen_log_probs = (
                torch.log_softmax(
                    legal_logits,
                    dim=0,
                )
            )

        else:

            sampling_logits = (
                legal_logits
                / self.temperature
            )

            chosen_log_probs = (
                torch.log_softmax(
                    sampling_logits,
                    dim=0,
                )
            )

            sampling_probs = torch.exp(
                chosen_log_probs
            )

            position = (
                torch.multinomial(
                    sampling_probs,
                    num_samples=1,
                ).item()
            )

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

            "log_prob":
                chosen_log_probs[
                    position
                ].item(),

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
        boards: Sequence[
            chess.variant.AtomicBoard
        ],
    ) -> list[dict]:
        """
        Select one action for each position in a batch.

        The mathematical semantics are identical to choose_move,
        while neural-network forward passes are batched.
        """

        if len(boards) == 0:
            return []

        batch_size = len(
            boards
        )

        # ====================================================
        # Legal moves
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
        # Batched RL forward
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
        # Legal RL logits
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
        # ====================================================

        if self.bc_model is not None:

            active_indices = [
                i
                for i, board in enumerate(
                    boards
                )
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
                ).to(
                    self.device
                )

                bc_policies, _ = (
                    self.bc_model(
                        active_x
                    )
                )

                for j, i in enumerate(
                    active_indices
                ):

                    n = len(
                        legal_indices[
                            i
                        ]
                    )

                    indices = (
                        legal_index_tensor[
                            i,
                            :n,
                        ]
                    )

                    bc_legal_logits = (
                        bc_policies[
                            j,
                            indices,
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
                            boards[
                                i
                            ].ply()
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
                        + alpha
                        * bc_log_probs
                    )

        # ====================================================
        # Guided-policy entropy before temperature scaling
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
        # Selection distribution
        # ====================================================

        if (
            self.deterministic
            or self.temperature <= 0.0
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

            sampling_logits = (
                legal_logits
                / self.temperature
            )

            chosen_log_probs = (
                torch.log_softmax(
                    sampling_logits,
                    dim=1,
                )
            )

            sampling_probs = torch.exp(
                chosen_log_probs
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
        # Convert back to chess moves
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