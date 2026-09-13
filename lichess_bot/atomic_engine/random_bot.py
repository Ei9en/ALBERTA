from __future__ import annotations

import random

import chess


class RandomBot:
    """
    Baseline bot that selects uniformly among legal moves.

    Randomness uses Python's global RNG.
    Reproducibility is controlled by the caller.
    """

    def choose_move(
        self,
        board: chess.Board,
    ) -> chess.Move:

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:
            raise RuntimeError(
                "choose_move() called on a position "
                "without legal moves."
            )

        return random.choice(
            legal_moves
        )