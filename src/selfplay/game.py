from __future__ import annotations

from typing import Any

import chess.variant


class SelfPlayGame:
    """
    Run one complete Atomic Chess game between two agents.

    Each agent must implement:

        choose_move(board) -> dict

    with at least:
        - "move"
        - "action"

    Optional fields:
        - "value"
        - "entropy"
    """

    def __init__(
        self,
        white_agent,
        black_agent,
    ) -> None:

        self.white = white_agent
        self.black = black_agent


    def play(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        str,
    ]:

        board = chess.variant.AtomicBoard()

        trajectory = []

        while not board.is_game_over():

            agent = (
                self.white
                if board.turn
                else self.black
            )

            info = agent.choose_move(
                board
            )

            trajectory.append(
                {
                    "fen":
                        board.fen(),

                    "action":
                        info["action"],

                    "player":
                        board.turn,

                    "value":
                        info.get(
                            "value",
                            0.0,
                        ),

                    "entropy":
                        info.get(
                            "entropy",
                            0.0,
                        ),

                    "legal_moves":
                        [
                            move.uci()
                            for move in board.legal_moves
                        ],
                }
            )

            board.push(
                info["move"]
            )

        return (
            trajectory,
            board.result(),
        )