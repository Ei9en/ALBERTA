from __future__ import annotations

import logging

import chess
from chess.engine import PlayResult

from atomic_engine.rl_bot import RLBot
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import HOMEMADE_ARGS_TYPE


logger = logging.getLogger(
    __name__
)


class ExampleEngine(MinimalEngine):
    """
    Base class required by lichess-bot homemade mode.
    """

    pass


class AtomicRL(ExampleEngine):
    """
    ALBERTA Atomic Chess engine for lichess-bot.

    Move selection is delegated to RLBot.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(
            *args,
            **kwargs,
        )

        self.bot = RLBot(
            temperature=2.0,
        )

    def search(
        self,
        board: chess.Board,
        *args: HOMEMADE_ARGS_TYPE,
    ) -> PlayResult:

        info = self.bot.choose_move(
            board
        )

        move = info[
            "move"
        ]

        logger.info(
            "RL move: %s",
            move,
        )

        return PlayResult(
            move,
            None,
        )