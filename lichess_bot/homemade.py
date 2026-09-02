from pathlib import Path
import chess
from chess.engine import PlayResult
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import HOMEMADE_ARGS_TYPE
import logging

from atomic_engine.rl_bot import RLBot
from atomic_engine.bc_bot_stochastic import BCBotStochastic


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExampleEngine(MinimalEngine):
    """Base class required by lichess-bot homemade mode."""
    pass


class AtomicRandom(ExampleEngine):
    """Bot Atomic utilisant RL."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.bot = RLBot(
            temperature=1,
        )


    def search(
        self,
        board: chess.Board,
        *args: HOMEMADE_ARGS_TYPE,
    ) -> PlayResult:

        info = self.bot.choose_move(board)

        move = info["move"]

        logger.info(
            f"RL joue : {move}"
        )

        return PlayResult(
            move,
            None,
        )