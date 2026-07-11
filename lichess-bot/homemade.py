import chess
from chess.engine import PlayResult
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import HOMEMADE_ARGS_TYPE
import logging

from atomic_engine import RandomBot

logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """Base class required by lichess-bot homemade mode."""
    pass


class AtomicRandom(ExampleEngine):
    """Bot Atomic utilisant notre moteur Python."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = RandomBot()

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        move = self.bot.choose_move(board)
        logger.info(f"AtomicRandom joue : {move}")
        return PlayResult(move, None)