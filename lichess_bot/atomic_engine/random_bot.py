import random


class RandomBot:
    """Bot qui choisit un coup légal au hasard."""

    def choose_move(self, board):
        """
        Retourne un coup légal choisi aléatoirement.

        Parameters
        ----------
        board : chess.Board
            Position actuelle.

        Returns
        -------
        chess.Move
            Coup sélectionné.
        """
        return random.choice(list(board.legal_moves))