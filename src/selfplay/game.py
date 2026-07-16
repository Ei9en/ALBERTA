# Game.py

import chess.variant

class SelfPlayGame:

    def __init__(self, white_agent, black_agent):

        self.white = white_agent
        self.black = black_agent


    def play(self):

        board = chess.variant.AtomicBoard()

        trajectory = []

        while not board.is_game_over():

            agent = (
                self.white
                if board.turn
                else self.black
            )

            info = agent.choose_move(board)

            trajectory.append({
                "fen": board.fen(),
                "action": info["action"],
                "player": board.turn,
                "value": info.get("value", 0.0),
                "legal_moves": [
                    move.uci()
                    for move in board.legal_moves
                ],
            })

            board.push(info["move"])

        return trajectory, board.result()