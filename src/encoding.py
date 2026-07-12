import chess
import chess.variant
import torch

PIECE_TO_CHANNEL = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def encode_fen(fen: str) -> torch.Tensor:

    board = chess.variant.AtomicBoard(fen)

    planes = torch.zeros((19, 8, 8), dtype=torch.float32)

    #
    # Pieces
    #
    for square, piece in board.piece_map().items():

        channel = PIECE_TO_CHANNEL[piece.symbol()]

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        planes[channel, row, col] = 1.0

    #
    # Side to move
    #
    if board.turn:
        planes[12].fill_(1.0)

    #
    # Castling
    #
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13].fill_(1.0)

    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14].fill_(1.0)

    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15].fill_(1.0)

    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16].fill_(1.0)

    #
    # En passant
    #
    if board.ep_square is not None:

        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)

        planes[17, row, col] = 1.0

    #
    # Halfmove clock
    #
    planes[18].fill_(board.halfmove_clock / 100.0)

    return planes