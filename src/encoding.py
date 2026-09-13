from __future__ import annotations

from collections.abc import Sequence

import chess
import chess.variant
import torch


# ============================================================
# Encoding specification
# ============================================================

NUM_PLANES = 19

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


# ============================================================
# Single-board encoding
# ============================================================

def encode_board(
    board: chess.variant.AtomicBoard,
) -> torch.Tensor:
    """
    Encode an Atomic Chess position as a 19x8x8 tensor.

    Plane layout
    ------------
    0-5:
        White pieces P, N, B, R, Q, K.

    6-11:
        Black pieces p, n, b, r, q, k.

    12:
        Side to move. Filled with 1 when White is to move,
        otherwise 0.

    13:
        White kingside castling right.

    14:
        White queenside castling right.

    15:
        Black kingside castling right.

    16:
        Black queenside castling right.

    17:
        En-passant target square.

    18:
        Halfmove clock normalized by 100.

    Returns
    -------
    torch.Tensor
        Float32 tensor with shape (19, 8, 8).
    """

    planes = torch.zeros(
        (NUM_PLANES, 8, 8),
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Pieces
    # --------------------------------------------------------

    for square, piece in board.piece_map().items():

        channel = (
            PIECE_TO_CHANNEL[
                piece.symbol()
            ]
        )

        row = (
            7
            - chess.square_rank(
                square
            )
        )

        col = chess.square_file(
            square
        )

        planes[
            channel,
            row,
            col,
        ] = 1.0

    # --------------------------------------------------------
    # Side to move
    # --------------------------------------------------------

    if board.turn == chess.WHITE:
        planes[12].fill_(1.0)

    # --------------------------------------------------------
    # Castling rights
    # --------------------------------------------------------

    if board.has_kingside_castling_rights(
        chess.WHITE
    ):
        planes[13].fill_(1.0)

    if board.has_queenside_castling_rights(
        chess.WHITE
    ):
        planes[14].fill_(1.0)

    if board.has_kingside_castling_rights(
        chess.BLACK
    ):
        planes[15].fill_(1.0)

    if board.has_queenside_castling_rights(
        chess.BLACK
    ):
        planes[16].fill_(1.0)

    # --------------------------------------------------------
    # En passant
    # --------------------------------------------------------

    if board.ep_square is not None:

        row = (
            7
            - chess.square_rank(
                board.ep_square
            )
        )

        col = chess.square_file(
            board.ep_square
        )

        planes[
            17,
            row,
            col,
        ] = 1.0

    # --------------------------------------------------------
    # Halfmove clock
    # --------------------------------------------------------

    planes[18].fill_(
        board.halfmove_clock / 100.0
    )

    return planes


# ============================================================
# Batched encoding
# ============================================================

def encode_boards(
    boards: Sequence[chess.variant.AtomicBoard],
) -> torch.Tensor:
    """
    Encode a sequence of Atomic Chess positions.

    Returns
    -------
    torch.Tensor
        Float32 tensor with shape
        (batch_size, 19, 8, 8).
    """

    batch_size = len(
        boards
    )

    planes = torch.zeros(
        (
            batch_size,
            NUM_PLANES,
            8,
            8,
        ),
        dtype=torch.float32,
    )

    for batch_idx, board in enumerate(
        boards
    ):

        # ----------------------------------------------------
        # Pieces
        # ----------------------------------------------------

        for square, piece in board.piece_map().items():

            channel = (
                PIECE_TO_CHANNEL[
                    piece.symbol()
                ]
            )

            row = (
                7
                - chess.square_rank(
                    square
                )
            )

            col = chess.square_file(
                square
            )

            planes[
                batch_idx,
                channel,
                row,
                col,
            ] = 1.0

        # ----------------------------------------------------
        # Side to move
        # ----------------------------------------------------

        if board.turn == chess.WHITE:

            planes[
                batch_idx,
                12,
            ].fill_(1.0)

        # ----------------------------------------------------
        # Castling rights
        # ----------------------------------------------------

        if board.has_kingside_castling_rights(
            chess.WHITE
        ):

            planes[
                batch_idx,
                13,
            ].fill_(1.0)

        if board.has_queenside_castling_rights(
            chess.WHITE
        ):

            planes[
                batch_idx,
                14,
            ].fill_(1.0)

        if board.has_kingside_castling_rights(
            chess.BLACK
        ):

            planes[
                batch_idx,
                15,
            ].fill_(1.0)

        if board.has_queenside_castling_rights(
            chess.BLACK
        ):

            planes[
                batch_idx,
                16,
            ].fill_(1.0)

        # ----------------------------------------------------
        # En passant
        # ----------------------------------------------------

        if board.ep_square is not None:

            row = (
                7
                - chess.square_rank(
                    board.ep_square
                )
            )

            col = chess.square_file(
                board.ep_square
            )

            planes[
                batch_idx,
                17,
                row,
                col,
            ] = 1.0

        # ----------------------------------------------------
        # Halfmove clock
        # ----------------------------------------------------

        planes[
            batch_idx,
            18,
        ].fill_(
            board.halfmove_clock / 100.0
        )

    return planes


# ============================================================
# FEN encoding
# ============================================================

def encode_fen(
    fen: str,
) -> torch.Tensor:
    """
    Create an AtomicBoard from a FEN string and encode it.
    """

    board = chess.variant.AtomicBoard(
        fen
    )

    return encode_board(
        board
    )