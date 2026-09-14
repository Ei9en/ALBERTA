from __future__ import annotations

from pathlib import Path

import chess
import chess.variant

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Assets
# ============================================================

HMI_ROOT = Path(__file__).resolve().parents[1]

PIECE_ASSET_DIR = (
    HMI_ROOT
    / "assets"
    / "pieces"
)

# ============================================================
# Promotion dialog
# ============================================================

class PromotionDialog(QDialog):
    """
    Promotion-piece chooser.
    """

    def __init__(
        self,
        parent=None,
    ) -> None:

        super().__init__(
            parent
        )

        self.choice: int | None = None

        self.setWindowTitle(
            "Choose promotion"
        )

        layout = QVBoxLayout(
            self
        )

        choices = [
            (
                "Queen",
                chess.QUEEN,
            ),
            (
                "Rook",
                chess.ROOK,
            ),
            (
                "Bishop",
                chess.BISHOP,
            ),
            (
                "Knight",
                chess.KNIGHT,
            ),
        ]

        for name, piece in choices:

            button = QPushButton(
                name
            )

            button.clicked.connect(
                lambda checked=False, p=piece: self.select(
                    p
                )
            )

            layout.addWidget(
                button
            )

    def select(
        self,
        piece: int,
    ) -> None:

        self.choice = piece

        self.accept()


# ============================================================
# Chessboard widget
# ============================================================

class ChessBoardWidget(QWidget):
    """
    Interactive Atomic Chess board used by the ALBERTA HMI.

    A selected move is previewed locally before being emitted.
    The underlying reference position remains unchanged until
    the HMI loads another query.
    """

    move_selected = pyqtSignal(
        str
    )

    BOARD_SIZE = 520

    def __init__(
        self,
        parent=None,
    ) -> None:

        super().__init__(
            parent
        )

        # ----------------------------------------------------
        # Fixed board geometry
        #
        # Keeping the widget fixed prevents Qt layout
        # recalculation in the right-side annotation panel
        # from slightly resizing the graphical chessboard.
        # ----------------------------------------------------

        self.setFixedSize(
            QSize(
                self.BOARD_SIZE,
                self.BOARD_SIZE,
            )
        )

        self.square_size = (
            self.BOARD_SIZE
            // 8
        )

        # ----------------------------------------------------
        # Position state
        # ----------------------------------------------------

        self.board = (
            chess.variant.AtomicBoard()
        )

        # Preview only. The reference board is not modified
        # when the user selects a candidate move.
        self.preview_board: (
            chess.variant.AtomicBoard
            | None
        ) = None

        self.pending_move: (
            chess.Move
            | None
        ) = None

        # ----------------------------------------------------
        # Selection state
        # ----------------------------------------------------

        self.selected_square: (
            chess.Square
            | None
        ) = None

        self.highlight_squares: list[
            chess.Square
        ] = []

        # ----------------------------------------------------
        # Display state
        # ----------------------------------------------------

        self.flipped = False

        self.piece_images: dict[
            str,
            QSvgRenderer,
        ] = {}

        self.load_piece_images()

    # ========================================================
    # Assets
    # ========================================================

    def load_piece_images(
        self,
    ) -> None:

        piece_names = [
            "wK",
            "wQ",
            "wR",
            "wB",
            "wN",
            "wP",
            "bK",
            "bQ",
            "bR",
            "bB",
            "bN",
            "bP",
        ]

        for name in piece_names:

            path = (
                PIECE_ASSET_DIR
                / f"{name}.svg"
            )

            if path.exists():

                self.piece_images[
                    name
                ] = QSvgRenderer(
                    str(
                        path
                    )
                )

    # ========================================================
    # Public API
    # ========================================================

    def set_fen(
        self,
        fen: str,
    ) -> None:

        self.board = (
            chess.variant.AtomicBoard(
                fen
            )
        )

        self.flipped = (
            self.board.turn
            == chess.BLACK
        )

        self.preview_board = None
        self.pending_move = None

        self.reset_selection()

    def san_from_uci(
        self,
        uci: str,
    ) -> str:

        move = chess.Move.from_uci(
            uci
        )

        return self.board.san(
            move
        )

    def take_back(
        self,
    ) -> None:

        self.pending_move = None
        self.preview_board = None

        self.reset_selection()

    def has_pending_move(
        self,
    ) -> bool:

        return (
            self.pending_move
            is not None
        )

    def reset_selection(
        self,
    ) -> None:

        self.selected_square = None
        self.highlight_squares = []

        self.update()

    # ========================================================
    # Preview
    # ========================================================

    def compute_preview(
        self,
        move: chess.Move,
    ) -> None:

        self.preview_board = (
            chess.variant.AtomicBoard(
                self.board.fen()
            )
        )

        self.preview_board.push(
            move
        )

    # ========================================================
    # Coordinate conversion
    # ========================================================

    def screen_to_square(
        self,
        file_index: int,
        rank_index: int,
    ) -> chess.Square:

        if self.flipped:

            return chess.square(
                7 - file_index,
                rank_index,
            )

        return chess.square(
            file_index,
            7 - rank_index,
        )

    # ========================================================
    # Rendering
    # ========================================================

    def resizeEvent(
        self,
        event,
    ) -> None:

        self.square_size = (
            min(
                self.width(),
                self.height(),
            )
            // 8
        )

        super().resizeEvent(
            event
        )

    def paintEvent(
        self,
        _event,
    ) -> None:

        painter = QPainter(
            self
        )

        size = self.square_size

        if size <= 0:
            return

        board = (
            self.preview_board
            if self.preview_board is not None
            else self.board
        )

        light = QColor(
            "#f0d9b5"
        )

        dark = QColor(
            "#b58863"
        )

        piece_symbols = {
            chess.KING:
                "K",

            chess.QUEEN:
                "Q",

            chess.ROOK:
                "R",

            chess.BISHOP:
                "B",

            chess.KNIGHT:
                "N",

            chess.PAWN:
                "P",
        }

        # ----------------------------------------------------
        # Squares
        # ----------------------------------------------------

        for rank_index in range(
            8
        ):

            for file_index in range(
                8
            ):

                square = self.screen_to_square(
                    file_index,
                    rank_index,
                )

                x = (
                    file_index
                    * size
                )

                y = (
                    rank_index
                    * size
                )

                painter.fillRect(
                    x,
                    y,
                    size,
                    size,
                    (
                        light
                        if (
                            file_index
                            + rank_index
                        )
                        % 2
                        == 0
                        else dark
                    ),
                )

                # ------------------------------------------------
                # Selected origin square
                # ------------------------------------------------

                if square == self.selected_square:

                    painter.fillRect(
                        x,
                        y,
                        size,
                        size,
                        QColor(
                            240,
                            220,
                            70,
                            130,
                        ),
                    )

                # ------------------------------------------------
                # Legal destination highlights
                # ------------------------------------------------

                if square in self.highlight_squares:

                    occupied = (
                        self.board.piece_at(
                            square
                        )
                        is not None
                    )

                    painter.setPen(
                        QColor(
                            40,
                            120,
                            230,
                        )
                    )

                    if occupied:

                        painter.setBrush(
                            Qt.BrushStyle.NoBrush
                        )

                        painter.drawRect(
                            x + 4,
                            y + 4,
                            size - 8,
                            size - 8,
                        )

                    else:

                        painter.setBrush(
                            QColor(
                                40,
                                120,
                                230,
                                190,
                            )
                        )

                        radius = int(
                            size
                            * 0.10
                        )

                        painter.drawEllipse(
                            int(
                                x
                                + size / 2
                                - radius
                            ),
                            int(
                                y
                                + size / 2
                                - radius
                            ),
                            radius * 2,
                            radius * 2,
                        )

                # ------------------------------------------------
                # Piece rendering
                # ------------------------------------------------

                piece = board.piece_at(
                    square
                )

                if piece is None:
                    continue

                color = (
                    "w"
                    if piece.color
                    else "b"
                )

                symbol = piece_symbols[
                    piece.piece_type
                ]

                renderer = self.piece_images.get(
                    color
                    + symbol
                )

                if renderer is None:
                    continue

                renderer.render(
                    painter,
                    QRectF(
                        x + 2,
                        y + 2,
                        size - 4,
                        size - 4,
                    ),
                )

        # ====================================================
        # Coordinates
        # ====================================================

        painter.setPen(
            QColor(
                "black"
            )
        )

        painter.setFont(
            QFont(
                "Arial",
                int(
                    size
                    * 0.15
                ),
            )
        )

        for i in range(
            8
        ):

            if self.flipped:

                file_label = chr(
                    ord(
                        "h"
                    )
                    - i
                )

                rank_label = str(
                    i + 1
                )

            else:

                file_label = chr(
                    ord(
                        "a"
                    )
                    + i
                )

                rank_label = str(
                    8 - i
                )

            painter.drawText(
                int(
                    i
                    * size
                    + 4
                ),
                int(
                    8
                    * size
                    - 5
                ),
                file_label,
            )

            painter.drawText(
                4,
                int(
                    i
                    * size
                    + 15
                ),
                rank_label,
            )

    # ========================================================
    # Mouse interaction
    # ========================================================

    def mousePressEvent(
        self,
        event,
    ) -> None:

        if (
            event.button()
            != Qt.MouseButton.LeftButton
        ):
            return

        # Only one previewed move at a time.
        if self.pending_move is not None:
            return

        size = self.square_size

        if size <= 0:
            return

        file_index = int(
            event.position().x()
            // size
        )

        rank_index = int(
            event.position().y()
            // size
        )

        if not (
            0
            <= file_index
            < 8
            and 0
            <= rank_index
            < 8
        ):
            return

        square = self.screen_to_square(
            file_index,
            rank_index,
        )

        # ====================================================
        # Select origin square
        # ====================================================

        if self.selected_square is None:

            piece = self.board.piece_at(
                square
            )

            if (
                piece is not None
                and piece.color
                == self.board.turn
            ):

                self.selected_square = (
                    square
                )

                self.highlight_squares = [
                    move.to_square
                    for move in self.board.legal_moves
                    if move.from_square
                    == square
                ]

                self.update()

            return

        # ====================================================
        # Select destination square
        # ====================================================

        candidates = [
            move
            for move in self.board.legal_moves
            if (
                move.from_square
                == self.selected_square
                and move.to_square
                == square
            )
        ]

        if not candidates:

            self.reset_selection()

            return

        # ====================================================
        # Promotion
        # ====================================================

        if len(
            candidates
        ) > 1:

            dialog = PromotionDialog(
                self
            )

            if (
                dialog.exec()
                != QDialog.DialogCode.Accepted
            ):

                self.reset_selection()

                return

            candidates = [
                move
                for move in candidates
                if move.promotion
                == dialog.choice
            ]

            if not candidates:

                self.reset_selection()

                return

        move = candidates[
            0
        ]

        # ====================================================
        # Preview and emit
        # ====================================================

        self.pending_move = move

        self.compute_preview(
            move
        )

        self.move_selected.emit(
            move.uci()
        )

        self.reset_selection()