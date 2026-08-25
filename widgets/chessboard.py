from __future__ import annotations

import chess
import chess.variant

from pathlib import Path

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QPushButton,
)


class PromotionDialog(QDialog):
    """
    Promotion choice.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.choice = None

        self.setWindowTitle(
            "Choose promotion"
        )

        layout = QVBoxLayout(
            self
        )


        choices = [
            ("Queen", chess.QUEEN),
            ("Rook", chess.ROOK),
            ("Bishop", chess.BISHOP),
            ("Knight", chess.KNIGHT),
        ]


        for name, piece in choices:

            button = QPushButton(
                name
            )

            button.clicked.connect(
                lambda checked=False, p=piece: self.select(p)
            )

            layout.addWidget(
                button
            )



    def select(self, piece):

        self.choice = piece

        self.accept()



class ChessBoardWidget(QWidget):

    move_selected = pyqtSignal(str)


    def __init__(
        self,
        parent=None,
    ):

        super().__init__(parent)


        self.setMinimumSize(
            QSize(520,520)
        )


        self.board = chess.variant.AtomicBoard()


        # preview only

        self.preview_board = None

        self.pending_move = None



        # selection

        self.selected_square = None

        self.highlight_squares = []



        self.square_size = 0

        self.flipped = False



        self.piece_images = {}

        self.load_piece_images()



    # =====================================================
    # Assets
    # =====================================================

    def load_piece_images(self):

        base = Path(
            "assets/pieces"
        )


        for name in [
            "wK","wQ","wR","wB","wN","wP",
            "bK","bQ","bR","bB","bN","bP",
        ]:

            path = base / f"{name}.svg"


            if path.exists():

                self.piece_images[name] = QSvgRenderer(
                    str(path)
                )



    # =====================================================
    # API
    # =====================================================

    def set_fen(
        self,
        fen: str,
    ):

        self.board = chess.variant.AtomicBoard(
            fen
        )


        self.flipped = (
            self.board.turn == chess.BLACK
        )


        self.preview_board = None

        self.pending_move = None


        self.reset_selection()



    def san_from_uci(
        self,
        uci: str,
    ):

        move = chess.Move.from_uci(
            uci
        )

        return self.board.san(
            move
        )



    def take_back(self):

        self.pending_move = None

        self.preview_board = None

        self.reset_selection()



    def has_pending_move(self):

        return self.pending_move is not None



    def reset_selection(self):

        self.selected_square = None

        self.highlight_squares = []

        self.update()



    # =====================================================
    # Preview
    # =====================================================

    def compute_preview(
        self,
        move,
    ):

        self.preview_board = chess.variant.AtomicBoard(
            self.board.fen()
        )

        self.preview_board.push(
            move
        )



    # =====================================================
    # Coordinates
    # =====================================================

    def screen_to_square(
        self,
        file,
        rank,
    ):

        if self.flipped:

            return chess.square(
                7-file,
                rank,
            )

        return chess.square(
            file,
            7-rank,
        )



    # =====================================================
    # Rendering
    # =====================================================

    def resizeEvent(self,event):

        self.square_size = min(
            self.width(),
            self.height(),
        ) // 8



    def paintEvent(self,_event):

        painter = QPainter(
            self
        )


        size = self.square_size


        if size <= 0:
            return


        board = (
            self.preview_board
            if self.preview_board
            else self.board
        )


        light = QColor("#f0d9b5")
        dark = QColor("#b58863")



        for rank in range(8):

            for file in range(8):

                square = self.screen_to_square(
                    file,
                    rank,
                )


                x = file * size

                y = rank * size



                painter.fillRect(
                    x,
                    y,
                    size,
                    size,
                    light if (file+rank)%2==0 else dark
                )



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
                            130
                        )
                    )



                if square in self.highlight_squares:

                    occupied = (
                        self.board.piece_at(square)
                        is not None
                    )


                    painter.setPen(
                        QColor(
                            40,
                            120,
                            230
                        )
                    )


                    if occupied:

                        painter.setBrush(
                            Qt.BrushStyle.NoBrush
                        )

                        painter.drawRect(
                            x+4,
                            y+4,
                            size-8,
                            size-8,
                        )

                    else:

                        painter.setBrush(
                            QColor(
                                40,
                                120,
                                230,
                                190
                            )
                        )

                        radius = int(
                            size*0.10
                        )

                        painter.drawEllipse(
                            int(x+size/2-radius),
                            int(y+size/2-radius),
                            radius*2,
                            radius*2,
                        )



                piece = board.piece_at(
                    square
                )


                if piece:

                    color = (
                        "w"
                        if piece.color
                        else "b"
                    )


                    symbols = {
                        chess.KING:"K",
                        chess.QUEEN:"Q",
                        chess.ROOK:"R",
                        chess.BISHOP:"B",
                        chess.KNIGHT:"N",
                        chess.PAWN:"P",
                    }


                    renderer = self.piece_images.get(
                        color + symbols[piece.piece_type]
                    )


                    if renderer:

                        renderer.render(
                            painter,
                            QRectF(
                                x+2,
                                y+2,
                                size-4,
                                size-4,
                            )
                        )



        # coordinates

        painter.setPen(
            QColor("black")
        )

        painter.setFont(
            QFont(
                "Arial",
                int(size*0.15)
            )
        )


        for i in range(8):

            if self.flipped:

                f = chr(
                    ord("h")-i
                )

                r = str(
                    i+1
                )

            else:

                f = chr(
                    ord("a")+i
                )

                r = str(
                    8-i
                )


            painter.drawText(
                int(i*size+4),
                int(8*size-5),
                f,
            )


            painter.drawText(
                4,
                int(i*size+15),
                r,
            )



    # =====================================================
    # Mouse
    # =====================================================

    def mousePressEvent(
        self,
        event,
    ):

        if event.button() != Qt.MouseButton.LeftButton:
            return


        if self.pending_move:
            return



        size = self.square_size


        file = int(
            event.position().x() // size
        )

        rank = int(
            event.position().y() // size
        )


        if not (
            0 <= file < 8
            and
            0 <= rank < 8
        ):
            return



        square = self.screen_to_square(
            file,
            rank,
        )



        if self.selected_square is None:


            piece = self.board.piece_at(
                square
            )


            if (
                piece
                and
                piece.color == self.board.turn
            ):

                self.selected_square = square


                self.highlight_squares = [

                    move.to_square

                    for move in self.board.legal_moves

                    if move.from_square == square

                ]


                self.update()


            return



        candidates = [

            move

            for move in self.board.legal_moves

            if (
                move.from_square == self.selected_square
                and
                move.to_square == square
            )

        ]



        if not candidates:

            self.reset_selection()

            return



        # Promotion

        if len(candidates) > 1:

            dialog = PromotionDialog(
                self
            )


            if dialog.exec() != QDialog.DialogCode.Accepted:

                self.reset_selection()

                return


            candidates = [

                m

                for m in candidates

                if m.promotion == dialog.choice

            ]



        move = candidates[0]


        self.pending_move = move


        self.compute_preview(
            move
        )


        self.move_selected.emit(
            move.uci()
        )


        self.reset_selection()