from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
)

from PyQt6.QtCore import Qt

from oracle_queue import OracleQueue

from widgets.chessboard import ChessBoardWidget
from widgets.annotation_panel import AnnotationPanel


class OracleHMI(QMainWindow):
    """
    ALBERTA Human Oracle Interface.
    """

    def __init__(
        self,
        queue_path="data/oracle_queue_1-10.jsonl",
    ):

        super().__init__()

        self.setWindowTitle(
            "ALBERTA - Atomic Oracle"
        )

        self.resize(
            1200,
            750,
        )

        # ====================================================
        # Backend
        # ====================================================

        self.queue = OracleQueue(
            queue_path
        )

        self.current_query = None

        self.current_move = None

        # ====================================================
        # Main layout
        # ====================================================

        container = QWidget()

        self.setCentralWidget(
            container
        )

        layout = QHBoxLayout(
            container
        )

        layout.setContentsMargins(
            20,
            20,
            20,
            20,
        )

        layout.setSpacing(
            25
        )

        # ====================================================
        # Board
        # ====================================================

        self.board = ChessBoardWidget()

        self.board.move_selected.connect(
            self.on_move_selected
        )

        layout.addWidget(
            self.board
        )

        # ====================================================
        # Right panel
        # ====================================================

        right_widget = QWidget()

        right_widget.setFixedWidth(
            330
        )

        right = QVBoxLayout(
            right_widget
        )

        right.setSpacing(
            20
        )

        right.addStretch()

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        self.info_label = QLabel()

        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.info_label.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                padding: 10px;
            }
            """
        )

        right.addWidget(
            self.info_label
        )

        # ----------------------------------------------------
        # Take back
        # ----------------------------------------------------

        self.takeback_button = QPushButton(
            "↩ Take back"
        )

        self.takeback_button.clicked.connect(
            self.take_back
        )

        self.takeback_button.setEnabled(
            False
        )

        right.addWidget(
            self.takeback_button
        )

        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        self.annotation = AnnotationPanel()

        self.annotation.validated.connect(
            self.submit_answer
        )

        right.addWidget(
            self.annotation
        )

        right.addStretch()

        layout.addWidget(
            right_widget
        )

        # ====================================================
        # Initial position
        # ====================================================

        self.load_next()

    # ========================================================
    # Information display
    # ========================================================

    def update_info(
        self,
        extra="",
    ):

        if self.current_query is None:

            return

        side = (
            "White"
            if self.current_query.fen.split()[1] == "w"
            else "Black"
        )

        uncertainty = (
            self.current_query.score * 100
        )

        stats = self.queue.stats()

        text = f"""
<b>Side to move:</b><br>
{side}

<br>

<b>Agent uncertainty (relative):</b><br>
{uncertainty:.2f}%

<br>

<b>Queue remaining:</b><br>
{self.remaining_count()}
"""

        if extra:

            text += f"""
<br>
{extra}
"""

        self.info_label.setText(
            text
        )

    # ========================================================
    # Remaining count
    # ========================================================

    def remaining_count(
        self,
    ):

        pending = self.queue.pending(
            reward_mode=self.annotation.reward_enabled(),
            oracle_mode=self.annotation.oracle_enabled(),
        )

        return len(pending)

    # ========================================================
    # Load next position
    # ========================================================

    def load_next(
        self,
    ):

        query = self.queue.next(
            reward_mode=self.annotation.reward_enabled(),
            oracle_mode=self.annotation.oracle_enabled(),
        )

        if query is None:

            self.current_query = None

            self.info_label.setText(
                """
                <b>Queue completed.</b>
                <br><br>
                All positions have been annotated
                for the active annotation modes.
                """
            )

            self.board.setEnabled(
                False
            )

            self.annotation.setEnabled(
                False
            )

            self.takeback_button.setEnabled(
                False
            )

            return

        self.current_query = query

        self.current_move = None

        self.board.setEnabled(
            True
        )

        self.annotation.setEnabled(
            True
        )

        self.board.set_fen(
            query.fen
        )

        self.annotation.reset()

        self.takeback_button.setEnabled(
            False
        )

        self.update_info(
            "<b>No move selected</b>"
        )

    # ========================================================
    # Move selected
    # ========================================================

    def on_move_selected(
        self,
        uci: str,
    ):

        self.current_move = uci

        san = self.board.san_from_uci(
            uci
        )

        self.takeback_button.setEnabled(
            True
        )

        self.update_info(
            f"<b>Selected move:</b><br>{san}"
        )

    # ========================================================
    # Take back
    # ========================================================

    def take_back(
        self,
    ):

        self.board.take_back()

        self.current_move = None

        self.takeback_button.setEnabled(
            False
        )

        self.update_info(
            "<b>No move selected</b>"
        )

    # ========================================================
    # Submit annotation
    # ========================================================

    def submit_answer(
        self,
        annotation: dict,
    ):

        if self.current_query is None:

            return

        # ----------------------------------------------------
        # Oracle mode requires a selected move.
        # Reward-only mode does not.
        # ----------------------------------------------------

        if self.annotation.oracle_enabled():

            if self.current_move is None:

                self.update_info(
                    "<b>Please select a move first.</b>"
                )

                return

            annotation = {
                **annotation,
                "oracle_move": self.current_move,
            }

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        try:

            self.queue.answer(
                query_id=self.current_query.query_id,
                **annotation,
            )

        except (
            ValueError,
            KeyError,
        ) as e:

            self.update_info(
                f"<b>Error:</b><br>{e}"
            )

            return

        # ----------------------------------------------------
        # Next
        # ----------------------------------------------------

        self.load_next()


def main():

    app = QApplication(
        sys.argv
    )

    window = OracleHMI()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":

    main()