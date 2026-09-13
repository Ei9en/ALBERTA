from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from AL.oracle_queue import OracleQueue
from AL.HMI.widgets.annotation_panel import AnnotationPanel
from AL.HMI.widgets.chessboard import ChessBoardWidget


# ============================================================
# Defaults
# ============================================================

DEFAULT_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_AL.jsonl"
)


# ============================================================
# Main window
# ============================================================

class OracleHMI(QMainWindow):
    """
    ALBERTA Human Oracle Interface.

    Supports reward annotation, Oracle move annotation,
    or both depending on the active modes exposed by the
    AnnotationPanel.
    """

    def __init__(
        self,
        queue_path: str | Path,
    ) -> None:

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
        self.current_move: str | None = None

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
        # Annotation panel
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
        extra: str = "",
    ) -> None:

        if self.current_query is None:
            return

        fen_parts = (
            self.current_query.fen.split()
        )

        side = (
            "White"
            if fen_parts[1] == "w"
            else "Black"
        )

        # ----------------------------------------------------
        # Acquisition score
        #
        # I_norm is a min-max representation of the active-
        # learning acquisition score. It is not a calibrated
        # probability or a pure uncertainty measure.
        # ----------------------------------------------------

        if self.current_query.I_norm is None:

            score_text = (
                "N/A"
            )

        else:

            score_text = (
                f"{100 * self.current_query.I_norm:.2f}%"
            )

        text = f"""
<b>Side to move:</b><br>
{side}

<br>

<b>Relative acquisition score:</b><br>
{score_text}

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
    ) -> int:

        pending = self.queue.pending(
            reward_mode=self.annotation.reward_enabled(),
            oracle_mode=self.annotation.oracle_enabled(),
        )

        return len(
            pending
        )

    # ========================================================
    # Load next position
    # ========================================================

    def load_next(
        self,
    ) -> None:

        query = self.queue.next(
            reward_mode=self.annotation.reward_enabled(),
            oracle_mode=self.annotation.oracle_enabled(),
        )

        if query is None:

            self.current_query = None
            self.current_move = None

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
    ) -> None:

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
    ) -> None:

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
    ) -> None:

        if self.current_query is None:
            return

        # ----------------------------------------------------
        # Oracle mode requires a selected move.
        #
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
                "oracle_move":
                    self.current_move,
            }

        # ----------------------------------------------------
        # Persist annotation
        # ----------------------------------------------------

        try:

            self.queue.answer(
                query_id=self.current_query.query_id,
                **annotation,
            )

        except (
            ValueError,
            KeyError,
        ) as exc:

            self.update_info(
                f"<b>Error:</b><br>{exc}"
            )

            return

        # ----------------------------------------------------
        # Load next required annotation
        # ----------------------------------------------------

        self.load_next()


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Launch the ALBERTA Human Oracle Interface."
        )
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
        help="Oracle queue JSONL file.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    app = QApplication(
        sys.argv
    )

    window = OracleHMI(
        queue_path=args.queue
    )

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()