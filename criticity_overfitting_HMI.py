from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import chess.variant
import torch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.criticality import CriticalityNet
from src.encoding import encode_board

from widgets.chessboard import ChessBoardWidget


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


NUM_POSITIONS = 50


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "criticality_epoch"
    / "criticality_epoch_20.pt"
)


CLASS_NAMES = {
    0: "outcome_independent",
    1: "non_critical",
    2: "critical",
}


CLASS_DESCRIPTIONS = {
    0: "Outcome independent",
    1: "Non-critical",
    2: "Critical",
}


# ============================================================
# Command line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "ALBERTA CriticalityNet "
            "overfitting test"
        )
    )


    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "JSON or JSONL file containing "
            "Atomic positions"
        ),
    )


    parser.add_argument(
        "--count",
        type=int,
        default=NUM_POSITIONS,
        help=(
            f"Number of positions to test "
            f"(default: {NUM_POSITIONS})"
        ),
    )


    return parser.parse_args()


# ============================================================
# Criticality model
# ============================================================

def load_criticality_model():

    model = CriticalityNet(
        in_channels=19,
        channels=32,
        blocks=4,
        num_classes=3,
    ).to(DEVICE)


    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Criticality checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    print()
    print("=" * 60)
    print("Loading criticality model")
    print("=" * 60)

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Device    : {DEVICE}"
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )


    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )


    model.eval()


    print(
        "Criticality model loaded."
    )


    return model


# ============================================================
# Random Atomic position generation
# ============================================================

def generate_random_position(
    min_plies=4,
    max_plies=80,
):

    """
    Génère une position Atomic aléatoire en jouant
    une partie légale avec des coups choisis aléatoirement.

    On évite les positions initiales et terminales.
    """

    while True:

        board = chess.variant.AtomicBoard()


        target_plies = random.randint(
            min_plies,
            max_plies,
        )


        for _ in range(target_plies):

            if board.is_game_over():
                break


            legal_moves = list(
                board.legal_moves
            )


            if not legal_moves:
                break


            move = random.choice(
                legal_moves
            )


            board.push(
                move
            )


        if board.is_game_over():
            continue


        if board.fullmove_number < 3:
            continue


        return board


def generate_positions(
    count,
):

    positions = []

    seen_fens = set()


    while len(positions) < count:

        board = generate_random_position()

        fen = board.fen()


        if fen in seen_fens:
            continue


        seen_fens.add(
            fen
        )


        positions.append(
            (
                board,
                fen,
            )
        )


    return positions


# ============================================================
# JSON / JSONL loading
# ============================================================

def extract_fen(item):

    """
    Extrait un FEN depuis une entrée JSON.

    Supporte :

        {"fen": "..."}
        {"FEN": "..."}
        {"position": "..."}

    ou directement :

        "fen ..."
    """

    if isinstance(
        item,
        str,
    ):

        return item.strip()


    if not isinstance(
        item,
        dict,
    ):

        return None


    for key in (
        "fen",
        "FEN",
        "position",
    ):

        value = item.get(
            key
        )


        if isinstance(
            value,
            str,
        ):

            return value.strip()


    return None


def load_positions_from_json(
    path,
    count,
):

    path = Path(
        path
    )


    if not path.exists():

        raise FileNotFoundError(
            f"JSON file not found:\n"
            f"{path}"
        )


    print()
    print("=" * 70)
    print("LOADING POSITIONS FROM JSON")
    print("=" * 70)


    print(
        f"File : {path}"
    )


    # ========================================================
    # JSONL
    # ========================================================

    if path.suffix.lower() == ".jsonl":

        entries = []


        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()


                if not line:
                    continue


                try:

                    entries.append(
                        json.loads(
                            line
                        )
                    )

                except json.JSONDecodeError as error:

                    print(
                        f"WARNING: invalid JSONL line "
                        f"{line_number}: {error}"
                    )


    # ========================================================
    # JSON
    # ========================================================

    else:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )


        if isinstance(
            data,
            list,
        ):

            entries = data


        elif isinstance(
            data,
            dict,
        ):

            entries = None


            for key in (
                "positions",
                "data",
                "samples",
                "annotations",
            ):

                value = data.get(
                    key
                )


                if isinstance(
                    value,
                    list,
                ):

                    entries = value

                    break


            if entries is None:

                raise ValueError(
                    "Could not find a position list "
                    "in the JSON file."
                )


        else:

            raise ValueError(
                "Unsupported JSON structure."
            )


    # ========================================================
    # Extract and validate FENs
    # ========================================================

    fens = []

    seen_fens = set()


    for entry in entries:

        fen = extract_fen(
            entry
        )


        if not fen:
            continue


        if fen in seen_fens:
            continue


        try:

            chess.variant.AtomicBoard(
                fen
            )

        except Exception:

            continue


        seen_fens.add(
            fen
        )

        fens.append(
            fen
        )


    print(
        f"Valid unique positions : "
        f"{len(fens)}"
    )


    if len(fens) < count:

        raise ValueError(
            f"The file contains only "
            f"{len(fens)} valid unique positions, "
            f"but {count} are required."
        )


    # ========================================================
    # Random selection
    # ========================================================

    selected_fens = random.sample(
        fens,
        count,
    )


    positions = []


    for fen in selected_fens:

        board = chess.variant.AtomicBoard(
            fen
        )


        positions.append(
            (
                board,
                fen,
            )
        )


    print(
        f"Randomly selected : "
        f"{len(positions)}"
    )


    print("=" * 70)


    return positions


# ============================================================
# Model predictions
# ============================================================

def predict_positions(
    model,
    positions,
):

    """
    Calcule les prédictions sur toutes les positions
    avant l'ouverture de l'IHM.

    Les prédictions ne sont pas montrées à l'utilisateur
    avant son annotation.
    """

    predictions = []


    counts = {
        0: 0,
        1: 0,
        2: 0,
    }


    print()
    print("=" * 70)
    print("Computing model predictions")
    print("=" * 70)


    with torch.no_grad():

        for index, (
            board,
            fen,
        ) in enumerate(
            positions
        ):

            encoded = encode_board(
                board
            )


            encoded = encoded.unsqueeze(
                0
            ).to(
                DEVICE
            )


            probabilities = torch.softmax(
                model(
                    encoded
                ),
                dim=1,
            )[0]


            model_class = int(
                torch.argmax(
                    probabilities
                ).item()
            )


            probability_list = (
                probabilities
                .detach()
                .cpu()
                .tolist()
            )


            predictions.append(
                {
                    "fen": fen,
                    "probabilities": probability_list,
                    "model_class": model_class,
                }
            )


            counts[
                model_class
            ] += 1


            if (
                (index + 1) % 25 == 0
                or index + 1 == len(positions)
            ):

                print(
                    f"Predicted "
                    f"{index + 1}/"
                    f"{len(positions)}"
                )


    # ========================================================
    # Distribution
    # ========================================================

    total = len(
        positions
    )


    print()
    print("=" * 70)
    print("MODEL PREDICTION DISTRIBUTION")
    print("=" * 70)


    print(
        f"Positions : {total}"
    )


    for class_id in [
        2,
        1,
        0,
    ]:

        count = counts[
            class_id
        ]


        percentage = (
            100.0
            * count
            / total
        )


        print(
            f"{CLASS_NAMES[class_id]:20s}"
            f": {count:3d} "
            f"({percentage:6.2f}%)"
        )


    print("=" * 70)


    return predictions


# ============================================================
# Result storage
# ============================================================

class AnnotationResult:

    def __init__(
        self,
        fen,
        human_class,
        model_class,
        probabilities,
    ):

        self.fen = fen

        self.human_class = human_class

        self.model_class = model_class

        self.probabilities = probabilities


# ============================================================
# Main HMI
# ============================================================

class CriticalityOverfittingHMI(
    QWidget
):

    def __init__(
        self,
        model,
        positions,
        predictions,
    ):

        super().__init__()


        self.model = model

        self.positions = positions

        self.predictions = predictions

        self.index = 0

        self.results = []

        self.current_prediction = None

        self.annotation_locked = False


        self.setWindowTitle(
            "ALBERTA - Criticality Overfitting Test"
        )


        self.resize(
            1000,
            720,
        )


        self.build_ui()

        self.show_position()


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        main_layout = QVBoxLayout(
            self
        )


        # ====================================================
        # Header
        # ====================================================

        self.progress_label = QLabel()


        self.progress_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.progress_label.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold;
            """
        )


        main_layout.addWidget(
            self.progress_label
        )


        # ====================================================
        # Board + controls
        # ====================================================

        content_layout = QHBoxLayout()


        # ====================================================
        # Board
        # ====================================================

        self.board_widget = ChessBoardWidget()


        self.board_widget.setMinimumSize(
            600,
            600,
        )


        # Quand un coup est joué sur le board,
        # on active le Take back.
        self.board_widget.move_selected.connect(
            self.on_move_selected
        )


        content_layout.addWidget(
            self.board_widget,
            stretch=1,
        )


        # ====================================================
        # Controls
        # ====================================================

        controls = QVBoxLayout()


        self.instruction_label = QLabel(
            "Quelle est la criticité "
            "de cette position ?"
        )


        self.instruction_label.setWordWrap(
            True
        )


        self.instruction_label.setStyleSheet(
            """
            font-size: 18px;
            font-weight: bold;
            """
        )


        controls.addWidget(
            self.instruction_label
        )


        controls.addSpacing(
            20
        )


        # ====================================================
        # Human annotation buttons
        # ====================================================

        self.annotation_buttons = []


        for class_id in range(3):

            button = QPushButton(
                CLASS_DESCRIPTIONS[
                    class_id
                ]
            )


            button.setMinimumHeight(
                55
            )


            button.setStyleSheet(
                """
                font-size: 16px;
                """
            )


            button.clicked.connect(
                lambda checked=False,
                c=class_id:
                    self.submit_annotation(c)
            )


            controls.addWidget(
                button
            )


            self.annotation_buttons.append(
                button
            )


        controls.addSpacing(
            20
        )


        # ====================================================
        # Take back
        # ====================================================

        self.take_back_button = QPushButton(
            "Take back"
        )


        self.take_back_button.setMinimumHeight(
            45
        )


        self.take_back_button.setEnabled(
            False
        )


        self.take_back_button.clicked.connect(
            self.take_back
        )


        controls.addWidget(
            self.take_back_button
        )


        controls.addSpacing(
            15
        )


        # ====================================================
        # Model result
        # ====================================================

        self.model_result_label = QLabel()


        self.model_result_label.setWordWrap(
            True
        )


        self.model_result_label.setAlignment(
            Qt.AlignmentFlag.AlignTop
        )


        self.model_result_label.setStyleSheet(
            """
            font-size: 15px;
            """
        )


        controls.addWidget(
            self.model_result_label
        )


        controls.addStretch()


        # ====================================================
        # Next
        # ====================================================

        self.next_button = QPushButton(
            "Next position"
        )


        self.next_button.setMinimumHeight(
            50
        )


        self.next_button.setEnabled(
            False
        )


        self.next_button.clicked.connect(
            self.next_position
        )


        controls.addWidget(
            self.next_button
        )


        content_layout.addLayout(
            controls,
            stretch=0,
        )


        main_layout.addLayout(
            content_layout
        )


    # ========================================================
    # Move selected
    # ========================================================

    def on_move_selected(
        self,
        uci,
    ):

        """
        Le coup joué est uniquement visuel.

        CriticalityNet continue d'être évalué sur
        la position FEN originale.
        """

        self.take_back_button.setEnabled(
            True
        )


    # ========================================================
    # Position
    # ========================================================

    def show_position(self):

        if self.index >= len(
            self.positions
        ):

            self.finish_test()

            return


        board, fen = self.positions[
            self.index
        ]


        self.annotation_locked = False


        self.current_prediction = (
            self.predictions[
                self.index
            ]
        )


        self.next_button.setEnabled(
            False
        )


        self.take_back_button.setEnabled(
            False
        )


        # ----------------------------------------------------
        # Reset annotation buttons
        # ----------------------------------------------------

        for button in self.annotation_buttons:

            button.setEnabled(
                True
            )


        # ----------------------------------------------------
        # Hide model prediction
        # ----------------------------------------------------

        self.model_result_label.setText(
            "Model prediction: "
            "hidden until your annotation."
        )


        self.instruction_label.setText(
            "Quelle est la criticité "
            "de cette position ?"
        )


        self.progress_label.setText(
            f"Position "
            f"{self.index + 1} / "
            f"{len(self.positions)}"
        )


        # ----------------------------------------------------
        # Reset board to original FEN
        # ----------------------------------------------------

        self.board_widget.set_fen(
            fen
        )


    # ========================================================
    # Take back
    # ========================================================

    def take_back(self):

        """
        Annule uniquement le coup joué visuellement.

        La position évaluée reste la position originale.
        """

        self.board_widget.take_back()


        self.take_back_button.setEnabled(
            False
        )


    # ========================================================
    # Human annotation
    # ========================================================

    def submit_annotation(
        self,
        human_class,
    ):

        if self.annotation_locked:
            return


        self.annotation_locked = True


        for button in self.annotation_buttons:

            button.setEnabled(
                False
            )


        prediction = (
            self.current_prediction
        )


        model_class = prediction[
            "model_class"
        ]


        probabilities = prediction[
            "probabilities"
        ]


        fen = prediction[
            "fen"
        ]


        result = AnnotationResult(
            fen=fen,
            human_class=human_class,
            model_class=model_class,
            probabilities=probabilities,
        )


        self.results.append(
            result
        )


        # ====================================================
        # Display model prediction
        # ====================================================

        human_name = CLASS_NAMES[
            human_class
        ]


        model_name = CLASS_NAMES[
            model_class
        ]


        p0, p1, p2 = probabilities


        agreement = (
            "AGREEMENT"
            if human_class == model_class
            else "DISAGREEMENT"
        )


        self.model_result_label.setText(
            f"""
            <b>Human annotation:</b>
            {human_name}<br><br>

            <b>Model prediction:</b>
            {model_name}<br><br>

            <b>Model probabilities:</b><br>
            outcome_independent:
            {p0:.4f}<br>
            non_critical:
            {p1:.4f}<br>
            critical:
            {p2:.4f}<br><br>

            <b>{agreement}</b>
            """
        )


        self.instruction_label.setText(
            "Annotation recorded."
        )


        self.next_button.setEnabled(
            True
        )


    # ========================================================
    # Next
    # ========================================================

    def next_position(self):

        self.index += 1

        self.show_position()


    # ========================================================
    # Final analysis
    # ========================================================

    def finish_test(self):

        if not self.results:
            return


        total = len(
            self.results
        )


        # ====================================================
        # Accuracy
        # ====================================================

        correct = sum(
            r.human_class
            == r.model_class

            for r in self.results
        )


        accuracy = (
            correct
            / total
        )


        # ====================================================
        # Class distance
        #
        # 0 = exact
        # 1 = adjacent
        # 2 = critical <-> independent
        # ====================================================

        distances = [

            abs(
                r.human_class
                - r.model_class
            )

            for r in self.results
        ]


        mean_distance = (
            sum(distances)
            / total
        )


        exact_count = sum(
            d == 0
            for d in distances
        )


        distance_1_count = sum(
            d == 1
            for d in distances
        )


        distance_2_count = sum(
            d == 2
            for d in distances
        )


        # ====================================================
        # Confusion matrix
        #
        # rows = human
        # cols = model
        # ====================================================

        confusion = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]


        for result in self.results:

            confusion[
                result.human_class
            ][
                result.model_class
            ] += 1


        # ====================================================
        # Mean probability assigned to human class
        # ====================================================

        mean_p_human = sum(

            r.probabilities[
                r.human_class
            ]

            for r in self.results

        ) / total


        # ====================================================
        # Cross entropy
        # ====================================================

        epsilon = 1e-12


        log_loss = -sum(

            torch.log(

                torch.tensor(

                    max(
                        r.probabilities[
                            r.human_class
                        ],
                        epsilon,
                    )

                )

            ).item()

            for r in self.results

        ) / total


        # ====================================================
        # Brier score
        # ====================================================

        brier = 0.0


        for result in self.results:

            for class_id in range(3):

                target = (
                    1.0
                    if class_id
                    == result.human_class
                    else 0.0
                )


                brier += (

                    result.probabilities[
                        class_id
                    ]

                    - target

                ) ** 2


        brier /= total


        # ====================================================
        # Class distributions
        # ====================================================

        human_counts = [
            0,
            0,
            0,
        ]


        model_counts = [
            0,
            0,
            0,
        ]


        for result in self.results:

            human_counts[
                result.human_class
            ] += 1


            model_counts[
                result.model_class
            ] += 1


        # ====================================================
        # Console output
        # ====================================================

        print()
        print("=" * 70)
        print("CRITICALITY OVERFITTING TEST")
        print("=" * 70)


        print(
            f"Positions              : "
            f"{total}"
        )


        print(
            f"Model / human accuracy : "
            f"{accuracy * 100:.2f}%"
        )


        # ====================================================
        # Distance
        # ====================================================

        print()
        print("CLASS DISTANCE")


        print(
            "0 = exact | "
            "1 = adjacent | "
            "2 = critical <-> outcome independent"
        )


        print(
            f"Mean absolute distance : "
            f"{mean_distance:.4f}"
        )


        print(
            f"Distance 0             : "
            f"{exact_count:3d} "
            f"({exact_count / total * 100:6.2f}%)"
        )


        print(
            f"Distance 1             : "
            f"{distance_1_count:3d} "
            f"({distance_1_count / total * 100:6.2f}%)"
        )


        print(
            f"Distance 2             : "
            f"{distance_2_count:3d} "
            f"({distance_2_count / total * 100:6.2f}%)"
        )


        # ====================================================
        # Probability metrics
        # ====================================================

        print()
        print("PROBABILITY METRICS")


        print(
            f"Mean P(human class)    : "
            f"{mean_p_human:.4f}"
        )


        print(
            f"Cross entropy          : "
            f"{log_loss:.6f}"
        )


        print(
            f"Brier score            : "
            f"{brier:.6f}"
        )


        # ====================================================
        # Human distribution
        # ====================================================

        print()
        print("HUMAN DISTRIBUTION")


        for class_id in range(3):

            print(
                f"  "
                f"{CLASS_NAMES[class_id]:20s}"
                f": "
                f"{human_counts[class_id]:3d}"
                f" "
                f"("
                f"{human_counts[class_id] / total * 100:6.2f}%"
                f")"
            )


        # ====================================================
        # Model distribution
        # ====================================================

        print()
        print("MODEL DISTRIBUTION")


        for class_id in range(3):

            print(
                f"  "
                f"{CLASS_NAMES[class_id]:20s}"
                f": "
                f"{model_counts[class_id]:3d}"
                f" "
                f"("
                f"{model_counts[class_id] / total * 100:6.2f}%"
                f")"
            )


        # ====================================================
        # Confusion matrix
        # ====================================================

        print()
        print("CONFUSION MATRIX")


        print(
            "Rows = human / columns = model"
        )


        print(
            f"{'':20s}"
            f"{'outcome':>10s}"
            f"{'non_crit':>10s}"
            f"{'critical':>10s}"
        )


        for row_id in range(3):

            print(
                f"{CLASS_NAMES[row_id]:20s}"
                f"{confusion[row_id][0]:10d}"
                f"{confusion[row_id][1]:10d}"
                f"{confusion[row_id][2]:10d}"
            )


        print()
        print("=" * 70)


        # ====================================================
        # GUI summary
        # ====================================================

        message = (

            f"<b>Test finished</b><br><br>"

            f"Positions: {total}<br>"

            f"Accuracy: "
            f"{accuracy * 100:.2f}%"
            f"<br><br>"


            f"<b>Class distance</b><br>"

            f"Mean absolute distance: "
            f"{mean_distance:.4f}<br>"

            f"Distance 0: "
            f"{exact_count} "
            f"({exact_count / total * 100:.2f}%)"
            f"<br>"

            f"Distance 1: "
            f"{distance_1_count} "
            f"({distance_1_count / total * 100:.2f}%)"
            f"<br>"

            f"Distance 2: "
            f"{distance_2_count} "
            f"({distance_2_count / total * 100:.2f}%)"
            f"<br><br>"


            f"<b>Probability metrics</b><br>"

            f"Mean P(human class): "
            f"{mean_p_human:.4f}<br>"

            f"Cross entropy: "
            f"{log_loss:.6f}<br>"

            f"Brier score: "
            f"{brier:.6f}"
            f"<br><br>"


            f"<b>Confusion matrix</b><br>"

            f"Rows = human / columns = model"
            f"<br><br>"


            f"{confusion[0][0]} &nbsp;&nbsp; "
            f"{confusion[0][1]} &nbsp;&nbsp; "
            f"{confusion[0][2]}"
            f"<br>"

            f"{confusion[1][0]} &nbsp;&nbsp; "
            f"{confusion[1][1]} &nbsp;&nbsp; "
            f"{confusion[1][2]}"
            f"<br>"

            f"{confusion[2][0]} &nbsp;&nbsp; "
            f"{confusion[2][1]} &nbsp;&nbsp; "
            f"{confusion[2][2]}"
        )


        QMessageBox.information(
            self,
            "Criticality test completed",
            message,
        )


        # ====================================================
        # Disable everything
        # ====================================================

        for button in self.annotation_buttons:

            button.setEnabled(
                False
            )


        self.take_back_button.setEnabled(
            False
        )


        self.next_button.setEnabled(
            False
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()


    print()
    print("=" * 70)
    print("ALBERTA - CRITICALITY OVERFITTING TEST")
    print("=" * 70)


    # ========================================================
    # Load / generate positions
    # ========================================================

    if args.json is not None:

        print()
        print(
            "Loading positions from:"
        )

        print(
            f"  {args.json}"
        )


        positions = load_positions_from_json(
            args.json,
            args.count,
        )


    else:

        print()
        print(
            f"Generating "
            f"{args.count} random Atomic positions..."
        )


        positions = generate_positions(
            args.count
        )


        print(
            f"Generated "
            f"{len(positions)} unique positions."
        )


    # ========================================================
    # Load model
    # ========================================================

    model = load_criticality_model()


    # ========================================================
    # Compute predictions BEFORE opening HMI
    # ========================================================

    predictions = predict_positions(
        model,
        positions,
    )


    # ========================================================
    # Launch HMI
    # ========================================================

    app = QApplication(
        sys.argv
    )


    window = CriticalityOverfittingHMI(
        model=model,
        positions=positions,
        predictions=predictions,
    )


    window.show()


    sys.exit(
        app.exec()
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()