from __future__ import annotations


from PyQt6.QtCore import pyqtSignal

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QRadioButton,
    QPushButton,
)



class AnnotationPanel(QWidget):

    validated = pyqtSignal(dict)



    CONFIDENCE_VALUES = {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
    }


    SITUATION_VALUES = {

        "Critical": "critical",  # Only one move is acceptable

        "Non-critical": "non_critical",  # Several moves are playable

        "Outcome-independent": "outcome_independent",  # Everything wins or loses

    }


    def __init__(
        self,
        parent=None,
    ):

        super().__init__(parent)


        self.confidence_buttons = {}
        self.situation_buttons = {}


        self.build_ui()



    # =====================================================
    # UI
    # =====================================================

    def build_ui(self):

        layout = QVBoxLayout(
            self
        )


        layout.setSpacing(
            15
        )


        self.setStyleSheet(
            """
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #bbbbbb;
                border-radius: 8px;
                margin-top: 18px;
                padding-top: 12px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            QRadioButton {
                font-size: 15px;
                padding: 6px;
            }

            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
            }

            QPushButton:hover {
                background-color: #dddddd;
            }
            """
        )



        confidence_group = QGroupBox(
            "Confidence"
        )


        confidence_layout = QVBoxLayout()


        for label, value in self.CONFIDENCE_VALUES.items():

            button = QRadioButton(
                label
            )

            self.confidence_buttons[value] = button


            confidence_layout.addWidget(
                button
            )


        confidence_group.setLayout(
            confidence_layout
        )


        layout.addWidget(
            confidence_group
        )



        situation_group = QGroupBox(
            "Decision criticality"
        )


        situation_layout = QVBoxLayout()


        for label, value in self.SITUATION_VALUES.items():

            button = QRadioButton(
                label
            )

            self.situation_buttons[value] = button


            situation_layout.addWidget(
                button
            )


        situation_group.setLayout(
            situation_layout
        )


        layout.addWidget(
            situation_group
        )



        self.confidence_buttons[
            "high"
        ].setChecked(True)


        self.situation_buttons[
            "critical"
        ].setChecked(True)



        self.validate_button = QPushButton(
            "✓ Validate annotation"
        )


        self.validate_button.clicked.connect(
            self.validate
        )


        layout.addWidget(
            self.validate_button
        )


        layout.addStretch()



    # =====================================================
    # API
    # =====================================================

    def get_annotation(
        self,
    ):

        confidence = None
        situation = None


        for value, button in self.confidence_buttons.items():

            if button.isChecked():

                confidence = value


        for value, button in self.situation_buttons.items():

            if button.isChecked():

                situation = value


        return {
            "confidence": confidence,
            "situation": situation,
        }



    def reset(
        self,
    ):

        self.confidence_buttons[
            "high"
        ].setChecked(True)


        self.situation_buttons[
            "critical"
        ].setChecked(True)



    def validate(
        self,
    ):

        self.validated.emit(
            self.get_annotation()
        )