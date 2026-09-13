from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class AnnotationPanel(QWidget):
    """
    Annotation controls for the ALBERTA Human Oracle Interface.

    The panel supports two independent annotation families:

        - terminal reward;
        - Oracle decision annotation.

    Both modes are enabled by default.
    """

    validated = pyqtSignal(dict)

    CONFIDENCE_VALUES = {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
    }

    SITUATION_VALUES = {
        "Critical": "critical",
        "Non-critical": "non_critical",
        "Outcome-independent": "outcome_independent",
    }

    REWARD_VALUES = {
        "+1": 1,
        "0": 0,
        "-1": -1,
    }

    def __init__(
        self,
        parent=None,
    ) -> None:

        super().__init__(
            parent
        )

        self.confidence_buttons: dict[
            str,
            QRadioButton,
        ] = {}

        self.situation_buttons: dict[
            str,
            QRadioButton,
        ] = {}

        self.reward_buttons: dict[
            int,
            QRadioButton,
        ] = {}

        self._build_ui()

    # ========================================================
    # UI
    # ========================================================

    def _build_ui(
        self,
    ) -> None:

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
                padding: 2px;
                spacing: 4px;
            }

            QCheckBox {
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

        # ====================================================
        # Annotation modes
        # ====================================================

        mode_group = QGroupBox(
            "Annotation modes"
        )

        mode_layout = QVBoxLayout()

        self.reward_mode = QCheckBox(
            "Reward"
        )

        self.oracle_mode = QCheckBox(
            "Oracle decision"
        )

        # Both annotation families are enabled by default.
        self.reward_mode.setChecked(
            True
        )

        self.oracle_mode.setChecked(
            True
        )

        self.reward_mode.toggled.connect(
            self.update_mode_state
        )

        self.oracle_mode.toggled.connect(
            self.update_mode_state
        )

        mode_layout.addWidget(
            self.reward_mode
        )

        mode_layout.addWidget(
            self.oracle_mode
        )

        mode_group.setLayout(
            mode_layout
        )

        layout.addWidget(
            mode_group
        )

        # ====================================================
        # Reward
        # ====================================================

        self.reward_group = QGroupBox(
            "Reward"
        )

        reward_layout = QHBoxLayout()

        reward_layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )

        reward_layout.setSpacing(
            10
        )

        for label, value in self.REWARD_VALUES.items():

            button = QRadioButton(
                label
            )

            self.reward_buttons[
                value
            ] = button

            reward_layout.addWidget(
                button
            )

        self.reward_group.setLayout(
            reward_layout
        )

        layout.addWidget(
            self.reward_group
        )

        # Default reward annotation.
        self.reward_buttons[
            1
        ].setChecked(
            True
        )

        # ====================================================
        # Oracle decision
        # ====================================================

        self.oracle_group = QWidget()

        oracle_layout = QVBoxLayout(
            self.oracle_group
        )

        oracle_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence_group = QGroupBox(
            "Confidence"
        )

        confidence_layout = QVBoxLayout()

        for label, value in self.CONFIDENCE_VALUES.items():

            button = QRadioButton(
                label
            )

            self.confidence_buttons[
                value
            ] = button

            confidence_layout.addWidget(
                button
            )

        confidence_group.setLayout(
            confidence_layout
        )

        oracle_layout.addWidget(
            confidence_group
        )

        # ----------------------------------------------------
        # Decision situation
        # ----------------------------------------------------

        situation_group = QGroupBox(
            "Decision situation"
        )

        situation_layout = QVBoxLayout()

        for label, value in self.SITUATION_VALUES.items():

            button = QRadioButton(
                label
            )

            self.situation_buttons[
                value
            ] = button

            situation_layout.addWidget(
                button
            )

        situation_group.setLayout(
            situation_layout
        )

        oracle_layout.addWidget(
            situation_group
        )

        layout.addWidget(
            self.oracle_group
        )

        # Default Oracle annotations.
        self.confidence_buttons[
            "high"
        ].setChecked(
            True
        )

        self.situation_buttons[
            "critical"
        ].setChecked(
            True
        )

        # ====================================================
        # Validate
        # ====================================================

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

        self.update_mode_state()

    # ========================================================
    # Mode state
    # ========================================================

    def update_mode_state(
        self,
    ) -> None:

        reward_enabled = (
            self.reward_mode.isChecked()
        )

        oracle_enabled = (
            self.oracle_mode.isChecked()
        )

        self.reward_group.setEnabled(
            reward_enabled
        )

        self.oracle_group.setEnabled(
            oracle_enabled
        )

        # Prevent validation when no annotation family is active.
        self.validate_button.setEnabled(
            reward_enabled
            or oracle_enabled
        )

    # ========================================================
    # Annotation API
    # ========================================================

    def get_annotation(
        self,
    ) -> dict:

        annotation = {}

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        if self.reward_enabled():

            reward = None

            for value, button in self.reward_buttons.items():

                if button.isChecked():

                    reward = value
                    break

            if reward is None:

                raise ValueError(
                    "Reward annotation is required."
                )

            annotation[
                "reward"
            ] = reward

        # ----------------------------------------------------
        # Oracle decision
        # ----------------------------------------------------

        if self.oracle_enabled():

            confidence = None
            situation = None

            for (
                value,
                button,
            ) in self.confidence_buttons.items():

                if button.isChecked():

                    confidence = value
                    break

            for (
                value,
                button,
            ) in self.situation_buttons.items():

                if button.isChecked():

                    situation = value
                    break

            if confidence is None:

                raise ValueError(
                    "Confidence annotation is required."
                )

            if situation is None:

                raise ValueError(
                    "Decision situation annotation is required."
                )

            annotation[
                "confidence"
            ] = confidence

            annotation[
                "situation"
            ] = situation

        if not annotation:

            raise ValueError(
                "At least one annotation mode must be enabled."
            )

        return annotation

    # ========================================================
    # Mode API
    # ========================================================

    def reward_enabled(
        self,
    ) -> bool:

        return (
            self.reward_mode.isChecked()
        )

    def oracle_enabled(
        self,
    ) -> bool:

        return (
            self.oracle_mode.isChecked()
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.reward_buttons[
            1
        ].setChecked(
            True
        )

        self.confidence_buttons[
            "high"
        ].setChecked(
            True
        )

        self.situation_buttons[
            "critical"
        ].setChecked(
            True
        )

    # ========================================================
    # Validate
    # ========================================================

    def validate(
        self,
    ) -> None:

        try:

            annotation = (
                self.get_annotation()
            )

        except ValueError:

            return

        self.validated.emit(
            annotation
        )