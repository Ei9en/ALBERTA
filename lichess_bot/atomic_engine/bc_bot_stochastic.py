from __future__ import annotations

import sys
from pathlib import Path

import chess
import torch


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.actions_space import ACTIONS, ACTION_TO_INDEX
from src.encoding import encode_board
from src.models.resnet import ChessResNet


# ============================================================
# Defaults
# ============================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_epoch_7.pt"
)

DEFAULT_CHANNELS = 32
DEFAULT_BLOCKS = 4
DEFAULT_TEMPERATURE = 2.0


# ============================================================
# Stochastic BC bot
# ============================================================

class BCBotStochastic:
    """
    Stochastic Behavioral Cloning inference wrapper.

    Move selection samples from the BC policy restricted to
    legal moves and scaled by temperature.

    Randomness uses PyTorch's global RNG and is controlled by
    the calling evaluation script.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        temperature: float = DEFAULT_TEMPERATURE,
        device: str | torch.device = "cpu",
        channels: int = DEFAULT_CHANNELS,
        blocks: int = DEFAULT_BLOCKS,
    ) -> None:

        if temperature <= 0.0:
            raise ValueError(
                "temperature must be strictly positive."
            )

        self.device = torch.device(
            device
        )

        self.checkpoint = Path(
            checkpoint
        )

        self.temperature = float(
            temperature
        )

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"BC checkpoint not found: {self.checkpoint}"
            )

        self.model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=channels,
            blocks=blocks,
        ).to(
            self.device
        )

        checkpoint_data = torch.load(
            self.checkpoint,
            map_location=self.device,
        )

        checkpoint_actions = checkpoint_data.get(
            "actions"
        )

        if (
            checkpoint_actions is not None
            and checkpoint_actions != len(ACTIONS)
        ):
            raise ValueError(
                "Action-space mismatch in checkpoint: "
                f"{checkpoint_actions} != {len(ACTIONS)}"
            )

        self.model.load_state_dict(
            checkpoint_data[
                "model_state_dict"
            ]
        )

        self.model.eval()

        print(
            f"Loaded BC checkpoint: {self.checkpoint}"
        )

    # ========================================================
    # Move selection
    # ========================================================

    @torch.no_grad()
    def choose_move(
        self,
        board: chess.Board,
    ) -> dict:

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:
            raise RuntimeError(
                "choose_move() called on a position "
                "without legal moves."
            )

        x = encode_board(
            board
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        logits = self.model(
            x
        )[0]

        legal_indices = [
            ACTION_TO_INDEX[
                move.uci()
            ]
            for move in legal_moves
        ]

        legal_logits = logits[
            legal_indices
        ]

        # ----------------------------------------------------
        # Temperature-scaled sampling distribution
        # ----------------------------------------------------

        sampling_logits = (
            legal_logits
            / self.temperature
        )

        sampling_log_probs = torch.log_softmax(
            sampling_logits,
            dim=0,
        )

        sampling_probs = torch.exp(
            sampling_log_probs
        )

        position = torch.multinomial(
            sampling_probs,
            num_samples=1,
        ).item()

        action = legal_indices[
            position
        ]

        move = legal_moves[
            position
        ]

        # Historical semantics:
        # entropy corresponds to the actual sampling policy.
        entropy = -(
            sampling_probs
            * sampling_log_probs
        ).sum().item()

        return {
            "action":
                action,

            "move":
                move,

            "value":
                0.0,

            "entropy":
                entropy,
        }