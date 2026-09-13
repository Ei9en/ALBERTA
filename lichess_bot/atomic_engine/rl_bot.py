from __future__ import annotations

from pathlib import Path

import chess.variant
import torch

from src.actions_space import (
    ACTIONS,
    ACTION_TO_INDEX,
)
from src.agents.actor_critic_agent import ActorCriticAgent
from src.encoding import encode_board
from src.models.actor_critic import ActorCritic
from src.models.resnet import ChessResNet


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "oracle_epoch"
    / "al_epoch_30.pt"
)


# ============================================================
# Default model configuration
# ============================================================

DEFAULT_CHANNELS = 32
DEFAULT_BLOCKS = 4

DEFAULT_TEMPERATURE = 2.0


# ============================================================
# RL bot
# ============================================================

class RLBot:
    """
    ALBERTA ActorCritic inference wrapper.

    This wrapper is used by the lichess-bot integration and by
    evaluation scripts that require the same deployment interface.

    Move selection is delegated to ActorCriticAgent.

    No BC opening prior is applied during inference.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        temperature: float = DEFAULT_TEMPERATURE,
        deterministic: bool = False,
        device: str | torch.device = "cpu",
        channels: int = DEFAULT_CHANNELS,
        blocks: int = DEFAULT_BLOCKS,
    ) -> None:

        self.device = torch.device(
            device
        )

        self.checkpoint = Path(
            checkpoint
        )

        if not self.checkpoint.exists():

            raise FileNotFoundError(
                f"RL checkpoint not found: {self.checkpoint}"
            )

        # ====================================================
        # ActorCritic architecture
        # ====================================================

        bc_model = ChessResNet(
            num_actions=len(
                ACTIONS
            ),
            channels=channels,
            blocks=blocks,
        )

        self.model = ActorCritic(
            bc_model
        ).to(
            self.device
        )

        # ====================================================
        # Checkpoint
        # ====================================================

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

        # ====================================================
        # Inference agent
        # ====================================================

        self.agent = ActorCriticAgent(
            self.model,
            device=self.device,
            deterministic=deterministic,
            temperature=temperature,
        )

        print(
            f"Loaded RL checkpoint: {self.checkpoint}"
        )

    # ========================================================
    # Move selection
    # ========================================================

    @torch.no_grad()
    def choose_move(
        self,
        board: chess.variant.AtomicBoard,
    ) -> dict:

        return self.agent.choose_move(
            board
        )

    # ========================================================
    # Policy inspection
    # ========================================================

    @torch.no_grad()
    def evaluate_policy(
        self,
        board: chess.variant.AtomicBoard,
    ) -> dict:
        """
        Return the intrinsic ActorCritic policy over legal moves.

        No temperature scaling is applied here.

        The returned distribution therefore corresponds to:

            softmax(legal RL logits)

        rather than to the possibly temperature-scaled
        move-selection distribution used by choose_move().
        """

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:

            raise RuntimeError(
                "evaluate_policy() called on a position "
                "without legal moves."
            )

        # ----------------------------------------------------
        # Encode position
        # ----------------------------------------------------

        x = encode_board(
            board
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        policy, value = self.model(
            x
        )

        logits = policy[
            0
        ]

        # ----------------------------------------------------
        # Legal actions only
        # ----------------------------------------------------

        legal_uci = [
            move.uci()
            for move in legal_moves
        ]

        legal_indices = [
            ACTION_TO_INDEX[
                uci
            ]
            for uci in legal_uci
        ]

        legal_logits = logits[
            legal_indices
        ]

        # ----------------------------------------------------
        # Intrinsic policy distribution
        # ----------------------------------------------------

        log_probs = torch.log_softmax(
            legal_logits,
            dim=0,
        )

        probs = torch.exp(
            log_probs
        )

        entropy = -(
            probs
            * log_probs
        ).sum().item()

        return {
            "moves":
                legal_uci,

            "probs":
                probs.cpu().tolist(),

            "log_probs":
                log_probs.cpu().tolist(),

            "entropy":
                entropy,

            "value":
                value.item(),
        }