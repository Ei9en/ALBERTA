from __future__ import annotations

import random
from typing import Any


class ReplayBuffer:
    """
    Buffer storing PPO rollout transitions.

    Transitions are stored in insertion order. When the maximum
    capacity is exceeded, the oldest transition is removed.

    Sampling uses Python's global random-number generator. The
    training script is responsible for seeding it reproducibly.
    """

    def __init__(
        self,
        capacity: int = 300_000,
    ) -> None:

        if capacity <= 0:
            raise ValueError(
                "capacity must be greater than zero."
            )

        self.capacity = capacity

        self.buffer: list[
            dict[str, Any]
        ] = []


    # ========================================================
    # Clear
    # ========================================================

    def clear(
        self,
    ) -> None:
        """
        Remove all stored transitions.
        """

        self.buffer.clear()


    # ========================================================
    # Add
    # ========================================================

    def add(
        self,
        fen: str,
        action: int,
        legal_moves,
        return_: float,
        value: float,
        old_log_prob: float,
        advantage: float,
        ply: int,
        game_result: str | None = None,
    ) -> None:
        """
        Add one PPO transition to the buffer.

        The dictionary keys are kept unchanged for compatibility
        with the existing PPO training pipeline.
        """

        self.buffer.append(
            {
                "fen":
                    fen,

                "action":
                    action,

                "legal_moves":
                    legal_moves,

                "return":
                    return_,

                "value":
                    value,

                "old_log_prob":
                    old_log_prob,

                "advantage":
                    advantage,

                "ply":
                    ply,

                "game_result":
                    game_result,
            }
        )

        # ----------------------------------------------------
        # FIFO capacity control
        # ----------------------------------------------------

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)


    # ========================================================
    # Sample
    # ========================================================

    def sample(
        self,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """
        Uniformly sample transitions without replacement.

        Randomness is controlled by the caller's global Python
        RNG seed.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if batch_size > len(self.buffer):
            raise ValueError(
                "Cannot sample more transitions than are "
                f"currently stored: requested {batch_size}, "
                f"available {len(self.buffer)}."
            )

        return random.sample(
            self.buffer,
            batch_size,
        )


    # ========================================================
    # Length
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.buffer
        )