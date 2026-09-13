from __future__ import annotations

import random


class OracleReplayBuffer:
    """
    FIFO replay buffer for expert / Oracle annotations.

    Random sampling uses Python's global RNG.
    Reproducibility is controlled by the caller.
    """

    def __init__(
        self,
        capacity: int = 50_000,
    ) -> None:

        if capacity <= 0:
            raise ValueError(
                "capacity must be greater than zero."
            )

        self.capacity = capacity
        self.buffer: list[dict] = []


    def clear(
        self,
    ) -> None:

        self.buffer.clear()


    def add(
        self,
        fen: str,
        oracle_move: str,
        confidence: float,
        criticality: float,
        reward: float,
    ) -> None:

        self.buffer.append(
            {
                "fen":
                    fen,

                "oracle_move":
                    oracle_move,

                "confidence":
                    confidence,

                "criticality":
                    criticality,

                "reward":
                    reward,
            }
        )

        if len(
            self.buffer
        ) > self.capacity:

            self.buffer.pop(
                0
            )


    def sample(
        self,
        batch_size: int,
    ) -> list[dict]:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if batch_size > len(
            self.buffer
        ):

            raise ValueError(
                "batch_size cannot exceed the number "
                "of stored Oracle samples."
            )

        return random.sample(
            self.buffer,
            batch_size,
        )


    def __len__(
        self,
    ) -> int:

        return len(
            self.buffer
        )