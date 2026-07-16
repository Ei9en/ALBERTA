# Replay_Buffer.py

import random

class ReplayBuffer:

    def __init__(
        self,
        capacity=50000,
    ):

        self.capacity = capacity
        self.buffer = []


    def clear(self):

        self.buffer.clear()


    def add(
        self,
        fen,
        action,
        legal_moves,
        target_return,
    ):

        self.buffer.append(
            {
                "fen": fen,
                "action": action,
                "legal_moves": legal_moves,
                "return": target_return,
            }
        )

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)


    def sample(
        self,
        batch_size,
    ):

        return random.sample(
            self.buffer,
            batch_size,
        )


    def __len__(self):

        return len(self.buffer)