import random


class OracleReplayBuffer:

    def __init__(self, capacity=50000):

        self.capacity = capacity

        self.buffer = []


    def clear(self):

        self.buffer.clear()


    def add(
        self,
        fen,
        oracle_move,
        confidence,
        criticality,
    ):

        self.buffer.append({

            "fen":
                fen,

            "oracle_move":
                oracle_move,

            "confidence":
                confidence,

            "criticality":
                criticality,
        })


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