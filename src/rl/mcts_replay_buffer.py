import random


class MCTSReplayBuffer:

    def __init__(
        self,
        capacity=100000,
    ):

        self.capacity = capacity

        self.buffer = []


    def __len__(self):

        return len(self.buffer)


    def add(
        self,
        fen,
        policy,
        value,
        result,
        visits=None,
    ):

        sample = {
            "fen": fen,

            "policy": policy,

            "value": float(value),

            "result": result,
        }


        if visits is not None:

            sample["visits"] = visits


        if len(self.buffer) >= self.capacity:

            self.buffer.pop(0)


        self.buffer.append(
            sample
        )


    def sample(
        self,
        batch_size,
    ):

        if batch_size > len(
            self.buffer
        ):

            batch_size = len(
                self.buffer
            )


        return random.sample(
            self.buffer,
            batch_size,
        )


    def clear(self):

        self.buffer.clear()


    def get_all(self):

        return self.buffer


    def state_dict(self):

        return {
            "capacity":
                self.capacity,

            "buffer":
                self.buffer,
        }


    def load_state_dict(
        self,
        state,
    ):

        self.capacity = state[
            "capacity"
        ]

        self.buffer = state[
            "buffer"
        ]


    def __getitem__(
        self,
        index,
    ):

        return self.buffer[index]