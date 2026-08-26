# ============================================================
# ReplayBuffer
# ============================================================

import random


class ReplayBuffer:

    def __init__(
        self,
        capacity=300000,
    ):

        self.capacity = capacity

        self.buffer = []


    # ========================================================
    # Clear
    # ========================================================

    def clear(self):

        self.buffer.clear()


    # ========================================================
    # Add
    # ========================================================

    def add(
        self,
        fen,
        action,
        legal_moves,
        return_,
        value,
        old_log_prob,
        advantage,
        ply,
        game_result=None,
    ):

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
        # FIFO
        # ----------------------------------------------------

        if len(self.buffer) > self.capacity:

            self.buffer.pop(0)


    # ========================================================
    # Sample
    # ========================================================

    def sample(
        self,
        batch_size,
    ):

        return random.sample(
            self.buffer,
            batch_size,
        )


    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.buffer)