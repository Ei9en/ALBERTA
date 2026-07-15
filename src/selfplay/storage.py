import json
from pathlib import Path


class ReplayBuffer:

    def __init__(self, path):

        self.path = Path(path)

        self.path.parent.mkdir(
            exist_ok=True,
            parents=True,
        )


    def save_game(
        self,
        trajectory,
        result,
        white,
        black,
    ):

        data = {
            "white": white,
            "black": black,
            "result": result,
            "trajectory": trajectory,
        }

        with open(self.path, "a") as f:

            f.write(
                json.dumps(data)
                + "\n"
            )