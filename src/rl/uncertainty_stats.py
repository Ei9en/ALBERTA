import json
from pathlib import Path

import torch


class UncertaintyStats:

    def __init__(self):

        self.data = []


    @torch.no_grad()
    def compute_uncertainty(
        self,
        x,
        league,
        current_model,
    ):

        values = []


        #
        # Snapshots league
        #
        for model in league.agents.values():

            model.eval()

            _, value = model(x)

            values.append(
                value.item()
            )


        #
        # Modèle courant
        #
        current_model.eval()

        _, value = current_model(x)

        values.append(
            value.item()
        )


        if len(values) < 2:

            return 0.0


        return torch.var(
            torch.tensor(values),
            unbiased=False,
        ).item()



    def add(
        self,
        fen,
        entropy,
        uncertainty,
    ):

        self.data.append(
            {
                "fen": fen,

                "H": entropy,

                "U": uncertainty,

                "M": None,
            }
        )



    def save(
        self,
        path,
    ):

        path = Path(path)

        with open(path, "w") as f:

            json.dump(
                self.data,
                f,
                indent=2,
            )