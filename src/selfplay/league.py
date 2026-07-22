# League.py

import random
import torch


class League:

    def __init__(self, max_agents=20):

        self.max_agents = max_agents

        self.agents = {}


    def add_agent(
        self,
        name,
        model,
    ):

        self.agents[name] = model

        #
        # On conserve toujours le BC initial.
        # Les autres snapshots sont supprimés du plus ancien au plus récent.
        #
        while len(self.agents) > self.max_agents:

            removable = [
                agent_name
                for agent_name in self.agents
                if agent_name not in [
                    "bc_epoch_4",
                    "bc_epoch_5",
                ]
            ]

            if not removable:
                break

            oldest = removable[0]

            del self.agents[oldest]


    def sample_opponent(
        self,
        exclude=None,
    ):

        candidates = [
            (name, model)
            for name, model in self.agents.items()
            if name != exclude
        ]

        return random.choice(candidates)


    #
    # =========================
    # Uncertainty U(s)
    # =========================
    #

    @torch.no_grad()
    def values(
        self,
        x,
    ):

        values = []


        for model in self.agents.values():

            model.eval()

            _, value = model(x)

            values.append(
                value.item()
            )


        return values


    @torch.no_grad()
    def uncertainty(
        self,
        x,
        current_model=None,
    ):

        values = []


        #
        # Valeurs des snapshots de la league
        #
        for model in self.agents.values():

            model.eval()

            _, value = model(x)

            values.append(
                value.item()
            )


        #
        # Valeur du modèle en entraînement
        #
        if current_model is not None:

            current_model.eval()

            _, value = current_model(x)

            values.append(
                value.item()
            )


        #
        # Pas assez d'agents pour une variance
        #
        if len(values) < 2:

            return 0.0


        values = torch.tensor(values)


        return torch.var(
            values,
            unbiased=False,
        ).item()

    def __len__(self):

        return len(self.agents)

    def names(self):

        return list(self.agents.keys())