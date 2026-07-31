import random
import torch


class League:

    def __init__(self, max_agents=22):

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
        # Snapshots de la league
        #
        for model in self.agents.values():

            model.eval()

            _, value = model(x)

            values.append(
                value.item()
            )

        #
        # Modèle courant
        #
        if current_model is not None:

            current_model.eval()

            _, value = current_model(x)

            values.append(
                value.item()
            )

        if len(values) < 2:

            return 0.0

        values = torch.tensor(
            values,
            device=x.device,
        )

        return torch.var(
            values,
            unbiased=False,
        ).item()


    #
    # =========================
    # Batched uncertainty
    # =========================
    #

    @torch.no_grad()
    def uncertainty_batch(
        self,
        x,
        current_model=None,
    ):

        #
        # x:
        #
        # [N, 19, 8, 8]
        #
        # Retour:
        #
        # [N]
        #
        all_values = []

        #
        # Snapshots de la league
        #
        for model in self.agents.values():

            model.eval()

            _, values = model(x)

            #
            # [N, 1] -> [N]
            #
            values = values.squeeze(-1)

            all_values.append(
                values
            )

        #
        # Modèle courant
        #
        if current_model is not None:

            current_model.eval()

            _, values = current_model(x)

            values = values.squeeze(-1)

            all_values.append(
                values
            )

        #
        # Pas assez d'agents
        #
        if len(all_values) < 2:

            return torch.zeros(
                x.shape[0],
                device=x.device,
            )

        #
        # [agents, N]
        #
        values = torch.stack(
            all_values,
            dim=0,
        )

        #
        # Variance entre agents
        #
        uncertainty = torch.var(
            values,
            dim=0,
            unbiased=False,
        )

        return uncertainty


    def __len__(self):

        return len(self.agents)


    def names(self):

        return list(
            self.agents.keys()
        )