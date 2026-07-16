# League.py

import random


class League:

    def __init__(self, max_agents=3):

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
                if agent_name != "league_bc_init"
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


    def __len__(self):

        return len(self.agents)


    def names(self):

        return list(self.agents.keys())