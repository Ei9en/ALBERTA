# League.py

import random


class League:

    def __init__(self, max_agents=10):

        self.max_agents = max_agents
        self.agents = {}


    def add_agent(self, name, model):

        self.agents[name] = model

        while len(self.agents) > self.max_agents:

            oldest = next(iter(self.agents))

            if oldest == "bc_init" and len(self.agents) > 1:

                keys = list(self.agents.keys())

                oldest = keys[1]

            del self.agents[oldest]


    def sample_opponent(self, exclude=None):

        names = list(self.agents.keys())

        if exclude is not None and exclude in names:
            names.remove(exclude)

        if len(names) == 0:
            raise ValueError(
                "No available opponent in league"
            )

        opponent = random.choice(names)

        return (
            opponent,
            self.agents[opponent]
        )


    def list_agents(self):

        return list(
            self.agents.keys()
        )