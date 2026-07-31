# Batched_Selfplay_Game.py

import chess.variant

from src.agents.actor_critic_agent import ActorCriticAgent


class BatchedSelfPlayGame:

    def __init__(
        self,
        white_agents,
        black_agents,
    ):

        if len(white_agents) != len(black_agents):
            raise ValueError(
                "white_agents and black_agents "
                "must have the same length."
            )

        self.white_agents = white_agents
        self.black_agents = black_agents


    def play(self):

        n_games = len(
            self.white_agents
        )

        boards = [
            chess.variant.AtomicBoard()
            for _ in range(n_games)
        ]

        trajectories = [
            []
            for _ in range(n_games)
        ]

        active = list(
            range(n_games)
        )

        results = [
            None
            for _ in range(n_games)
        ]


        while active:

            #
            # Group active games by agent
            #
            groups = {}

            for game_idx in active:

                board = boards[game_idx]

                agent = (
                    self.white_agents[game_idx]
                    if board.turn
                    else self.black_agents[game_idx]
                )

                key = id(agent)

                if key not in groups:

                    groups[key] = {
                        "agent": agent,
                        "indices": [],
                    }

                groups[key]["indices"].append(
                    game_idx
                )


            #
            # Each agent performs one batched
            # forward pass for its games.
            #
            infos = {}

            for group in groups.values():

                agent = group["agent"]

                indices = group["indices"]

                group_boards = [
                    boards[i]
                    for i in indices
                ]

                group_infos = agent.choose_moves(
                    group_boards
                )

                for game_idx, info in zip(
                    indices,
                    group_infos,
                ):

                    infos[game_idx] = info


            #
            # Apply the selected moves
            #
            finished = []

            for game_idx in active:

                board = boards[game_idx]

                info = infos[game_idx]

                trajectories[game_idx].append(
                    {
                        "fen": board.fen(),

                        "action": info["action"],

                        "player": board.turn,

                        "value": info.get(
                            "value",
                            0.0,
                        ),

                        "entropy": info.get(
                            "entropy",
                            0.0,
                        ),

                        "legal_moves": [
                            move.uci()
                            for move in board.legal_moves
                        ],
                    }
                )

                board.push(
                    info["move"]
                )

                if board.is_game_over():

                    results[game_idx] = (
                        board.result()
                    )

                    finished.append(
                        game_idx
                    )


            #
            # Remove finished games
            #
            active = [
                i
                for i in active
                if i not in finished
            ]


        return [
            (
                trajectories[i],
                results[i],
            )
            for i in range(n_games)
        ]