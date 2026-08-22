import torch

from src.mcts.search import MCTS


class MCTSAgent:

    def __init__(
        self,
        model,
        simulations=100,
        c_puct=1.5,
        device="cpu",
        temperature=1.0,
        deterministic=True,
    ):

        self.model = model

        self.simulations = simulations

        self.c_puct = c_puct

        self.device = device

        self.temperature = temperature

        self.deterministic = deterministic

        self.mcts = MCTS(
            model=model,
            simulations=simulations,
            c_puct=c_puct,
            device=device,
        )


    def choose_move(
        self,
        board,
    ):

        result = self.mcts.search(
            board
        )

        move = result["move"]

        policy = result["policy"]

        visits = result.get(
            "visits",
            {},
        )

        root = result.get(
            "root",
            None,
        )


        if move is None:

            return {
                "move": None,
                "action": None,
                "policy": policy,
                "visits": visits,
                "value": 0.0,
                "root": root,
            }


        #
        # Valeur estimée à la racine
        #

        if root is not None:

            value = root.value

        else:

            value = 0.0


        return {
            "move": move,

            "action": move.uci(),

            "policy": policy,

            "visits": visits,

            "value": value,

            "root": root,
        }


    def choose_moves(
        self,
        boards,
    ):

        results = []

        for board in boards:

            results.append(
                self.choose_move(
                    board
                )
            )

        return results