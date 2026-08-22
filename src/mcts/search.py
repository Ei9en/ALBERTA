import torch

from src.encoding import encode_boards
from src.actions_space import (
    ACTION_TO_INDEX,
)

from src.mcts.node import MCTSNode
from src.mcts.tree import MCTSTree


class MCTS:

    def __init__(
        self,
        model,
        simulations=100,
        c_puct=1.5,
        device="cpu",
    ):

        self.model = model

        self.simulations = (
            simulations
        )

        self.c_puct = c_puct

        self.device = device


    @torch.no_grad()
    def evaluate(
        self,
        board,
    ):

        x = encode_boards(
            [board]
        ).to(self.device)


        policy, value = (
            self.model(x)
        )


        policy = policy[0]

        value = (
            value[0]
            .item()
        )


        legal_moves = list(
            board.legal_moves
        )


        legal_indices = [
            ACTION_TO_INDEX[
                move.uci()
            ]
            for move in legal_moves
        ]


        legal_logits = policy[
            legal_indices
        ]


        priors = torch.softmax(
            legal_logits,
            dim=0,
        )


        return (
            legal_moves,
            priors.cpu().tolist(),
            value,
        )


    def expand(
        self,
        node,
    ):

        if node.is_terminal:

            return 0.0


        (
            legal_moves,
            priors,
            value,
        ) = self.evaluate(
            node.board
        )


        for move, prior in zip(
            legal_moves,
            priors,
        ):

            board = (
                node.board.copy()
            )


            board.push(
                move
            )


            child = MCTSNode(
                board=board,
                parent=node,
                action=move,
                prior=prior,
            )


            node.children[
                move.uci()
            ] = child


        return value


    def search(
        self,
        board,
    ):

        root = MCTSNode(
            board=board.copy()
        )


        #
        # Initial expansion
        #

        if root.is_terminal:

            return {
                "move": None,
                "policy": {},
                "root": root,
            }


        self.expand(
            root
        )


        tree = MCTSTree(
            root,
            c_puct=self.c_puct,
        )


        #
        # Simulations
        #

        for _ in range(
            self.simulations
        ):

            node = root


            #
            # Selection
            #

            while (
                node.is_expanded
                and not node.is_terminal
            ):

                node = (
                    tree.select_child(
                        node
                    )
                )


            #
            # Expansion
            #

            if node.is_terminal:

                value = (
                    self.terminal_value(
                        node
                    )
                )

            else:

                value = (
                    self.expand(
                        node
                    )
                )


            #
            # Backup
            #

            tree.backup(
                node,
                value,
            )


        #
        # Visit distribution
        #

        visits = {}

        total_visits = 0


        for (
            action,
            child,
        ) in root.children.items():

            visits[action] = (
                child.visit_count
            )

            total_visits += (
                child.visit_count
            )


        if total_visits > 0:

            policy = {
                action:
                    count
                    / total_visits

                for (
                    action,
                    count
                ) in visits.items()
            }

        else:

            policy = {
                action:
                    1.0
                    / len(visits)

                for action in visits
            }


        #
        # Best move
        #

        best_action = max(
            visits,
            key=visits.get,
        )


        best_move = (
            chess_move_from_uci(
                board,
                best_action,
            )
        )


        return {
            "move": best_move,
            "policy": policy,
            "visits": visits,
            "root": root,
        }


    def terminal_value(
        self,
        node,
    ):

        board = node.board


        result = (
            board.result()
        )


        if result == "1/2-1/2":

            return 0.0


        if result == "1-0":

            return (
                1.0
                if board.turn
                else -1.0
            )


        if result == "0-1":

            return (
                -1.0
                if board.turn
                else 1.0
            )


        return 0.0


def chess_move_from_uci(
    board,
    uci,
):

    for move in board.legal_moves:

        if move.uci() == uci:

            return move


    raise ValueError(
        f"Illegal move: {uci}"
    )