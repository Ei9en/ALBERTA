from __future__ import annotations

import chess


# ============================================================
# Promotion types
# ============================================================

PROMOTIONS = (
    None,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)


# ============================================================
# Action space
# ============================================================

def build_action_space() -> list[str]:
    """
    Build ALBERTA's fixed UCI action space.

    The action space contains every ordered pair of distinct
    chess squares, with each possible promotion type:
    no promotion, queen, rook, bishop, or knight.

    Legality is NOT encoded in the action space. Illegal moves
    are masked according to the current board position by the
    agent.

    IMPORTANT:
        The iteration order is part of the model and dataset
        specification. Changing it changes the mapping between
        policy indices and UCI moves and therefore breaks
        compatibility with existing datasets and checkpoints.
    """

    actions: list[str] = []

    for from_sq in chess.SQUARES:

        for to_sq in chess.SQUARES:

            if from_sq == to_sq:
                continue

            for promotion in PROMOTIONS:

                move = chess.Move(
                    from_sq,
                    to_sq,
                    promotion=promotion,
                )

                actions.append(
                    move.uci()
                )

    return actions


# ============================================================
# Fixed mappings
# ============================================================

ACTIONS = build_action_space()

ACTION_TO_INDEX = {
    move: index
    for index, move in enumerate(ACTIONS)
}

INDEX_TO_ACTION = {
    index: move
    for index, move in enumerate(ACTIONS)
}