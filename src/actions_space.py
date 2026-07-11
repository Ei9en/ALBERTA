import chess


PROMOTIONS = [
    None,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
]


def build_action_space():

    actions = []

    for from_sq in chess.SQUARES:

        for to_sq in chess.SQUARES:

            if from_sq == to_sq:
                continue

            for promo in PROMOTIONS:

                move = chess.Move(
                    from_sq,
                    to_sq,
                    promotion=promo
                )

                actions.append(move.uci())

    return actions


ACTIONS = build_action_space()

ACTION_TO_INDEX = {
    move: idx
    for idx, move in enumerate(ACTIONS)
}


INDEX_TO_ACTION = {
    idx: move
    for idx, move in enumerate(ACTIONS)
}