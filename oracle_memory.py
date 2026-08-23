from __future__ import annotations

import chess

from oracle_queue import OracleQueue


# ============================================================
# Configuration
# ============================================================

QUEUE_PATH = "data/oracle_queue.jsonl"


# ============================================================
# Display
# ============================================================

def print_separator():
    print()
    print("=" * 70)
    print()


def display_query(query):
    """
    Display all relevant information about an oracle query.
    """

    board = chess.Board(query.fen)

    print_separator()

    print("ALBERTA - ORACLE QUERY")
    print_separator()

    print(f"Query ID : {query.query_id}")
    print(f"Model    : {query.model}")
    print(f"Epoch    : {query.epoch}")
    print(f"Game     : {query.game_id}")
    print(f"Ply      : {query.ply}")

    print()

    print("ACTIVE LEARNING SIGNAL")
    print("-" * 70)

    print(f"H         : {query.H:.6f}")
    print(f"U         : {query.U:.6f}")
    print(f"HU        : {query.HU:.6f}")
    print(f"I         : {query.score:.6f}")
    print(f"Threshold : {query.threshold:.6f}")

    print()

    print("POSITION")
    print("-" * 70)

    print(board)

    print()

    print(f"Turn      : {'White' if board.turn else 'Black'}")
    print(f"Move      : {board.fullmove_number}")

    print()

    print("FEN")
    print("-" * 70)

    print(query.fen)

    print()

    print("LEGAL MOVES")
    print("-" * 70)

    legal_moves = list(board.legal_moves)

    print(
        " ".join(
            move.uci()
            for move in legal_moves
        )
    )

    print()


# ============================================================
# Move parsing
# ============================================================

def parse_move(
    board: chess.Board,
    text: str,
):
    """
    Parse either UCI or SAN notation.

    Examples:

        e2e4
        Nf3
        Qxh7+
    """

    text = text.strip()

    #
    # Try UCI first.
    #
    try:

        move = chess.Move.from_uci(text)

        if move in board.legal_moves:
            return move

    except ValueError:
        pass

    #
    # Then try SAN.
    #
    try:

        return board.parse_san(text)

    except ValueError:

        return None


# ============================================================
# Query handling
# ============================================================

def process_query(
    queue: OracleQueue,
    query,
):
    """
    Interactively process one oracle query.
    """

    display_query(query)

    while True:

        answer = input(
            "Your move "
            "[s = skip, q = quit] > "
        ).strip()

        #
        # Quit.
        #
        if answer.lower() == "q":

            return "quit"

        #
        # Skip.
        #
        if answer.lower() == "s":

            queue.discard(
                query.query_id
            )

            print()
            print(
                "Query discarded."
            )

            return "done"

        #
        # Parse move.
        #
        board = chess.Board(
            query.fen
        )

        move = parse_move(
            board,
            answer,
        )

        if move is None:

            print()
            print(
                "Invalid or illegal move."
            )

            print(
                "Use UCI (e2e4) "
                "or SAN (Nf3, Qxh7+, ...)."
            )

            continue

        #
        # Confirm before saving.
        #
        print()

        print(
            f"Selected move: "
            f"{board.san(move)} "
            f"({move.uci()})"
        )

        confirmation = input(
            "Confirm? [y/n] > "
        ).strip().lower()

        if confirmation != "y":

            print(
                "Move cancelled."
            )

            continue

        #
        # Save answer.
        #
        queue.answer(
            query.query_id,
            board.san(move),
        )

        print()
        print(
            "Oracle answer saved."
        )

        return "done"


# ============================================================
# Main loop
# ============================================================

def main():

    queue = OracleQueue(
        QUEUE_PATH
    )

    print_separator()

    print(
        "ALBERTA - ORACLE INTERFACE"
    )

    print_separator()

    while True:

        stats = queue.stats()

        print(
            f"Pending:   {stats['pending']}"
        )

        print(
            f"Answered:  {stats['answered']}"
        )

        print(
            f"Discarded: {stats['discarded']}"
        )

        query = queue.next()

        if query is None:

            print()
            print(
                "No pending oracle queries."
            )

            print()

            return

        result = process_query(
            queue,
            query,
        )

        if result == "quit":

            print()
            print(
                "Oracle interface closed."
            )

            return


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()