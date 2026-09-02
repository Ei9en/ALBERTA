import uuid
import argparse


def fen_to_query_id(fen):
    return uuid.uuid5(
        uuid.NAMESPACE_DNS,
        fen
    ).hex


def main():

    parser = argparse.ArgumentParser(
        description="Convert an Atomic FEN to an ALBERTA query_id."
    )

    parser.add_argument(
        "fen",
        type=str,
        help="Atomic FEN"
    )

    args = parser.parse_args()

    query_id = fen_to_query_id(
        args.fen
    )

    print(query_id)


if __name__ == "__main__":
    main()