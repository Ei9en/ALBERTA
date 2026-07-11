from pathlib import Path
import sys
import re
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chess.variant

from dataset.fast_reader import iter_games

INPUT = "data/filtered/atomic_2000_2026-06.pgn"
OUTPUT = "data/processed/positions.jsonl"

COMMENT_RE = re.compile(r"\{.*?\}")
MOVE_NUMBER_RE = re.compile(r"\d+\.(\.\.)?")


def extract_moves(game_text):

    _, moves = game_text.split("\n\n", 1)

    moves = COMMENT_RE.sub("", moves)
    moves = MOVE_NUMBER_RE.sub("", moves)

    for result in ("1-0", "0-1", "1/2-1/2", "*"):
        moves = moves.replace(result, "")

    return [move.rstrip("!?") for move in moves.split()]


def main():

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    games = 0
    failed = 0
    positions = 0

    with open(OUTPUT, "w", encoding="utf-8") as out:

        for headers, game_text in iter_games(INPUT):

            games += 1

            board = chess.variant.AtomicBoard()

            try:

                moves = extract_moves(game_text)

                for san in moves:

                    move = board.parse_san(san)

                    sample = {
                        "fen": board.fen(),
                        "uci": move.uci(),
                        "result": headers["Result"],
                        "white_elo": int(headers["WhiteElo"]),
                        "black_elo": int(headers["BlackElo"]),
                    }

                    out.write(json.dumps(sample))
                    out.write("\n")

                    board.push(move)

                    positions += 1

            except Exception as e:

                failed += 1

                print("=" * 60)
                print(f"Game #{games}")
                print(headers.get("White"), "-", headers.get("Black"))
                print("SAN :", san)
                print(e)

            if games % 1000 == 0:
                print(f"{games:,} games")

    print()
    print("=" * 60)
    print(f"Games processed : {games:,}")
    print(f"Games failed    : {failed:,}")
    print(f"Games kept      : {games - failed:,}")
    print(f"Positions       : {positions:,}")
    print(f"Average/game    : {positions / max(1, games - failed):.2f}")
    print(f"Saved to        : {OUTPUT}")


if __name__ == "__main__":
    main()