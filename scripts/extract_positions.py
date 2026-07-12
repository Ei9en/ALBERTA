from pathlib import Path
import sys
import re
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chess.variant

from dataset.fast_reader import iter_games


RAW_DIR = Path("data/raw")
OUTPUT = "data/processed/positions_2300.jsonl"

MIN_ELO = 2300

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
    skipped_elo = 0
    failed = 0
    positions = 0

    input_files = sorted(RAW_DIR.glob("*.pgn.zst"))

    print(f"Found {len(input_files)} PGN files")
    print(f"Elo filter: >= {MIN_ELO}")

    with open(OUTPUT, "w", encoding="utf-8") as out:

        for input_file in input_files:

            print()
            print(f"Processing {input_file.name}")

            for headers, game_text in iter_games(input_file):

                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))

                except ValueError:
                    failed += 1
                    continue

                if white_elo < MIN_ELO or black_elo < MIN_ELO:
                    skipped_elo += 1
                    continue

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
                            "white_elo": white_elo,
                            "black_elo": black_elo,
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
                    print("SAN :", san if "san" in locals() else "?")
                    print(e)

                if games % 1000 == 0:
                    print(f"{games:,} kept games")


    print()
    print("=" * 60)
    print(f"Games kept      : {games:,}")
    print(f"Games skipped   : {skipped_elo:,}")
    print(f"Games failed    : {failed:,}")
    print(f"Positions       : {positions:,}")
    print(f"Average/game    : {positions / max(1, games):.2f}")
    print(f"Saved to        : {OUTPUT}")


if __name__ == "__main__":
    main()