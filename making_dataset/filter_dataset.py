from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset.fast_reader import iter_games

INPUT = "data/raw/lichess_db_atomic_rated_2026-06.pgn.zst"
OUTPUT = "data/filtered/atomic_2000_2026-06.pgn"


kept = 0
total = 0

with open(OUTPUT, "w", encoding="utf-8") as out:

    for headers, game_text in iter_games(INPUT):

        total += 1

        try:
            white = int(headers.get("WhiteElo", 0))
            black = int(headers.get("BlackElo", 0))

        except ValueError:
            continue

        if white >= 2000 and black >= 2000:
            out.write(game_text)
            out.write("\n\n")
            kept += 1


print(f"Total : {total}")
print(f"Kept  : {kept}")