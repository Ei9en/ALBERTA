from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset.fast_reader import iter_headers
DATASET = "data/raw/lichess_db_atomic_rated_2026-06.pgn.zst"

total = 0
games_2000 = 0

for headers in iter_headers(DATASET):

    total += 1

    try:
        w = int(headers.get("WhiteElo", 0))
        b = int(headers.get("BlackElo", 0))

        if w >= 2000 and b >= 2000:
            games_2000 += 1

    except ValueError:
        pass

print(total)
print(games_2000)
print(round(games_2000/total*100, 2),'%')