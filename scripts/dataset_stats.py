from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset.fast_reader import iter_headers

RAW_DIR = Path("data/raw")
MIN_ELO = 2300

total = 0
games_kept = 0

datasets = sorted(RAW_DIR.glob("*.pgn.zst"))

print(f"Found {len(datasets)} datasets")

for dataset in datasets:
    print(f"Processing {dataset.name}")

    for headers in iter_headers(dataset):

        total += 1

        try:
            w = int(headers.get("WhiteElo", 0))
            b = int(headers.get("BlackElo", 0))

            if w >= MIN_ELO and b >= MIN_ELO:
                games_kept += 1

        except ValueError:
            pass

print()
print(f"Total games       : {total}")
print(f"Games >= {MIN_ELO} : {games_kept}")

if total:
    print(f"Percentage        : {games_kept / total * 100:.2f}%")