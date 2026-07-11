from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

from src.actions_space import ACTION_TO_INDEX, INDEX_TO_ACTION

INPUT = "data/processed/positions.jsonl"
OUTPUT = "data/processed/positions_bc.jsonl"


def main():

    converted = 0
    skipped = 0

    with open(INPUT, "r") as fin, open(OUTPUT, "w") as fout:

        for line in fin:

            sample = json.loads(line)

            uci = sample.pop("uci")

            action = ACTION_TO_INDEX.get(uci)

            if action is None:
                skipped += 1
                continue

            sample["action"] = action

            fout.write(json.dumps(sample))
            fout.write("\n")

            converted += 1

    print(f"Converted : {converted:,}")
    print(f"Skipped   : {skipped:,}")
    print(f"Saved to  : {OUTPUT}")


if __name__ == "__main__":
    main()