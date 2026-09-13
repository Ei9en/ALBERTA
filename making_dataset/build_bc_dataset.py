from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions_space import ACTION_TO_INDEX


# ============================================================
# Default paths
# ============================================================

DEFAULT_INPUT = Path(
    "data/processed/positions_2300.jsonl"
)

DEFAULT_OUTPUT = Path(
    "data/processed/positions_2300_bc.jsonl"
)


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Convert UCI moves from the processed Atomic Chess "
            "dataset into ALBERTA action-space indices for "
            "behavioral cloning."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input JSONL dataset containing a 'uci' field "
            f"(default: {DEFAULT_INPUT})."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSONL dataset containing an 'action' field "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )

    args = parser.parse_args()

    input_file: Path = args.input
    output_file: Path = args.output

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_file}"
        )

    if not input_file.is_file():
        raise ValueError(
            f"Input path is not a file: {input_file}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Convert dataset
    #
    # No random seed is required here:
    # this transformation is fully deterministic.
    # --------------------------------------------------------

    converted = 0
    skipped = 0

    with open(
        input_file,
        "r",
        encoding="utf-8",
    ) as fin, open(
        output_file,
        "w",
        encoding="utf-8",
    ) as fout:

        for line_number, line in enumerate(
            fin,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at line {line_number} "
                    f"in {input_file}"
                ) from exc

            if "uci" not in sample:
                raise KeyError(
                    f"Missing 'uci' field at line "
                    f"{line_number} in {input_file}"
                )

            uci = sample.pop("uci")

            action = ACTION_TO_INDEX.get(
                uci
            )

            if action is None:
                skipped += 1
                continue

            sample["action"] = action

            fout.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
            )
            fout.write("\n")

            converted += 1

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("=" * 60)
    print("ALBERTA - BUILD BC DATASET")
    print("=" * 60)

    print(
        f"Input     : {input_file}"
    )
    print(
        f"Converted : {converted:,}"
    )
    print(
        f"Skipped   : {skipped:,}"
    )
    print(
        f"Saved to  : {output_file}"
    )


if __name__ == "__main__":
    main()