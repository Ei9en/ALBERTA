from __future__ import annotations

import argparse
import json
from pathlib import Path


# ============================================================
# Merge
# ============================================================

def merge_json_lists(
    input_files: list[Path],
) -> list:
    """
    Merge multiple JSON files containing top-level lists.

    Files are processed in the exact order provided on the CLI.
    """

    merged = []

    for path in input_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(
                f
            )

        if not isinstance(
            data,
            list,
        ):

            raise ValueError(
                f"{path} does not contain a JSON list "
                f"(found {type(data).__name__})."
            )

        print(
            f"{path.name}: {len(data):,} records"
        )

        merged.extend(
            data
        )

    return merged


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Merge ALBERTA uncertainty-statistics JSON files "
            "into a single ordered JSON list."
        )
    )

    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "Input JSON files, in the order they should be "
            "concatenated."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Merged output JSON file.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    merged = merge_json_lists(
        args.inputs
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            merged,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("MERGE COMPLETED")
    print("=" * 60)

    print(
        f"Files merged  : {len(args.inputs)}"
    )

    print(
        f"Total records : {len(merged):,}"
    )

    print(
        f"Output        : {args.output}"
    )


if __name__ == "__main__":
    main()