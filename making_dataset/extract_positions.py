from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import chess.variant


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fast_reader import iter_games


# ============================================================
# Default paths
# ============================================================

DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT = Path("data/processed/positions_2300.jsonl")


# ============================================================
# Default configuration
# ============================================================

DEFAULT_MIN_ELO = 2300


# ============================================================
# PGN cleanup
# ============================================================

COMMENT_RE = re.compile(r"\{.*?\}")
MOVE_NUMBER_RE = re.compile(r"\d+\.(\.\.)?")


def extract_moves(game_text: str) -> list[str]:
    """
    Extract the SAN move sequence from one PGN game.

    The fast PGN reader returns the complete game as text.
    Comments, move numbers, game results and SAN annotations
    are removed before the moves are replayed with python-chess.

    This parser intentionally preserves the historical ALBERTA
    preprocessing behavior used to build the BC dataset.
    """

    _, moves = game_text.split("\n\n", 1)

    moves = COMMENT_RE.sub("", moves)
    moves = MOVE_NUMBER_RE.sub("", moves)

    for result in (
        "1-0",
        "0-1",
        "1/2-1/2",
        "*",
    ):
        moves = moves.replace(result, "")

    return [
        move.rstrip("!?")
        for move in moves.split()
    ]


# ============================================================
# Main extraction
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract Atomic Chess positions from Lichess PGN "
            "archives, retaining only games in which both "
            "players satisfy the Elo threshold."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing the input .pgn.zst files "
            f"(default: {DEFAULT_INPUT_DIR})."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSONL file "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )

    parser.add_argument(
        "--min-elo",
        type=int,
        default=DEFAULT_MIN_ELO,
        help=(
            "Minimum Elo required for both players "
            f"(default: {DEFAULT_MIN_ELO})."
        ),
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output: Path = args.output
    min_elo: int = args.min_elo

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if min_elo < 0:
        raise ValueError(
            "--min-elo must be non-negative."
        )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Deterministic input ordering
    #
    # No random seed is required by this script. Given the
    # same PGN archives and configuration, files are processed
    # in a fixed lexicographic order.
    # --------------------------------------------------------

    input_files = sorted(
        input_dir.glob("*.pgn.zst")
    )

    if not input_files:
        raise FileNotFoundError(
            f"No .pgn.zst files found in {input_dir}"
        )

    print("=" * 60)
    print("ALBERTA - EXTRACT ATOMIC POSITIONS")
    print("=" * 60)

    print(f"Input directory : {input_dir}")
    print(f"Output file     : {output}")
    print(f"PGN files       : {len(input_files)}")
    print(f"Elo filter      : >= {min_elo}")

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    games = 0
    skipped_elo = 0
    failed = 0
    positions = 0

    # --------------------------------------------------------
    # Extract positions
    # --------------------------------------------------------

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as out:

        for input_file in input_files:

            print()
            print(
                f"Processing {input_file.name}"
            )

            for headers, game_text in iter_games(
                input_file
            ):

                # ------------------------------------------------
                # Elo filtering
                # ------------------------------------------------

                try:
                    white_elo = int(
                        headers.get(
                            "WhiteElo",
                            0,
                        )
                    )

                    black_elo = int(
                        headers.get(
                            "BlackElo",
                            0,
                        )
                    )

                except (TypeError, ValueError):

                    failed += 1
                    continue

                if (
                    white_elo < min_elo
                    or black_elo < min_elo
                ):
                    skipped_elo += 1
                    continue

                games += 1

                # ------------------------------------------------
                # Replay the Atomic Chess game.
                #
                # Each sample contains the position BEFORE the
                # played move and the corresponding move in UCI.
                # ------------------------------------------------

                board = chess.variant.AtomicBoard()

                san = None

                try:

                    moves = extract_moves(
                        game_text
                    )

                    for san in moves:

                        move = board.parse_san(
                            san
                        )

                        sample = {
                            "fen": board.fen(),
                            "uci": move.uci(),
                            "result": headers["Result"],
                            "white_elo": white_elo,
                            "black_elo": black_elo,
                        }

                        out.write(
                            json.dumps(sample)
                        )
                        out.write("\n")

                        board.push(move)

                        positions += 1

                except Exception as exc:

                    failed += 1

                    print("=" * 60)
                    print(
                        f"Game #{games}"
                    )
                    print(
                        headers.get("White", "?"),
                        "-",
                        headers.get("Black", "?"),
                    )
                    print(
                        f"SAN : "
                        f"{san if san is not None else '?'}"
                    )
                    print(exc)

                if games % 1000 == 0:
                    print(
                        f"{games:,} kept games"
                    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        f"Games kept      : {games:,}"
    )
    print(
        f"Games skipped   : {skipped_elo:,}"
    )
    print(
        f"Games failed    : {failed:,}"
    )
    print(
        f"Positions       : {positions:,}"
    )
    print(
        f"Average/game    : "
        f"{positions / max(1, games):.2f}"
    )
    print(
        f"Saved to        : {output}"
    )


if __name__ == "__main__":
    main()