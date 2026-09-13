from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import zstandard as zstd


# ============================================================
# PGN header parsing
# ============================================================

def parse_header(
    line: str,
) -> tuple[str, str] | None:
    """
    Parse one PGN header line.

    Example:
        [WhiteElo "2400"]

    Returns:
        ("WhiteElo", "2400")

    Returns None for non-header lines.
    """

    line = line.strip()

    if not (
        line.startswith("[")
        and line.endswith("]")
    ):
        return None

    content = line[1:-1]

    if " " not in content:
        return None

    key, value = content.split(
        " ",
        1,
    )

    return (
        key,
        value.strip('"'),
    )


# ============================================================
# Text input
# ============================================================

@contextmanager
def open_text(
    path: str | Path,
) -> Iterator[TextIO]:
    """
    Open a plain-text or Zstandard-compressed PGN file.

    Files ending in '.zst' are streamed through the Zstandard
    decompressor rather than being fully decompressed in memory.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    if path.suffix == ".zst":

        with open(
            path,
            "rb",
        ) as compressed:

            with zstd.ZstdDecompressor().stream_reader(
                compressed
            ) as reader:

                with io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                ) as text:

                    yield text

    else:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as text:

            yield text


# ============================================================
# Game iterator
# ============================================================

def iter_games(
    path: str | Path,
) -> Iterator[tuple[dict[str, str], str]]:
    """
    Iterate over PGN games without loading the whole file.

    Yields:
        headers:
            Dictionary containing the PGN headers.

        game_text:
            Complete PGN text for the game, preserving the
            original lines from the source file.

    A new game is detected from its '[Event ...]' header,
    matching the structure of the Lichess PGN archives used
    by ALBERTA.
    """

    with open_text(path) as text:

        headers: dict[str, str] = {}
        game_lines: list[str] = []

        for line in text:

            # ------------------------------------------------
            # The Lichess archives start every game with an
            # Event header. If we already accumulated lines,
            # this marks the beginning of the next game.
            # ------------------------------------------------

            if (
                line.startswith("[Event ")
                and game_lines
            ):

                yield (
                    headers.copy(),
                    "".join(game_lines),
                )

                headers.clear()
                game_lines.clear()

            game_lines.append(line)

            # ------------------------------------------------
            # Collect PGN headers
            # ------------------------------------------------

            if line.startswith("["):

                parsed = parse_header(
                    line
                )

                if parsed is not None:

                    key, value = parsed
                    headers[key] = value

        # ----------------------------------------------------
        # Final game
        # ----------------------------------------------------

        if game_lines:

            yield (
                headers.copy(),
                "".join(game_lines),
            )


# ============================================================
# Header-only iterator
# ============================================================

def iter_headers(
    path: str | Path,
) -> Iterator[dict[str, str]]:
    """
    Iterate over PGN headers while discarding the move text.
    """

    for headers, _ in iter_games(path):
        yield headers