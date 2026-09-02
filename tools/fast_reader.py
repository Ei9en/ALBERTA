import io
from pathlib import Path

import zstandard as zstd


def parse_header(line: str):
    line = line.strip()

    if not line.startswith("["):
        return None

    line = line[1:-1]

    key, value = line.split(" ", 1)

    return key, value.strip('"')


def open_text(path):

    path = Path(path)

    if path.suffix == ".zst":
        compressed = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        return io.TextIOWrapper(reader, encoding="utf-8")

    return open(path, "r", encoding="utf-8")


def iter_games(path):
    """
    Yield:
        headers (dict)
        game_text (str) : PGN EXACTEMENT comme dans le fichier.
    """

    with open_text(path) as text:

        headers = {}
        game_lines = []

        for line in text:

            # Début d'une nouvelle partie
            if line.startswith("[Event ") and game_lines:

                yield headers.copy(), "".join(game_lines)

                headers.clear()
                game_lines.clear()

            game_lines.append(line)

            if line.startswith("["):

                parsed = parse_header(line)

                if parsed:
                    key, value = parsed
                    headers[key] = value

        # Dernière partie
        if game_lines:
            yield headers.copy(), "".join(game_lines)


def iter_headers(path):

    for headers, _ in iter_games(path):
        yield headers