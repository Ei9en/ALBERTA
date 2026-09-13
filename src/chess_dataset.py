from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.encoding import encode_fen


class ChessDataset(Dataset):
    """
    Behavioral-cloning dataset for ALBERTA.

    Each JSONL sample is expected to contain:
        - "fen": Atomic Chess position in FEN format
        - "action": target move index in ALBERTA's fixed action space

    Samples are loaded into memory at initialization. Board
    encoding is performed lazily in __getitem__.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)
        self.samples: list[dict] = []

        bad = 0

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset file does not exist: {self.path}"
            )

        if not self.path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {self.path}"
            )

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    sample = json.loads(line)

                except json.JSONDecodeError:
                    bad += 1
                    continue

                if (
                    "fen" not in sample
                    or "action" not in sample
                ):
                    bad += 1
                    continue

                self.samples.append(
                    sample
                )

        print(
            f"Loaded samples   : {len(self.samples):,}"
        )
        print(
            f"Ignored bad lines: {bad:,}"
        )

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        sample = self.samples[idx]

        x = encode_fen(
            sample["fen"]
        )

        y = sample["action"]

        return (
            torch.as_tensor(
                x,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                y,
                dtype=torch.long,
            ),
        )