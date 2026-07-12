import json
import torch

from torch.utils.data import Dataset

from src.encoding import encode_fen


class ChessDataset(Dataset):

    def __init__(self, path):

        self.samples = []

        bad = 0

        with open(path, "r") as f:

            for line in f:

                try:
                    sample = json.loads(line)

                    self.samples.append(sample)

                except Exception:
                    bad += 1

        print(f"Loaded samples: {len(self.samples):,}")
        print(f"Ignored bad lines: {bad:,}")


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]

        x = encode_fen(
            sample["fen"]
        )

        y = sample["action"]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )