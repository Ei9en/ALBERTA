from pathlib import Path
import json

import torch
from torch.utils.data import Dataset

from src.encoding import encode_fen


class ChessDataset(Dataset):

    def __init__(self, path):

        self.path = Path(path)

        with open(self.path, "r") as f:
            self.samples = f.readlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = json.loads(self.samples[idx])

        x = encode_fen(sample["fen"])
        y = torch.tensor(sample["action"], dtype=torch.long)

        return x, y