import json

from torch.utils.data import Dataset

from encoding import encode_fen


class ChessDataset(Dataset):

    def __init__(self, filename, move_to_idx):

        self.samples = []

        with open(filename) as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.move_to_idx = move_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        x = encode_fen(sample["fen"])

        y = self.move_to_idx[sample["uci"]]

        return x, y