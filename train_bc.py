import torch
import torch.nn as nn
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.chess_dataset import ChessDataset
from src.models.resnet import ChessResNet
from src.actions_space import ACTIONS


DATASET = "data/processed/positions_bc.jsonl"


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    dataset = ChessDataset(DATASET)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    ).to(device)


    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )


    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()


    model.train()

    epochs = 1


    for epoch in range(epochs):

        start = time.time()

        total_loss = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}"
        )

        for batch, (x, y) in enumerate(pbar):

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += loss.item()


            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


        epoch_loss = total_loss / len(loader)

        print()
        print(
            f"Epoch {epoch} loss:",
            epoch_loss
        )

        print(
            f"Epoch time: {(time.time()-start)/60:.1f} min"
        )


if __name__ == "__main__":
    main()