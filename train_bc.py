import torch
import torch.nn as nn
import time
import json

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.chess_dataset import ChessDataset
from src.models.resnet import ChessResNet
from src.actions_space import ACTIONS


DATASET = "data/processed/positions_2300_bc.jsonl"

CHECKPOINT_DIR = Path("checkpoints")

# checkpoint V2 utilisé comme point de départ
#PRETRAINED_CHECKPOINT = CHECKPOINT_DIR / "bc_epoch_0.pt"
PRETRAINED_CHECKPOINT = "/content/bc_epoch_0.pt"
# nouveau checkpoint V2.5
START_EPOCH = 0

EPOCHS = 10

LOSS_LOG = CHECKPOINT_DIR / "training_loss.json"


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)


    CHECKPOINT_DIR.mkdir(exist_ok=True)


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


    # =========================
    # LOAD V2 CHECKPOINT
    # =========================

    if PRETRAINED_CHECKPOINT.exists():

        print("Loading:", PRETRAINED_CHECKPOINT)

        checkpoint = torch.load(
            PRETRAINED_CHECKPOINT,
            map_location=device
        )

        assert checkpoint["actions"] == len(ACTIONS), (
            "Action space mismatch"
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Loaded previous model, loss:",
            checkpoint["loss"]
        )

    else:
        print("No checkpoint found, training from scratch")


    criterion = nn.CrossEntropyLoss()


    # historique loss

    if LOSS_LOG.exists():

        with open(LOSS_LOG) as f:
            history = json.load(f)

    else:
        history = []


    model.train()


    for epoch in range(START_EPOCH, EPOCHS):

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


        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "time_min": (time.time()-start)/60
            }
        )


        with open(LOSS_LOG, "w") as f:
            json.dump(
                history,
                f,
                indent=2
            )


        print()
        print(
            f"Epoch {epoch} loss:",
            epoch_loss
        )

        print(
            f"Epoch time: {(time.time()-start)/60:.1f} min"
        )


        # =========================
        # SAVE CHECKPOINT
        # =========================

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                epoch_loss,

            "actions":
                len(ACTIONS),

            "loss_history":
                history,
        }


        path = CHECKPOINT_DIR / f"bc_v2_5_epoch_{epoch}.pt"


        torch.save(
            checkpoint,
            path
        )


        print(
            "Saved:",
            path
        )


if __name__ == "__main__":
    main()