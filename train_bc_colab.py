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


# ============================================================
# Configuration
# ============================================================

DATASET = (
    "/content/drive/MyDrive/ALBERTA/positions_2300_bc.jsonl"
)

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/ALBERTA/checkpoints/bc_epoch"
)

# ============================================================
# Checkpoint de départ
# ============================================================

PRETRAINED_CHECKPOINT = (
    CHECKPOINT_DIR
    / "bc_v3_epoch_0.pt"
)

EPOCHS = 10

SAVE_EVERY = 40000

LOSS_LOG = (
    CHECKPOINT_DIR
    / "training_loss_v3.json"
)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    epoch,
    batch,
    model,
    optimizer,
    loss,
    history
):

    checkpoint = {

        "epoch":
            epoch,

        "batch":
            batch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "loss":
            loss,

        "actions":
            len(ACTIONS),

        "loss_history":
            history,
    }

    torch.save(
        checkpoint,
        path
    )

    print()
    print(
        "Saved checkpoint:",
        path
    )


# ============================================================
# Main
# ============================================================

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "======================================"
    )
    print(
        "BC TRAINING - 32 CHANNEL RESNET"
    )
    print(
        "======================================"
    )
    print()

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

        torch.backends.cudnn.benchmark = True


    # ========================================================
    # Directories
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "Loading dataset..."
    )

    dataset = ChessDataset(
        DATASET
    )

    print(
        "Dataset loaded."
    )


    # ========================================================
    # DataLoader
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )


    # ========================================================
    # Model
    # ========================================================

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    ).to(device)


    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model:"
    )

    print(
        "Channels:",
        32
    )

    print(
        "Residual blocks:",
        4
    )

    print(
        "Actions:",
        len(ACTIONS)
    )

    print(
        "Parameters:",
        f"{parameters:,}"
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    if PRETRAINED_CHECKPOINT.exists():

        print()
        print(
            "======================================"
        )

        print(
            "Loading checkpoint:"
        )

        print(
            PRETRAINED_CHECKPOINT
        )

        print(
            "======================================"
        )


        checkpoint = torch.load(
            PRETRAINED_CHECKPOINT,
            map_location=device
        )


        # ----------------------------------------------------
        # Vérification action space
        # ----------------------------------------------------

        assert checkpoint["actions"] == len(ACTIONS), (
            "Action space mismatch"
        )


        # ----------------------------------------------------
        # Charger modèle
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        # ----------------------------------------------------
        # Charger optimizer
        # ----------------------------------------------------

        if "optimizer_state_dict" in checkpoint:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            print(
                "Optimizer state loaded."
            )


        # ----------------------------------------------------
        # Reprendre après l'epoch du checkpoint
        # ----------------------------------------------------

        START_EPOCH = (
            checkpoint["epoch"] + 1
        )


        print(
            "Loaded epoch:",
            checkpoint["epoch"]
        )

        print(
            "Previous loss:",
            checkpoint["loss"]
        )

        print(
            "Resume from epoch:",
            START_EPOCH
        )


    else:

        print()
        print(
            "WARNING: checkpoint not found:"
        )

        print(
            PRETRAINED_CHECKPOINT
        )

        print(
            "Training from scratch."
        )

        START_EPOCH = 0


    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # History
    # ========================================================

    if LOSS_LOG.exists():

        with open(
            LOSS_LOG,
            "r"
        ) as f:

            history = json.load(f)

    else:

        history = []


    # ========================================================
    # Training
    # ========================================================

    model.train()

    for epoch in range(
        START_EPOCH,
        EPOCHS
    ):

        start = time.time()

        total_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}"
        )


        for batch, (x, y) in enumerate(
            pbar
        ):

            # ------------------------------------------------
            # CPU -> GPU
            # ------------------------------------------------

            x = x.to(
                device
            )

            y = y.to(
                device
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                x
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                y
            )


            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            total_loss += (
                loss.item()
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


            # ------------------------------------------------
            # Temporary checkpoint
            # ------------------------------------------------

            if (
                batch > 0
                and batch % SAVE_EVERY == 0
            ):

                temp_path = (
                    CHECKPOINT_DIR
                    /
                    f"bc_v3_temp_e"
                    f"{epoch}_b"
                    f"{batch}.pt"
                )

                save_checkpoint(
                    temp_path,
                    epoch,
                    batch,
                    model,
                    optimizer,
                    loss.item(),
                    history
                )


        # ====================================================
        # Epoch statistics
        # ====================================================

        epoch_loss = (
            total_loss
            /
            len(loader)
        )

        epoch_time = (
            time.time()
            -
            start
        ) / 60


        history.append(
            {
                "epoch":
                    epoch,

                "loss":
                    epoch_loss,

                "time_min":
                    epoch_time
            }
        )


        # ====================================================
        # Save loss history
        # ====================================================

        with open(
            LOSS_LOG,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )


        # ====================================================
        # Print
        # ====================================================

        print()

        print(
            f"Epoch {epoch} loss:",
            epoch_loss
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f} min"
        )


        # ====================================================
        # End epoch checkpoint
        # ====================================================

        path = (
            CHECKPOINT_DIR
            /
            f"bc_v3_epoch_{epoch}.pt"
        )

        save_checkpoint(
            path,
            epoch,
            -1,
            model,
            optimizer,
            epoch_loss,
            history
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()