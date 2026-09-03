# ============================================================
# Train_BC.py
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import time
import json

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
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


# ============================================================
# Training
# ============================================================

EPOCHS = 10

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4


# ============================================================
# Validation
# ============================================================

VALIDATION_RATIO = 0.20

SPLIT_SEED = 42


# ============================================================
# Checkpoints temporaires
# ============================================================

SAVE_EVERY = 40000


# ============================================================
# Loss log
# ============================================================

LOSS_LOG = (
    CHECKPOINT_DIR
    / "training_loss_bc.json"
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
    train_loss,
    val_loss,
    val_accuracy,
    history,
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

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

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
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()


    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    with torch.no_grad():

        for x, y in loader:

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
            # Cross-Entropy
            # ------------------------------------------------

            loss = criterion(
                logits,
                y
            )


            # ------------------------------------------------
            # Accumulation loss
            #
            # On pondère par le nombre d'exemples afin
            # d'obtenir une vraie moyenne sur le dataset.
            # ------------------------------------------------

            batch_size = (
                y.size(0)
            )


            total_loss += (
                loss.item()
                * batch_size
            )


            # ------------------------------------------------
            # Action agreement
            # ------------------------------------------------

            predictions = (
                logits.argmax(
                    dim=1
                )
            )


            total_correct += (
                predictions == y
            ).sum().item()


            total_samples += (
                batch_size
            )


    # ========================================================
    # Final metrics
    # ========================================================

    average_loss = (
        total_loss
        / total_samples
    )


    accuracy = (
        total_correct
        / total_samples
    )


    return (
        average_loss,
        accuracy,
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Device
    # ========================================================

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


    print(
        "Total positions:",
        len(dataset)
    )


    # ========================================================
    # Train / validation split
    # ========================================================

    validation_size = int(
        len(dataset)
        * VALIDATION_RATIO
    )


    train_size = (
        len(dataset)
        - validation_size
    )


    generator = (
        torch.Generator()
        .manual_seed(
            SPLIT_SEED
        )
    )


    train_dataset, val_dataset = (
        random_split(
            dataset,
            [
                train_size,
                validation_size,
            ],
            generator=generator,
        )
    )


    print()

    print(
        "======================================"
    )

    print(
        "DATASET SPLIT"
    )

    print(
        "======================================"
    )


    print(
        f"Total:      {len(dataset)}"
    )

    print(
        f"Training:   {len(train_dataset)} "
        f"({100 * len(train_dataset) / len(dataset):.1f}%)"
    )

    print(
        f"Validation: {len(val_dataset)} "
        f"({100 * len(val_dataset) / len(dataset):.1f}%)"
    )

    print(
        f"Seed:       {SPLIT_SEED}"
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
        "======================================"
    )

    print(
        "MODEL"
    )

    print(
        "======================================"
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
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
            map_location=device,
        )


        # ----------------------------------------------------
        # Vérification action space
        # ----------------------------------------------------

        assert (
            checkpoint["actions"]
            == len(ACTIONS)
        ), (
            "Action space mismatch"
        )


        # ----------------------------------------------------
        # Charger modèle
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        # ----------------------------------------------------
        # Charger optimizer
        # ----------------------------------------------------

        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )


            print(
                "Optimizer state loaded."
            )


        # ----------------------------------------------------
        # Reprendre après l'epoch
        # ----------------------------------------------------

        START_EPOCH = (
            checkpoint["epoch"]
            + 1
        )


        print(
            "Loaded epoch:",
            checkpoint["epoch"]
        )


        # ----------------------------------------------------
        # Anciennes métriques
        # ----------------------------------------------------

        if "train_loss" in checkpoint:

            print(
                "Previous train loss:",
                checkpoint["train_loss"]
            )

        elif "loss" in checkpoint:

            print(
                "Previous loss:",
                checkpoint["loss"]
            )


        if "val_loss" in checkpoint:

            print(
                "Previous val loss:",
                checkpoint["val_loss"]
            )


        if "val_accuracy" in checkpoint:

            print(
                "Previous val accuracy:",
                checkpoint[
                    "val_accuracy"
                ]
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

    for epoch in range(
        START_EPOCH,
        EPOCHS
    ):

        model.train()


        start = time.time()


        total_train_loss = 0.0

        total_train_samples = 0


        print()

        print(
            "======================================"
        )

        print(
            f"Epoch {epoch}"
        )

        print(
            "======================================"
        )


        pbar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch}"
        )


        # ====================================================
        # Training batches
        # ====================================================

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

            current_batch_size = (
                y.size(0)
            )


            total_train_loss += (
                loss.item()
                * current_batch_size
            )


            total_train_samples += (
                current_batch_size
            )


            current_train_loss = (
                total_train_loss
                /
                total_train_samples
            )


            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{current_train_loss:.4f}",
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
                    f"bc_temp_e"
                    f"{epoch}_b"
                    f"{batch}.pt"
                )


                save_checkpoint(
                    temp_path,
                    epoch,
                    batch,
                    model,
                    optimizer,
                    current_train_loss,
                    None,
                    None,
                    history,
                )


        # ====================================================
        # Training loss
        # ====================================================

        train_loss = (
            total_train_loss
            /
            total_train_samples
        )


        # ====================================================
        # Validation
        # ====================================================

        print()

        print(
            "Running validation..."
        )


        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion,
                device,
            )
        )


        # ====================================================
        # Epoch time
        # ====================================================

        epoch_time = (
            time.time()
            - start
        ) / 60


        # ====================================================
        # Gap
        # ====================================================

        loss_gap = (
            val_loss
            - train_loss
        )


        # ====================================================
        # History
        # ====================================================

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "loss_gap":
                    loss_gap,

                "time_min":
                    epoch_time,
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
        # Print metrics
        # ====================================================

        print()

        print(
            "======================================"
        )

        print(
            f"Epoch {epoch} RESULTS"
        )

        print(
            "======================================"
        )


        print(
            f"Training loss:     "
            f"{train_loss:.6f}"
        )


        print(
            f"Validation loss:   "
            f"{val_loss:.6f}"
        )


        print(
            f"Action agreement:  "
            f"{val_accuracy:.2%}"
        )


        print(
            f"Train/Val gap:     "
            f"{loss_gap:+.6f}"
        )


        print(
            f"Epoch time:        "
            f"{epoch_time:.1f} min"
        )


        print(
            "======================================"
        )


        # ====================================================
        # End epoch checkpoint
        # ====================================================

        path = (
            CHECKPOINT_DIR
            /
            f"bc_epoch_{epoch}.pt"
        )


        save_checkpoint(
            path,
            epoch,
            -1,
            model,
            optimizer,
            train_loss,
            val_loss,
            val_accuracy,
            history,
        )


    # ========================================================
    # Final
    # ========================================================

    print()

    print(
        "======================================"
    )

    print(
        "BC TRAINING FINISHED"
    )

    print(
        "======================================"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()