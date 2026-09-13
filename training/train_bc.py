from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions_space import ACTIONS
from src.chess_dataset import ChessDataset
from src.models.resnet import ChessResNet


# ============================================================
# Default paths
# ============================================================

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "positions_2300_bc.jsonl"
)

DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
)

DEFAULT_LOSS_LOG = (
    DEFAULT_CHECKPOINT_DIR
    / "training_loss_bc.json"
)


# ============================================================
# Default training configuration
# ============================================================

DEFAULT_SEED = 42

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_VALIDATION_RATIO = 0.20

DEFAULT_SAVE_EVERY = 40000

DEFAULT_CHANNELS = 32
DEFAULT_BLOCKS = 4

DEFAULT_NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:
    """
    Seed all random-number generators used by the BC pipeline.

    The script aims at deterministic seeded reproduction.
    Exact bitwise equality can still depend on the PyTorch,
    CUDA, cuDNN and hardware versions used for execution.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Disable cuDNN autotuning because it may select different
    # convolution algorithms across executions.
    torch.backends.cudnn.benchmark = False

    # Prefer deterministic cuDNN implementations.
    torch.backends.cudnn.deterministic = True

    # Request deterministic PyTorch algorithms when available.
    # warn_only=True avoids aborting a long run if a specific
    # operation has no deterministic implementation.
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    batch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float | None,
    val_accuracy: float | None,
    history: list[dict],
    seed: int,
    config: dict,
) -> None:

    checkpoint = {
        "epoch": epoch,
        "batch": batch,

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

        # Reproducibility metadata
        "seed":
            seed,

        "config":
            config,
    }

    torch.save(
        checkpoint,
        path,
    )

    print()
    print(
        f"Saved checkpoint: {path}"
    )


# ============================================================
# Validation
# ============================================================

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            batch_size = y.size(0)

            # Weight each batch loss by its number of samples
            # so the final metric is a true dataset average.
            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions == y
            ).sum().item()

            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

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
# DataLoader
# ============================================================

def build_train_loader(
    dataset,
    batch_size: int,
    epoch: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Build the training DataLoader for one epoch.

    The shuffle generator is seeded from the master seed and
    the epoch number. This makes the sample order of each epoch
    reproducible independently, including after checkpoint
    resume at an epoch boundary.
    """

    generator = torch.Generator()

    generator.manual_seed(
        seed + epoch
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train ALBERTA's behavioral-cloning policy "
            "network on the processed Atomic Chess dataset."
        )
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Behavioral-cloning JSONL dataset "
            f"(default: {DEFAULT_DATASET})."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Directory used for checkpoints "
            f"(default: {DEFAULT_CHECKPOINT_DIR})."
        ),
    )

    parser.add_argument(
        "--loss-log",
        type=Path,
        default=None,
        help=(
            "JSON file used for training history. "
            "Defaults to training_loss_bc.json inside "
            "--checkpoint-dir."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint from which to resume. "
            "Training starts from scratch when omitted."
        ),
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Master random seed "
            f"(default: {DEFAULT_SEED})."
        ),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help=(
            "Save a temporary checkpoint every N batches. "
            "Set to 0 to disable temporary checkpoints."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=DEFAULT_BLOCKS,
    )

    args = parser.parse_args()

    # ========================================================
    # Validation of configuration
    # ========================================================

    if not args.dataset.exists():

        raise FileNotFoundError(
            f"Dataset does not exist: {args.dataset}"
        )

    if args.epochs <= 0:

        raise ValueError(
            "--epochs must be greater than zero."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if not (
        0.0
        < args.validation_ratio
        < 1.0
    ):

        raise ValueError(
            "--validation-ratio must be between 0 and 1."
        )

    if args.num_workers < 0:

        raise ValueError(
            "--num-workers cannot be negative."
        )

    # ========================================================
    # Paths
    # ========================================================

    checkpoint_dir: Path = (
        args.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loss_log = (
        args.loss_log
        if args.loss_log is not None
        else checkpoint_dir
        / "training_loss_bc.json"
    )

    loss_log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Reproducibility
    #
    # IMPORTANT:
    # This must happen before model initialization, dataset
    # splitting and DataLoader creation.
    # ========================================================

    seed_everything(
        args.seed
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pin_memory = (
        device.type == "cuda"
    )

    print()
    print("=" * 70)
    print("ALBERTA - BEHAVIORAL CLONING")
    print("=" * 70)

    print(
        f"Device           : {device}"
    )

    print(
        f"Seed             : {args.seed}"
    )

    print(
        f"Dataset          : {args.dataset}"
    )

    print(
        f"Checkpoint dir   : {checkpoint_dir}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU              : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA             : "
            f"{torch.version.cuda}"
        )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print("Loading dataset...")

    dataset = ChessDataset(
        args.dataset
    )

    if len(dataset) < 2:

        raise RuntimeError(
            "Dataset must contain at least two samples."
        )

    print(
        f"Total positions  : {len(dataset):,}"
    )

    # ========================================================
    # Deterministic train / validation split
    # ========================================================

    validation_size = int(
        len(dataset)
        * args.validation_ratio
    )

    validation_size = max(
        1,
        validation_size,
    )

    train_size = (
        len(dataset)
        - validation_size
    )

    if train_size <= 0:

        raise RuntimeError(
            "Training split is empty."
        )

    split_generator = (
        torch.Generator()
        .manual_seed(
            args.seed
        )
    )

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [
                train_size,
                validation_size,
            ],
            generator=split_generator,
        )
    )

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Total            : {len(dataset):,}"
    )

    print(
        f"Training         : "
        f"{len(train_dataset):,} "
        f"({100 * len(train_dataset) / len(dataset):.1f}%)"
    )

    print(
        f"Validation       : "
        f"{len(val_dataset):,} "
        f"({100 * len(val_dataset) / len(dataset):.1f}%)"
    )

    print(
        f"Split seed       : {args.seed}"
    )

    # ========================================================
    # Validation DataLoader
    # ========================================================

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ========================================================
    # Model
    #
    # Model initialization is deterministic because the master
    # seed was set before constructing the network.
    # ========================================================

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=args.channels,
        blocks=args.blocks,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        f"Channels         : {args.channels}"
    )

    print(
        f"Residual blocks  : {args.blocks}"
    )

    print(
        f"Actions          : {len(ACTIONS):,}"
    )

    print(
        f"Parameters       : {parameters:,}"
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Configuration stored in checkpoints
    # ========================================================

    config = {
        "dataset":
            str(args.dataset),

        "epochs":
            args.epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "validation_ratio":
            args.validation_ratio,

        "seed":
            args.seed,

        "channels":
            args.channels,

        "blocks":
            args.blocks,

        "num_actions":
            len(ACTIONS),
    }

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 0
    history: list[dict] = []

    if args.resume is not None:

        if not args.resume.exists():

            raise FileNotFoundError(
                f"Resume checkpoint does not exist: "
                f"{args.resume}"
            )

        print()
        print("=" * 70)
        print("LOADING CHECKPOINT")
        print("=" * 70)

        print(
            args.resume
        )

        checkpoint = torch.load(
            args.resume,
            map_location=device,
        )

        if (
            checkpoint.get("actions")
            != len(ACTIONS)
        ):

            raise ValueError(
                "Action-space mismatch between checkpoint "
                "and current ALBERTA configuration."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

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

        checkpoint_batch = (
            checkpoint.get(
                "batch",
                -1,
            )
        )

        # Temporary checkpoints are useful as backups, but
        # this script only guarantees deterministic resume
        # from completed epoch checkpoints.
        if checkpoint_batch != -1:

            raise ValueError(
                "The selected checkpoint was saved during "
                "an epoch. Deterministic resume is supported "
                "only from completed epoch checkpoints."
            )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        history = list(
            checkpoint.get(
                "loss_history",
                [],
            )
        )

        print(
            f"Loaded epoch      : "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Resume from epoch : "
            f"{start_epoch}"
        )

        if "seed" in checkpoint:

            print(
                f"Checkpoint seed   : "
                f"{checkpoint['seed']}"
            )

            if (
                checkpoint["seed"]
                != args.seed
            ):

                print(
                    "WARNING: current --seed differs from "
                    "the checkpoint seed."
                )

    else:

        print()
        print(
            "Training from scratch."
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs,
    ):

        # ----------------------------------------------------
        # Recreate the shuffled DataLoader for every epoch.
        #
        # Its order depends only on:
        #     master seed + epoch
        #
        # This means epoch N has the same sample order whether
        # reached continuously or after an epoch checkpoint
        # resume.
        # ----------------------------------------------------

        train_loader = build_train_loader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            epoch=epoch,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        model.train()

        start_time = time.time()

        total_train_loss = 0.0
        total_train_samples = 0

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch}"
        )
        print("=" * 70)

        pbar = tqdm(
            train_loader,
            desc=f"Training epoch {epoch}",
        )

        # ====================================================
        # Training batches
        # ====================================================

        for batch, (x, y) in enumerate(
            pbar
        ):

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

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
                / total_train_samples
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{current_train_loss:.4f}",
            )

            # ------------------------------------------------
            # Temporary backup checkpoint
            #
            # These checkpoints preserve progress but are not
            # used for deterministic mid-epoch resume.
            # ------------------------------------------------

            if (
                args.save_every > 0
                and batch > 0
                and batch % args.save_every == 0
            ):

                temp_path = (
                    checkpoint_dir
                    / (
                        f"bc_temp_e"
                        f"{epoch}_b"
                        f"{batch}.pt"
                    )
                )

                save_checkpoint(
                    path=temp_path,
                    epoch=epoch,
                    batch=batch,
                    model=model,
                    optimizer=optimizer,
                    train_loss=current_train_loss,
                    val_loss=None,
                    val_accuracy=None,
                    history=history,
                    seed=args.seed,
                    config=config,
                )

        # ====================================================
        # Training loss
        # ========================================================

        if total_train_samples == 0:

            raise RuntimeError(
                "Training dataset produced no samples."
            )

        train_loss = (
            total_train_loss
            / total_train_samples
        )

        # ====================================================
        # Validation
        # ====================================================

        print()
        print(
            "Running validation..."
        )

        val_loss, val_accuracy = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ====================================================
        # Epoch statistics
        # ====================================================

        epoch_time = (
            time.time()
            - start_time
        ) / 60.0

        loss_gap = (
            val_loss
            - train_loss
        )

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
        # Save training history
        # ====================================================

        with open(
            loss_log,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        # ====================================================
        # Print metrics
        # ====================================================

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch} RESULTS"
        )
        print("=" * 70)

        print(
            f"Training loss    : "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss  : "
            f"{val_loss:.6f}"
        )

        print(
            f"Action agreement : "
            f"{val_accuracy:.2%}"
        )

        print(
            f"Train/Val gap    : "
            f"{loss_gap:+.6f}"
        )

        print(
            f"Epoch time       : "
            f"{epoch_time:.1f} min"
        )

        # ====================================================
        # End-of-epoch checkpoint
        # ====================================================

        checkpoint_path = (
            checkpoint_dir
            / f"bc_epoch_{epoch}.pt"
        )

        save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            batch=-1,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            history=history,
            seed=args.seed,
            config=config,
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)
    print("BC TRAINING FINISHED")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()