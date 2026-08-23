import torch
import torch.nn as nn
import time
import json

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import chess
import chess.variant

from src.encoding import encode_fen
from src.models.resnet import ChessResNet
from src.actions_space import ACTIONS


# ============================================================
# Configuration
# ============================================================

DATASET = "//content/drive/MyDrive/ALBERTA/positions_2300_bc.jsonl"

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/ALBERTA/checkpoints/bc_sym_epoch"
)

EPOCHS = 10

SAVE_EVERY = 40000

LOSS_LOG = CHECKPOINT_DIR / "training_loss.json"


# ============================================================
# Action mirror map
# ============================================================

def build_action_mirror_map():

    action_to_index = {
        action: idx
        for idx, action in enumerate(ACTIONS)
    }

    mirror_map = {}

    missing = []

    for action in ACTIONS:

        move = chess.Move.from_uci(action)

        mirrored_from = chess.square_mirror(
            move.from_square
        )

        mirrored_to = chess.square_mirror(
            move.to_square
        )

        mirrored_move = chess.Move(
            mirrored_from,
            mirrored_to,
            promotion=move.promotion
        )

        mirrored_action = mirrored_move.uci()

        if mirrored_action not in action_to_index:

            missing.append(
                (action, mirrored_action)
            )

            continue

        mirror_map[
            action_to_index[action]
        ] = action_to_index[mirrored_action]

    if missing:

        print()
        print("WARNING: missing mirrored actions:")
        print()

        for original, mirrored in missing[:20]:

            print(
                f"  {original} -> {mirrored}"
            )

        raise RuntimeError(
            f"{len(missing)} mirrored actions "
            f"are missing from ACTIONS."
        )

    return mirror_map


# ============================================================
# Symmetric dataset
# ============================================================

class SymmetricChessDataset(Dataset):

    def __init__(self, path, mirror_map):

        self.samples = []

        self.mirror_map = mirror_map

        bad = 0

        with open(path, "r") as f:

            for line in f:

                try:

                    sample = json.loads(line)

                    self.samples.append(
                        sample
                    )

                except Exception:

                    bad += 1

        print(
            f"Loaded original samples: "
            f"{len(self.samples):,}"
        )

        print(
            f"Ignored bad lines: "
            f"{bad:,}"
        )

        print(
            f"Effective samples per epoch: "
            f"{2 * len(self.samples):,}"
        )


    def __len__(self):

        return (
            2 * len(self.samples)
        )


    def __getitem__(self, idx):

        #
        # First half:
        # original examples
        #
        if idx < len(self.samples):

            sample = self.samples[idx]

            x = encode_fen(
                sample["fen"]
            )

            y = sample["action"]

            y = torch.tensor(
                y,
                dtype=torch.long
            )

            return (
                torch.tensor(
                    x,
                    dtype=torch.float32
                ),
                y
            )


        #
        # Second half:
        # mirrored examples
        #
        original_idx = (
            idx - len(self.samples)
        )

        sample = self.samples[
            original_idx
        ]

        #
        # Original Atomic position
        #
        board = chess.variant.AtomicBoard(
            sample["fen"]
        )

        #
        # Mirror:
        #
        # - board rank flip
        # - white <-> black
        # - side to move changes
        # - castling rights are transformed
        # - en passant is transformed
        #
        mirrored_board = board.mirror()

        mirrored_fen = (
            mirrored_board.fen()
        )

        x = encode_fen(
            mirrored_fen
        )


        #
        # Mirror action
        #
        original_action = sample["action"]

        original_index = int(
            original_action
        )

        mirrored_index = self.mirror_map[
            original_index
        ]

        y = torch.tensor(
            mirrored_index,
            dtype=torch.long
        )


        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            y
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

        "symmetric_training":
            True,
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
    print("======================================")
    print("SYMMETRIC BC TRAINING")
    print("======================================")
    print()

    print(
        "Device:",
        device
    )


    CHECKPOINT_DIR.mkdir(
        exist_ok=True
    )


    # ========================================================
    # Build action mirror map
    # ========================================================

    print()
    print(
        "Actions:",
        len(ACTIONS)
    )

    print(
        "Building action mirror map..."
    )

    mirror_map = build_action_mirror_map()

    print(
        "Mirrored actions:",
        len(mirror_map)
    )

    print(
        "Action mirror map ready."
    )


    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "Loading dataset..."
    )

    dataset = SymmetricChessDataset(
        DATASET,
        mirror_map
    )


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
        channels=64,
        blocks=4,
    ).to(device)


    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
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
    # IMPORTANT:
    # Train from scratch
    # ========================================================

    START_EPOCH = 0

    print()
    print(
        "Training from scratch."
    )

    print(
        "No asymmetric BC checkpoint loaded."
    )


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

            x = x.to(device)

            y = y.to(device)


            # =================================================
            # Forward
            # =================================================

            logits = model(x)


            # =================================================
            # Loss
            # =================================================

            loss = criterion(
                logits,
                y
            )


            # =================================================
            # Backprop
            # =================================================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += (
                loss.item()
            )


            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


            # =================================================
            # Temporary checkpoint
            # =================================================

            if (
                batch > 0
                and batch % SAVE_EVERY == 0
            ):

                temp_path = (
                    CHECKPOINT_DIR
                    /
                    f"bc_sym_temp_e{epoch}_b{batch}.pt"
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
            time.time() - start
        ) / 60


        history.append(
            {
                "epoch":
                    epoch,

                "loss":
                    epoch_loss,

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
            f"bc_sym_epoch_{epoch}.pt"
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