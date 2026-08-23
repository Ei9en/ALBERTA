import time
import json
from pathlib import Path

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import chess
import chess.variant

from src.encoding import encode_board
from src.models.resnet import ChessResNet
from src.actions_space import ACTIONS


# ============================================================
# Configuration
# ============================================================

DATASET = "/content/drive/MyDrive/ALBERTA/positions_2300_bc.jsonl"

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/ALBERTA/checkpoints/bc_sym_epoch"
)

CACHE_DIR = Path(
    "/content/alberta_bc_sym_cache"
)

EPOCHS = 10

SAVE_EVERY = 40000

LOSS_LOG = CHECKPOINT_DIR / "training_loss.json"


# ============================================================
# Model configuration
# ============================================================

CHANNELS = 32
BLOCKS = 4
POLICY_HIDDEN = 512


# ============================================================
# Performance
# ============================================================

NUM_WORKERS = 8

PREFETCH_FACTOR = 2

PIN_MEMORY = True

BATCH_SIZE = 2048

USE_AMP = True

AMP_DTYPE = torch.bfloat16

USE_COMPILE = True


# ============================================================
# Cache configuration
# ============================================================

SHARD_SIZE = 10000


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
            promotion=move.promotion,
        )

        mirrored_action = mirrored_move.uci()

        if mirrored_action not in action_to_index:

            missing.append(
                (action, mirrored_action)
            )

            continue

        mirror_map[
            action_to_index[action]
        ] = action_to_index[
            mirrored_action
        ]

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
# Cache builder
# ============================================================

def build_symmetric_cache(
    path,
    mirror_map,
    cache_dir,
):

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        cache_dir /
        "metadata.json"
    )

    existing_shards = sorted(
        cache_dir.glob("sym_*.pt")
    )

    if metadata_path.exists() and existing_shards:

        print()
        print("Symmetric cache already exists.")

        print(
            f"Found {len(existing_shards)} shard(s)."
        )

        with open(
            metadata_path,
            "r"
        ) as f:

            metadata = json.load(f)

        print(
            "Cached symmetric samples:",
            f"{metadata['samples']:,}"
        )

        # Vérification de cohérence
        if metadata.get("shape") != [19, 8, 8]:

            raise RuntimeError(
                "Symmetric cache has an unexpected shape."
            )

        return metadata

    print()
    print("======================================")
    print("BUILDING SYMMETRIC CACHE")
    print("======================================")
    print()

    print(
        "This is a one-time preprocessing step."
    )

    print(
        "Original dataset:",
        path
    )

    print(
        "Cache directory:",
        cache_dir
    )

    x_buffer = []

    y_buffer = []

    shard_index = 0

    total = 0

    bad = 0

    start = time.time()

    with open(
        path,
        "r"
    ) as f:

        pbar = tqdm(
            f,
            desc="Building mirror cache",
            unit=" lines",
        )

        for line in pbar:

            try:

                sample = json.loads(line)

            except Exception:

                bad += 1

                continue

            fen = sample["fen"]

            original_action = int(
                sample["action"]
            )

            board = chess.variant.AtomicBoard(
                fen
            )

            mirrored_board = board.mirror()

            x = encode_board(
                mirrored_board
            )

            y = mirror_map[
                original_action
            ]

            x_buffer.append(x)

            y_buffer.append(y)

            total += 1

            if len(x_buffer) >= SHARD_SIZE:

                save_cache_shard(
                    cache_dir,
                    shard_index,
                    x_buffer,
                    y_buffer,
                )

                shard_index += 1

                x_buffer.clear()
                y_buffer.clear()

            if total % 10000 == 0:

                elapsed = time.time() - start

                rate = (
                    total / elapsed
                    if elapsed > 0
                    else 0
                )

                pbar.set_postfix(
                    examples=f"{total:,}",
                    rate=f"{rate:.0f}/s",
                )

    if x_buffer:

        save_cache_shard(
            cache_dir,
            shard_index,
            x_buffer,
            y_buffer,
        )

        shard_index += 1

    elapsed = time.time() - start

    metadata = {

        "samples":
            total,

        "bad_lines":
            bad,

        "shards":
            shard_index,

        "shard_size":
            SHARD_SIZE,

        "shape":
            [19, 8, 8],

        "dtype":
            "float32",

        "actions":
            len(ACTIONS),

        "symmetric":
            True,
    }

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "Symmetric cache completed."
    )

    print(
        f"Samples: {total:,}"
    )

    print(
        f"Shards: {shard_index:,}"
    )

    print(
        f"Time: {elapsed / 60:.1f} min"
    )

    print(
        f"Bad lines: {bad:,}"
    )

    return metadata


# ============================================================
# Save cache shard
# ============================================================

def save_cache_shard(
    cache_dir,
    shard_index,
    x_buffer,
    y_buffer,
):

    x = torch.stack(
        x_buffer
    )

    y = torch.tensor(
        y_buffer,
        dtype=torch.long,
    )

    path = (
        cache_dir /
        f"sym_{shard_index:05d}.pt"
    )

    torch.save(
        {
            "x": x,
            "y": y,
        },
        path,
    )


# ============================================================
# Original dataset
# ============================================================

class OriginalChessDataset(Dataset):

    def __init__(self, path):

        self.samples = []

        bad = 0

        with open(
            path,
            "r"
        ) as f:

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

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        board = chess.variant.AtomicBoard(
            sample["fen"]
        )

        x = encode_board(
            board
        )

        y = int(
            sample["action"]
        )

        return (
            x,
            y,
        )


# ============================================================
# Symmetric cache dataset
# ============================================================

class SymmetricCacheDataset(Dataset):

    def __init__(
        self,
        cache_dir,
        metadata,
    ):

        self.cache_dir = cache_dir

        self.shard_paths = sorted(
            cache_dir.glob("sym_*.pt")
        )

        self.shard_size = metadata[
            "shard_size"
        ]

        self.total_samples = metadata[
            "samples"
        ]

        self.shards = []

        for path in self.shard_paths:

            data = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            n = len(
                data["y"]
            )

            self.shards.append(
                {
                    "path": path,
                    "length": n,
                }
            )

        self.offsets = []

        current = 0

        for shard in self.shards:

            self.offsets.append(
                current
            )

            current += shard["length"]

        assert current == self.total_samples

        self._loaded_shard_idx = None
        self._loaded_shard = None

        print(
            f"Loaded symmetric cache: "
            f"{self.total_samples:,} samples"
        )

    def __len__(self):

        return self.total_samples

    def __getitem__(
        self,
        idx
    ):

        shard_idx = (
            idx // self.shard_size
        )

        local_idx = (
            idx
            -
            self.offsets[shard_idx]
        )

        if shard_idx != self._loaded_shard_idx:

            self._loaded_shard = torch.load(
                self.shard_paths[shard_idx],
                map_location="cpu",
                weights_only=True,
            )

            self._loaded_shard_idx = shard_idx

        shard = self._loaded_shard

        return (
            shard["x"][local_idx],
            shard["y"][local_idx],
        )


# ============================================================
# Combined symmetric dataset
# ============================================================

class CombinedSymmetricDataset(Dataset):

    def __init__(
        self,
        original_dataset,
        symmetric_dataset,
    ):

        self.original = original_dataset

        self.symmetric = symmetric_dataset

        self.original_len = len(
            original_dataset
        )

    def __len__(self):

        return (
            self.original_len
            +
            len(self.symmetric)
        )

    def __getitem__(
        self,
        idx
    ):

        if idx < self.original_len:

            return self.original[idx]

        return self.symmetric[
            idx - self.original_len
        ]


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

        "loss":
            loss,

        "actions":
            len(ACTIONS),

        "channels":
            CHANNELS,

        "blocks":
            BLOCKS,

        "policy_hidden":
            POLICY_HIDDEN,

        "loss_history":
            history,

        "symmetric_training":
            True,
    }

    torch.save(
        checkpoint,
        path,
    )

    print()
    print(
        "Saved checkpoint:",
        path,
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
    print("SYMMETRIC BC TRAINING - OPTIMIZED")
    print("======================================")
    print()

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA:",
            torch.version.cuda,
        )

    # --------------------------------------------------------
    # CUDA performance
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Action map
    # --------------------------------------------------------

    print()
    print(
        "Actions:",
        len(ACTIONS),
    )

    print(
        "Building action mirror map..."
    )

    mirror_map = (
        build_action_mirror_map()
    )

    print(
        "Mirrored actions:",
        len(mirror_map),
    )

    print(
        "Action mirror map ready."
    )

    # --------------------------------------------------------
    # Build symmetric cache
    # --------------------------------------------------------

    metadata = build_symmetric_cache(
        DATASET,
        mirror_map,
        CACHE_DIR,
    )

    # --------------------------------------------------------
    # Original dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading original dataset..."
    )

    original_dataset = (
        OriginalChessDataset(
            DATASET
        )
    )

    # --------------------------------------------------------
    # Symmetric dataset
    # --------------------------------------------------------

    symmetric_dataset = (
        SymmetricCacheDataset(
            CACHE_DIR,
            metadata,
        )
    )

    # --------------------------------------------------------
    # Combined dataset
    # --------------------------------------------------------

    dataset = CombinedSymmetricDataset(
        original_dataset,
        symmetric_dataset,
    )

    print()
    print(
        "======================================"
    )

    print(
        "Combined dataset:"
    )

    print(
        f"Original:  "
        f"{len(original_dataset):,}"
    )

    print(
        f"Symmetric: "
        f"{len(symmetric_dataset):,}"
    )

    print(
        f"Total:     "
        f"{len(dataset):,}"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader_kwargs = {

        "batch_size":
            BATCH_SIZE,

        "shuffle":
            True,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            PIN_MEMORY,

        "persistent_workers":
            NUM_WORKERS > 0,

        "drop_last":
            True,

    }

    if NUM_WORKERS > 0:

        loader_kwargs[
            "prefetch_factor"
        ] = PREFETCH_FACTOR

    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    print()
    print(
        "DataLoader:"
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Workers:",
        NUM_WORKERS,
    )

    print(
        "Pin memory:",
        PIN_MEMORY,
    )

    print(
        "Persistent workers:",
        NUM_WORKERS > 0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=CHANNELS,
        blocks=BLOCKS,
    ).to(device)

    raw_model = model

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model configuration:"
    )

    print(
        "Channels:",
        CHANNELS,
    )

    print(
        "Residual blocks:",
        BLOCKS,
    )

    print(
        "Policy hidden:",
        POLICY_HIDDEN,
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    if (
        USE_COMPILE
        and hasattr(
            torch,
            "compile"
        )
    ):

        print()
        print(
            "Compiling model..."
        )

        model = torch.compile(
            model,
            mode="max-autotune",
        )

        print(
            "torch.compile enabled."
        )

    else:

        print(
            "torch.compile disabled."
        )

    # --------------------------------------------------------
    # From scratch
    # --------------------------------------------------------

    START_EPOCH = 0

    print()
    print(
        "Training from scratch."
    )

    print(
        "No asymmetric BC checkpoint loaded."
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    amp_enabled = (
        USE_AMP
        and device == "cuda"
    )

    print(
        "AMP:",
        amp_enabled,
    )

    if amp_enabled:

        print(
            "AMP dtype:",
            AMP_DTYPE,
        )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    if LOSS_LOG.exists():

        with open(
            LOSS_LOG,
            "r"
        ) as f:

            history = json.load(f)

    else:

        history = []

    # --------------------------------------------------------
    # GradScaler
    #
    # bfloat16 ne nécessite pas de GradScaler.
    # --------------------------------------------------------

    scaler = None

    if (
        amp_enabled
        and AMP_DTYPE == torch.float16
    ):

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(
        START_EPOCH,
        EPOCHS,
    ):

        start = time.time()

        total_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}",
        )

        for batch, (x, y) in enumerate(
            pbar
        ):

            # ------------------------------------------------
            # CPU -> GPU
            # ------------------------------------------------

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Forward + loss
            # ------------------------------------------------

            if amp_enabled:

                with torch.autocast(
                    device_type="cuda",
                    dtype=AMP_DTYPE,
                ):

                    logits = model(x)

                    loss = criterion(
                        logits,
                        y,
                    )

            else:

                logits = model(x)

                loss = criterion(
                    logits,
                    y,
                )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            if scaler is not None:

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            # ------------------------------------------------
            # Temporary checkpoint
            #
            # IMPORTANT :
            # on sauvegarde raw_model, pas le wrapper
            # torch.compile.
            # ------------------------------------------------

            if (
                batch > 0
                and batch % SAVE_EVERY == 0
            ):

                temp_path = (
                    CHECKPOINT_DIR
                    /
                    f"bc_sym_temp_e"
                    f"{epoch}_b"
                    f"{batch}.pt"
                )

                save_checkpoint(
                    temp_path,
                    epoch,
                    batch,
                    raw_model,
                    optimizer,
                    loss.item(),
                    history,
                )

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

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
                    epoch_time,
            }
        )

        # ----------------------------------------------------
        # Save loss history
        # ----------------------------------------------------

        with open(
            LOSS_LOG,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()

        print(
            f"Epoch {epoch} loss:",
            epoch_loss,
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f} min"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        path = (
            CHECKPOINT_DIR
            /
            f"bc_sym_epoch_{epoch}.pt"
        )

        save_checkpoint(
            path,
            epoch,
            -1,
            raw_model,
            optimizer,
            epoch_loss,
            history,
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()