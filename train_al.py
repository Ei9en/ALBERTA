from pathlib import Path
import sys
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.variant
import torch
import torch.nn.functional as F

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.actions_space import ACTION_TO_INDEX
from src.encoding import encode_boards


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "data"
    / "oracle_queue_1-10.jsonl"
)

DEFAULT_HISTORY = (
    PROJECT_ROOT
    / "data"
    / "al_history_1-10.jsonl"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "checkpoints"
    / "al_epoch"
    / "al_epoch_10.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

def build_model():

    bc_model = ChessResNet(
        num_actions=len(ACTION_TO_INDEX),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        bc_model
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(checkpoint_path):

    model = build_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    return model, checkpoint


# ============================================================
# LOAD QUEUE
# ============================================================

def load_queue(queue_path):

    samples = []

    with open(
        queue_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)

            # ------------------------------------------------
            # AL ne travaille que sur les annotations terminées
            # ------------------------------------------------

            if sample.get("status") != "answered":
                continue

            required = [
                "query_id",
                "fen",
                "oracle_move",
                "oracle_confidence",
                "oracle_situation",
            ]

            if not all(
                key in sample
                for key in required
            ):
                continue

            samples.append(sample)

    return samples


# ============================================================
# LOAD AL HISTORY
# ============================================================

def load_history(history_path):

    if not history_path.exists():
        return set()

    consumed = set()

    with open(
        history_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            query_id = record.get(
                "query_id"
            )

            if query_id is not None:
                consumed.add(query_id)

    return consumed


# ============================================================
# SOFT TARGET
# ============================================================

def make_soft_target(
    oracle_move,
    confidence,
    position_evaluation,
    legal_moves,
):
    """
    Construct the target distribution over LEGAL actions.

    confidence:
        high   -> 0.99
        medium -> 0.75
        low    -> 0.50

    position_evaluation:
        unique_move     -> alpha = 1.0
        multiple_good   -> alpha = 0.6
        everything_wins -> alpha = 0.2
    """

    p_conf = {
        "high": 0.99,
        "medium": 0.75,
        "low": 0.50,
    }[confidence]

    alpha = {
        "unique_move": 1.0,
        "multiple_good": 0.6,
        "everything_wins": 0.2,
    }[position_evaluation]

    n = len(legal_moves)

    if n < 2:
        raise ValueError(
            "Position has fewer than 2 legal moves."
        )

    if oracle_move not in legal_moves:
        raise ValueError(
            f"Oracle move {oracle_move} "
            "is not legal."
        )

    # --------------------------------------------------------
    # Effective preference for oracle_move
    # --------------------------------------------------------

    p_eff = (
        alpha * p_conf
        + (1.0 - alpha) * (1.0 / n)
    )

    # --------------------------------------------------------
    # Residual mass
    # --------------------------------------------------------

    residual = (
        (1.0 - p_eff)
        / (n - 1)
    )

    # --------------------------------------------------------
    # Target in EXACTLY the same order as legal_moves
    # --------------------------------------------------------

    target = torch.full(
        (n,),
        residual,
        dtype=torch.float32,
    )

    oracle_index = legal_moves.index(
        oracle_move
    )

    target[
        oracle_index
    ] = p_eff

    return target


# ============================================================
# PREPARE BATCH
# ============================================================

def prepare_batch(samples):

    boards = []

    legal_indices = []

    legal_moves_all = []

    targets = []

    for sample in samples:

        fen = sample["fen"]

        board = chess.variant.AtomicBoard(
            fen
        )

        # ----------------------------------------------------
        # Legal moves
        # ----------------------------------------------------

        legal_moves = list(
            board.legal_moves
        )

        legal_uci = [
            move.uci()
            for move in legal_moves
        ]

        oracle_move = sample[
            "oracle_move"
        ]

        # ----------------------------------------------------
        # Verify annotation
        # ----------------------------------------------------

        if oracle_move not in legal_uci:

            raise ValueError(
                "\n"
                f"Illegal oracle move.\n"
                f"query_id : {sample['query_id']}\n"
                f"fen      : {fen}\n"
                f"oracle   : {oracle_move}\n"
            )

        # ----------------------------------------------------
        # Global action indices
        # ----------------------------------------------------

        indices = [
            ACTION_TO_INDEX[
                move.uci()
            ]
            for move in legal_moves
        ]

        # ----------------------------------------------------
        # Soft target
        # ----------------------------------------------------

        target = make_soft_target(
            oracle_move=oracle_move,
            confidence=sample[
                "oracle_confidence"
            ],
            position_evaluation=sample[
                "oracle_situation"
            ],
            legal_moves=legal_uci,
        )

        boards.append(board)
        legal_indices.append(indices)
        legal_moves_all.append(
            legal_moves
        )
        targets.append(target)

    # --------------------------------------------------------
    # Encode positions
    # --------------------------------------------------------

    x = encode_boards(
        boards
    ).to(DEVICE)

    return (
        x,
        legal_indices,
        targets,
    )


# ============================================================
# SUPERVISED AL LOSS
# ============================================================

def compute_al_loss(
    model,
    x,
    legal_indices,
    targets,
):
    """
    Compute soft cross-entropy only over legal actions.
    """

    policies, _ = model(x)

    losses = []

    for i in range(
        len(legal_indices)
    ):

        indices = legal_indices[i]

        target = targets[i].to(
            DEVICE
        )

        index_tensor = torch.tensor(
            indices,
            dtype=torch.long,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # Keep only legal action logits
        # ----------------------------------------------------

        legal_logits = policies[
            i
        ].index_select(
            0,
            index_tensor,
        )

        # ----------------------------------------------------
        # log pi(a|s)
        # ----------------------------------------------------

        log_probs = F.log_softmax(
            legal_logits,
            dim=0,
        )

        # ----------------------------------------------------
        # Soft cross entropy
        # ----------------------------------------------------

        loss = -(
            target * log_probs
        ).sum()

        losses.append(loss)

    return torch.stack(
        losses
    ).mean()


# ============================================================
# SAVE HISTORY
# ============================================================

def append_history(
    history_path,
    samples,
    checkpoint_epoch,
):

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        history_path,
        "a",
        encoding="utf-8",
    ) as f:

        for sample in samples:

            record = {
                "query_id":
                    sample["query_id"],

                "checkpoint_epoch":
                    checkpoint_epoch,
            }

            f.write(
                json.dumps(
                    record
                )
                + "\n"
            )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    checkpoint,
    model,
    output_path,
    consumed_samples,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_checkpoint = dict(checkpoint)

    new_checkpoint["model_state_dict"] = (
        model.state_dict()
    )

    new_checkpoint.pop(
        "optimizer_state_dict",
        None,
    )

    new_checkpoint["al_applied"] = True

    new_checkpoint["al_num_annotations"] = len(
        consumed_samples
    )

    new_checkpoint["al_query_ids"] = [
        sample["query_id"]
        for sample in consumed_samples
    ]

    torch.save(
        new_checkpoint,
        output_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
    )

    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ALBERTA ACTIVE LEARNING")
    print("=" * 70)

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Checkpoint  : {args.checkpoint}"
    )

    print(
        f"Queue       : {args.queue}"
    )

    # ========================================================
    # Load queue
    # ========================================================

    samples = load_queue(
        args.queue
    )

    print(
        f"Answered annotations : "
        f"{len(samples)}"
    )

    if len(samples) == 0:

        print(
            "No answered annotations."
        )

        return

    # ========================================================
    # Remove already consumed annotations
    # ========================================================

    consumed_ids = load_history(
        args.history
    )

    new_samples = [
        sample
        for sample in samples
        if sample["query_id"]
        not in consumed_ids
    ]

    print(
        f"Already consumed      : "
        f"{len(consumed_ids)}"
    )

    print(
        f"New annotations       : "
        f"{len(new_samples)}"
    )

    if len(new_samples) == 0:

        print(
            "Nothing to train on."
        )

        return

    # ========================================================
    # Load model
    # ========================================================

    model, checkpoint = load_model(
        args.checkpoint
    )

    model.train()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    # ========================================================
    # AL training
    # ========================================================

    print()
    print(
        f"AL epochs   : {args.epochs}"
    )

    print(
        f"Batch size  : {args.batch_size}"
    )

    print(
        f"Learning rate : {args.lr}"
    )

    print()

    for epoch in range(
        args.epochs
    ):

        total_loss = 0.0
        n_batches = 0

        # ----------------------------------------------------
        # Shuffle annotations
        # ----------------------------------------------------

        permutation = torch.randperm(
            len(new_samples)
        ).tolist()

        shuffled = [
            new_samples[i]
            for i in permutation
        ]

        # ----------------------------------------------------
        # Mini-batches
        # ----------------------------------------------------

        for start in range(
            0,
            len(shuffled),
            args.batch_size,
        ):

            batch_samples = shuffled[
                start:
                start + args.batch_size
            ]

            (
                x,
                legal_indices,
                targets,
            ) = prepare_batch(
                batch_samples
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = compute_al_loss(
                model,
                x,
                legal_indices,
                targets,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            n_batches += 1

        mean_loss = (
            total_loss
            / n_batches
        )

        print(
            f"AL epoch "
            f"{epoch + 1}/{args.epochs} "
            f"- loss: {mean_loss:.6f}"
        )

    # ========================================================
    # Save checkpoint
    # ========================================================

    save_checkpoint(
        checkpoint=checkpoint,
        model=model,
        output_path=args.output,
        consumed_samples=new_samples,
    )

    # ========================================================
    # Update history
    # ========================================================

    append_history(
        history_path=args.history,
        samples=new_samples,
        checkpoint_epoch=args.checkpoint_epoch,
    )

    print()
    print("=" * 70)
    print("AL UPDATE COMPLETED")
    print("=" * 70)

    print(
        f"Annotations used : "
        f"{len(new_samples)}"
    )

    print(
        f"Output checkpoint : "
        f"{args.output}"
    )

    print(
        f"History           : "
        f"{args.history}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()