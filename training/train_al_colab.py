# ============================================================
# train_al.py
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import multiprocessing as mp

import chess
import chess.variant

import torch
import torch.nn.functional as F
from torch.optim import Adam

import train_rl_league_colab as rl

from src.encoding import encode_boards
from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX
from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.rl.oracle_replay_buffer import OracleReplayBuffer


# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/ALBERTA"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RL starting point
# ============================================================

START_EPOCH = 10

AL_START_EPOCH = 11
AL_END_EPOCH = 20


# ============================================================
# Oracle queue
# ============================================================

ORACLE_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_random.jsonl"
)


# ============================================================
# Oracle replay buffer
# ============================================================

ORACLE_CAPACITY = 50000

ORACLE_BATCH_SIZE = 4096


# ============================================================
# Oracle loss coefficients
# ============================================================

ORACLE_POLICY_COEF = 0.10

ORACLE_VALUE_COEF = 0.10


# ============================================================
# Oracle confidence
# ============================================================

CONFIDENCE_WEIGHTS = {
    "low": 0.50,
    "medium": 0.75,
    "high": 0.99,
}


# ============================================================
# Oracle criticality
# ============================================================

CRITICALITY_TEMPERATURES = {
    "critical": 0.25,
    "non_critical": 0.50,
    "outcome_independent": 1.00,
}


# ============================================================
# Load Oracle queue
# ============================================================

def load_oracle_queue(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Oracle queue not found:\n{path}"
        )

    annotations = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            # ------------------------------------------------
            # Ignore unanswered entries
            # ------------------------------------------------

            if (
                "oracle_move" not in record
                or "reward" not in record
            ):

                continue

            fen = record["fen"]

            oracle_move = record[
                "oracle_move"
            ]

            reward = float(
                record["reward"]
            )

            confidence = record.get(
                "confidence",
                "medium",
            )

            criticality = record.get(
                "criticality",
                "non_critical",
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if reward not in (
                -1.0,
                0.0,
                1.0,
            ):

                raise ValueError(
                    f"Invalid reward at line "
                    f"{line_number}: {reward}"
                )

            if confidence not in (
                CONFIDENCE_WEIGHTS
            ):

                raise ValueError(
                    f"Invalid confidence at line "
                    f"{line_number}: {confidence}"
                )

            if criticality not in (
                CRITICALITY_TEMPERATURES
            ):

                raise ValueError(
                    f"Invalid criticality at line "
                    f"{line_number}: {criticality}"
                )

            if oracle_move not in (
                ACTION_TO_INDEX
            ):

                raise ValueError(
                    f"Unknown oracle move at line "
                    f"{line_number}: {oracle_move}"
                )

            annotations.append(
                {
                    "fen":
                        fen,

                    "oracle_move":
                        oracle_move,

                    "confidence":
                        confidence,

                    "criticality":
                        criticality,

                    "reward":
                        reward,
                }
            )

    if not annotations:

        raise RuntimeError(
            "No answered Oracle annotations found."
        )

    print()
    print(
        "======================================"
    )

    print(
        "ORACLE QUEUE"
    )

    print(
        "======================================"
    )

    print(
        f"Path:        {path}"
    )

    print(
        f"Annotations: {len(annotations)}"
    )

    reward_counts = {
        -1.0: 0,
        0.0: 0,
        1.0: 0,
    }

    for record in annotations:

        reward_counts[
            record["reward"]
        ] += 1

    print(
        f"Losses:      {reward_counts[-1.0]}"
    )

    print(
        f"Draws:       {reward_counts[0.0]}"
    )

    print(
        f"Wins:        {reward_counts[1.0]}"
    )

    print(
        "======================================"
    )

    return annotations


# ============================================================
# Build Oracle replay buffer
# ============================================================

def build_oracle_buffer(
    annotations,
):

    buffer = OracleReplayBuffer(
        capacity=ORACLE_CAPACITY
    )

    for record in annotations:

        buffer.add(
            record["fen"],
            record["oracle_move"],
            record["confidence"],
            record["criticality"],
            record["reward"],
        )

    print(
        f"Oracle buffer size: {len(buffer)}"
    )

    return buffer


# ============================================================
# Oracle policy target
# ============================================================

def build_oracle_target(
    legal_ids,
    oracle_action,
    temperature,
    device,
):

    logits = torch.zeros(
        len(ACTIONS),
        device=device,
    )

    logits[
        oracle_action
    ] = 1.0

    logits = (
        logits
        / temperature
    )

    legal_mask = torch.zeros(
        len(ACTIONS),
        dtype=torch.bool,
        device=device,
    )

    legal_mask[
        legal_ids
    ] = True

    logits = logits.masked_fill(
        ~legal_mask,
        float("-inf"),
    )

    target_probs = F.softmax(
        logits,
        dim=0,
    )

    return (
        target_probs,
        legal_mask,
    )


# ============================================================
# Oracle loss
# ============================================================

def compute_oracle_loss(
    model,
    oracle_buffer,
):

    if len(oracle_buffer) == 0:

        zero = torch.zeros(
            (),
            device=DEVICE,
        )

        return {
            "loss":
                zero,

            "policy_loss":
                zero,

            "value_loss":
                zero,
        }

    batch_size = min(
        ORACLE_BATCH_SIZE,
        len(oracle_buffer),
    )

    batch = oracle_buffer.sample(
        batch_size
    )

    boards = [
        chess.variant.AtomicBoard(
            record["fen"]
        )
        for record in batch
    ]

    x = encode_boards(
        boards
    ).to(DEVICE)

    policy, values = model(x)

    policy_losses = []
    value_losses = []
    weights = []

    for i, record in enumerate(
        batch
    ):

        board = boards[i]

        oracle_move = record[
            "oracle_move"
        ]

        oracle_action = (
            ACTION_TO_INDEX[
                oracle_move
            ]
        )

        legal_ids = [
            ACTION_TO_INDEX[
                move.uci()
            ]
            for move in board.legal_moves
            if move.uci()
            in ACTION_TO_INDEX
        ]

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if oracle_action not in legal_ids:

            continue

        # ----------------------------------------------------
        # Criticality -> temperature
        # ----------------------------------------------------

        temperature = (
            CRITICALITY_TEMPERATURES[
                record["criticality"]
            ]
        )

        target_probs, legal_mask = (
            build_oracle_target(
                legal_ids,
                oracle_action,
                temperature,
                DEVICE,
            )
        )

        # ----------------------------------------------------
        # Current policy
        # ----------------------------------------------------

        current_logits = (
            policy[i]
        )

        current_logits = (
            current_logits.masked_fill(
                ~legal_mask,
                float("-inf"),
            )
        )

        current_log_probs = (
            F.log_softmax(
                current_logits,
                dim=0,
            )
        )

        safe_log_probs = (
            current_log_probs.masked_fill(
                ~legal_mask,
                0.0,
            )
        )

        # ----------------------------------------------------
        # Oracle policy loss
        # ----------------------------------------------------

        policy_loss = -(
            target_probs
            * safe_log_probs
        ).sum()

        # ----------------------------------------------------
        # Oracle value loss
        # ----------------------------------------------------

        value_target = torch.tensor(
            record["reward"],
            device=DEVICE,
            dtype=torch.float32,
        )

        value_loss = F.mse_loss(
            values[i, 0],
            value_target,
        )

        # ----------------------------------------------------
        # Confidence -> supervision weight
        # ----------------------------------------------------

        weight = torch.tensor(
            CONFIDENCE_WEIGHTS[
                record["confidence"]
            ],
            device=DEVICE,
            dtype=torch.float32,
        )

        policy_losses.append(
            policy_loss * weight
        )

        value_losses.append(
            value_loss * weight
        )

        weights.append(
            weight
        )

    # ========================================================
    # No valid annotations
    # ========================================================

    if not weights:

        zero = torch.zeros(
            (),
            device=DEVICE,
        )

        return {
            "loss":
                zero,

            "policy_loss":
                zero,

            "value_loss":
                zero,
        }

    weights = torch.stack(
        weights
    )

    policy_loss = (
        torch.stack(
            policy_losses
        ).sum()
        /
        weights.sum()
    )

    value_loss = (
        torch.stack(
            value_losses
        ).sum()
        /
        weights.sum()
    )

    # ========================================================
    # Total Oracle loss
    # ========================================================

    loss = (
        ORACLE_POLICY_COEF
        * policy_loss
        +
        ORACLE_VALUE_COEF
        * value_loss
    )

    return {
        "loss":
            loss,

        "policy_loss":
            policy_loss,

        "value_loss":
            value_loss,
    }


# ============================================================
# Callback used by RL train_epoch()
# ============================================================

def make_oracle_loss_fn(
    oracle_buffer,
):

    def oracle_loss_fn(model):

        return compute_oracle_loss(
            model,
            oracle_buffer,
        )

    return oracle_loss_fn


# ============================================================
# Load RL starting checkpoint
# ============================================================

def load_rl_start():

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "rl_epoch"
        / f"rl_epoch_{START_EPOCH}.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"RL starting checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print(
        "======================================"
    )

    print(
        "RL STARTING CHECKPOINT"
    )

    print(
        "======================================"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        base_model
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer = Adam(
        model.parameters(),
        lr=rl.LR,
    )

    if "optimizer_state_dict" not in checkpoint:

        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not contain "
            "optimizer_state_dict. Cannot guarantee an identical "
            "RL10 starting state."
        )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    print(
        f"Loaded RL epoch "
        f"{checkpoint.get('epoch', '?')}."
    )

    # ========================================================
    # Load BC7
    # ========================================================

    bc_model = rl.load_bc_agent(7)

    # ========================================================
    # Load league up to RL10
    # ========================================================

    league = rl.League(
        max_agents=rl.LEAGUE_MAX_AGENTS
    )

    bc6 = rl.load_bc_agent(6)

    league.add_agent(
        "bc_epoch_6",
        bc6,
    )

    bc7 = rl.load_bc_agent(7)

    league.add_agent(
        "bc_epoch_7",
        bc7,
    )

    for epoch in range(
        1,
        START_EPOCH + 1,
    ):

        path = (
            rl.LEAGUE_DIR
            / f"league_epoch_{epoch:03d}.pt"
        )

        if not path.exists():
            continue

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
        )

        snapshot_base = ChessResNet(
            num_actions=len(ACTIONS),
            channels=32,
            blocks=4,
        )

        snapshot = ActorCritic(
            snapshot_base
        ).to(DEVICE)

        snapshot.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        snapshot.eval()

        league.add_agent(
            f"league_epoch_{epoch:03d}",
            snapshot,
        )

    print(
        f"League loaded: {len(league)} agents"
    )

    return (
        model,
        optimizer,
        bc_model,
        league,
    )


# ============================================================
# Replay buffer from self-play
# ============================================================

def build_rl_buffer(
    completed_games,
):

    buffer = rl.ReplayBuffer()

    for game in completed_games:

        trajectory = game[
            "trajectory"
        ]

        current_white = game[
            "current_white"
        ]

        result = game[
            "result"
        ]

        # ----------------------------------------------------
        # Result from current agent perspective
        # ----------------------------------------------------

        if result == "1-0":

            game_result = (
                1.0
                if current_white
                else -1.0
            )

        elif result == "0-1":

            game_result = (
                -1.0
                if current_white
                else 1.0
            )

        else:

            game_result = 0.0

        # ----------------------------------------------------
        # Sparse reward
        # ----------------------------------------------------

        rewards = [
            0.0
        ] * len(trajectory)

        if trajectory:

            rewards[-1] = game_result

        # ----------------------------------------------------
        # GAE
        # ----------------------------------------------------

        advantages, returns = (
            rl.compute_gae(
                trajectory,
                rewards,
                gamma=rl.GAMMA,
                gae_lambda=rl.GAE_LAMBDA,
            )
        )

        # ====================================================
        # Add to replay buffer
        # ====================================================

        for (
            step,
            advantage,
            ret,
        ) in zip(
            trajectory,
            advantages,
            returns,
        ):

            buffer.add(
                step["fen"],
                step["action"],
                step["legal_moves"],
                ret,
                step["value"],
                step["old_log_prob"],
                advantage,
                step["ply"],
                game_result,
            )

    return buffer


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "============================================================"
    )

    print(
        "ALBERTA — RL + ORACLE"
    )

    print(
        "============================================================"
    )

    print(
        f"Starting checkpoint: RL{START_EPOCH}"
    )

    print(
        f"Training epochs:     "
        f"{AL_START_EPOCH} -> {AL_END_EPOCH}"
    )

    print(
        f"Oracle queue:        "
        f"{ORACLE_QUEUE_PATH}"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # Load RL starting point
    # ========================================================

    (
        model,
        optimizer,
        bc_model,
        league,
    ) = load_rl_start()

    # ========================================================
    # Load Oracle annotations
    # ========================================================

    annotations = load_oracle_queue(
        ORACLE_QUEUE_PATH
    )

    oracle_buffer = (
        build_oracle_buffer(
            annotations
        )
    )

    # ========================================================
    # Shared current model
    # ========================================================

    shared_current_model = (
        rl._prepare_shared_model(
            model
        )
    )

    # ========================================================
    # Shared league models
    # ========================================================

    shared_league_models = {}

    for name, agent in (
        league.agents.items()
    ):

        shared_league_models[
            name
        ] = rl._prepare_shared_model(
            agent
        )

    # ========================================================
    # Manager registry
    # ========================================================

    manager = mp.Manager()

    league_registry = manager.list(
        league.names()
    )

    # ========================================================
    # Shared BC model
    # ========================================================

    shared_bc_model = copy.deepcopy(
        bc_model
    ).to("cpu")

    shared_bc_model.eval()

    shared_bc_model.share_memory()

    # ========================================================
    # Preallocate future league snapshot slots
    # ========================================================

    for epoch in range(
        AL_START_EPOCH,
        AL_END_EPOCH + 1,
    ):

        snapshot_name = (
            f"league_epoch_{epoch:03d}"
        )

        shared_league_models[
            snapshot_name
        ] = rl._prepare_shared_model(
            model
        )

    # ========================================================
    # Worker pool
    # ========================================================

    pool = mp.Pool(
        processes=12,
        initializer=rl._init_selfplay_worker,
        initargs=(
            shared_current_model,
            shared_league_models,
            league_registry,
            shared_bc_model,
        ),
    )

    try:

        # ====================================================
        # AL / Oracle training
        # ====================================================

        for epoch in range(
            AL_START_EPOCH,
            AL_END_EPOCH + 1,
        ):

            print()
            print(
                "============================================================"
            )

            print(
                f"RL + ORACLE — EPOCH {epoch}"
            )

            print(
                "============================================================"
            )

            # =================================================
            # Self-play
            # =================================================

            stats = rl.UncertaintyStats()

            completed_games = (
                rl.collect_games_parallel(
                    pool,
                    shared_current_model,
                    shared_league_models,
                    model,
                    league,
                    rl.GAMES_PER_EPOCH,
                    stats,
                    num_workers=12,
                    batch_size=256,
                )
            )

            # =================================================
            # RL replay buffer
            # =================================================

            rl_buffer = build_rl_buffer(
                completed_games
            )

            print(
                f"RL replay buffer: "
                f"{len(rl_buffer)} positions"
            )

            # =================================================
            # Oracle loss
            # =================================================

            oracle_loss_fn = (
                make_oracle_loss_fn(
                    oracle_buffer
                )
            )

            # =================================================
            # PPO + Oracle
            # =================================================

            metrics = rl.train_epoch(
                model,
                optimizer,
                rl_buffer,
                bc_model,
                epoch,
                extra_loss_fn=oracle_loss_fn,
            )

            loss = metrics[0]

            # =================================================
            # Save checkpoint
            # =================================================

            AL_CHECKPOINT_DIR = (
                PROJECT_ROOT
                / "checkpoints"
                / "al_epoch"
            )

            AL_CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint_path = (
                AL_CHECKPOINT_DIR
                / f"al_epoch_{epoch}.pt"
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "loss":
                        loss,
                },
                checkpoint_path,
            )

            print(
                "AL checkpoint saved:",
                checkpoint_path,
                flush=True,
            )

            # =================================================
            # Save league snapshot
            # =================================================

            AL_LEAGUE_DIR = (
                PROJECT_ROOT
                / "checkpoints"
                / "league_al"
            )

            AL_LEAGUE_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            league_path = (
                AL_LEAGUE_DIR
                / f"league_epoch_{epoch:03d}.pt"
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),
                },
                league_path,
            )

            print(
                "AL league snapshot saved:",
                league_path,
                flush=True,
            )

            # =================================================
            # Add snapshot to local league
            # =================================================

            snapshot = copy.deepcopy(
                model
            ).to(DEVICE)

            snapshot.eval()

            snapshot_name = (
                f"league_epoch_{epoch:03d}"
            )

            league.add_agent(
                snapshot_name,
                snapshot,
            )

            # =================================================
            # Update shared snapshot slot
            # =================================================

            shared_league_models[
                snapshot_name
            ].load_state_dict(
                model.state_dict()
            )

            # =================================================
            # Synchronize registry
            # =================================================

            league_registry[:] = (
                league.names()
            )

            # =================================================
            # Synchronize current model
            # =================================================

            shared_current_model.load_state_dict(
                model.state_dict()
            )

            print(
                f"League size: {len(league)}"
            )

            # =================================================
            # Cleanup
            # =================================================

            del rl_buffer

    finally:

        pool.close()
        pool.join()

        manager.shutdown()

    print()
    print(
        "============================================================"
    )

    print(
        "RL + ORACLE TRAINING FINISHED"
    )

    print(
        "============================================================"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()