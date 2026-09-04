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

ORACLE_POLICY_COEF = 0.001

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

    # ========================================================
    # Load RL10 + optimizer + BC7 + league
    # ========================================================

    (
        model,
        optimizer,
        bc_model,
        league,
    ) = load_rl_start()

    # ========================================================
    # BC model for self-play workers
    # ========================================================

    bc_model_selfplay = copy.deepcopy(
        bc_model
    ).to("cpu")

    bc_model_selfplay.eval()
    bc_model_selfplay.share_memory()

    # ========================================================
    # Load Oracle annotations
    # ========================================================

    annotations = load_oracle_queue(
        ORACLE_QUEUE_PATH
    )

    oracle_buffer = build_oracle_buffer(
        annotations
    )

    oracle_loss_fn = make_oracle_loss_fn(
        oracle_buffer
    )

    # ========================================================
    # Replay buffer
    # ========================================================

    buffer = rl.ReplayBuffer(
        capacity=300000
    )

    # ========================================================
    # IMPORTANT:
    # Keep ONE UncertaintyStats instance for the
    # entire AL11 -> AL20 experiment.
    #
    # If this were created inside the epoch loop,
    # stats.save() would overwrite the JSON with only
    # the current epoch's positions.
    # ========================================================

    stats = rl.UncertaintyStats()

    best_loss = None

    # ========================================================
    # Parallel self-play
    # ========================================================

    NUM_WORKERS = 12
    SELFPLAY_BATCH_SIZE = 256

    # ========================================================
    # Shared current model
    # ========================================================

    print(
        "\nPreparing shared CPU models...",
        flush=True,
    )

    shared_current_model = (
        rl._prepare_shared_model(
            model
        )
    )

    # ========================================================
    # Shared league models
    # ========================================================

    shared_league_models = {}

    for (
        name,
        league_model,
    ) in league.agents.items():

        shared_league_models[name] = (
            rl._prepare_shared_model(
                league_model
            )
        )

    # ========================================================
    # Preallocate future league slots
    # ========================================================

    for epoch in range(
        AL_START_EPOCH,
        AL_END_EPOCH + 1,
    ):

        name = (
            f"league_epoch_{epoch:03d}"
        )

        if name in shared_league_models:

            continue

        placeholder = (
            copy.deepcopy(
                model
            ).to("cpu")
        )

        placeholder.eval()

        placeholder.share_memory()

        shared_league_models[name] = (
            placeholder
        )

    print(
        f"Shared models ready: "
        f"{len(shared_league_models)} league slots",
        flush=True,
    )

    # ========================================================
    # Multiprocessing
    # ========================================================

    ctx = mp.get_context(
        "spawn"
    )

    manager = ctx.Manager()

    league_registry = manager.list(
        league.names()
    )

    print(
        "\nInitial league registry:",
        flush=True,
    )

    for name in league_registry:

        print(
            f"  - {name}",
            flush=True,
        )

    # ========================================================
    # Pool
    # ========================================================

    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=rl._init_selfplay_worker,
        initargs=(
            shared_current_model,
            shared_league_models,
            league_registry,
            bc_model_selfplay,
        ),
    ) as pool:

        # ====================================================
        # RL + Oracle loop
        # ====================================================

        for epoch in range(
            AL_START_EPOCH,
            AL_END_EPOCH + 1,
        ):

            print(
                "\n============================================================",
                flush=True,
            )

            print(
                f"===== RL + ORACLE — Epoch {epoch} =====",
                flush=True,
            )

            print(
                "============================================================",
                flush=True,
            )

            wins = 0
            losses = 0
            draws = 0

            # =================================================
            # Self-play
            # =================================================

            games = rl.collect_games_parallel(
                pool,
                shared_current_model,
                shared_league_models,
                model,
                league,
                rl.GAMES_PER_EPOCH,
                stats,
                num_workers=NUM_WORKERS,
                batch_size=SELFPLAY_BATCH_SIZE,
            )

            # =================================================
            # Construction replay buffer
            # =================================================

            for game in games:

                trajectory = (
                    game["trajectory"]
                )

                result = (
                    game["result"]
                )

                current_white = (
                    game["current_white"]
                )

                # =============================================
                # Résultat
                # =============================================

                if result == "1-0":

                    if current_white:

                        wins += 1

                    else:

                        losses += 1

                elif result == "0-1":

                    if current_white:

                        losses += 1

                    else:

                        wins += 1

                else:

                    draws += 1

                # =============================================
                # Rewards
                # =============================================

                rewards = [
                    0.0
                ] * len(trajectory)

                if trajectory:

                    if result == "1-0":

                        terminal_reward = (
                            1.0
                            if current_white
                            else -1.0
                        )

                    elif result == "0-1":

                        terminal_reward = (
                            -1.0
                            if current_white
                            else 1.0
                        )

                    else:

                        terminal_reward = 0.0

                    rewards[-1] = (
                        terminal_reward
                    )

                # =============================================
                # GAE
                # =============================================

                advantages, returns = (
                    rl.compute_gae(
                        trajectory,
                        rewards,
                        gamma=rl.GAMMA,
                        gae_lambda=rl.GAE_LAMBDA,
                    )
                )

                # =============================================
                # Replay
                # =============================================

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
                        game.get(
                            "result"
                        ),
                    )

            # =================================================
            # Stats
            # =================================================

            total_games = (
                wins
                + losses
                + draws
            )

            score_rate = (
                wins
                + 0.5 * draws
            ) / total_games

            print(
                f"Replay buffer size: "
                f"{len(buffer)}",
                flush=True,
            )

            print(
                f"Results: "
                f"W={wins} "
                f"L={losses} "
                f"D={draws} "
                f"Score={score_rate:.1%}",
                flush=True,
            )

            # =================================================
            # PPO + Oracle
            # =================================================

            (
                loss,
                actor_loss,
                critic_loss,
                approx_kl,
                dkl,
                dkl_loss,
            ) = rl.train_epoch(
                model,
                optimizer,
                buffer,
                bc_model,
                epoch,
                extra_loss_fn=oracle_loss_fn,
            )

            print(
                f"Loss={loss:.4f} "
                f"| Actor={actor_loss:.4f} "
                f"| Critic={critic_loss:.4f} "
                f"| KL={approx_kl:.6f} "
                f"| DKL(RL||BC)={dkl:.6f} "
                f"| DKL loss={dkl_loss:.6e}",
                flush=True,
            )

            # =================================================
            # Replay buffer sauvegarde
            # =================================================

            if epoch % 5 == 0:

                rl.save_replay_buffer(
                    buffer,
                    epoch,
                )

            # =================================================
            # On-policy
            # =================================================

            buffer.clear()

            print(
                "Replay buffer cleared after PPO update.",
                flush=True,
            )

            # =================================================
            # Uncertainty stats
            #
            # IMPORTANT:
            # stats is persistent across all epochs.
            # save() therefore writes the cumulative dataset.
            # =================================================

            stats_path = (
                PROJECT_ROOT
                / "checkpoints"
                / "uncertainty_stats_random_oracle.json"
            )

            stats.save(
                stats_path
            )

            print(
                f"Uncertainty JSON updated: "
                f"{len(stats.data)} positions",
                flush=True,
            )

            # =================================================
            # AL checkpoint
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
                f"AL checkpoint saved: "
                f"{checkpoint_path}",
                flush=True,
            )

            # =================================================
            # AL league snapshot
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

            snapshot = (
                copy.deepcopy(
                    model
                ).to(DEVICE)
            )

            snapshot.eval()

            agent_name = (
                f"league_epoch_{epoch:03d}"
            )

            league.add_agent(
                agent_name,
                snapshot,
            )

            snapshot_path = (
                AL_LEAGUE_DIR
                / f"{agent_name}.pt"
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        snapshot.state_dict(),
                },
                snapshot_path,
            )

            print(
                f"AL league snapshot saved: "
                f"{snapshot_path}",
                flush=True,
            )

            # =================================================
            # Shared snapshot
            # =================================================

            if agent_name not in (
                shared_league_models
            ):

                raise RuntimeError(
                    f"Missing shared slot "
                    f"for {agent_name}"
                )

            shared_snapshot = (
                shared_league_models[
                    agent_name
                ]
            )

            for (
                key,
                value,
            ) in snapshot.state_dict().items():

                shared_snapshot.state_dict()[
                    key
                ].copy_(
                    value.detach().cpu()
                )

            shared_snapshot.eval()

            # =================================================
            # Registry
            # =================================================

            league_registry[:] = (
                league.names()
            )

            print(
                "Updated league registry:",
                list(league_registry),
                flush=True,
            )

            # =================================================
            # Shared current model
            # =================================================

            for (
                key,
                value,
            ) in model.state_dict().items():

                shared_current_model.state_dict()[
                    key
                ].copy_(
                    value.detach().cpu()
                )

            # =================================================
            # Summary
            # =================================================

            print(
                f"\n===== Epoch {epoch} summary =====",
                flush=True,
            )

            print(
                f"Self-play: "
                f"{wins}W / "
                f"{losses}L / "
                f"{draws}D "
                f"({score_rate:.1%})",
                flush=True,
            )

            print(
                f"DKL(RL || BC): "
                f"{dkl:.6e}",
                flush=True,
            )

            print(
                f"DKL lambda: "
                f"{rl.get_dkl_lambda(epoch).item():.6e}",
                flush=True,
            )

            print(
                f"DKL loss: "
                f"{dkl_loss:.6e}",
                flush=True,
            )

            print(
                f"League size: "
                f"{len(league)}",
                flush=True,
            )

            print(
                f"Oracle annotations: "
                f"{len(oracle_buffer)}",
                flush=True,
            )

    manager.shutdown()

    print(
        "\nRL + ORACLE training finished.",
        flush=True,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()