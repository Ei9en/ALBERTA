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
import numpy as np

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
AL_END_EPOCH = 30


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
# Oracle injection frequency
#
# Oracle is injected once every N PPO minibatch updates.
#
# N = 1:
#     Oracle on every PPO update.
#
# N = 5:
#     Oracle on 1 PPO update out of 5.
#
# The remaining PPO updates are pure RL.
# ============================================================

ORACLE_INJECTION_FREQUENCY = 1


# ============================================================
# Oracle loss coefficients
# ============================================================

ORACLE_POLICY_COEF = 0.05

ORACLE_VALUE_COEF = 0.5


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
#
# Criticality represents the importance of selecting the
# correct action.
#
# It is therefore used as a supervision weight rather than
# as a temperature defining an artificial target distribution.
# ============================================================

CRITICALITY_WEIGHTS = {
    "critical": 1.00,
    "non_critical": 0.50,
    "outcome_independent": 0.25,
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
                CRITICALITY_WEIGHTS
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
# Oracle loss
# ============================================================

def compute_oracle_loss(
    model,
    oracle_batch,
    device=DEVICE,
    policy_coef=ORACLE_POLICY_COEF,
    value_coef=ORACLE_VALUE_COEF,
):
    """
    Compute Oracle supervision loss.

    Oracle annotation semantics
    ----------------------------

    oracle_move:
        Best move, or at least one of the best moves.

    confidence:
        Confidence that oracle_move is a best move.

    criticality:
        Importance of choosing the correct action.

    reward:
        Oracle evaluation of the position under Nash-equilibrium
        assumption, in {-1, 0, +1}.

    Policy supervision
    ------------------

    The Oracle annotation specifies an action, not a probability
    distribution over legal actions.

    Therefore the policy loss directly maximizes the probability
    of the annotated Oracle move:

        L_policy = -log pi(a* | s)

    No artificial target distribution or temperature is introduced.

    Confidence and criticality determine the supervision weight:

        w = w_confidence * w_criticality

    Value supervision
    -----------------

    The Oracle reward is used as an optional value target:

        L_value = MSE(V(s), R_oracle)

    The value component can be disabled with:

        value_coef = 0
    """

    if not oracle_batch:

        zero = sum(
            (
                parameter.sum()
                * 0.0
            )
            for parameter
            in model.parameters()
        )

        return {
            "loss":
                zero,

            "policy_loss":
                zero,

            "value_loss":
                zero,
        }

    # ========================================================
    # Prepare Oracle data
    # ========================================================

    board_objects = []

    oracle_actions = []
    weights = []
    oracle_rewards = []

    for record in oracle_batch:

        # ----------------------------------------------------
        # Atomic board
        # ----------------------------------------------------

        board = chess.variant.AtomicBoard(
            record["fen"]
        )

        board_objects.append(
            board
        )

        # ----------------------------------------------------
        # Oracle action
        # ----------------------------------------------------

        oracle_actions.append(
            ACTION_TO_INDEX[
                record["oracle_move"]
            ]
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = record.get(
            "confidence",
            "medium",
        )

        confidence_weight = (
            CONFIDENCE_WEIGHTS[
                confidence
            ]
        )

        # ----------------------------------------------------
        # Criticality
        # ----------------------------------------------------

        criticality = record.get(
            "criticality",
            "non_critical",
        )

        criticality_weight = (
            CRITICALITY_WEIGHTS[
                criticality
            ]
        )

        # ----------------------------------------------------
        # Combined supervision weight
        # ----------------------------------------------------

        weights.append(
            confidence_weight
            * criticality_weight
        )

        # ----------------------------------------------------
        # Oracle reward
        # ----------------------------------------------------

        oracle_rewards.append(
            float(
                record["reward"]
            )
        )

    # ========================================================
    # Encode boards
    # ========================================================

    boards = encode_boards(
        board_objects
    ).to(device)

    # ========================================================
    # Tensor construction
    # ========================================================

    oracle_actions = torch.tensor(
        oracle_actions,
        dtype=torch.long,
        device=device,
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    oracle_rewards = torch.tensor(
        oracle_rewards,
        dtype=torch.float32,
        device=device,
    )

    # ========================================================
    # Forward pass
    # ========================================================

    logits, values = model(
        boards
    )

    # ========================================================
    # Legal move masking
    # ========================================================

    masked_logits = logits.clone()

    for i, board in enumerate(
        board_objects
    ):

        legal_indices = []

        for move in board.legal_moves:

            move_uci = move.uci()

            if move_uci not in ACTION_TO_INDEX:

                raise ValueError(
                    f"Legal Atomic move "
                    f"{move_uci} is not present "
                    f"in ACTION_TO_INDEX."
                )

            legal_indices.append(
                ACTION_TO_INDEX[
                    move_uci
                ]
            )

        legal_mask = torch.zeros(
            logits.shape[1],
            dtype=torch.bool,
            device=device,
        )

        legal_mask[
            legal_indices
        ] = True

        masked_logits[
            i
        ] = masked_logits[
            i
        ].masked_fill(
            ~legal_mask,
            float("-inf"),
        )

    # ========================================================
    # Policy loss
    # ========================================================

    log_probs = F.log_softmax(
        masked_logits,
        dim=1,
    )

    oracle_log_probs = (
        log_probs[
            torch.arange(
                len(oracle_batch),
                device=device,
            ),
            oracle_actions,
        ]
    )

    policy_losses = (
        -oracle_log_probs
    )

    policy_loss = (
        (
            weights
            * policy_losses
        ).sum()
        /
        weights.sum().clamp_min(
            1e-8
        )
    )

    # ========================================================
    # Value loss
    # ========================================================

    predicted_values = (
        values[:, 0]
    )

    value_losses = F.mse_loss(
        predicted_values,
        oracle_rewards,
        reduction="none",
    )

    value_loss = (
        (
            weights
            * value_losses
        ).sum()
        /
        weights.sum().clamp_min(
            1e-8
        )
    )

    # ========================================================
    # Total Oracle loss
    # ========================================================

    oracle_loss = (
        policy_coef
        * policy_loss
        +
        value_coef
        * value_loss
    )

    return {
        "loss":
            oracle_loss,

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

    state = {
        "calls": 0,
        "injections": 0,
    }

    def oracle_loss_fn(
        model
    ):

        state["calls"] += 1

        # ----------------------------------------------------
        # Skip Oracle injection
        #
        # Example with frequency = 5:
        #
        # call 1  -> Oracle
        # call 2  -> skip
        # call 3  -> skip
        # call 4  -> skip
        # call 5  -> skip
        # call 6  -> Oracle
        # ...
        # ----------------------------------------------------

        if (
            (state["calls"] - 1)
            % ORACLE_INJECTION_FREQUENCY
            != 0
        ):

            # IMPORTANT:
            #
            # The zero must remain connected to the model
            # computation graph.
            #
            # Otherwise train_epoch() will fail when it calls:
            #
            #     torch.autograd.grad(extra_loss, ...)
            #
            # A plain torch.zeros(...) has no grad_fn.

            zero = sum(
                (
                    parameter.sum()
                    * 0.0
                )
                for parameter
                in model.parameters()
            )

            return {
                "loss":
                    zero,

                "policy_loss":
                    zero,

                "value_loss":
                    zero,
            }

        # ----------------------------------------------------
        # Actual Oracle injection
        # ----------------------------------------------------

        state["injections"] += 1

        effective_batch_size = min(
            ORACLE_BATCH_SIZE,
            len(oracle_buffer),
        )

        oracle_batch = oracle_buffer.sample(
            effective_batch_size
        )

        return compute_oracle_loss(
            model,
            oracle_batch,
            device=DEVICE,
            policy_coef=ORACLE_POLICY_COEF,
            value_coef=ORACLE_VALUE_COEF,
        )

    # --------------------------------------------------------
    # Reset diagnostics at the beginning of each epoch
    # --------------------------------------------------------

    def reset_epoch_stats():

        state["calls"] = 0
        state["injections"] = 0

    # --------------------------------------------------------
    # Retrieve diagnostics
    # --------------------------------------------------------

    def get_epoch_stats():

        return {
            "calls":
                state["calls"],

            "injections":
                state["injections"],
        }

    oracle_loss_fn.reset_epoch_stats = (
        reset_epoch_stats
    )

    oracle_loss_fn.get_epoch_stats = (
        get_epoch_stats
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
    # Oracle injection configuration
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        "ORACLE INJECTION"
    )

    print(
        "======================================"
    )

    print(
        f"Injection frequency: "
        f"1 / {ORACLE_INJECTION_FREQUENCY} PPO updates"
    )

    print(
        f"Oracle policy coefficient: "
        f"{ORACLE_POLICY_COEF:.4f}"
    )

    print(
        f"Oracle value coefficient:  "
        f"{ORACLE_VALUE_COEF:.4f}"
    )

    print(
        f"Oracle batch size: "
        f"{ORACLE_BATCH_SIZE}"
    )

    print(
        "======================================"
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
    # entire AL11 -> AL30 experiment.
    #
    # If this were created inside the epoch loop,
    # stats.save() would overwrite the JSON with only
    # the current epoch's positions.
    # ========================================================

    stats = rl.UncertaintyStats()

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

            oracle_loss_fn.reset_epoch_stats()

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

            oracle_epoch_stats = (
                oracle_loss_fn
                .get_epoch_stats()
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

            print(
                f"Oracle callbacks: "
                f"{oracle_epoch_stats['calls']} "
                f"| injections: "
                f"{oracle_epoch_stats['injections']}",
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
                / "uncertainty_stats_random_oracle_1in5.json"
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
                / "al_epoch_1in5"
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

                    "oracle_injection_frequency":
                        ORACLE_INJECTION_FREQUENCY,

                    "oracle_policy_coef":
                        ORACLE_POLICY_COEF,

                    "oracle_value_coef":
                        ORACLE_VALUE_COEF,
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
                / "league_al_1in5"
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
                f"Oracle callbacks: "
                f"{oracle_epoch_stats['calls']}",
                flush=True,
            )

            print(
                f"Oracle injections: "
                f"{oracle_epoch_stats['injections']}",
                flush=True,
            )

            print(
                f"Oracle injection: "
                f"1 / {ORACLE_INJECTION_FREQUENCY} PPO updates",
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