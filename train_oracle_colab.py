# ============================================================
# Train_Oracle.py
# ============================================================

import json
import pickle
import chess
import chess.variant
import copy
import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.selfplay.league import League

from src.encoding import encode_boards

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.agents.ppo_agent import PPOAgent

from src.rl.replay_buffer import ReplayBuffer
from src.rl.oracle_replay_buffer import OracleReplayBuffer
from src.rl.uncertainty_stats import UncertaintyStats

from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX


from train_rl_league_colab import (
    load_bc_agent,
    _prepare_shared_model,
    _init_selfplay_worker,
    _selfplay_worker,
    collect_games_parallel,
    compute_gae,
)


# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/ALBERTA"
)


# ============================================================
# Temperature
# ============================================================

TEMPERATURE_SELFPLAY = 2

# ============================================================
# Reprise RL
# ============================================================

START_EPOCH = 11

RESUME_RL = True

RESUME_EPOCH = START_EPOCH - 1

CHECKPOINT_EPOCH = START_EPOCH - 1


LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league"
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Hyperparameters
# ============================================================

LR = 3e-4

GAMES_PER_EPOCH = 2500

RL_EPOCHS = 10

CHECKPOINT_EVERY = 5

VALUE_COEF = 0.1

BATCH_SIZE = 4096

SGD_EPOCHS = 3

GAMMA = 0.99

GAE_LAMBDA = 0.95


# ============================================================
# PPO diagnostics
# ============================================================

PPO_CLIP = 0.2

ENTROPY_COEF = 0.01


# ============================================================
# League
# ============================================================

LEAGUE_MAX_AGENTS = 12

LEAGUE_END_EPOCH = START_EPOCH - 1

LEAGUE_START_EPOCH = LEAGUE_END_EPOCH - 9


# ============================================================
# Oracle
# ============================================================

ORACLE_QUEUE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "oracle_queue.jsonl"
)


# ------------------------------------------------------------
# Oracle policy supervision
#
# Confidence = confiance dans le coup annoté.
# Elle agit UNIQUEMENT sur la policy loss.
# ------------------------------------------------------------

ORACLE_CONFIDENCE_WEIGHTS = {
    "low": 0.50,
    "medium": 0.75,
    "high": 0.99,
}


# ------------------------------------------------------------
# Oracle value supervision
#
# Criticality = importance décisionnelle de la position.
# Elle agit UNIQUEMENT sur le critic loss.
# ------------------------------------------------------------

ORACLE_CRITICALITY_WEIGHTS = {
    "critical": 1.00,
    "non_critical": 0.66,
    "outcome_independent": 0.33,
}


# ------------------------------------------------------------
# Coefficients globaux
#
# Ces deux coefficients sont indépendants.
# ------------------------------------------------------------

ORACLE_POLICY_COEF = 1.0

ORACLE_VALUE_COEF = 1.0



# ============================================================
# RL checkpoint
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / f"rl_epoch_{CHECKPOINT_EPOCH}.pt"
)


# ============================================================
# Model Loading
# ============================================================



def load_model():

    print()
    print(
        "======================================"
    )


    if RESUME_RL:

        print("Resuming RL")

        print(
            "======================================"
        )


        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "rl_epoch"
            / f"rl_epoch_{RESUME_EPOCH}.pt"
        )


        print(
            f"Checkpoint: {checkpoint_path}"
        )


        bc_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=32,
            blocks=4,
        )


        model = ActorCritic(
            bc_model
        ).to(DEVICE)


        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        optimizer = Adam(
            model.parameters(),
            lr=LR,
        )


        if "optimizer_state_dict" in checkpoint:

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )


            print(
                "Optimizer state loaded."
            )


        print(
            f"RL checkpoint loaded "
            f"(epoch {checkpoint.get('epoch', '?')})."
        )


    else:

        print(
            "Initializing Oracle training from BC5"
        )

        print(
            "======================================"
        )


        bc5_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "bc_epoch"
            / "bc_v3_epoch_5.pt"
        )


        bc_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=32,
            blocks=4,
        )


        checkpoint = torch.load(
            bc5_path,
            map_location=DEVICE,
        )


        bc_model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        model = ActorCritic(
            bc_model
        ).to(DEVICE)


        optimizer = Adam(
            model.parameters(),
            lr=LR,
        )


        print(
            "Initial policy loaded from BC5."
        )


    # ========================================================
    # League initiale
    # ========================================================

    league = League(
        max_agents=LEAGUE_MAX_AGENTS
    )


    bc4 = load_bc_agent(4)


    league.add_agent(
        "bc_epoch_4",
        bc4,
    )


    bc5 = load_bc_agent(5)


    league.add_agent(
        "bc_epoch_5",
        bc5,
    )


    # ========================================================
    # Reprise : charger snapshots
    # ========================================================

    if RESUME_RL:

        for epoch in range(
            1,
            RESUME_EPOCH + 1,
        ):

            path = (
                LEAGUE_DIR
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
            )


            snapshot.load_state_dict(
                checkpoint["model_state_dict"]
            )


            snapshot = snapshot.to(DEVICE)

            snapshot.eval()


            league.add_agent(
                f"league_epoch_{epoch:03d}",
                snapshot,
            )


            print(
                f"Loaded league_epoch_{epoch:03d}"
            )


    print()

    print(
        f"League loaded: {len(league)} agents"
    )


    return model, optimizer, league


# ============================================================
# Oracle queue
# ============================================================

def load_oracle_queue():

    if not ORACLE_QUEUE_PATH.exists():

        raise FileNotFoundError(
            f"Oracle queue not found: "
            f"{ORACLE_QUEUE_PATH}"
        )


    entries = []


    with open(
        ORACLE_QUEUE_PATH,
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


            try:

                entry = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                print(
                    f"WARNING: invalid JSON "
                    f"at line {line_number}: "
                    f"{exc}"
                )

                continue


            if entry.get("status") != "answered":
                continue


            entries.append(
                entry
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
        f"Answered annotations: "
        f"{len(entries)}"
    )


    return entries


# ============================================================
# Oracle preparation
# ============================================================

def prepare_oracle_data(
    entries,
):

    oracle_data = []


    skipped = 0


    for entry in entries:

        fen = entry.get(
            "fen"
        )


        oracle_move = entry.get(
            "oracle_move"
        )


        confidence = entry.get(
            "oracle_confidence"
        )


        criticality = entry.get(
            "oracle_situation"
        )


        if fen is None:

            skipped += 1

            continue


        if oracle_move is None:

            skipped += 1

            continue


        if confidence not in (
            ORACLE_CONFIDENCE_WEIGHTS
        ):

            print(
                "WARNING: unknown "
                f"oracle_confidence={confidence}"
            )

            skipped += 1

            continue


        if criticality not in (
            ORACLE_CRITICALITY_WEIGHTS
        ):

            print(
                "WARNING: unknown "
                f"oracle_criticality={criticality}"
            )

            skipped += 1

            continue


        try:

            action_index = (
                ACTION_TO_INDEX[
                    oracle_move
                ]
            )

        except KeyError:

            print(
                f"WARNING: oracle move "
                f"{oracle_move} not found "
                f"in ACTION_TO_INDEX."
            )

            skipped += 1

            continue


        confidence_weight = (
            ORACLE_CONFIDENCE_WEIGHTS[
                confidence
            ]
        )


        criticality_weight = (
            ORACLE_CRITICALITY_WEIGHTS[
                criticality
            ]
        )


        oracle_data.append(
            {
                "fen":
                    fen,

                "oracle_move":
                    oracle_move,

                "action":
                    action_index,

                "confidence":
                    confidence,

                "confidence_weight":
                    confidence_weight,

                "criticality":
                    criticality,

                "criticality_weight":
                    criticality_weight,
            }
        )


    print(
        f"Valid Oracle annotations: "
        f"{len(oracle_data)}"
    )


    print(
        f"Skipped Oracle annotations: "
        f"{skipped}"
    )


    return oracle_data


# ============================================================
# Self-play globals
# ============================================================

_WORKER_CURRENT_MODEL = None

_WORKER_LEAGUE_MODELS = None

_WORKER_CURRENT_AGENT = None

_WORKER_LEAGUE_AGENTS = None

_WORKER_LEAGUE_REGISTRY = None


# ============================================================
# Oracle PPO training
# ============================================================

def train_oracle_epoch(
    model,
    optimizer,
    buffer,
    bc_model,
    oracle_buffer,
):

    model.train()

    bc_model.eval()


    # --------------------------------------------------------
    # BatchNorm
    # --------------------------------------------------------

    for module in model.modules():

        if isinstance(
            module,
            torch.nn.BatchNorm2d,
        ):

            module.eval()


    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    if len(buffer) < BATCH_SIZE:

        print(
            "Replay buffer too small."
        )

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )


    if len(oracle_buffer) == 0:

        print(
            "Oracle replay buffer empty."
        )


    TRAIN_STEPS = (
        len(buffer)
        // BATCH_SIZE
    )


    # ========================================================
    # Oracle weights
    # ========================================================

    confidence_weights = {
        "low":
            0.50,

        "medium":
            0.75,

        "high":
            0.99,
    }


    criticality_weights = {
        "critical":
            1.00,

        "non_critical":
            0.66,

        "outcome_independent":
            0.33,
    }


    # ========================================================
    # RL buffer index
    #
    # Permet de retrouver le game_result pour une position
    # Oracle rencontrée pendant le self-play.
    # ========================================================

    rl_by_fen = {}


    for entry in buffer.buffer:

        fen = entry["fen"]


        # Une position peut apparaître plusieurs fois.
        # On conserve la première occurrence disposant
        # d'un résultat valide.

        if fen not in rl_by_fen:

            rl_by_fen[fen] = entry


        elif (
            rl_by_fen[fen].get("game_result") is None
            and
            entry.get("game_result") is not None
        ):

            rl_by_fen[fen] = entry


    # ========================================================
    # Accumulators
    # ========================================================

    total_loss = 0.0

    total_actor = 0.0

    total_critic = 0.0

    total_oracle_policy = 0.0

    total_oracle_value = 0.0

    total_kl = 0.0

    total_entropy = 0.0

    total_oracle_policy_positions = 0

    total_oracle_value_positions = 0


    total_updates = (
        TRAIN_STEPS
        * SGD_EPOCHS
    )


    progress = tqdm(
        total=total_updates,
        desc="Oracle PPO Training",
    )


    # ========================================================
    # PPO updates
    # ========================================================

    for _ in range(
        SGD_EPOCHS
    ):

        for _ in range(
            TRAIN_STEPS
        ):

            # =================================================
            # RL batch
            # =================================================

            batch = buffer.sample(
                BATCH_SIZE
            )


            boards = [
                chess.variant.AtomicBoard(
                    s["fen"]
                )
                for s in batch
            ]


            x = encode_boards(
                boards
            ).to(DEVICE)


            policy, values = model(x)


            # =================================================
            # BC policy
            # =================================================

            with torch.no_grad():

                bc_policy, _ = bc_model(x)


            # =================================================
            # Basic tensors
            # =================================================

            plys = torch.tensor(
                [
                    s["ply"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            )


            returns = torch.tensor(
                [
                    s["return"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            ).unsqueeze(1)


            raw_advantages = torch.tensor(
                [
                    s["advantage"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            )


            advantages = (
                raw_advantages
                - raw_advantages.mean()
            ) / (
                raw_advantages.std()
                + 1e-8
            )


            old_log_probs = torch.tensor(
                [
                    s["old_log_prob"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            )


            actions = torch.tensor(
                [
                    s["action"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.long,
            )


            # =================================================
            # Legal move mask
            # =================================================

            legal_mask = torch.zeros(
                (
                    len(batch),
                    policy.shape[1],
                ),
                dtype=torch.bool,
                device=DEVICE,
            )


            for i, s in enumerate(batch):

                ids = [
                    ACTION_TO_INDEX[m]
                    for m in s[
                        "legal_moves"
                    ]
                ]


                legal_mask[
                    i,
                    ids,
                ] = True


            # =================================================
            # RL policy
            # =================================================

            legal_policy = (
                policy.masked_fill(
                    ~legal_mask,
                    float("-inf"),
                )
            )


            legal_bc_policy = (
                bc_policy.masked_fill(
                    ~legal_mask,
                    float("-inf"),
                )
            )


            rl_log_probs = F.log_softmax(
                legal_policy,
                dim=1,
            )


            bc_log_probs = F.log_softmax(
                legal_bc_policy,
                dim=1,
            )


            safe_bc_log_probs = (
                bc_log_probs.masked_fill(
                    ~legal_mask,
                    0.0,
                )
            )


            # =================================================
            # BC opening prior
            # =================================================

            BC_PRIOR_STRENGTH = 1.0

            BC_PRIOR_PLIES = 6.0


            bc_weight = (
                BC_PRIOR_STRENGTH
                *
                torch.clamp(
                    1.0
                    - plys
                    / BC_PRIOR_PLIES,
                    min=0.0,
                )
            )


            combined_log_probs = (
                rl_log_probs
                +
                bc_weight.unsqueeze(1)
                *
                safe_bc_log_probs
            )


            combined_log_probs = F.log_softmax(
                combined_log_probs,
                dim=1,
            )


            # =================================================
            # Self-play temperature
            # =================================================

            ppo_logits = (
                combined_log_probs
                /
                TEMPERATURE_SELFPLAY
            )


            log_probs = F.log_softmax(
                ppo_logits,
                dim=1,
            )


            selected_log_probs = (
                log_probs
                .gather(
                    1,
                    actions.unsqueeze(1),
                )
                .squeeze(1)
            )


            # =================================================
            # PPO ratio
            # =================================================

            log_ratio = (
                selected_log_probs
                - old_log_probs
            )


            ratio = torch.exp(
                log_ratio
            )


            approx_kl = (
                (
                    ratio
                    - 1.0
                    - log_ratio
                ).mean()
            )


            clipped_mask = (
                (ratio < 1.0 - PPO_CLIP)
                |
                (ratio > 1.0 + PPO_CLIP)
            )


            clip_fraction = (
                clipped_mask
                .float()
                .mean()
            )


            unclipped = (
                ratio
                * advantages
            )


            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - PPO_CLIP,
                    1.0 + PPO_CLIP,
                )
                * advantages
            )


            actor_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()


            # =================================================
            # Entropy
            # =================================================

            probs = torch.exp(
                log_probs
            )


            safe_log_probs = (
                log_probs.masked_fill(
                    ~legal_mask,
                    0.0,
                )
            )


            entropy = -(
                probs
                *
                safe_log_probs
            ).sum(
                dim=1
            ).mean()


            # =================================================
            # Standard critic loss
            # =================================================

            critic_loss = F.mse_loss(
                values,
                returns,
            )


            # =================================================
            # Oracle batch
            # =================================================
            #
            # IMPORTANT :
            #
            # confidence -> policy
            # criticality -> value
            #
            # Les deux sont indépendants.
            # =================================================

            oracle_policy_loss = torch.tensor(
                0.0,
                device=DEVICE,
            )


            oracle_value_loss = torch.tensor(
                0.0,
                device=DEVICE,
            )


            oracle_policy_count = 0

            oracle_value_count = 0


            if len(oracle_buffer) > 0:

                oracle_batch_size = min(
                    BATCH_SIZE,
                    len(oracle_buffer),
                )


                oracle_batch = (
                    oracle_buffer.sample(
                        oracle_batch_size
                    )
                )


                # =================================================
                # Oracle boards
                # =================================================

                oracle_boards = [
                    chess.variant.AtomicBoard(
                        entry["fen"]
                    )
                    for entry in oracle_batch
                ]


                oracle_x = encode_boards(
                    oracle_boards
                ).to(DEVICE)


                oracle_policy_logits, oracle_values = (
                    model(
                        oracle_x
                    )
                )


                # =================================================
                # Oracle policy supervision
                # =================================================

                policy_losses = []

                policy_weights = []


                for i, entry in enumerate(
                    oracle_batch
                ):

                    oracle_move = (
                        entry["oracle_move"]
                    )


                    confidence = (
                        entry["confidence"]
                    )


                    if confidence not in (
                        confidence_weights
                    ):

                        continue


                    try:

                        oracle_action = (
                            ACTION_TO_INDEX[
                                oracle_move
                            ]
                        )

                    except KeyError:

                        print(
                            "WARNING: Oracle move "
                            f"{oracle_move} not found "
                            "in ACTION_TO_INDEX."
                        )

                        continue


                    board = (
                        oracle_boards[i]
                    )


                    legal_ids = {
                        ACTION_TO_INDEX[
                            move.uci()
                        ]
                        for move
                        in board.legal_moves
                    }


                    if oracle_action not in legal_ids:

                        print(
                            "WARNING: Oracle move "
                            f"{oracle_move} is illegal "
                            f"for FEN {entry['fen']}"
                        )

                        continue


                    # ------------------------------------------------
                    # Legal policy
                    # ------------------------------------------------

                    legal_oracle_policy = (
                        oracle_policy_logits[
                            i
                        ]
                        .clone()
                    )


                    illegal_mask = torch.ones(
                        oracle_policy_logits.shape[1],
                        dtype=torch.bool,
                        device=DEVICE,
                    )


                    illegal_mask[
                        list(legal_ids)
                    ] = False


                    legal_oracle_policy = (
                        legal_oracle_policy.masked_fill(
                            illegal_mask,
                            float("-inf"),
                        )
                    )


                    oracle_log_probs = (
                        F.log_softmax(
                            legal_oracle_policy,
                            dim=0,
                        )
                    )


                    policy_losses.append(
                        -oracle_log_probs[
                            oracle_action
                        ]
                    )


                    policy_weights.append(
                        confidence_weights[
                            confidence
                        ]
                    )


                # =================================================
                # Weighted Oracle policy loss
                # =================================================

                if policy_losses:

                    policy_losses_tensor = (
                        torch.stack(
                            policy_losses
                        )
                    )


                    policy_weights_tensor = (
                        torch.tensor(
                            policy_weights,
                            device=DEVICE,
                            dtype=torch.float32,
                        )
                    )


                    oracle_policy_loss = (
                        (
                            policy_losses_tensor
                            *
                            policy_weights_tensor
                        ).sum()
                        /
                        (
                            policy_weights_tensor.sum()
                            + 1e-8
                        )
                    )


                    oracle_policy_count = (
                        len(policy_losses)
                    )


                # =================================================
                # Oracle value supervision
                # =================================================
                #
                # On ne peut utiliser la criticality que si la
                # position annotée a également été rencontrée
                # dans le self-play courant.
                #
                # Le résultat de la partie fournit alors la cible.
                # =================================================

                value_losses = []

                value_weights = []


                for i, entry in enumerate(
                    oracle_batch
                ):

                    situation = (
                        entry["situation"]
                    )


                    if situation not in (
                        criticality_weights
                    ):

                        continue


                    rl_entry = (
                        rl_by_fen.get(
                            entry["fen"]
                        )
                    )


                    if rl_entry is None:

                        continue


                    game_result = (
                        rl_entry.get(
                            "game_result"
                        )
                    )


                    if game_result is None:

                        continue


                    # ------------------------------------------------
                    # Final outcome from White's perspective
                    # ------------------------------------------------

                    if game_result == "1-0":

                        white_value = 1.0

                    elif game_result == "0-1":

                        white_value = -1.0

                    else:

                        white_value = 0.0


                    # ------------------------------------------------
                    # Convert to side-to-move perspective
                    # ------------------------------------------------

                    board = (
                        oracle_boards[i]
                    )


                    target_value = (
                        white_value
                        if board.turn
                        else -white_value
                    )


                    prediction = (
                        oracle_values[
                            i,
                            0,
                        ]
                    )


                    value_losses.append(
                        (
                            prediction
                            - target_value
                        ) ** 2
                    )


                    value_weights.append(
                        criticality_weights[
                            situation
                        ]
                    )


                # =================================================
                # Weighted Oracle value loss
                # =================================================

                if value_losses:

                    value_losses_tensor = (
                        torch.stack(
                            value_losses
                        )
                    )


                    value_weights_tensor = (
                        torch.tensor(
                            value_weights,
                            device=DEVICE,
                            dtype=torch.float32,
                        )
                    )


                    oracle_value_loss = (
                        (
                            value_losses_tensor
                            *
                            value_weights_tensor
                        ).sum()
                        /
                        (
                            value_weights_tensor.sum()
                            + 1e-8
                        )
                    )


                    oracle_value_count = (
                        len(value_losses)
                    )


                del oracle_x

                del oracle_policy_logits

                del oracle_values


            # =================================================
            # Total loss
            # =================================================
            #
            # IMPORTANT :
            #
            # Oracle policy et Oracle value sont deux termes
            # séparés.
            #
            # Il n'existe AUCUN :
            #
            # confidence * criticality
            #
            # =================================================

            loss = (
                actor_loss
                +
                VALUE_COEF
                *
                critic_loss
                -
                ENTROPY_COEF
                *
                entropy
                +
                ORACLE_POLICY_COEF
                *
                oracle_policy_loss
                +
                ORACLE_VALUE_COEF
                *
                oracle_value_loss
            )


            # =================================================
            # Backward
            # =================================================

            optimizer.zero_grad()


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )


            optimizer.step()


            # =================================================
            # Accumulators
            # =================================================

            total_loss += (
                loss.item()
            )


            total_actor += (
                actor_loss.item()
            )


            total_critic += (
                critic_loss.item()
            )


            total_oracle_policy += (
                oracle_policy_loss.item()
            )


            total_oracle_value += (
                oracle_value_loss.item()
            )


            total_kl += (
                approx_kl.item()
            )


            total_entropy += (
                entropy.item()
            )


            total_oracle_policy_positions += (
                oracle_policy_count
            )


            total_oracle_value_positions += (
                oracle_value_count
            )


            progress.update(1)


    progress.close()


    # ========================================================
    # Averages
    # ========================================================

    avg_loss = (
        total_loss
        / total_updates
    )


    avg_actor = (
        total_actor
        / total_updates
    )


    avg_critic = (
        total_critic
        / total_updates
    )


    avg_oracle_policy = (
        total_oracle_policy
        / total_updates
    )


    avg_oracle_value = (
        total_oracle_value
        / total_updates
    )


    avg_kl = (
        total_kl
        / total_updates
    )


    avg_entropy = (
        total_entropy
        / total_updates
    )


    oracle_policy_positions_per_update = (
        total_oracle_policy_positions
        / total_updates
    )


    oracle_value_positions_per_update = (
        total_oracle_value_positions
        / total_updates
    )


    # ========================================================
    # Diagnostics
    # ========================================================

    print()

    print(
        "======================================"
    )

    print(
        "ORACLE PPO DIAGNOSTICS"
    )

    print(
        "======================================"
    )


    print(
        f"Self-play temperature: "
        f"{TEMPERATURE_SELFPLAY:.2f}"
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
        "Oracle confidence weights: "
        "low=0.50 / medium=0.75 / high=0.99"
    )


    print(
        "Oracle criticality weights: "
        "critical=1.00 / "
        "non_critical=0.66 / "
        "outcome_independent=0.33"
    )


    print(
        f"Oracle policy loss:        "
        f"{avg_oracle_policy:.6f}"
    )


    print(
        f"Oracle value loss:         "
        f"{avg_oracle_value:.6f}"
    )


    print(
        f"Oracle policy positions/update: "
        f"{oracle_policy_positions_per_update:.2f}"
    )


    print(
        f"Oracle value positions/update:  "
        f"{oracle_value_positions_per_update:.2f}"
    )


    print(
        f"Policy KL:                 "
        f"{avg_kl:.6e}"
    )


    print(
        f"Entropy:                   "
        f"{avg_entropy:.6f}"
    )


    print(
        "======================================"
    )


    return (
        avg_loss,
        avg_actor,
        avg_critic,
        avg_oracle_policy,
        avg_oracle_value,
        avg_kl,
    )


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
):

    path = (
        PROJECT_ROOT
        / "checkpoints"
        / "oracle_epoch"
        / f"oracle_epoch_{epoch}.pt"
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        path,
    )


    print(
        "Oracle checkpoint saved:",
        path,
        flush=True,
    )


# ============================================================
# Replay Buffer
# ============================================================

def save_replay_buffer(
    buffer,
    epoch,
):

    path = (
        PROJECT_ROOT
        / "checkpoints"
        / f"oracle_replay_buffer_epoch_{epoch}.pkl"
    )


    with open(
        path,
        "wb",
    ) as f:

        pickle.dump(
            buffer,
            f,
        )


    print(
        "Replay buffer saved:",
        path,
        flush=True,
    )


# ============================================================
# Main
# ============================================================

# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Oracle queue
    # ========================================================

    oracle_entries = (
        load_oracle_queue()
    )


    oracle_data = (
        prepare_oracle_data(
            oracle_entries
        )
    )


    # ========================================================
    # Load model + optimizer + league
    # ========================================================

    model, optimizer, league = (
        load_model()
    )


    # ========================================================
    # BC model
    # ========================================================

    bc_model = load_bc_agent(5)


    bc_model_selfplay = copy.deepcopy(
        bc_model
    ).to("cpu")


    bc_model_selfplay.eval()


    # ========================================================
    # Oracle replay buffer
    #
    # Contient uniquement les annotations humaines.
    #
    # confidence :
    #     policy supervision
    #
    # situation :
    #     critic supervision
    # ========================================================

    oracle_buffer = OracleReplayBuffer(
        capacity=300000
    )


    for entry in oracle_data:

        oracle_buffer.add(
            entry["fen"],
            entry["oracle_move"],
            entry["confidence"],
            entry["criticality"],
        )


    print(
        f"Oracle replay buffer: "
        f"{len(oracle_buffer)} annotations",
        flush=True,
    )


    # ========================================================
    # RL replay buffer
    #
    # Celui-ci contient les trajectoires self-play.
    # Il reste nécessaire pour le PPO et pour récupérer
    # le résultat final des positions annotées.
    # ========================================================

    buffer = ReplayBuffer(
        capacity=300000
    )


    # ========================================================
    # Uncertainty statistics
    # ========================================================

    stats = UncertaintyStats()


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
        _prepare_shared_model(
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
            _prepare_shared_model(
                league_model
            )
        )


    # ========================================================
    # Préparer futurs slots
    # ========================================================

    future_end = (
        START_EPOCH
        + RL_EPOCHS
        - 1
    )


    for epoch in range(
        START_EPOCH,
        future_end + 1,
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
        initializer=_init_selfplay_worker,
        initargs=(
            shared_current_model,
            shared_league_models,
            league_registry,
            bc_model_selfplay,
        ),
    ) as pool:

        # ====================================================
        # Oracle RL loop
        # ====================================================

        for epoch in range(
            START_EPOCH,
            START_EPOCH + RL_EPOCHS,
        ):

            print(
                "\n======================================",
                flush=True,
            )


            print(
                f"===== Oracle Epoch {epoch} =====",
                flush=True,
            )


            print(
                "======================================",
                flush=True,
            )


            wins = 0

            losses = 0

            draws = 0


            # =================================================
            # Self-play
            # =================================================

            games = collect_games_parallel(
                pool,
                shared_current_model,
                shared_league_models,
                model,
                league,
                GAMES_PER_EPOCH,
                stats,
                num_workers=NUM_WORKERS,
                batch_size=SELFPLAY_BATCH_SIZE,
            )


            # =================================================
            # Construction RL replay buffer
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


                # ------------------------------------------------
                # Game result
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Rewards
                # ------------------------------------------------

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


                # ------------------------------------------------
                # GAE
                # ------------------------------------------------

                advantages, returns = (
                    compute_gae(
                        trajectory,
                        rewards,
                        gamma=GAMMA,
                        gae_lambda=GAE_LAMBDA,
                    )
                )


                # ------------------------------------------------
                # Replay buffer
                # ------------------------------------------------

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
                        game_result=result,
                    )


            # =================================================
            # Diagnostics
            # =================================================

            score_rate = (
                wins
                + 0.5 * draws
            ) / GAMES_PER_EPOCH


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


            print(
                f"Oracle annotations available: "
                f"{len(oracle_buffer)}",
                flush=True,
            )


            # =================================================
            # Oracle PPO
            # =================================================

            (
                loss,
                actor_loss,
                critic_loss,
                oracle_policy_loss,
                oracle_value_loss,
                approx_kl,
            ) = train_oracle_epoch(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                bc_model=bc_model,
                oracle_buffer=oracle_buffer,
            )


            print(
                f"Loss={loss:.4f} "
                f"| Actor={actor_loss:.4f} "
                f"| Critic={critic_loss:.4f} "
                f"| OraclePolicy={oracle_policy_loss:.4f} "
                f"| OracleValue={oracle_value_loss:.4f} "
                f"| KL={approx_kl:.6f}",
                flush=True,
            )


            # =================================================
            # Replay buffer sauvegarde
            # =================================================

            if epoch % 5 == 0:

                save_replay_buffer(
                    buffer,
                    epoch,
                )


            # =================================================
            # On-policy
            # =================================================

            buffer.clear()


            print(
                "RL replay buffer cleared after PPO update.",
                flush=True,
            )


            # =================================================
            # Uncertainty stats
            # =================================================

            stats_path = (
                PROJECT_ROOT
                / "checkpoints"
                / "uncertainty_stats_oracle.json"
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
            # Oracle checkpoint
            # =================================================

            if (
                epoch
                % CHECKPOINT_EVERY
                == 0
            ):

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    loss,
                )


            # =================================================
            # Snapshot league
            # =================================================

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
                LEAGUE_DIR
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
                f"League snapshot saved: "
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
            # Best checkpoint
            # =================================================

            if (
                best_loss is None
                or loss < best_loss
            ):

                best_loss = loss


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
                    PROJECT_ROOT
                    / "checkpoints"
                    / "oracle_best.pt",
                )


                print(
                    "New best Oracle checkpoint saved.",
                    flush=True,
                )


            # =================================================
            # Summary
            # =================================================

            print(
                f"\n===== Oracle Epoch {epoch} summary =====",
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
                f"Oracle annotations: "
                f"{len(oracle_buffer)}",
                flush=True,
            )


            print(
                f"League size: "
                f"{len(league)}",
                flush=True,
            )


    manager.shutdown()


    print(
        "\nOracle training finished.",
        flush=True,
    )


# ============================================================

if __name__ == "__main__":

    main()