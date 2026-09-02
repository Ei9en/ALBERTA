# ============================================================
# Train_Oracle.py
# ============================================================

import json
import pickle
import copy
import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm

import chess
import chess.variant

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.selfplay.league import League

from src.encoding import encode_boards

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.rl.replay_buffer import ReplayBuffer
from src.rl.oracle_replay_buffer import OracleReplayBuffer
from src.rl.uncertainty_stats import UncertaintyStats

from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX

from train_rl_league_colab import (
    load_bc_agent,
    _prepare_shared_model,
    _init_selfplay_worker,
    collect_games_parallel,
    compute_gae,
    get_dkl_lambda,
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

TEMPERATURE_SELFPLAY = 2.0


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
    / "league_al"
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

AL_EPOCHS = 1

CHECKPOINT_EVERY = 5

VALUE_COEF = 0.1

BATCH_SIZE = 4096

SGD_EPOCHS = 3

GAMMA = 0.99

GAE_LAMBDA = 0.95


# ============================================================
# Oracle value
# ============================================================

# Strength of the Oracle value supervision.
#
# This is deliberately separate from VALUE_COEF because:
#
# VALUE_COEF:
#     ordinary self-play critic
#
# ORACLE_VALUE_COEF:
#     human Oracle value supervision
#
# The Oracle policy and Oracle value therefore have
# independently controllable strengths.
# ============================================================

ORACLE_VALUE_COEF = 0.10


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
    / "queue"
    / "oracle_queue_1-10.jsonl"
)


# ============================================================
# Oracle policy confidence
# ============================================================

ORACLE_CONFIDENCE_WEIGHTS = {

    "low":
        0.50,

    "medium":
        0.75,

    "high":
        0.99,
}


ORACLE_POLICY_COEF = 0.1


# ============================================================
# Oracle criticality
# ============================================================

ORACLE_CRITICALITY_TEMPERATURES = {

    "critical":
        0.25,

    "non_critical":
        0.50,

    "outcome_independent":
        1.00,
}


# ============================================================
# Numerical safety
# ============================================================

FINITE_EPS = 1e-8


# ============================================================
# Utility: finite tensor check
# ============================================================

def assert_finite_tensor(
    tensor,
    name,
):

    if not torch.isfinite(
        tensor
    ).all():

        bad_mask = ~torch.isfinite(
            tensor
        )

        bad_count = (
            bad_mask.sum().item()
        )

        raise RuntimeError(
            f"NON-FINITE TENSOR: {name} "
            f"contains {bad_count} non-finite values."
        )


# ============================================================
# Utility: finite model check
# ============================================================

def assert_model_finite(
    model,
    name="model",
):

    for (
        parameter_name,
        parameter,
    ) in model.named_parameters():

        if not torch.isfinite(
            parameter
        ).all():

            raise RuntimeError(
                f"NON-FINITE MODEL PARAMETER: "
                f"{name}.{parameter_name}"
            )


# ============================================================
# Utility: finite gradients check
# ============================================================

def assert_gradients_finite(
    model,
):

    for (
        parameter_name,
        parameter,
    ) in model.named_parameters():

        if parameter.grad is None:

            continue


        if not torch.isfinite(
            parameter.grad
        ).all():

            raise RuntimeError(
                "NON-FINITE GRADIENT: "
                f"{parameter_name}"
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

        print(
            "Resuming RL"
        )

        print(
            "======================================"
        )

        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "oracle_epoch"
            / f"oracle_epoch_{RESUME_EPOCH}.pt"
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
            checkpoint[
                "model_state_dict"
            ]
        )

        assert_model_finite(
            model,
            "loaded RL model",
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
            f"(epoch "
            f"{checkpoint.get('epoch', '?')})."
        )

    else:

        print(
            "Initializing Oracle training from BC7"
        )

        print(
            "======================================"
        )

        bc7_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "bc_epoch"
            / "bc_epoch_7.pt"
        )

        bc_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=32,
            blocks=4,
        )

        checkpoint = torch.load(
            bc7_path,
            map_location=DEVICE,
        )

        bc_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model = ActorCritic(
            bc_model
        ).to(DEVICE)

        assert_model_finite(
            model,
            "initial BC model",
        )

        optimizer = Adam(
            model.parameters(),
            lr=LR,
        )

        print(
            "Initial policy loaded from BC5."
        )

    # ========================================================
    # League
    # ========================================================

    league = League(
        max_agents=LEAGUE_MAX_AGENTS
    )

    bc6 = load_bc_agent(6)

    league.add_agent(
        "bc_epoch_6",
        bc6,
    )

    bc7 = load_bc_agent(7)

    league.add_agent(
        "bc_epoch_7",
        bc7,
    )

    # ========================================================
    # Reprise league
    # ========================================================

    if RESUME_RL:

        # Keep only the 10 most recent RL snapshots.
        #
        # With BC6 + BC7:
        #
        # 10 RL snapshots
        # + 2 BC snapshots
        # = 12 agents maximum.
        #
        # Example:
        # RESUME_EPOCH = 30
        # -> league_epoch_021 ... league_epoch_030

        league_start = max(
            1,
            RESUME_EPOCH - 9
        )

        print()
        print(
            f"Loading league snapshots "
            f"{league_start:03d} -> "
            f"{RESUME_EPOCH:03d}"
        )

        for epoch in range(
            league_start,
            RESUME_EPOCH + 1,
        ):

            path = (
                LEAGUE_DIR
                / f"league_epoch_{epoch:03d}.pt"
            )

            if not path.exists():

                print(
                    f"WARNING: missing league snapshot: "
                    f"{path}"
                )

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
                checkpoint[
                    "model_state_dict"
                ]
            )

            assert_model_finite(
                snapshot,
                f"league_epoch_{epoch:03d}",
            )

            snapshot = snapshot.to(
                DEVICE
            )

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

    return (
        model,
        optimizer,
        league,
    )


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

        for (
            line_number,
            line,
        ) in enumerate(
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


            if entry.get(
                "status"
            ) != "answered":

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


    # ========================================================
    # Reward diagnostics
    # ========================================================

    reward_counts = {
        -1.0: 0,
         0.0: 0,
         1.0: 0,
    }


    for entry in entries:

        reward = entry.get(
            "reward"
        )


        if reward is None:

            continue


        try:

            reward = float(
                reward
            )

        except (
            TypeError,
            ValueError,
        ):

            continue


        if reward in reward_counts:

            reward_counts[
                reward
            ] += 1


    print()

    print(
        "Oracle reward distribution:"
    )


    for reward in (
        -1.0,
         0.0,
         1.0,
    ):

        print(
            f"  reward={reward:+.1f}: "
            f"{reward_counts[reward]}"
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


        reward = entry.get(
            "reward"
        )


        confidence = entry.get(
            "confidence"
        )


        if confidence is None:

            confidence = entry.get(
                "oracle_confidence"
            )


        criticality = entry.get(
            "criticality"
        )


        if criticality is None:

            criticality = entry.get(
                "oracle_situation"
            )


        if fen is None:

            skipped += 1

            continue


        if oracle_move is None:

            skipped += 1

            continue


        # ----------------------------------------------------
        # Reward is fundamental.
        #
        # Every answered Oracle annotation must have one.
        # ----------------------------------------------------

        if reward is None:

            print(
                "WARNING: Oracle annotation "
                "has no reward."
            )

            skipped += 1

            continue


        try:

            reward = float(
                reward
            )

        except (
            TypeError,
            ValueError,
        ):

            print(
                f"WARNING: invalid Oracle "
                f"reward={reward}"
            )

            skipped += 1

            continue


        if reward not in (
            -1.0,
             0.0,
             1.0,
        ):

            print(
                f"WARNING: Oracle reward must "
                f"be -1, 0 or +1, got {reward}"
            )

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
            ORACLE_CRITICALITY_TEMPERATURES
        ):

            print(
                "WARNING: unknown "
                f"oracle_situation={criticality}"
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


        temperature = (
            ORACLE_CRITICALITY_TEMPERATURES[
                criticality
            ]
        )


        if not torch.isfinite(
            torch.tensor(
                confidence_weight
            )
        ):

            raise RuntimeError(
                f"Non-finite confidence weight: "
                f"{confidence_weight}"
            )


        if not torch.isfinite(
            torch.tensor(
                temperature
            )
        ):

            raise RuntimeError(
                f"Non-finite temperature: "
                f"{temperature}"
            )


        if temperature <= 0.0:

            raise RuntimeError(
                f"Invalid Oracle temperature: "
                f"{temperature}"
            )


        oracle_data.append(
            {

                "fen":
                    fen,

                "oracle_move":
                    oracle_move,

                "action":
                    action_index,

                "reward":
                    reward,

                "confidence":
                    confidence,

                "confidence_weight":
                    confidence_weight,

                "criticality":
                    criticality,

                "criticality_temperature":
                    temperature,
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


    # ========================================================
    # Reward distribution
    # ========================================================

    counts = {
        -1.0: 0,
         0.0: 0,
         1.0: 0,
    }


    for entry in oracle_data:

        counts[
            entry["reward"]
        ] += 1


    print()

    print(
        "Oracle reward distribution:"
    )


    for reward in (
        -1.0,
         0.0,
         1.0,
    ):

        print(
            f"  reward={reward:+.1f}: "
            f"{counts[reward]}"
        )


    # ========================================================
    # Criticality distribution
    # ========================================================

    criticality_counts = {

        "outcome_independent":
            0,

        "non_critical":
            0,

        "critical":
            0,
    }


    for entry in oracle_data:

        criticality_counts[
            entry["criticality"]
        ] += 1


    print()

    print(
        "Criticality distribution:"
    )


    for (
        name,
        count,
    ) in criticality_counts.items():

        print(
            f"  {name}: {count}"
        )


    print()

    print(
        "Oracle criticality temperatures:"
    )


    for (
        name,
        temperature,
    ) in ORACLE_CRITICALITY_TEMPERATURES.items():

        print(
            f"  {name}: T={temperature:.2f}"
        )


    return oracle_data


# ============================================================
# Oracle target distribution
# ============================================================

def build_oracle_target(
    legal_ids,
    oracle_action,
    temperature,
    device,
):

    num_actions = len(ACTIONS)


    if temperature <= 0.0:

        raise ValueError(
            f"Temperature must be > 0, "
            f"got {temperature}"
        )


    target_logits = torch.full(
        (
            num_actions,
        ),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )


    legal_mask = torch.zeros(
        (
            num_actions,
        ),
        device=device,
        dtype=torch.bool,
    )


    legal_indices = list(
        legal_ids
    )


    if len(
        legal_indices
    ) == 0:

        raise RuntimeError(
            "Oracle target has no legal moves."
        )


    legal_mask[
        legal_indices
    ] = True


    target_logits[
        legal_mask
    ] = 0.0


    if not legal_mask[
        oracle_action
    ]:

        raise RuntimeError(
            "Oracle action is not legal."
        )


    target_logits[
        oracle_action
    ] = 1.0


    target_log_probs = F.log_softmax(
        target_logits / temperature,
        dim=0,
    )


    target_probs = torch.exp(
        target_log_probs
    )


    assert_finite_tensor(
        target_probs,
        "Oracle target probabilities",
    )


    target_probs = (
        target_probs
        .masked_fill(
            ~legal_mask,
            0.0,
        )
    )


    target_sum = (
        target_probs.sum()
    )


    if not torch.isfinite(
        target_sum
    ):

        raise RuntimeError(
            "Oracle target normalization "
            "became non-finite."
        )


    if target_sum <= 0.0:

        raise RuntimeError(
            "Oracle target has zero total probability."
        )


    target_probs = (
        target_probs
        /
        target_sum
    )


    assert_finite_tensor(
        target_probs,
        "Normalized Oracle target probabilities",
    )


    return target_probs


# ============================================================
# Oracle value target
# ============================================================

def compute_oracle_value_target(
    model,
    oracle_board,
    oracle_move,
    reward,
):

    """
    Compute the Oracle value target.

    The Oracle reward is defined in the reference frame
    of the player to move in the annotated FEN.

    After oracle_move, the player to move changes.

    Therefore:

        V(s) = r + gamma * V_next_in_same_reference_frame

    but the model evaluates s' from the opponent's
    point of view, so:

        V(s) = r - gamma * V(s')

    where V(s') is the value from the new side-to-move
    perspective.

    Terminal successor:
        no bootstrap is used.

    Non-terminal successor:
        one-step Oracle TD target is used.

    The bootstrap value is detached.
    """

    board_after = (
        oracle_board.copy(
            stack=False
        )
    )


    board_after.push(
        oracle_move
    )


    # --------------------------------------------------------
    # Terminal successor
    # --------------------------------------------------------

    if board_after.is_game_over():

        return torch.tensor(
            reward,
            device=DEVICE,
            dtype=torch.float32,
        )


    # --------------------------------------------------------
    # Non-terminal successor
    # --------------------------------------------------------

    next_x = encode_boards(
        [board_after]
    ).to(DEVICE)


    with torch.no_grad():

        _, next_value = model(
            next_x
        )


    assert_finite_tensor(
        next_value,
        "Oracle successor value",
    )


    next_value = (
        next_value
        .reshape(-1)[0]
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # next_value is from the opponent's perspective.
    #
    # Oracle reward is from the current player's
    # perspective.
    #
    # Therefore the bootstrap is NEGATED.
    # --------------------------------------------------------

    target = (
        torch.tensor(
            reward,
            device=DEVICE,
            dtype=torch.float32,
        )
        -
        GAMMA
        *
        next_value
    )


    target = torch.clamp(
        target,
        min=-1.0,
        max=1.0,
    )


    assert_finite_tensor(
        target,
        "Oracle value target",
    )


    return target.detach()


# ============================================================
# Oracle policy + value training
# ============================================================

# ============================================================
# Oracle policy + value training
# ============================================================

def train_oracle_epoch(
    model,
    optimizer,
    buffer,
    bc_model,
    oracle_buffer,
    epoch,
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
    # Model safety
    # --------------------------------------------------------

    assert_model_finite(
        model,
        "model before epoch",
    )


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
            0.0,
            0.0,
        )


    TRAIN_STEPS = (
        len(buffer)
        //
        BATCH_SIZE
    )


    total_loss = 0.0

    total_actor = 0.0

    total_critic = 0.0

    total_oracle_policy = 0.0

    total_oracle_value = 0.0

    total_kl = 0.0

    total_entropy = 0.0

    total_dkl = 0.0

    total_dkl_loss = 0.0

    total_dkl_lambda = 0.0


    total_oracle_confidence_weight = 0.0

    total_oracle_temperature = 0.0

    total_oracle_annotations = 0


    total_updates = (
        TRAIN_STEPS
        *
        SGD_EPOCHS
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


            policy, values = model(
                x
            )


            assert_finite_tensor(
                policy,
                "RL policy logits",
            )


            assert_finite_tensor(
                values,
                "RL values",
            )


            # =================================================
            # BC policy
            # =================================================

            with torch.no_grad():

                bc_policy, _ = bc_model(
                    x
                )


            assert_finite_tensor(
                bc_policy,
                "BC policy logits",
            )


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


            assert_finite_tensor(
                plys,
                "PPO plys",
            )

            assert_finite_tensor(
                returns,
                "PPO returns",
            )

            assert_finite_tensor(
                raw_advantages,
                "PPO raw advantages",
            )

            assert_finite_tensor(
                old_log_probs,
                "PPO old log probabilities",
            )


            # =================================================
            # Advantages
            # =================================================

            advantage_mean = (
                raw_advantages.mean()
            )


            advantage_std = (
                raw_advantages.std()
            )


            if not torch.isfinite(
                advantage_mean
            ):

                raise RuntimeError(
                    "Non-finite advantage mean."
                )


            if not torch.isfinite(
                advantage_std
            ):

                raise RuntimeError(
                    "Non-finite advantage std."
                )


            advantages = (
                raw_advantages
                -
                advantage_mean
            ) / (
                advantage_std
                +
                FINITE_EPS
            )


            assert_finite_tensor(
                advantages,
                "Normalized advantages",
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

                    for m
                    in s["legal_moves"]
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


            assert_finite_tensor(
                rl_log_probs[
                    legal_mask
                ],
                "RL legal log probabilities",
            )


            assert_finite_tensor(
                bc_log_probs[
                    legal_mask
                ],
                "BC legal log probabilities",
            )


            # =================================================
            # DKL anchor: RL || BC
            # =================================================
            #
            # The anchor is computed between the current RL
            # policy and the frozen BC policy, both restricted
            # to legal moves.
            #
            # DKL(RL || BC)
            #
            # The lambda schedule is epoch-dependent and is
            # imported directly from train_rl_league_colab.
            # =================================================

            # ------------------------------------------------------------
            # DKL(RL || BC)
            # ------------------------------------------------------------

            # Sanitize masked / invalid values BEFORE any arithmetic.
            safe_rl_log_probs = torch.where(
                legal_mask,
                torch.nan_to_num(
                    rl_log_probs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                torch.zeros_like(rl_log_probs),
            )

            safe_bc_log_probs = torch.where(
                legal_mask,
                torch.nan_to_num(
                    safe_bc_log_probs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                torch.zeros_like(safe_bc_log_probs),
            )

            # Convert only sanitized values.
            rl_probs = torch.exp(
                safe_rl_log_probs
            )

            # KL contribution per action.
            kl_per_action = (
                rl_probs
                *
                (
                    safe_rl_log_probs
                    -
                    safe_bc_log_probs
                )
            )

            # Ignore masked actions completely.
            kl_per_action = torch.where(
                legal_mask,
                kl_per_action,
                torch.zeros_like(kl_per_action),
            )

            dkl_per_sample = (
                kl_per_action
                .sum(dim=1)
            )

            # Final safety guard.
            dkl_per_sample = torch.nan_to_num(
                dkl_per_sample,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            assert_finite_tensor(
                dkl_per_sample,
                "DKL per sample",
            )

            dkl_loss = (
                dkl_per_sample.mean()
            )

            assert_finite_tensor(
                dkl_loss,
                "DKL(RL || BC)",
            )

            dkl_lambda = get_dkl_lambda(
                epoch
            )

            if not torch.isfinite(dkl_lambda):
                raise RuntimeError(
                    f"Non-finite DKL lambda: "
                    f"{dkl_lambda}"
                )

            dkl_anchor_loss = (
                dkl_lambda
                *
                dkl_loss.pow(2)
            )

            assert_finite_tensor(
                dkl_anchor_loss,
                "DKL anchor loss",
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
                    -
                    plys
                    /
                    BC_PRIOR_PLIES,
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


            assert_finite_tensor(
                combined_log_probs[
                    legal_mask
                ],
                "Combined legal log probabilities",
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


            assert_finite_tensor(
                log_probs[
                    legal_mask
                ],
                "PPO legal log probabilities",
            )


            selected_log_probs = (
                log_probs
                .gather(
                    1,
                    actions.unsqueeze(1),
                )
                .squeeze(1)
            )


            assert_finite_tensor(
                selected_log_probs,
                "Selected PPO log probabilities",
            )


            # =================================================
            # PPO ratio
            # =================================================

            log_ratio = (
                selected_log_probs
                -
                old_log_probs
            )


            assert_finite_tensor(
                log_ratio,
                "PPO log ratio",
            )


            ratio = torch.exp(
                log_ratio
            )


            assert_finite_tensor(
                ratio,
                "PPO ratio",
            )


            approx_kl = (
                (
                    ratio
                    -
                    1.0
                    -
                    log_ratio
                ).mean()
            )


            assert_finite_tensor(
                approx_kl,
                "Approximate KL",
            )


            clipped_mask = (
                (ratio < 1.0 - PPO_CLIP)
                |
                (ratio > 1.0 + PPO_CLIP)
            )


            clipped_fraction = (
                clipped_mask
                .float()
                .mean()
            )


            unclipped = (
                ratio
                *
                advantages
            )


            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - PPO_CLIP,
                    1.0 + PPO_CLIP,
                )
                *
                advantages
            )


            actor_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()


            assert_finite_tensor(
                actor_loss,
                "Actor loss",
            )


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


            assert_finite_tensor(
                entropy,
                "Entropy",
            )


            # =================================================
            # Standard RL critic
            # =================================================

            critic_loss = F.mse_loss(
                values,
                returns,
            )


            assert_finite_tensor(
                critic_loss,
                "Critic loss",
            )


            # =================================================
            # Oracle policy
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


            if len(
                oracle_buffer
            ) > 0:

                oracle_batch_size = min(
                    BATCH_SIZE,
                    len(oracle_buffer),
                )


                oracle_batch = (
                    oracle_buffer.sample(
                        oracle_batch_size
                    )
                )


                oracle_boards = [

                    chess.variant.AtomicBoard(
                        entry["fen"]
                    )

                    for entry in oracle_batch
                ]


                oracle_x = encode_boards(
                    oracle_boards
                ).to(DEVICE)


                oracle_policy_logits, _ = (
                    model(
                        oracle_x
                    )
                )


                assert_finite_tensor(
                    oracle_policy_logits,
                    "Oracle policy logits",
                )


                policy_losses = []

                policy_weights = []

                value_losses = []

                value_weights = []


                batch_confidence_weight = 0.0

                batch_temperature = 0.0


                # =================================================
                # Oracle annotations
                # =================================================

                for i, entry in enumerate(
                    oracle_batch
                ):

                    oracle_move = (
                        entry["oracle_move"]
                    )


                    reward = float(
                        entry["reward"]
                    )


                    confidence = (
                        entry["confidence"]
                    )


                    criticality = (
                        entry["criticality"]
                    )


                    confidence_weight = (
                        ORACLE_CONFIDENCE_WEIGHTS[
                            confidence
                        ]
                    )


                    temperature = (
                        ORACLE_CRITICALITY_TEMPERATURES[
                            criticality
                        ]
                    )


                    try:

                        oracle_action = (
                            ACTION_TO_INDEX[
                                oracle_move
                            ]
                        )

                    except KeyError:

                        print(
                            f"WARNING: skipping unknown "
                            f"oracle move {oracle_move}"
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


                    if len(
                        legal_ids
                    ) == 0:

                        print(
                            "WARNING: skipping Oracle "
                            "position with no legal moves."
                        )

                        continue


                    if oracle_action not in legal_ids:

                        print(
                            "WARNING: skipping Oracle "
                            "position because oracle move "
                            "is not legal."
                        )

                        continue


                    if (
                        confidence_weight
                        <=
                        0.0
                    ):

                        print(
                            "WARNING: skipping Oracle "
                            "annotation with non-positive "
                            "confidence weight."
                        )

                        continue


                    if (
                        temperature
                        <=
                        0.0
                    ):

                        raise RuntimeError(
                            "Oracle temperature "
                            "must be positive."
                        )


                    # =================================================
                    # Oracle legal mask
                    # =================================================

                    oracle_legal_mask = (
                        torch.zeros(
                            oracle_policy_logits.shape[1],
                            dtype=torch.bool,
                            device=DEVICE,
                        )
                    )


                    oracle_legal_mask[
                        list(legal_ids)
                    ] = True


                    # =================================================
                    # Current policy over legal moves
                    # =================================================

                    legal_oracle_policy = (
                        oracle_policy_logits[i]
                        .masked_fill(
                            ~oracle_legal_mask,
                            float("-inf"),
                        )
                    )


                    oracle_log_probs = (
                        F.log_softmax(
                            legal_oracle_policy,
                            dim=0,
                        )
                    )


                    safe_oracle_log_probs = (
                        oracle_log_probs.masked_fill(
                            ~oracle_legal_mask,
                            0.0,
                        )
                    )


                    assert_finite_tensor(
                        safe_oracle_log_probs,
                        "Safe Oracle log probabilities",
                    )


                    # =================================================
                    # Soft Oracle target
                    # =================================================

                    target_probs = (
                        build_oracle_target(
                            legal_ids=legal_ids,
                            oracle_action=oracle_action,
                            temperature=temperature,
                            device=DEVICE,
                        )
                    )


                    assert_finite_tensor(
                        target_probs,
                        "Oracle target probabilities",
                    )


                    # =================================================
                    # Policy cross entropy
                    # =================================================

                    cross_entropy = -(
                        target_probs
                        *
                        safe_oracle_log_probs
                    ).sum()


                    if not torch.isfinite(
                        cross_entropy
                    ):

                        raise RuntimeError(
                            "Oracle cross-entropy "
                            "became non-finite."
                        )


                    policy_losses.append(
                        cross_entropy
                    )


                    policy_weights.append(
                        confidence_weight
                    )


                    # =================================================
                    # Oracle VALUE target
                    # =================================================
                    #
                    # reward is defined from the perspective
                    # of the player to move in the annotated FEN.
                    #
                    # After oracle_move, the side to move changes.
                    #
                    # Therefore:
                    #
                    #   target = reward - gamma * V(next)
                    #
                    # The minus sign is essential.
                    # =================================================

                    oracle_value_target = (
                        compute_oracle_value_target(
                            model=model,
                            oracle_board=board,
                            oracle_move=board.parse_uci(
                                oracle_move
                            ),
                            reward=reward,
                        )
                    )


                    # ------------------------------------------------
                    # Current V(s)
                    # ------------------------------------------------

                    current_value = (
                        model(
                            oracle_x[i:i + 1]
                        )[1]
                        .reshape(-1)[0]
                    )


                    assert_finite_tensor(
                        current_value,
                        "Oracle current value",
                    )


                    assert_finite_tensor(
                        oracle_value_target,
                        "Oracle value target",
                    )


                    value_loss = F.mse_loss(
                        current_value,
                        oracle_value_target,
                    )


                    assert_finite_tensor(
                        value_loss,
                        "Oracle value loss",
                    )


                    value_losses.append(
                        value_loss
                    )


                    value_weights.append(
                        confidence_weight
                    )


                    batch_confidence_weight += (
                        confidence_weight
                    )


                    batch_temperature += (
                        temperature
                    )


                # =================================================
                # Aggregate Oracle policy loss
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


                    weight_sum = (
                        policy_weights_tensor.sum()
                    )


                    if not torch.isfinite(
                        weight_sum
                    ):

                        raise RuntimeError(
                            "Oracle policy weight sum "
                            "became non-finite."
                        )


                    if weight_sum <= 0.0:

                        raise RuntimeError(
                            "Oracle policy weight sum "
                            "is zero."
                        )


                    oracle_policy_loss = (
                        (
                            policy_losses_tensor
                            *
                            policy_weights_tensor
                        ).sum()
                        /
                        (
                            weight_sum
                            +
                            FINITE_EPS
                        )
                    )


                    assert_finite_tensor(
                        oracle_policy_loss,
                        "Oracle policy loss",
                    )


                # =================================================
                # Aggregate Oracle value loss
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


                    value_weight_sum = (
                        value_weights_tensor.sum()
                    )


                    if not torch.isfinite(
                        value_weight_sum
                    ):

                        raise RuntimeError(
                            "Oracle value weight sum "
                            "became non-finite."
                        )


                    if value_weight_sum <= 0.0:

                        raise RuntimeError(
                            "Oracle value weight sum "
                            "is zero."
                        )


                    oracle_value_loss = (
                        (
                            value_losses_tensor
                            *
                            value_weights_tensor
                        ).sum()
                        /
                        (
                            value_weight_sum
                            +
                            FINITE_EPS
                        )
                    )


                    assert_finite_tensor(
                        oracle_value_loss,
                        "Oracle value loss",
                    )


                    oracle_policy_count = (
                        len(value_losses)
                    )


                    total_oracle_confidence_weight += (
                        batch_confidence_weight
                    )


                    total_oracle_temperature += (
                        batch_temperature
                    )


                    total_oracle_annotations += (
                        oracle_policy_count
                    )


                del oracle_x

                del oracle_policy_logits


            # =================================================
            # Total loss
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

                +

                dkl_anchor_loss
            )


            # =================================================
            # TOTAL LOSS SAFETY
            # =================================================

            if not torch.isfinite(
                loss
            ):

                print()
                print(
                    "======================================"
                )
                print(
                    "FATAL: NON-FINITE TOTAL LOSS"
                )
                print(
                    "======================================"
                )

                print(
                    f"Actor        : "
                    f"{actor_loss.item()}"
                )

                print(
                    f"Critic       : "
                    f"{critic_loss.item()}"
                )

                print(
                    f"Entropy      : "
                    f"{entropy.item()}"
                )

                print(
                    f"OraclePolicy : "
                    f"{oracle_policy_loss.item()}"
                )

                print(
                    f"OracleValue  : "
                    f"{oracle_value_loss.item()}"
                )

                print(
                    f"DKL(RL||BC)  : "
                    f"{dkl_loss.item()}"
                )

                print(
                    f"DKL lambda    : "
                    f"{float(dkl_lambda)}"
                )

                print(
                    f"DKL anchor    : "
                    f"{dkl_anchor_loss.item()}"
                )

                print(
                    f"KL           : "
                    f"{approx_kl.item()}"
                )

                raise RuntimeError(
                    "Total loss became non-finite."
                )


            # =================================================
            # Backward
            # =================================================

            optimizer.zero_grad(
                set_to_none=True
            )


            loss.backward()


            assert_gradients_finite(
                model
            )


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )


            assert_gradients_finite(
                model
            )


            optimizer.step()


            assert_model_finite(
                model,
                "model after optimizer step",
            )


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


            total_dkl += (
                dkl_loss.item()
            )


            total_dkl_loss += (
                dkl_anchor_loss.item()
            )


            total_dkl_lambda += (
                float(dkl_lambda)
            )


            progress.update(1)


    progress.close()


    # ========================================================
    # Averages
    # ========================================================

    avg_loss = (
        total_loss
        /
        total_updates
    )


    avg_actor = (
        total_actor
        /
        total_updates
    )


    avg_critic = (
        total_critic
        /
        total_updates
    )


    avg_oracle_policy = (
        total_oracle_policy
        /
        total_updates
    )


    avg_oracle_value = (
        total_oracle_value
        /
        total_updates
    )


    avg_kl = (
        total_kl
        /
        total_updates
    )


    avg_entropy = (
        total_entropy
        /
        total_updates
    )


    avg_dkl = (
        total_dkl
        /
        total_updates
    )


    avg_dkl_loss = (
        total_dkl_loss
        /
        total_updates
    )


    avg_dkl_lambda = (
        total_dkl_lambda
        /
        total_updates
    )


    # ========================================================
    # Final safety
    # ========================================================

    for (
        name,
        value,
    ) in {

        "avg_loss":
            avg_loss,

        "avg_actor":
            avg_actor,

        "avg_critic":
            avg_critic,

        "avg_oracle_policy":
            avg_oracle_policy,

        "avg_oracle_value":
            avg_oracle_value,

        "avg_kl":
            avg_kl,

        "avg_entropy":
            avg_entropy,

        "avg_dkl":
            avg_dkl,

        "avg_dkl_loss":
            avg_dkl_loss,

        "avg_dkl_lambda":
            avg_dkl_lambda,

    }.items():

        if not torch.isfinite(
            torch.tensor(
                value
            )
        ):

            raise RuntimeError(
                f"Non-finite training average: "
                f"{name}={value}"
            )


    # ========================================================
    # Oracle diagnostics
    # ========================================================

    if total_oracle_annotations > 0:

        avg_oracle_temperature = (
            total_oracle_temperature
            /
            total_oracle_annotations
        )


        avg_oracle_confidence = (
            total_oracle_confidence_weight
            /
            total_oracle_annotations
        )

    else:

        avg_oracle_temperature = 0.0

        avg_oracle_confidence = 0.0


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
        f"Epoch: "
        f"{epoch}"
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
        f"Oracle value coefficient: "
        f"{ORACLE_VALUE_COEF:.4f}"
    )


    print(
        "Oracle confidence weights: "
        "low=0.50 / medium=0.75 / high=0.99"
    )


    print(
        "Oracle criticality temperatures:"
    )


    for (
        name,
        temperature,
    ) in ORACLE_CRITICALITY_TEMPERATURES.items():

        print(
            f"  {name}: T={temperature:.2f}"
        )


    print(
        f"Mean Oracle confidence weight: "
        f"{avg_oracle_confidence:.4f}"
    )


    print(
        f"Mean Oracle target temperature: "
        f"{avg_oracle_temperature:.4f}"
    )


    print(
        f"Oracle annotations used: "
        f"{total_oracle_annotations}"
    )


    print(
        f"Oracle policy loss: "
        f"{avg_oracle_policy:.6f}"
    )


    print(
        f"Oracle value loss: "
        f"{avg_oracle_value:.6f}"
    )


    print(
        f"Policy KL: "
        f"{avg_kl:.6e}"
    )


    print(
        f"Entropy: "
        f"{avg_entropy:.6f}"
    )


    print(
        f"DKL(RL || BC): "
        f"{avg_dkl:.6e}"
    )


    print(
        f"DKL lambda: "
        f"{avg_dkl_lambda:.6e}"
    )


    print(
        f"DKL anchor loss: "
        f"{avg_dkl_loss:.6e}"
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
        avg_dkl,
        avg_dkl_loss,
    )


# ============================================================
# Replay buffer
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
# Safe checkpoint
# ============================================================

def save_model_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
):

    assert_model_finite(
        model,
        "model before checkpoint",
    )


    if not torch.isfinite(
        torch.tensor(
            loss
        )
    ):

        raise RuntimeError(
            f"Refusing to save checkpoint: "
            f"loss={loss} is non-finite."
        )


    path = Path(
        path
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
        "Checkpoint saved:",
        path,
        flush=True,
    )


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
    # RL model
    # ========================================================

    (
        model,
        optimizer,
        league,
    ) = load_model()


    # ========================================================
    # BC model
    # ========================================================

    bc_model = load_bc_agent(5)


    assert_model_finite(
        bc_model,
        "BC5 model",
    )


    bc_model_selfplay = copy.deepcopy(
        bc_model
    ).to("cpu")


    bc_model_selfplay.eval()


    # ========================================================
    # Oracle replay buffer
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
            entry["reward"],
        )


    print(
        f"Oracle replay buffer: "
        f"{len(oracle_buffer)} annotations",
        flush=True,
    )


    # ========================================================
    # DEBUG BUFFER
    # ========================================================

    if len(
        oracle_buffer
    ) > 0:

        print(
            "DEBUG BUFFER:",
            oracle_buffer.buffer[
                :min(
                    2,
                    len(oracle_buffer.buffer),
                )
            ],
            flush=True,
        )


    # ========================================================
    # RL replay buffer
    # ========================================================

    buffer = ReplayBuffer(
        capacity=300000
    )


    # ========================================================
    # Uncertainty
    # ========================================================

    stats = UncertaintyStats()


    best_training_loss = None


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
    # Future league slots
    # ========================================================

    future_end = (
        START_EPOCH
        +
        AL_EPOCHS
        -
        1
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


        placeholder = copy.deepcopy(
            model
        ).to("cpu")


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
        # Epoch loop
        # ====================================================

        for epoch in range(
            START_EPOCH,
            START_EPOCH + AL_EPOCHS,
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
                # Result
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
                # Replay
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
            # Diagnostics self-play
            # =================================================

            score_rate = (

                wins
                +
                0.5 * draws

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
            # Oracle is HISTORICAL.
            #
            # There is deliberately NO:
            #
            #     oracle reward hit
            #
            #     FEN matching against self-play
            #
            #     injection into current trajectory
            #
            # Oracle annotations are trained directly from
            # OracleReplayBuffer.
            # =================================================

            # =================================================
            # RL / Oracle policy + value
            # =================================================

            (
                rl_loss,
                actor_loss,
                critic_loss,
                oracle_policy_loss,
                oracle_value_loss,
                approx_kl,
                dkl_loss,
                dkl_anchor_loss,
            ) = train_oracle_epoch(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                bc_model=bc_model,
                oracle_buffer=oracle_buffer,
                epoch=epoch,
            )


            print(
                f"RL Loss={rl_loss:.4f} "
                f"| Actor={actor_loss:.4f} "
                f"| Critic={critic_loss:.4f} "
                f"| OraclePolicy={oracle_policy_loss:.4f} "
                f"| OracleValue={oracle_value_loss:.4f} "
                f"| KL={approx_kl:.6f} "
                f"| DKL(RL||BC)={dkl_loss:.6e} "
                f"| DKL Anchor={dkl_anchor_loss:.6e}",
                flush=True,
            )


            # =================================================
            # Replay buffer save
            # =================================================

            if epoch % 5 == 0:

                save_replay_buffer(
                    buffer,
                    epoch,
                )


            # =================================================
            # On-policy clear
            # =================================================

            buffer.clear()


            print(
                "RL replay buffer cleared after PPO update.",
                flush=True,
            )


            # =================================================
            # Uncertainty
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
            # RL checkpoint
            # =================================================

            if (
                epoch
                %
                CHECKPOINT_EVERY
                ==
                0
            ):

                save_path = (
                    PROJECT_ROOT
                    / "checkpoints"
                    / "oracle_epoch"
                    / f"oracle_epoch_{epoch}.pt"
                )


                save_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=rl_loss,
                    path=save_path,
                )


            # =================================================
            # Best RL
            # =================================================

            if (
                best_training_loss is None
                or rl_loss < best_training_loss
            ):

                best_training_loss = (
                    rl_loss
                )


                save_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=rl_loss,
                    path=(
                        PROJECT_ROOT
                        /
                        "checkpoints"
                        /
                        "oracle_best.pt"
                    ),
                )


                print(
                    "New best Oracle RL checkpoint saved.",
                    flush=True,
                )


            # =================================================
            # Snapshot league
            # =================================================

            snapshot = copy.deepcopy(
                model
            ).to(DEVICE)


            assert_model_finite(
                snapshot,
                f"snapshot epoch {epoch}",
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
                f"Oracle policy loss: "
                f"{oracle_policy_loss:.6f}",
                flush=True,
            )


            print(
                f"Oracle value loss: "
                f"{oracle_value_loss:.6f}",
                flush=True,
            )


            print(
                f"RL critic loss: "
                f"{critic_loss:.6f}",
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