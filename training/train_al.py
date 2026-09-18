from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import chess.variant
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.train_rl as rl

from src.actions_space import ACTIONS, ACTION_TO_INDEX
from src.encoding import encode_boards
from src.models.actor_critic import ActorCritic
from src.models.resnet import ChessResNet
from src.rl.oracle_replay_buffer import OracleReplayBuffer


# ============================================================
# Default paths
# ============================================================

DEFAULT_BC_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
)

DEFAULT_START_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_BASELINE_LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league"
)

DEFAULT_ORACLE_QUEUE = (
    PROJECT_ROOT
    / "data"
    / "queue"
    / "oracle_queue_1-10_random.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "al_runs"
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_SEED = 42

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DEFAULT_RUN_NAME = "random_001"
DEFAULT_END_EPOCH = 20

DEFAULT_NUM_WORKERS = 12
DEFAULT_SELFPLAY_BATCH_SIZE = 256

DEFAULT_ORACLE_CAPACITY = 50_000
DEFAULT_ORACLE_BATCH_SIZE = 4096
DEFAULT_ORACLE_INJECTION_FREQUENCY = 1

DEFAULT_ORACLE_POLICY_COEF = 0.05
DEFAULT_ORACLE_VALUE_COEF = 0.5


# ============================================================
# Oracle supervision weights
# ============================================================

CONFIDENCE_WEIGHTS = {
    "low": 0.50,
    "medium": 0.75,
    "high": 0.99,
}

SITUATION_WEIGHTS = {
    "critical": 1.00,
    "non_critical": 0.50,
    "outcome_independent": 0.25,
}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Continue an ALBERTA RL checkpoint with sparse "
            "Oracle supervision."
        )
    )

    # ========================================================
    # Experiment
    # ========================================================

    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
    )

    parser.add_argument(
        "--acquisition",
        type=str,
        default="unspecified",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    # ========================================================
    # Model / BC
    # ========================================================

    parser.add_argument(
        "--bc-checkpoint-dir",
        type=Path,
        default=DEFAULT_BC_CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=rl.DEFAULT_CHANNELS,
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=rl.DEFAULT_BLOCKS,
    )

    # ========================================================
    # Starting state
    # ========================================================

    parser.add_argument(
        "--start-checkpoint",
        type=Path,
        default=DEFAULT_START_CHECKPOINT,
    )

    parser.add_argument(
        "--league-dir",
        type=Path,
        default=DEFAULT_BASELINE_LEAGUE_DIR,
    )

    parser.add_argument(
        "--end-epoch",
        type=int,
        default=DEFAULT_END_EPOCH,
    )

    # ========================================================
    # Oracle
    # ========================================================

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_ORACLE_QUEUE,
    )

    parser.add_argument(
        "--oracle-capacity",
        type=int,
        default=DEFAULT_ORACLE_CAPACITY,
    )

    parser.add_argument(
        "--oracle-batch-size",
        type=int,
        default=DEFAULT_ORACLE_BATCH_SIZE,
    )

    parser.add_argument(
        "--oracle-frequency",
        type=int,
        default=DEFAULT_ORACLE_INJECTION_FREQUENCY,
    )

    parser.add_argument(
        "--oracle-policy-coef",
        type=float,
        default=DEFAULT_ORACLE_POLICY_COEF,
    )

    parser.add_argument(
        "--oracle-value-coef",
        type=float,
        default=DEFAULT_ORACLE_VALUE_COEF,
    )

    # ========================================================
    # Runtime
    # ========================================================

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    parser.add_argument(
        "--selfplay-batch-size",
        type=int,
        default=DEFAULT_SELFPLAY_BATCH_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--resume-stats",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# train_rl compatibility
# ============================================================

def complete_rl_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """
    train_al intentionally delegates the canonical PPO/self-play
    implementation to training.train_rl.

    Populate every RL configuration field consumed by those
    functions using exactly the historical train_rl defaults.
    """

    args.num_workers = args.workers

    args.games_per_epoch = (
        rl.DEFAULT_GAMES_PER_EPOCH
    )

    args.temperature_selfplay = (
        rl.DEFAULT_TEMPERATURE_SELFPLAY
    )

    args.opening_prior_strength = (
        rl.DEFAULT_OPENING_PRIOR_STRENGTH
    )

    args.opening_prior_plies = (
        rl.DEFAULT_OPENING_PRIOR_PLIES
    )

    args.buffer_capacity = (
        rl.DEFAULT_BUFFER_CAPACITY
    )

    args.batch_size = (
        rl.DEFAULT_BATCH_SIZE
    )

    args.sgd_epochs = (
        rl.DEFAULT_SGD_EPOCHS
    )

    args.value_coef = (
        rl.DEFAULT_VALUE_COEF
    )

    args.gamma = (
        rl.DEFAULT_GAMMA
    )

    args.gae_lambda = (
        rl.DEFAULT_GAE_LAMBDA
    )

    args.ppo_clip = (
        rl.DEFAULT_PPO_CLIP
    )

    args.entropy_coef = (
        rl.DEFAULT_ENTROPY_COEF
    )

    args.grad_clip = (
        rl.DEFAULT_GRAD_CLIP
    )

    args.uncertainty_batch_size = (
        rl.DEFAULT_UNCERTAINTY_BATCH_SIZE
    )

    args.rl_total_epochs = (
        rl.DEFAULT_RL_TOTAL_EPOCHS
    )

    args.dkl_fit_epoch_stride = (
        rl.DEFAULT_DKL_FIT_EPOCH_STRIDE
    )

    args.dkl_inf = (
        rl.DEFAULT_DKL_INF
    )

    args.dkl_decay_per_fit_unit = (
        rl.DEFAULT_DKL_DECAY_PER_FIT_UNIT
    )

    args.dkl_alpha = (
        rl.DEFAULT_DKL_ALPHA
    )

    args.lambda_dkl = (
        rl.DEFAULT_LAMBDA_DKL
    )

    return args


# ============================================================
# Validation
# ============================================================

def validate_args(
    args: argparse.Namespace,
) -> None:

    if args.end_epoch < 1:
        raise ValueError(
            "--end-epoch must be >= 1."
        )

    if args.workers <= 0:
        raise ValueError(
            "--workers must be >= 1."
        )

    if args.selfplay_batch_size <= 0:
        raise ValueError(
            "--selfplay-batch-size must be >= 1."
        )

    if args.uncertainty_batch_size <= 0:
        raise ValueError(
            "uncertainty_batch_size must be >= 1."
        )

    if args.oracle_batch_size <= 0:
        raise ValueError(
            "--oracle-batch-size must be >= 1."
        )

    if args.oracle_frequency <= 0:
        raise ValueError(
            "--oracle-frequency must be >= 1."
        )

    if args.channels <= 0:
        raise ValueError(
            "--channels must be >= 1."
        )

    if args.blocks <= 0:
        raise ValueError(
            "--blocks must be >= 1."
        )

    if args.games_per_epoch != 2500:

        raise ValueError(
            "This AL experiment expects exactly "
            "2500 self-play games per epoch."
        )


# ============================================================
# Oracle queue
# ============================================================

def load_oracle_queue(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Oracle queue not found:\n{path}"
        )

    annotations = []

    skipped_incomplete = 0
    skipped_discarded = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Invalid JSON at line "
                    f"{line_number} in:\n{path}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):

                raise ValueError(
                    f"Expected object at line {line_number}."
                )

            if record.get(
                "status"
            ) == "discarded":

                skipped_discarded += 1
                continue

            oracle_move = record.get(
                "oracle_move"
            )

            confidence = record.get(
                "oracle_confidence",
                record.get(
                    "confidence"
                ),
            )

            situation = record.get(
                "oracle_situation",
                record.get(
                    "criticality"
                ),
            )

            reward_raw = record.get(
                "reward"
            )

            if (
                oracle_move is None
                or confidence is None
                or situation is None
                or reward_raw is None
            ):

                skipped_incomplete += 1
                continue

            fen = record.get(
                "fen"
            )

            if not isinstance(
                fen,
                str,
            ) or not fen:

                raise ValueError(
                    f"Missing FEN at line {line_number}."
                )

            try:

                reward = float(
                    reward_raw
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"Invalid reward at line "
                    f"{line_number}: {reward_raw}"
                ) from exc

            if reward not in {
                -1.0,
                0.0,
                1.0,
            }:

                raise ValueError(
                    f"Invalid reward at line "
                    f"{line_number}: {reward}"
                )

            if confidence not in CONFIDENCE_WEIGHTS:

                raise ValueError(
                    f"Invalid Oracle confidence at line "
                    f"{line_number}: {confidence}"
                )

            if situation not in SITUATION_WEIGHTS:

                raise ValueError(
                    f"Invalid Oracle situation at line "
                    f"{line_number}: {situation}"
                )

            if oracle_move not in ACTION_TO_INDEX:

                raise ValueError(
                    f"Unknown Oracle move at line "
                    f"{line_number}: {oracle_move}"
                )

            try:

                board = chess.variant.AtomicBoard(
                    fen
                )

            except Exception as exc:

                raise ValueError(
                    f"Invalid Atomic FEN at line "
                    f"{line_number}:\n{fen}"
                ) from exc

            legal_moves = {
                move.uci()
                for move in board.legal_moves
            }

            if oracle_move not in legal_moves:

                raise ValueError(
                    "Oracle move is illegal at line "
                    f"{line_number}:\n"
                    f"FEN:  {fen}\n"
                    f"Move: {oracle_move}"
                )

            annotations.append(
                {
                    "query_id":
                        record.get(
                            "query_id"
                        ),

                    "fen":
                        fen,

                    "oracle_move":
                        oracle_move,

                    "confidence":
                        confidence,

                    "situation":
                        situation,

                    "reward":
                        reward,
                }
            )

    if not annotations:

        raise RuntimeError(
            "No complete Oracle annotations found."
        )

    reward_counts = {
        -1.0: 0,
        0.0: 0,
        1.0: 0,
    }

    confidence_counts = {
        key: 0
        for key in CONFIDENCE_WEIGHTS
    }

    situation_counts = {
        key: 0
        for key in SITUATION_WEIGHTS
    }

    for record in annotations:

        reward_counts[
            record[
                "reward"
            ]
        ] += 1

        confidence_counts[
            record[
                "confidence"
            ]
        ] += 1

        situation_counts[
            record[
                "situation"
            ]
        ] += 1

    print()
    print("=" * 70)
    print("ORACLE QUEUE")
    print("=" * 70)

    print(
        f"Path:       {path}"
    )

    print(
        f"Usable:     {len(annotations):,}"
    )

    print(
        f"Incomplete: {skipped_incomplete:,}"
    )

    print(
        f"Discarded:  {skipped_discarded:,}"
    )

    print(
        f"Loss/Draw/Win: "
        f"{reward_counts[-1.0]}/"
        f"{reward_counts[0.0]}/"
        f"{reward_counts[1.0]}"
    )

    print(
        f"Confidence: {confidence_counts}"
    )

    print(
        f"Situation:  {situation_counts}"
    )

    return annotations


# ============================================================
# Oracle replay buffer
# ============================================================

def build_oracle_buffer(
    annotations: list[dict],
    capacity: int,
) -> OracleReplayBuffer:

    if capacity <= 0:

        raise ValueError(
            "Oracle capacity must be positive."
        )

    if capacity < len(
        annotations
    ):

        raise ValueError(
            "Oracle capacity is smaller than the number "
            "of loaded annotations."
        )

    buffer = OracleReplayBuffer(
        capacity=capacity
    )

    for record in annotations:

        buffer.add(
            record[
                "fen"
            ],
            record[
                "oracle_move"
            ],
            record[
                "confidence"
            ],
            record[
                "situation"
            ],
            record[
                "reward"
            ],
        )

    print(
        f"Oracle buffer size: {len(buffer):,}"
    )

    return buffer


# ============================================================
# Oracle loss
# ============================================================

def compute_oracle_loss(
    model: ActorCritic,
    oracle_batch: list[dict],
    *,
    device: torch.device,
    policy_coef: float,
    value_coef: float,
) -> dict[str, torch.Tensor]:

    if not oracle_batch:

        zero = sum(
            parameter.sum()
            * 0.0
            for parameter
            in model.parameters()
        )

        return {
            "loss": zero,
            "policy_loss": zero,
            "value_loss": zero,
        }

    boards = []
    oracle_actions = []
    weights = []
    oracle_rewards = []

    for record in oracle_batch:

        board = chess.variant.AtomicBoard(
            record[
                "fen"
            ]
        )

        boards.append(
            board
        )

        oracle_move = record[
            "oracle_move"
        ]

        if oracle_move not in ACTION_TO_INDEX:

            raise ValueError(
                f"Unknown Oracle action: {oracle_move}"
            )

        legal_moves = {
            move.uci()
            for move in board.legal_moves
        }

        if oracle_move not in legal_moves:

            raise ValueError(
                "Oracle move became illegal during loss "
                "construction:\n"
                f"FEN:  {record['fen']}\n"
                f"Move: {oracle_move}"
            )

        oracle_actions.append(
            ACTION_TO_INDEX[
                oracle_move
            ]
        )

        confidence = record.get(
            "confidence",
            "medium",
        )

        situation = record.get(
            "situation",
            record.get(
                "criticality",
                "non_critical",
            ),
        )

        confidence_weight = (
            CONFIDENCE_WEIGHTS[
                confidence
            ]
        )

        situation_weight = (
            SITUATION_WEIGHTS[
                situation
            ]
        )

        weights.append(
            confidence_weight
            * situation_weight
        )

        oracle_rewards.append(
            float(
                record[
                    "reward"
                ]
            )
        )

    encoded_boards = encode_boards(
        boards
    ).to(
        device
    )

    oracle_actions_tensor = torch.tensor(
        oracle_actions,
        dtype=torch.long,
        device=device,
    )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    rewards_tensor = torch.tensor(
        oracle_rewards,
        dtype=torch.float32,
        device=device,
    )

    logits, values = model(
        encoded_boards
    )

    masked_logits = logits.clone()

    for batch_index, board in enumerate(
        boards
    ):

        legal_indices = []

        for move in board.legal_moves:

            uci = move.uci()

            if uci not in ACTION_TO_INDEX:

                raise ValueError(
                    "Legal Atomic move missing from action "
                    f"space: {uci}"
                )

            legal_indices.append(
                ACTION_TO_INDEX[
                    uci
                ]
            )

        if not legal_indices:

            raise RuntimeError(
                "Oracle loss received a position with no "
                "legal actions."
            )

        legal_mask = torch.zeros(
            logits.shape[
                1
            ],
            dtype=torch.bool,
            device=device,
        )

        legal_mask[
            legal_indices
        ] = True

        masked_logits[
            batch_index
        ] = masked_logits[
            batch_index
        ].masked_fill(
            ~legal_mask,
            float(
                "-inf"
            ),
        )

    log_probs = F.log_softmax(
        masked_logits,
        dim=1,
    )

    batch_indices = torch.arange(
        len(
            oracle_batch
        ),
        device=device,
    )

    oracle_log_probs = log_probs[
        batch_indices,
        oracle_actions_tensor,
    ]

    policy_losses = (
        -oracle_log_probs
    )

    weight_sum = weights_tensor.sum().clamp_min(
        1e-8
    )

    policy_loss = (
        (
            weights_tensor
            * policy_losses
        ).sum()
        / weight_sum
    )

    predicted_values = values[
        :,
        0,
    ]

    value_losses = F.mse_loss(
        predicted_values,
        rewards_tensor,
        reduction="none",
    )

    value_loss = (
        (
            weights_tensor
            * value_losses
        ).sum()
        / weight_sum
    )

    oracle_loss = (
        policy_coef
        * policy_loss
        + value_coef
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
# Oracle callback
# ============================================================

def make_oracle_loss_fn(
    oracle_buffer: OracleReplayBuffer,
    *,
    device: torch.device,
    batch_size: int,
    injection_frequency: int,
    policy_coef: float,
    value_coef: float,
):

    if injection_frequency <= 0:

        raise ValueError(
            "Oracle injection frequency must be >= 1."
        )

    if batch_size <= 0:

        raise ValueError(
            "Oracle batch size must be >= 1."
        )

    state = {
        "calls": 0,
        "injections": 0,
    }

    def connected_zero(
        model,
    ) -> torch.Tensor:

        return sum(
            parameter.sum()
            * 0.0
            for parameter
            in model.parameters()
        )

    def oracle_loss_fn(
        model,
    ):

        state[
            "calls"
        ] += 1

        should_inject = (
            (
                state[
                    "calls"
                ]
                - 1
            )
            % injection_frequency
            == 0
        )

        if not should_inject:

            zero = connected_zero(
                model
            )

            return {
                "loss":
                    zero,

                "policy_loss":
                    zero,

                "value_loss":
                    zero,
            }

        state[
            "injections"
        ] += 1

        effective_batch_size = min(
            batch_size,
            len(
                oracle_buffer
            ),
        )

        oracle_batch = oracle_buffer.sample(
            effective_batch_size
        )

        return compute_oracle_loss(
            model,
            oracle_batch,
            device=device,
            policy_coef=policy_coef,
            value_coef=value_coef,
        )

    def reset_epoch_stats() -> None:

        state[
            "calls"
        ] = 0

        state[
            "injections"
        ] = 0

    def get_epoch_stats() -> dict[str, int]:

        return {
            "calls":
                state[
                    "calls"
                ],

            "injections":
                state[
                    "injections"
                ],
        }

    oracle_loss_fn.reset_epoch_stats = (
        reset_epoch_stats
    )

    oracle_loss_fn.get_epoch_stats = (
        get_epoch_stats
    )

    return oracle_loss_fn


# ============================================================
# Model
# ============================================================

def build_actor_critic(
    args: argparse.Namespace,
    device: torch.device,
) -> ActorCritic:

    backbone = ChessResNet(
        num_actions=len(
            ACTIONS
        ),
        channels=args.channels,
        blocks=args.blocks,
    )

    return ActorCritic(
        backbone
    ).to(
        device
    )


# ============================================================
# Starting checkpoint
# ============================================================

def load_starting_state(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Starting checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    if "optimizer_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain optimizer_state_dict."
        )

    checkpoint_actions = checkpoint.get(
        "actions"
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(
            ACTIONS
        )
    ):

        raise ValueError(
            "Action-space mismatch in starting checkpoint: "
            f"{checkpoint_actions} != {len(ACTIONS)}"
        )

    model = build_actor_critic(
        args,
        device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer = Adam(
        model.parameters(),
        lr=rl.DEFAULT_LR,
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    start_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )

    if start_epoch < 0:

        raise RuntimeError(
            "Starting checkpoint does not contain "
            "a valid epoch."
        )

    print()
    print("=" * 70)
    print("STARTING CHECKPOINT")
    print("=" * 70)

    print(
        f"Path:  {checkpoint_path}"
    )

    print(
        f"Epoch: {start_epoch}"
    )

    print(
        "Optimizer LR: "
        f"{[group['lr'] for group in optimizer.param_groups]}"
    )

    return (
        model,
        optimizer,
        start_epoch,
    )


# ============================================================
# League reconstruction
# ============================================================

def load_league(
    *,
    start_epoch: int,
    league_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
):
    """
    Reconstruct exact historical league up to start_epoch.

    BC6 is protected and excluded from uncertainty.
    RL snapshots participate in uncertainty.
    """

    protected_agents = {
        "bc_epoch_6",
    }

    league = rl.League(
        max_agents=rl.DEFAULT_LEAGUE_MAX_AGENTS,
        protected_agents=protected_agents,
    )

    # ========================================================
    # BC6
    # ========================================================

    bc6 = rl.load_bc_actor_critic(
        epoch=6,
        args=args,
        device=device,
        evaluation=True,
    )

    league.add_agent(
        "bc_epoch_6",
        bc6,
        use_for_uncertainty=False,
    )

    # ========================================================
    # Historical RL snapshots
    # ========================================================

    loaded_snapshots = 0

    for epoch in range(
        1,
        start_epoch + 1,
    ):

        path = (
            league_dir
            / f"league_epoch_{epoch:03d}.pt"
        )

        if not path.exists():
            continue

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        checkpoint_actions = checkpoint.get(
            "actions"
        )

        if (
            checkpoint_actions is not None
            and checkpoint_actions
            != len(
                ACTIONS
            )
        ):

            raise ValueError(
                f"Action-space mismatch in {path}: "
                f"{checkpoint_actions} != {len(ACTIONS)}"
            )

        snapshot = build_actor_critic(
            args,
            device,
        )

        snapshot.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        snapshot.eval()

        league.add_agent(
            f"league_epoch_{epoch:03d}",
            snapshot,
            use_for_uncertainty=True,
        )

        loaded_snapshots += 1

    print()
    print("=" * 70)
    print("LEAGUE")
    print("=" * 70)

    print(
        f"Source: {league_dir}"
    )

    print(
        f"BC checkpoint directory: "
        f"{args.bc_checkpoint_dir}"
    )

    print(
        f"Historical RL snapshots loaded: "
        f"{loaded_snapshots}"
    )

    print(
        f"Opponent agents: {len(league)}"
    )

    print(
        "Uncertainty contributors: "
        f"{league.uncertainty_names()}"
    )

    # --------------------------------------------------------
    # Fail fast. An RL10 AL branch must have RL1...RL10.
    # --------------------------------------------------------

    if start_epoch == 10 and loaded_snapshots != 10:

        raise RuntimeError(
            "Expected 10 historical RL league snapshots "
            f"for an RL10 branch, but loaded {loaded_snapshots}. "
            "Check --league-dir before spending compute."
        )

    return (
        league,
        bc6,
    )


# ============================================================
# Uncertainty
# ============================================================

def load_uncertainty_stats(
    path: Path,
    resume: bool,
):

    stats = rl.UncertaintyStats()

    if (
        resume
        and path.exists()
    ):

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            list,
        ):

            raise ValueError(
                "Existing uncertainty statistics "
                "must be a JSON list."
            )

        stats.data = data

        print(
            "Resumed uncertainty statistics: "
            f"{len(stats.data):,} records"
        )

    return stats


# ============================================================
# Shared-model helper
# ============================================================

def copy_model_state_to_shared(
    source_model,
    shared_model,
) -> None:

    shared_state = (
        shared_model.state_dict()
    )

    for key, value in (
        source_model
        .state_dict()
        .items()
    ):

        shared_state[
            key
        ].copy_(
            value
            .detach()
            .cpu()
        )


# ============================================================
# Saving
# ============================================================

def save_al_checkpoint(
    *,
    path: Path,
    epoch: int,
    model,
    optimizer,
    loss: float,
    seed: int,
    acquisition: str,
    oracle_annotations: int,
    oracle_frequency: int,
    oracle_policy_coef: float,
    oracle_value_coef: float,
    source_checkpoint: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch":
                epoch,

            "seed":
                seed,

            "actions":
                len(
                    ACTIONS
                ),

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                loss,

            "training_mode":
                "RL+Oracle",

            "source_checkpoint":
                str(
                    source_checkpoint
                ),

            "oracle_acquisition":
                acquisition,

            "oracle_annotations":
                oracle_annotations,

            "oracle_injection_frequency":
                oracle_frequency,

            "oracle_policy_coef":
                oracle_policy_coef,

            "oracle_value_coef":
                oracle_value_coef,
        },
        path,
    )


def save_league_snapshot(
    *,
    path: Path,
    epoch: int,
    model,
    seed: int,
    acquisition: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch":
                epoch,

            "seed":
                seed,

            "actions":
                len(
                    ACTIONS
                ),

            "model_state_dict":
                model.state_dict(),

            "training_mode":
                "RL+Oracle",

            "oracle_acquisition":
                acquisition,
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    args = complete_rl_args(
        args
    )

    validate_args(
        args
    )

    # ========================================================
    # Determinism
    # ========================================================

    seed_everything(
        args.seed
    )

    device = rl.resolve_device(
        args.device
    )

    # ========================================================
    # Output
    # ========================================================

    run_dir = (
        args.output_dir
        / args.run_name
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    league_output_dir = (
        run_dir
        / "league"
    )

    uncertainty_stats_path = (
        run_dir
        / "uncertainty_stats.json"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 70)
    print("ALBERTA - RL + ORACLE TRAINING")
    print("=" * 70)

    print(
        f"Run:              {args.run_name}"
    )

    print(
        f"Acquisition:      {args.acquisition}"
    )

    print(
        f"Seed:             {args.seed}"
    )

    print(
        f"Device:           {device}"
    )

    print(
        f"Channels:         {args.channels}"
    )

    print(
        f"Blocks:           {args.blocks}"
    )

    print(
        f"Games / epoch:    {args.games_per_epoch}"
    )

    print(
        f"Self-play T:      {args.temperature_selfplay}"
    )

    print(
        f"Workers:          {args.num_workers}"
    )

    print(
        f"Self-play batch:  {args.selfplay_batch_size}"
    )

    print(
        f"Uncertainty batch:{args.uncertainty_batch_size}"
    )

    print(
        f"BC directory:     {args.bc_checkpoint_dir}"
    )

    print(
        f"Start checkpoint: {args.start_checkpoint}"
    )

    print(
        f"League directory: {args.league_dir}"
    )

    print(
        f"Oracle queue:     {args.queue}"
    )

    print(
        f"Output:           {run_dir}"
    )

    # ========================================================
    # Starting state
    # ========================================================

    (
        model,
        optimizer,
        start_epoch,
    ) = load_starting_state(
        args.start_checkpoint,
        args,
        device,
    )

    first_al_epoch = (
        start_epoch
        + 1
    )

    if args.end_epoch < first_al_epoch:

        raise ValueError(
            f"--end-epoch={args.end_epoch} "
            f"is earlier than epoch {first_al_epoch}."
        )

    # ========================================================
    # League
    # ========================================================

    (
        league,
        bc_model,
    ) = load_league(
        start_epoch=start_epoch,
        league_dir=args.league_dir,
        args=args,
        device=device,
    )

    # ========================================================
    # Oracle
    # ========================================================

    annotations = load_oracle_queue(
        args.queue
    )

    oracle_buffer = build_oracle_buffer(
        annotations,
        capacity=args.oracle_capacity,
    )

    oracle_loss_fn = make_oracle_loss_fn(
        oracle_buffer,
        device=device,
        batch_size=args.oracle_batch_size,
        injection_frequency=args.oracle_frequency,
        policy_coef=args.oracle_policy_coef,
        value_coef=args.oracle_value_coef,
    )

    print()
    print("=" * 70)
    print("ORACLE OBJECTIVE")
    print("=" * 70)

    print(
        f"Annotations:        {len(oracle_buffer):,}"
    )

    print(
        "Batch size:         "
        f"{min(args.oracle_batch_size, len(oracle_buffer))}"
    )

    print(
        f"Injection:          1 / {args.oracle_frequency}"
    )

    print(
        f"Policy coefficient: {args.oracle_policy_coef}"
    )

    print(
        f"Value coefficient:  {args.oracle_value_coef}"
    )

    # ========================================================
    # RL buffer
    # ========================================================

    buffer = rl.ReplayBuffer(
        capacity=args.buffer_capacity
    )

    # ========================================================
    # Uncertainty
    # ========================================================

    stats = load_uncertainty_stats(
        uncertainty_stats_path,
        resume=args.resume_stats,
    )

    # ========================================================
    # Shared models
    # ========================================================

    bc_model_selfplay = copy.deepcopy(
        bc_model
    ).to(
        "cpu"
    )

    bc_model_selfplay.eval()
    bc_model_selfplay.share_memory()

    shared_current_model = (
        rl.prepare_shared_model(
            model
        )
    )

    shared_league_models = {}

    for (
        name,
        league_model,
    ) in league.agents.items():

        shared_league_models[
            name
        ] = rl.prepare_shared_model(
            league_model
        )

    # ========================================================
    # Preallocate AL league slots
    # ========================================================

    for epoch in range(
        first_al_epoch,
        args.end_epoch + 1,
    ):

        name = (
            f"league_epoch_{epoch:03d}"
        )

        if name in shared_league_models:
            continue

        placeholder = copy.deepcopy(
            model
        ).to(
            "cpu"
        )

        placeholder.eval()
        placeholder.share_memory()

        shared_league_models[
            name
        ] = placeholder

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

    try:

        with ctx.Pool(
            processes=args.num_workers,
            initializer=rl._init_selfplay_worker,
            initargs=(
                shared_current_model,
                shared_league_models,
                league_registry,
                bc_model_selfplay,
                args.temperature_selfplay,
                args.opening_prior_strength,
                args.opening_prior_plies,
            ),
        ) as pool:

            # =================================================
            # Epoch loop
            # =================================================

            for epoch in range(
                first_al_epoch,
                args.end_epoch + 1,
            ):

                print()
                print("=" * 70)
                print(
                    f"RL + ORACLE — EPOCH {epoch}"
                )
                print("=" * 70)

                # =============================================
                # Self-play
                # =============================================

                games = rl.collect_games_parallel(
                    pool,
                    model,
                    league,
                    args.games_per_epoch,
                    stats,
                    epoch,
                    args,
                    device,
                )

                wins = 0
                losses = 0
                draws = 0

                # =============================================
                # Build PPO buffer
                # =============================================

                for game in games:

                    trajectory = game[
                        "trajectory"
                    ]

                    result = game[
                        "result"
                    ]

                    current_white = game[
                        "current_white"
                    ]

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

                    rewards = [
                        0.0
                    ] * len(
                        trajectory
                    )

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

                        rewards[
                            -1
                        ] = terminal_reward

                    advantages, returns = (
                        rl.compute_gae(
                            trajectory,
                            rewards,
                            gamma=args.gamma,
                            gae_lambda=args.gae_lambda,
                        )
                    )

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
                            step[
                                "fen"
                            ],
                            step[
                                "action"
                            ],
                            step[
                                "legal_moves"
                            ],
                            ret,
                            step[
                                "value"
                            ],
                            step[
                                "old_log_prob"
                            ],
                            advantage,
                            step[
                                "ply"
                            ],
                            result,
                        )

                total_games = (
                    wins
                    + losses
                    + draws
                )

                if total_games != args.games_per_epoch:

                    raise RuntimeError(
                        "Self-play returned "
                        f"{total_games} games instead of "
                        f"{args.games_per_epoch}."
                    )

                score_rate = (
                    wins
                    + 0.5
                    * draws
                ) / total_games

                print(
                    f"Results: "
                    f"{wins}W / "
                    f"{losses}L / "
                    f"{draws}D "
                    f"({score_rate:.1%})"
                )

                print(
                    f"On-policy buffer: "
                    f"{len(buffer):,} transitions"
                )

                # =============================================
                # PPO + Oracle
                # =============================================

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
                    args,
                    device,
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
                    f"| DKL={dkl:.6f} "
                    f"| DKL loss={dkl_loss:.6f}"
                )

                print(
                    "Oracle callbacks: "
                    f"{oracle_epoch_stats['calls']} "
                    "| injections: "
                    f"{oracle_epoch_stats['injections']}"
                )

                # =============================================
                # On-policy semantics
                # =============================================

                buffer.clear()

                # =============================================
                # Save uncertainty
                # =============================================

                stats.save(
                    uncertainty_stats_path
                )

                # =============================================
                # Save AL checkpoint
                # =============================================

                checkpoint_path = (
                    checkpoint_dir
                    / f"al_epoch_{epoch}.pt"
                )

                save_al_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    seed=args.seed,
                    acquisition=args.acquisition,
                    oracle_annotations=len(
                        oracle_buffer
                    ),
                    oracle_frequency=args.oracle_frequency,
                    oracle_policy_coef=args.oracle_policy_coef,
                    oracle_value_coef=args.oracle_value_coef,
                    source_checkpoint=args.start_checkpoint,
                )

                print(
                    f"Checkpoint: {checkpoint_path}"
                )

                # =============================================
                # League snapshot
                # =============================================

                snapshot = copy.deepcopy(
                    model
                ).to(
                    device
                )

                snapshot.eval()

                agent_name = (
                    f"league_epoch_{epoch:03d}"
                )

                league.add_agent(
                    agent_name,
                    snapshot,
                    use_for_uncertainty=True,
                )

                snapshot_path = (
                    league_output_dir
                    / f"{agent_name}.pt"
                )

                save_league_snapshot(
                    path=snapshot_path,
                    epoch=epoch,
                    model=snapshot,
                    seed=args.seed,
                    acquisition=args.acquisition,
                )

                # =============================================
                # Synchronize workers
                # =============================================

                if agent_name not in shared_league_models:

                    raise RuntimeError(
                        "Missing preallocated shared "
                        f"league slot: {agent_name}"
                    )

                copy_model_state_to_shared(
                    snapshot,
                    shared_league_models[
                        agent_name
                    ],
                )

                shared_league_models[
                    agent_name
                ].eval()

                league_registry[:] = (
                    league.names()
                )

                copy_model_state_to_shared(
                    model,
                    shared_current_model,
                )

                # =============================================
                # Summary
                # =============================================

                print()
                print(
                    f"Epoch {epoch} complete"
                )

                print(
                    f"League size: "
                    f"{len(league)}"
                )

                print(
                    "Uncertainty contributors: "
                    f"{league.uncertainty_names()}"
                )

                print(
                    f"Oracle annotations: "
                    f"{len(oracle_buffer)}"
                )

                print(
                    f"Uncertainty records: "
                    f"{len(stats.data):,}"
                )

    finally:

        manager.shutdown()

    print()
    print("=" * 70)
    print("RL + ORACLE TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Run directory: {run_dir}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()