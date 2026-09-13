from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import pickle
import random
import sys
import time

from pathlib import Path

import chess
import chess.variant
import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import Adam
from tqdm import tqdm


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions_space import ACTIONS, ACTION_TO_INDEX
from src.agents.ppo_agent import PPOAgent
from src.encoding import encode_boards
from src.models.actor_critic import ActorCritic
from src.models.resnet import ChessResNet
from src.rl.replay_buffer import ReplayBuffer
from src.rl.uncertainty_stats import UncertaintyStats
from src.selfplay.league import League


# ============================================================
# Default paths
# ============================================================

DEFAULT_BC_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
)

DEFAULT_RL_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
)

DEFAULT_LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league_rl"
)

DEFAULT_REPLAY_BUFFER_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "replay_buffer"
)

DEFAULT_UNCERTAINTY_STATS = (
    PROJECT_ROOT
    / "checkpoints"
    / "uncertainty_stats.json"
)


# ============================================================
# Historical default configuration
# ============================================================

DEFAULT_SEED = 42

DEFAULT_BC_INIT_EPOCH = 7
DEFAULT_BC_REFERENCE_EPOCH = 7
DEFAULT_BC_ANCHOR_EPOCHS = [6, 7]

DEFAULT_CHANNELS = 32
DEFAULT_BLOCKS = 4

DEFAULT_START_EPOCH = 1
DEFAULT_RL_EPOCHS = 10

DEFAULT_LR = 3e-4

DEFAULT_GAMES_PER_EPOCH = 2500

DEFAULT_CHECKPOINT_EVERY = 5
DEFAULT_SAVE_BUFFER_EVERY = 5

DEFAULT_BUFFER_CAPACITY = 300_000

DEFAULT_VALUE_COEF = 0.1
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SGD_EPOCHS = 3

DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95

DEFAULT_PPO_CLIP = 0.2
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_GRAD_CLIP = 1.0

DEFAULT_TEMPERATURE_SELFPLAY = 2.0

DEFAULT_OPENING_PRIOR_STRENGTH = 1.0
DEFAULT_OPENING_PRIOR_PLIES = 6

DEFAULT_LEAGUE_MAX_AGENTS = 12

DEFAULT_NUM_WORKERS = 12
DEFAULT_SELFPLAY_BATCH_SIZE = 256
DEFAULT_UNCERTAINTY_BATCH_SIZE = 256


# ============================================================
# DKL regularization defaults
# ============================================================

DEFAULT_RL_TOTAL_EPOCHS = 60

DEFAULT_DKL_FIT_EPOCH_STRIDE = 10.0

DEFAULT_DKL_INF = 1.8681333083457523

DEFAULT_DKL_DECAY_PER_FIT_UNIT = (
    0.2635650124178356
)

DEFAULT_DKL_ALPHA = 0.50

DEFAULT_LAMBDA_DKL = 0.12


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:
    """
    Seed the random generators used by the main process.

    Multiprocessing self-play tasks receive independently
    derived deterministic seeds.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def derive_seed(
    master_seed: int,
    epoch: int,
    stream: int,
    index: int = 0,
) -> int:
    """
    Deterministically derive independent RNG streams.

    stream:
        1 -> self-play task
        2 -> PPO update
    """

    modulus = (
        2_147_483_647
    )

    seed = (
        master_seed
        + epoch * 1_000_003
        + stream * 104_729
        + index * 7_919
    ) % modulus

    if seed == 0:
        seed = 1

    return seed


# ============================================================
# Device
# ============================================================

def resolve_device(
    requested: str,
) -> torch.device:

    if requested == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return device


# ============================================================
# Configuration serialization
# ============================================================

def serializable_config(
    args: argparse.Namespace,
) -> dict:

    config = {}

    for key, value in vars(args).items():

        if isinstance(
            value,
            Path,
        ):

            config[key] = str(
                value
            )

        elif isinstance(
            value,
            list,
        ):

            config[key] = [
                str(item)
                if isinstance(item, Path)
                else item
                for item in value
            ]

        else:

            config[key] = value

    return config


# ============================================================
# DKL schedule
# ============================================================

def get_natural_dkl(
    epoch: int | float,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:

    e = torch.as_tensor(
        epoch,
        dtype=torch.float32,
        device=device,
    )

    fit_time = (
        e
        / args.dkl_fit_epoch_stride
    )

    return (
        args.dkl_inf
        * (
            -torch.expm1(
                -args.dkl_decay_per_fit_unit
                * fit_time
            )
        )
    )


def get_dkl_lambda(
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:

    natural_dkl = get_natural_dkl(
        epoch,
        args,
        device,
    )

    if natural_dkl.item() <= 0.0:

        raise ValueError(
            "Natural DKL is zero or negative. "
            "RL epochs must start at epoch >= 1."
        )

    reference_dkl = get_natural_dkl(
        args.rl_total_epochs / 2,
        args,
        device,
    )

    return (
        args.lambda_dkl
        * reference_dkl
        / natural_dkl
    )


# ============================================================
# BC checkpoints
# ============================================================

def bc_checkpoint_path(
    epoch: int,
    args: argparse.Namespace,
) -> Path:

    return (
        args.bc_checkpoint_dir
        / f"bc_epoch_{epoch}.pt"
    )


def load_bc_actor_critic(
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    evaluation: bool = True,
) -> ActorCritic:
    """
    Build an ActorCritic whose backbone and policy head come
    from a BC checkpoint.

    The value head is newly initialized.

    For BC anchors and the BC regularization reference this
    value head is intentionally ignored.
    """

    path = bc_checkpoint_path(
        epoch,
        args,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"BC checkpoint not found: {path}"
        )

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=args.channels,
        blocks=args.blocks,
    )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    checkpoint_actions = (
        checkpoint.get(
            "actions"
        )
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(ACTIONS)
    ):

        raise ValueError(
            f"Action-space mismatch in {path}: "
            f"{checkpoint_actions} != {len(ACTIONS)}"
        )

    bc_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = ActorCritic(
        bc_model
    ).to(
        device
    )

    if evaluation:
        model.eval()

    return model


# ============================================================
# League snapshots
# ============================================================

def league_snapshot_epoch(
    path: Path,
) -> int:

    try:

        return int(
            path.stem.split("_")[-1]
        )

    except ValueError as exc:

        raise ValueError(
            "Cannot extract epoch from league snapshot: "
            f"{path.name}"
        ) from exc


def load_league_snapshot(
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> ActorCritic:

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=args.channels,
        blocks=args.blocks,
    )

    model = ActorCritic(
        base_model
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model


def load_historical_league(
    league: League,
    up_to_epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Restore the most recent historical PPO snapshots that fit
    inside the league alongside protected BC anchors.
    """

    historical_capacity = (
        args.league_max_agents
        - len(args.bc_anchor_epochs)
    )

    if historical_capacity <= 0:
        return

    snapshots = []

    for path in args.league_dir.glob(
        "league_epoch_*.pt"
    ):

        epoch = league_snapshot_epoch(
            path
        )

        if epoch <= up_to_epoch:

            snapshots.append(
                (
                    epoch,
                    path,
                )
            )

    snapshots.sort(
        key=lambda item: item[0]
    )

    snapshots = snapshots[
        -historical_capacity:
    ]

    for epoch, path in snapshots:

        snapshot = load_league_snapshot(
            path=path,
            args=args,
            device=device,
        )

        name = (
            f"league_epoch_{epoch:03d}"
        )

        league.add_agent(
            name,
            snapshot,
            use_for_uncertainty=True,
        )

        print(
            f"Loaded {name}"
        )


# ============================================================
# RL model initialization / resume
# ============================================================

def initialize_training_state(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    ActorCritic,
    Adam,
    League,
    int,
]:
    """
    Initialize or resume the current RL model and historical
    league.

    Returns:
        model,
        optimizer,
        league,
        first epoch to run.
    """

    print()
    print("=" * 70)

    # ========================================================
    # Current model
    # ========================================================

    if args.resume is None:

        print(
            f"Initializing RL from "
            f"BC{args.bc_init_epoch}"
        )

        model = load_bc_actor_critic(
            epoch=args.bc_init_epoch,
            args=args,
            device=device,
            evaluation=False,
        )

        optimizer = Adam(
            model.parameters(),
            lr=args.learning_rate,
        )

        start_epoch = (
            args.start_epoch
        )

    else:

        print(
            "Resuming RL"
        )

        if not args.resume.exists():

            raise FileNotFoundError(
                f"RL checkpoint not found: {args.resume}"
            )

        print(
            f"Checkpoint: {args.resume}"
        )

        base_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=args.channels,
            blocks=args.blocks,
        )

        model = ActorCritic(
            base_model
        ).to(
            device
        )

        checkpoint = torch.load(
            args.resume,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer = Adam(
            model.parameters(),
            lr=args.learning_rate,
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

        else:

            print(
                "WARNING: optimizer state missing."
            )

        checkpoint_seed = (
            checkpoint.get(
                "seed"
            )
        )

        if (
            checkpoint_seed is not None
            and int(checkpoint_seed)
            != args.seed
        ):

            raise ValueError(
                "Resume seed differs from checkpoint seed: "
                f"{args.seed} != {checkpoint_seed}"
            )

        checkpoint_epoch = int(
            checkpoint[
                "epoch"
            ]
        )

        start_epoch = (
            checkpoint_epoch
            + 1
        )

        print(
            f"RL checkpoint loaded "
            f"(epoch {checkpoint_epoch})."
        )

    # ========================================================
    # League
    # ========================================================

    protected_names = [
        f"bc_epoch_{epoch}"
        for epoch in args.bc_anchor_epochs
    ]

    league = League(
        max_agents=args.league_max_agents,
        protected_agents=protected_names,
    )

    # --------------------------------------------------------
    # BC anchors remain valid self-play opponents.
    #
    # Their value heads are newly/randomly initialized and
    # therefore MUST NOT contribute to U(s).
    # --------------------------------------------------------

    for epoch in args.bc_anchor_epochs:

        name = (
            f"bc_epoch_{epoch}"
        )

        anchor = load_bc_actor_critic(
            epoch=epoch,
            args=args,
            device=device,
            evaluation=True,
        )

        league.add_agent(
            name,
            anchor,
            use_for_uncertainty=False,
        )

    # --------------------------------------------------------
    # Historical trained critics on resume
    # --------------------------------------------------------

    if args.resume is not None:

        load_historical_league(
            league=league,
            up_to_epoch=start_epoch - 1,
            args=args,
            device=device,
        )

    print("-" * 70)

    print(
        f"League opponents:       "
        f"{len(league)}"
    )

    print(
        f"Uncertainty critics:    "
        f"{len(league.uncertainty_names())}"
    )

    print(
        "League members:"
    )

    for name in league.names():

        marker = (
            "U"
            if name
            in league.uncertainty_names()
            else "-"
        )

        print(
            f"  [{marker}] {name}"
        )

    print("=" * 70)

    return (
        model,
        optimizer,
        league,
        start_epoch,
    )


# ============================================================
# Self-play worker globals
# ============================================================

_WORKER_CURRENT_MODEL = None

_WORKER_LEAGUE_MODELS = None

_WORKER_CURRENT_AGENT = None

_WORKER_LEAGUE_AGENTS = None

_WORKER_LEAGUE_REGISTRY = None


# ============================================================
# Shared-model preparation
# ============================================================

def prepare_shared_model(
    model,
):

    cpu_model = copy.deepcopy(
        model
    ).to(
        "cpu"
    )

    cpu_model.eval()

    cpu_model.share_memory()

    return cpu_model


def copy_model_state_to_shared(
    shared_model,
    source_model,
) -> None:
    """
    Copy parameters and buffers into an already shared CPU
    model without replacing its storage.
    """

    shared_state = (
        shared_model.state_dict()
    )

    source_state = (
        source_model.state_dict()
    )

    with torch.no_grad():

        for key, value in source_state.items():

            shared_state[
                key
            ].copy_(
                value.detach().cpu()
            )


# ============================================================
# Worker initialization
# ============================================================

def _init_selfplay_worker(
    current_model,
    league_models,
    league_registry,
    bc_model,
    temperature_selfplay,
    opening_prior_strength,
    opening_prior_plies,
):

    global _WORKER_CURRENT_MODEL
    global _WORKER_LEAGUE_MODELS
    global _WORKER_CURRENT_AGENT
    global _WORKER_LEAGUE_AGENTS
    global _WORKER_LEAGUE_REGISTRY

    torch.set_num_threads(
        1
    )

    _WORKER_CURRENT_MODEL = (
        current_model
    )

    _WORKER_CURRENT_MODEL.eval()

    _WORKER_LEAGUE_MODELS = (
        league_models
    )

    for model in (
        _WORKER_LEAGUE_MODELS.values()
    ):

        model.eval()

    _WORKER_LEAGUE_REGISTRY = (
        league_registry
    )

    # --------------------------------------------------------
    # Current agent keeps the BC opening prior.
    # --------------------------------------------------------

    _WORKER_CURRENT_AGENT = PPOAgent(
        _WORKER_CURRENT_MODEL,
        deterministic=False,
        temperature=temperature_selfplay,
        device="cpu",
        bc_model=bc_model,
        opening_prior_strength=opening_prior_strength,
        opening_prior_plies=opening_prior_plies,
    )

    # --------------------------------------------------------
    # Historical opponents reproduce their frozen policy
    # directly, without adding the current BC opening prior.
    # --------------------------------------------------------

    _WORKER_LEAGUE_AGENTS = {}

    for name in _WORKER_LEAGUE_REGISTRY:

        if (
            name
            not in _WORKER_LEAGUE_MODELS
        ):
            continue

        model = (
            _WORKER_LEAGUE_MODELS[
                name
            ]
        )

        model.eval()

        _WORKER_LEAGUE_AGENTS[
            name
        ] = PPOAgent(
            model,
            deterministic=False,
            temperature=temperature_selfplay,
            device="cpu",
        )


# ============================================================
# Worker self-play
# ============================================================

def _selfplay_worker(
    worker_args,
):

    (
        n_games,
        task_id,
        batch_size,
        task_seed,
        global_game_start,
    ) = worker_args

    global _WORKER_CURRENT_AGENT
    global _WORKER_LEAGUE_MODELS
    global _WORKER_LEAGUE_AGENTS
    global _WORKER_LEAGUE_REGISTRY

    # ========================================================
    # Deterministic per-task RNG
    # ========================================================

    random.seed(
        task_seed
    )

    torch.manual_seed(
        task_seed
    )

    # ========================================================
    # Current active league
    # ========================================================

    active_names = list(
        _WORKER_LEAGUE_REGISTRY
    )

    # New snapshots can become active after the worker process
    # was initially created.
    for name in active_names:

        if (
            name
            in _WORKER_LEAGUE_AGENTS
        ):

            continue

        if (
            name
            not in _WORKER_LEAGUE_MODELS
        ):

            continue

        model = (
            _WORKER_LEAGUE_MODELS[
                name
            ]
        )

        model.eval()

        _WORKER_LEAGUE_AGENTS[
            name
        ] = PPOAgent(
            model,
            deterministic=False,
            temperature=(
                _WORKER_CURRENT_AGENT.temperature
            ),
            device="cpu",
        )

    current_agent = (
        _WORKER_CURRENT_AGENT
    )

    league_agents = (
        _WORKER_LEAGUE_AGENTS
    )

    opponent_names = [
        name
        for name in active_names
        if name in league_agents
    ]

    if not opponent_names:

        raise RuntimeError(
            "No opponent available in league."
        )

    # ========================================================
    # Initialize games
    # ========================================================

    active_games = []

    completed_games = []

    for local_game_index in range(
        n_games
    ):

        global_game_index = (
            global_game_start
            + local_game_index
        )

        opponent_name = random.choice(
            opponent_names
        )

        opponent_agent = (
            league_agents[
                opponent_name
            ]
        )

        # ----------------------------------------------------
        # Global alternation guarantees color balance even when
        # games are split across multiprocessing tasks.
        # ----------------------------------------------------

        if global_game_index % 2 == 0:

            white_agent = (
                current_agent
            )

            black_agent = (
                opponent_agent
            )

            current_is_white = True

        else:

            white_agent = (
                opponent_agent
            )

            black_agent = (
                current_agent
            )

            current_is_white = False

        active_games.append(
            {
                "game_index":
                    global_game_index,

                "board":
                    chess.variant.AtomicBoard(),

                "white":
                    white_agent,

                "black":
                    black_agent,

                "current_white":
                    current_is_white,

                "trajectory":
                    [],
            }
        )

    # ========================================================
    # Batched self-play
    # ========================================================

    with torch.no_grad():

        while active_games:

            games_by_agent = {}

            for game in active_games:

                board = (
                    game[
                        "board"
                    ]
                )

                agent = (
                    game["white"]
                    if board.turn
                    else game["black"]
                )

                games_by_agent.setdefault(
                    agent,
                    [],
                ).append(
                    game
                )

            # ------------------------------------------------
            # Forward each policy in batches.
            # ------------------------------------------------

            for (
                agent,
                agent_games,
            ) in games_by_agent.items():

                for start in range(
                    0,
                    len(agent_games),
                    batch_size,
                ):

                    batch_games = (
                        agent_games[
                            start:
                            start + batch_size
                        ]
                    )

                    if not batch_games:
                        continue

                    boards = [
                        game["board"]
                        for game in batch_games
                    ]

                    infos = agent.choose_moves(
                        boards
                    )

                    for (
                        game,
                        info,
                    ) in zip(
                        batch_games,
                        infos,
                    ):

                        board = (
                            game[
                                "board"
                            ]
                        )

                        # ------------------------------------
                        # PPO transitions are recorded only for
                        # the current learner's own turns.
                        # ------------------------------------

                        if agent is current_agent:

                            game[
                                "trajectory"
                            ].append(
                                {
                                    "fen":
                                        board.fen(),

                                    "action":
                                        info[
                                            "action"
                                        ],

                                    "player":
                                        board.turn,

                                    "value":
                                        info[
                                            "value"
                                        ],

                                    "entropy":
                                        info[
                                            "entropy"
                                        ],

                                    "old_log_prob":
                                        info[
                                            "log_prob"
                                        ],

                                    "legal_moves":
                                        [
                                            move.uci()
                                            for move
                                            in board.legal_moves
                                        ],

                                    "ply":
                                        board.ply(),
                                }
                            )

                        board.push(
                            info[
                                "move"
                            ]
                        )

            # ------------------------------------------------
            # Terminal games
            # ------------------------------------------------

            still_active = []

            for game in active_games:

                board = (
                    game[
                        "board"
                    ]
                )

                if board.is_game_over():

                    completed_games.append(
                        {
                            "game_index":
                                game[
                                    "game_index"
                                ],

                            "trajectory":
                                game[
                                    "trajectory"
                                ],

                            "result":
                                board.result(),

                            "current_white":
                                game[
                                    "current_white"
                                ],
                        }
                    )

                else:

                    still_active.append(
                        game
                    )

            active_games = (
                still_active
            )

    # --------------------------------------------------------
    # Completion order depends on game length. Restore original
    # game ordering inside the task.
    # --------------------------------------------------------

    completed_games.sort(
        key=lambda game: game[
            "game_index"
        ]
    )

    return (
        task_id,
        completed_games,
    )


# ============================================================
# Parallel self-play collection
# ============================================================

def collect_games_parallel(
    pool,
    model,
    league,
    n_games: int,
    stats: UncertaintyStats,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
):

    selfplay_start = (
        time.perf_counter()
    )

    if len(league) == 0:

        raise RuntimeError(
            "League is empty."
        )

    if n_games <= 0:

        return []

    num_workers = min(
        args.num_workers,
        n_games,
    )

    games_per_task = max(
        12,
        n_games
        // (
            num_workers
            * 4
        ),
    )

    worker_args = []

    remaining = n_games

    task_id = 0

    global_game_start = 0

    while remaining > 0:

        task_games = min(
            games_per_task,
            remaining,
        )

        task_seed = derive_seed(
            master_seed=args.seed,
            epoch=epoch,
            stream=1,
            index=task_id,
        )

        worker_args.append(
            (
                task_games,
                task_id,
                args.selfplay_batch_size,
                task_seed,
                global_game_start,
            )
        )

        remaining -= (
            task_games
        )

        global_game_start += (
            task_games
        )

        task_id += 1

    # ========================================================
    # Multiprocessing execution
    # ========================================================

    task_results = {}

    progress = tqdm(
        total=n_games,
        desc="League self-play",
    )

    for (
        returned_task_id,
        worker_games,
    ) in pool.imap_unordered(
        _selfplay_worker,
        worker_args,
        chunksize=1,
    ):

        task_results[
            returned_task_id
        ] = worker_games

        progress.update(
            len(
                worker_games
            )
        )

    progress.close()

    # --------------------------------------------------------
    # imap_unordered() scheduling is nondeterministic.
    # Restore deterministic task and game order.
    # --------------------------------------------------------

    completed_games = []

    for task_id in sorted(
        task_results
    ):

        completed_games.extend(
            task_results[
                task_id
            ]
        )

    completed_games.sort(
        key=lambda game: game[
            "game_index"
        ]
    )

    if len(completed_games) != n_games:

        raise RuntimeError(
            "Incorrect number of completed games: "
            f"{len(completed_games)} / {n_games}"
        )

    # ========================================================
    # Self-play diagnostics
    # ========================================================

    selfplay_time = (
        time.perf_counter()
        - selfplay_start
    )

    total_positions = sum(
        len(
            game[
                "trajectory"
            ]
        )
        for game in completed_games
    )

    print(
        f"Self-play time: "
        f"{selfplay_time:.2f}s "
        f"({selfplay_time / n_games:.2f}s/game)"
    )

    print(
        f"Self-play positions: "
        f"{total_positions} "
        f"({total_positions / n_games:.1f}/game)"
    )

    # ========================================================
    # H / U / HU
    # ========================================================

    uncertainty_start = (
        time.perf_counter()
    )

    all_steps = []

    for game in completed_games:

        result = (
            game[
                "result"
            ]
        )

        for step in game[
            "trajectory"
        ]:

            step[
                "_game_result"
            ] = result

            all_steps.append(
                step
            )

    if all_steps:

        boards = [
            chess.variant.AtomicBoard(
                step[
                    "fen"
                ]
            )
            for step in all_steps
        ]

        uncertainties = []

        with torch.no_grad():

            for start in range(
                0,
                len(boards),
                args.uncertainty_batch_size,
            ):

                batch_boards = (
                    boards[
                        start:
                        start
                        + args.uncertainty_batch_size
                    ]
                )

                x = encode_boards(
                    batch_boards
                ).to(
                    device
                )

                batch_uncertainties = (
                    league.uncertainty_batch(
                        x=x,
                        current_model=model,
                    )
                )

                uncertainties.extend(
                    batch_uncertainties
                    .detach()
                    .cpu()
                    .tolist()
                )

                del x

        for (
            step,
            uncertainty,
        ) in zip(
            all_steps,
            uncertainties,
        ):

            entropy = float(
                step.get(
                    "entropy",
                    0.0,
                )
            )

            uncertainty = float(
                uncertainty
            )

            interaction = (
                entropy
                * uncertainty
            )

            step[
                "uncertainty"
            ] = uncertainty

            step[
                "HU"
            ] = interaction

            stats.add(
                fen=step[
                    "fen"
                ],
                action=step[
                    "action"
                ],
                entropy=entropy,
                uncertainty=uncertainty,
                HU=interaction,
                result=step[
                    "_game_result"
                ],
            )

    uncertainty_time = (
        time.perf_counter()
        - uncertainty_start
    )

    milliseconds_per_position = (
        uncertainty_time
        / max(
            len(
                all_steps
            ),
            1,
        )
        * 1000.0
    )

    print(
        f"U computation time: "
        f"{uncertainty_time:.2f}s "
        f"({milliseconds_per_position:.2f}ms/position)"
    )

    print(
        f"Uncertainty records added: "
        f"{len(all_steps)}",
        flush=True,
    )

    return completed_games


# ============================================================
# GAE
# ============================================================

def compute_gae(
    trajectory,
    rewards,
    gamma: float,
    gae_lambda: float,
):

    n = len(
        trajectory
    )

    if n == 0:

        return (
            [],
            [],
        )

    values = [
        float(
            step[
                "value"
            ]
        )
        for step in trajectory
    ]

    advantages = [
        0.0
    ] * n

    gae = 0.0

    for t in reversed(
        range(
            n
        )
    ):

        if t == n - 1:

            next_value = 0.0

        else:

            next_value = (
                values[
                    t + 1
                ]
            )

        delta = (
            rewards[t]
            + gamma * next_value
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * gae
        )

        advantages[
            t
        ] = gae

    returns = [
        advantages[t]
        + values[t]
        for t in range(n)
    ]

    return (
        advantages,
        returns,
    )


# ============================================================
# PPO training
# ============================================================

def train_epoch(
    model,
    optimizer,
    buffer,
    bc_model,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    extra_loss_fn=None,
):

    model.train()

    bc_model.eval()

    # ========================================================
    # Freeze BatchNorm statistics during PPO.
    #
    # Parameters remain trainable; running statistics do not.
    # ========================================================

    for module in model.modules():

        if isinstance(
            module,
            torch.nn.BatchNorm2d,
        ):

            module.eval()

    if len(buffer) < args.batch_size:

        print(
            "Rollout buffer too small."
        )

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    train_steps = (
        len(buffer)
        // args.batch_size
    )

    total_updates = (
        train_steps
        * args.sgd_epochs
    )

    # ========================================================
    # Accumulators
    # ========================================================

    total_loss = 0.0

    total_actor = 0.0

    total_critic = 0.0

    total_kl = 0.0

    total_entropy = 0.0

    total_clip_fraction = 0.0

    total_actor_grad_norm = 0.0

    total_critic_grad_norm = 0.0

    total_dkl_grad_norm = 0.0

    total_adv_mean = 0.0

    total_adv_std = 0.0

    total_return_mean = 0.0

    total_return_std = 0.0

    total_value_mean = 0.0

    total_value_std = 0.0

    total_explained_variance = 0.0

    total_dkl = 0.0

    total_dkl_loss = 0.0

    total_extra_loss = 0.0

    total_extra_policy_loss = 0.0

    total_extra_value_loss = 0.0

    total_grad_cosine = 0.0

    valid_grad_cosines = 0

    lambda_dkl = get_dkl_lambda(
        epoch=epoch,
        args=args,
        device=device,
    )

    natural_dkl = get_natural_dkl(
        epoch=epoch,
        args=args,
        device=device,
    )

    target_dkl = (
        (1.0 - args.dkl_alpha)
        * natural_dkl
    )

    progress = tqdm(
        total=total_updates,
        desc="PPO Training",
    )

    # ========================================================
    # PPO updates
    # ========================================================

    for _ in range(
        args.sgd_epochs
    ):

        for _ in range(
            train_steps
        ):

            batch = buffer.sample(
                args.batch_size
            )

            # =================================================
            # Encode boards
            # =================================================

            boards = [
                chess.variant.AtomicBoard(
                    sample[
                        "fen"
                    ]
                )
                for sample in batch
            ]

            x = encode_boards(
                boards
            ).to(
                device
            )

            # =================================================
            # Current actor-critic
            # =================================================

            policy, values = model(
                x
            )

            # =================================================
            # Frozen BC reference policy
            # =================================================

            with torch.no_grad():

                bc_policy, _ = bc_model(
                    x
                )

            # =================================================
            # Ply-dependent BC opening prior
            # =================================================

            plys = torch.tensor(
                [
                    sample[
                        "ply"
                    ]
                    for sample in batch
                ],
                dtype=torch.float32,
                device=device,
            )

            if (
                args.opening_prior_strength <= 0.0
                or args.opening_prior_plies <= 0
            ):

                bc_weight = (
                    torch.zeros_like(
                        plys
                    )
                )

            else:

                bc_weight = (
                    args.opening_prior_strength
                    * torch.clamp(
                        1.0
                        - plys
                        / float(
                            args.opening_prior_plies
                        ),
                        min=0.0,
                    )
                )

            # =================================================
            # Returns
            # =================================================

            returns = torch.tensor(
                [
                    sample[
                        "return"
                    ]
                    for sample in batch
                ],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(
                1
            )

            # =================================================
            # Raw advantages
            # =================================================

            raw_advantages = torch.tensor(
                [
                    sample[
                        "advantage"
                    ]
                    for sample in batch
                ],
                dtype=torch.float32,
                device=device,
            )

            adv_mean = (
                raw_advantages.mean()
            )

            adv_std = (
                raw_advantages.std()
            )

            # =================================================
            # Advantage normalization
            # =================================================

            advantages = (
                raw_advantages
                - raw_advantages.mean()
            ) / (
                raw_advantages.std()
                + 1e-8
            )

            # =================================================
            # Old action log probabilities
            # =================================================

            old_log_probs = torch.tensor(
                [
                    sample[
                        "old_log_prob"
                    ]
                    for sample in batch
                ],
                dtype=torch.float32,
                device=device,
            )

            # =================================================
            # Legal action mask
            # =================================================

            legal_mask = torch.zeros(
                (
                    len(
                        batch
                    ),
                    policy.shape[
                        1
                    ],
                ),
                dtype=torch.bool,
                device=device,
            )

            actions = torch.tensor(
                [
                    sample[
                        "action"
                    ]
                    for sample in batch
                ],
                dtype=torch.long,
                device=device,
            )

            for i, sample in enumerate(
                batch
            ):

                legal_ids = [
                    ACTION_TO_INDEX[
                        move
                    ]
                    for move
                    in sample[
                        "legal_moves"
                    ]
                ]

                legal_mask[
                    i,
                    legal_ids,
                ] = True

            # =================================================
            # Legal RL / BC policies
            # =================================================

            legal_policy = (
                policy.masked_fill(
                    ~legal_mask,
                    float(
                        "-inf"
                    ),
                )
            )

            legal_bc_policy = (
                bc_policy.masked_fill(
                    ~legal_mask,
                    float(
                        "-inf"
                    ),
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

            # =================================================
            # DKL(RL || BC)
            #
            # This regularizer concerns the unguided RL policy,
            # before opening prior and sampling temperature.
            # =================================================

            safe_rl_log_probs = (
                rl_log_probs.masked_fill(
                    ~legal_mask,
                    0.0,
                )
            )

            safe_bc_log_probs = (
                bc_log_probs.masked_fill(
                    ~legal_mask,
                    0.0,
                )
            )

            rl_probs_for_dkl = (
                torch.exp(
                    rl_log_probs
                )
            )

            dkl_per_position = (
                rl_probs_for_dkl
                * (
                    safe_rl_log_probs
                    - safe_bc_log_probs
                )
            ).sum(
                dim=1
            )

            delta_dkl = (
                dkl_per_position.mean()
            )

            # =================================================
            # Unilateral DKL budget
            # =================================================

            dkl_error = (
                delta_dkl
                - target_dkl
            )

            excess_dkl = torch.relu(
                dkl_error
            )

            dkl_loss = (
                0.5
                * lambda_dkl
                * excess_dkl.pow(
                    2
                )
            )

            # =================================================
            # DKL head-gradient diagnostic
            #
            # Historical diagnostic: policy head only.
            # =================================================

            dkl_gradients = (
                torch.autograd.grad(
                    dkl_loss,
                    model.policy.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
            )

            dkl_grad_sq = 0.0

            for gradient in dkl_gradients:

                if gradient is None:
                    continue

                dkl_grad_sq += (
                    gradient
                    .detach()
                    .norm(
                        2
                    )
                    .item()
                    ** 2
                )

            dkl_grad_norm = (
                dkl_grad_sq
                ** 0.5
            )

            # =================================================
            # BC-guided policy
            #
            # Equivalent to:
            #
            #   logits_RL + alpha * log pi_BC
            #
            # up to a state-dependent normalizing constant.
            # =================================================

            combined_log_probs = (
                rl_log_probs
                + bc_weight.unsqueeze(
                    1
                )
                * safe_bc_log_probs
            )

            combined_log_probs = (
                F.log_softmax(
                    combined_log_probs,
                    dim=1,
                )
            )

            # =================================================
            # Self-play temperature
            #
            # Must match PPOAgent rollout semantics.
            # =================================================

            ppo_logits = (
                combined_log_probs
                / args.temperature_selfplay
            )

            log_probs = F.log_softmax(
                ppo_logits,
                dim=1,
            )

            selected_log_probs = (
                log_probs
                .gather(
                    1,
                    actions.unsqueeze(
                        1
                    ),
                )
                .squeeze(
                    1
                )
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
                (
                    ratio
                    < 1.0
                    - args.ppo_clip
                )
                |
                (
                    ratio
                    > 1.0
                    + args.ppo_clip
                )
            )

            clip_fraction = (
                clipped_mask
                .float()
                .mean()
            )

            # =================================================
            # PPO actor objective
            # =================================================

            unclipped = (
                ratio
                * advantages
            )

            clipped = (
                torch.clamp(
                    ratio,
                    1.0
                    - args.ppo_clip,
                    1.0
                    + args.ppo_clip,
                )
                * advantages
            )

            actor_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()

            # =================================================
            # Entropy of actual PPO sampling distribution
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
                * safe_log_probs
            ).sum(
                dim=1
            ).mean()

            # =================================================
            # Critic
            #
            # Historical ALBERTA value clipping is preserved
            # exactly for experimental continuity.
            # =================================================

            values_old = torch.tensor(
                [
                    sample[
                        "value"
                    ]
                    for sample in batch
                ],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(
                1
            )

            values_clipped = (
                values_old
                + (
                    values
                    - values_old
                ).clamp(
                    -args.ppo_clip,
                    args.ppo_clip,
                )
            )

            critic_loss = F.mse_loss(
                values_clipped,
                returns,
            )

            # =================================================
            # Critic diagnostics
            # =================================================

            value_flat = (
                values.squeeze(
                    1
                )
            )

            return_flat = (
                returns.squeeze(
                    1
                )
            )

            value_mean = (
                value_flat.mean()
            )

            value_std = (
                value_flat.std()
            )

            return_mean = (
                return_flat.mean()
            )

            return_std = (
                return_flat.std()
            )

            return_variance = torch.var(
                return_flat,
                unbiased=False,
            )

            residual_variance = torch.var(
                return_flat
                - value_flat,
                unbiased=False,
            )

            explained_variance = (
                1.0
                - residual_variance
                / (
                    return_variance
                    + 1e-8
                )
            )

            # =================================================
            # Optional extra loss
            #
            # Retained for compatibility with AL training.
            # =================================================

            if extra_loss_fn is not None:

                extra = (
                    extra_loss_fn(
                        model
                    )
                )

                extra_loss = (
                    extra[
                        "loss"
                    ]
                )

                extra_policy_loss = (
                    extra.get(
                        "policy_loss",
                        torch.zeros(
                            (),
                            device=device,
                        ),
                    )
                )

                extra_value_loss = (
                    extra.get(
                        "value_loss",
                        torch.zeros(
                            (),
                            device=device,
                        ),
                    )
                )

            else:

                extra_loss = torch.zeros(
                    (),
                    device=device,
                )

                extra_policy_loss = (
                    torch.zeros(
                        (),
                        device=device,
                    )
                )

                extra_value_loss = (
                    torch.zeros(
                        (),
                        device=device,
                    )
                )

            # =================================================
            # PPO loss before optional Oracle loss
            # =================================================

            ppo_loss = (
                actor_loss
                + args.value_coef
                * critic_loss
                - args.entropy_coef
                * entropy
                + dkl_loss
            )

            loss = (
                ppo_loss
                + extra_loss
            )

            # =================================================
            # Actor head gradient diagnostic
            # =================================================

            actor_gradients = (
                torch.autograd.grad(
                    actor_loss,
                    model.policy.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
            )

            actor_grad_sq = 0.0

            for gradient in actor_gradients:

                if gradient is None:
                    continue

                actor_grad_sq += (
                    gradient
                    .detach()
                    .norm(
                        2
                    )
                    .item()
                    ** 2
                )

            actor_grad_norm = (
                actor_grad_sq
                ** 0.5
            )

            # =================================================
            # Critic head gradient diagnostic
            # =================================================

            critic_gradients = (
                torch.autograd.grad(
                    critic_loss,
                    model.value.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
            )

            critic_grad_sq = 0.0

            for gradient in critic_gradients:

                if gradient is None:
                    continue

                critic_grad_sq += (
                    gradient
                    .detach()
                    .norm(
                        2
                    )
                    .item()
                    ** 2
                )

            critic_grad_norm = (
                critic_grad_sq
                ** 0.5
            )

            # =================================================
            # PPO / Oracle gradient cosine diagnostic
            # =================================================

            if extra_loss_fn is not None:

                ppo_gradients = (
                    torch.autograd.grad(
                        ppo_loss,
                        model.parameters(),
                        retain_graph=True,
                        allow_unused=True,
                    )
                )

                oracle_gradients = (
                    torch.autograd.grad(
                        extra_loss,
                        model.parameters(),
                        retain_graph=True,
                        allow_unused=True,
                    )
                )

                ppo_flat = []

                oracle_flat = []

                for (
                    ppo_gradient,
                    oracle_gradient,
                ) in zip(
                    ppo_gradients,
                    oracle_gradients,
                ):

                    if (
                        ppo_gradient is None
                        and oracle_gradient
                        is None
                    ):

                        continue

                    if ppo_gradient is None:

                        ppo_gradient = (
                            torch.zeros_like(
                                oracle_gradient
                            )
                        )

                    if oracle_gradient is None:

                        oracle_gradient = (
                            torch.zeros_like(
                                ppo_gradient
                            )
                        )

                    ppo_flat.append(
                        ppo_gradient
                        .detach()
                        .reshape(
                            -1
                        )
                    )

                    oracle_flat.append(
                        oracle_gradient
                        .detach()
                        .reshape(
                            -1
                        )
                    )

                if (
                    ppo_flat
                    and oracle_flat
                ):

                    ppo_vector = torch.cat(
                        ppo_flat
                    )

                    oracle_vector = torch.cat(
                        oracle_flat
                    )

                    ppo_norm = (
                        torch.linalg.vector_norm(
                            ppo_vector
                        )
                    )

                    oracle_norm = (
                        torch.linalg.vector_norm(
                            oracle_vector
                        )
                    )

                    if (
                        ppo_norm.item()
                        > 1e-12
                        and oracle_norm.item()
                        > 1e-12
                    ):

                        grad_cosine = (
                            torch.dot(
                                ppo_vector,
                                oracle_vector,
                            )
                            / (
                                ppo_norm
                                * oracle_norm
                            )
                        ).item()

                        total_grad_cosine += (
                            grad_cosine
                        )

                        valid_grad_cosines += (
                            1
                        )

            # =================================================
            # Optimization
            # =================================================

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
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

            total_kl += (
                approx_kl.item()
            )

            total_entropy += (
                entropy.item()
            )

            total_clip_fraction += (
                clip_fraction.item()
            )

            total_actor_grad_norm += (
                actor_grad_norm
            )

            total_critic_grad_norm += (
                critic_grad_norm
            )

            total_dkl_grad_norm += (
                dkl_grad_norm
            )

            total_adv_mean += (
                adv_mean.item()
            )

            total_adv_std += (
                adv_std.item()
            )

            total_return_mean += (
                return_mean.item()
            )

            total_return_std += (
                return_std.item()
            )

            total_value_mean += (
                value_mean.item()
            )

            total_value_std += (
                value_std.item()
            )

            total_explained_variance += (
                explained_variance.item()
            )

            total_dkl += (
                delta_dkl.item()
            )

            total_dkl_loss += (
                dkl_loss.item()
            )

            total_extra_loss += (
                extra_loss.item()
            )

            total_extra_policy_loss += (
                extra_policy_loss.item()
            )

            total_extra_value_loss += (
                extra_value_loss.item()
            )

            progress.update(
                1
            )

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

    avg_kl = (
        total_kl
        / total_updates
    )

    avg_entropy = (
        total_entropy
        / total_updates
    )

    avg_clip_fraction = (
        total_clip_fraction
        / total_updates
    )

    avg_actor_grad_norm = (
        total_actor_grad_norm
        / total_updates
    )

    avg_critic_grad_norm = (
        total_critic_grad_norm
        / total_updates
    )

    avg_dkl_grad_norm = (
        total_dkl_grad_norm
        / total_updates
    )

    avg_adv_mean = (
        total_adv_mean
        / total_updates
    )

    avg_adv_std = (
        total_adv_std
        / total_updates
    )

    avg_return_mean = (
        total_return_mean
        / total_updates
    )

    avg_return_std = (
        total_return_std
        / total_updates
    )

    avg_value_mean = (
        total_value_mean
        / total_updates
    )

    avg_value_std = (
        total_value_std
        / total_updates
    )

    avg_explained_variance = (
        total_explained_variance
        / total_updates
    )

    avg_dkl = (
        total_dkl
        / total_updates
    )

    avg_dkl_loss = (
        total_dkl_loss
        / total_updates
    )

    avg_extra_loss = (
        total_extra_loss
        / total_updates
    )

    avg_extra_policy_loss = (
        total_extra_policy_loss
        / total_updates
    )

    avg_extra_value_loss = (
        total_extra_value_loss
        / total_updates
    )

    if valid_grad_cosines > 0:

        avg_grad_cosine = (
            total_grad_cosine
            / valid_grad_cosines
        )

    else:

        avg_grad_cosine = float(
            "nan"
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    print()
    print("=" * 70)
    print("PPO DIAGNOSTICS")
    print("=" * 70)

    print(
        f"Epoch:                  "
        f"{epoch}"
    )

    print(
        f"Self-play temperature:  "
        f"{args.temperature_selfplay:.2f}"
    )

    print(
        f"BC prior strength:      "
        f"{args.opening_prior_strength:.2f}"
    )

    print(
        f"BC prior decay ply:     "
        f"{args.opening_prior_plies}"
    )

    print(
        f"Advantage mean:         "
        f"{avg_adv_mean:+.6f}"
    )

    print(
        f"Advantage std:          "
        f"{avg_adv_std:.6f}"
    )

    print(
        f"Return mean:            "
        f"{avg_return_mean:+.6f}"
    )

    print(
        f"Return std:             "
        f"{avg_return_std:.6f}"
    )

    print(
        f"Value mean:             "
        f"{avg_value_mean:+.6f}"
    )

    print(
        f"Value std:              "
        f"{avg_value_std:.6f}"
    )

    print(
        f"Critic MSE:             "
        f"{avg_critic:.6f}"
    )

    print(
        f"Explained variance:     "
        f"{avg_explained_variance:+.6f}"
    )

    print(
        f"Actor gradient norm:    "
        f"{avg_actor_grad_norm:.6e}"
    )

    print(
        f"Critic gradient norm:   "
        f"{avg_critic_grad_norm:.6e}"
    )

    print(
        f"DKL gradient norm:      "
        f"{avg_dkl_grad_norm:.6e}"
    )

    print(
        f"Policy KL:              "
        f"{avg_kl:.6e}"
    )

    print(
        f"Clip fraction:          "
        f"{avg_clip_fraction:.2%}"
    )

    print(
        f"Entropy:                "
        f"{avg_entropy:.6f}"
    )

    if extra_loss_fn is not None:

        print(
            f"PPO/Oracle grad cosine: "
            f"{avg_grad_cosine:+.6f}"
        )

    else:

        print(
            "PPO/Oracle grad cosine: "
            "N/A (no Oracle loss)"
        )

    print("-" * 70)

    print(
        f"Natural DKL:            "
        f"{natural_dkl.item():.6e}"
    )

    print(
        f"Target DKL:             "
        f"{target_dkl.item():.6e}"
    )

    print(
        f"DKL(RL || BC):          "
        f"{avg_dkl:.6e}"
    )

    print(
        f"DKL lambda:             "
        f"{lambda_dkl.item():.6e}"
    )

    print(
        f"DKL loss:               "
        f"{avg_dkl_loss:.6e}"
    )

    if extra_loss_fn is not None:

        print("-" * 70)

        print(
            f"Extra loss:             "
            f"{avg_extra_loss:.6e}"
        )

        print(
            f"Extra policy loss:      "
            f"{avg_extra_policy_loss:.6e}"
        )

        print(
            f"Extra value loss:       "
            f"{avg_extra_value_loss:.6e}"
        )

    print("=" * 70)

    return (
        avg_loss,
        avg_actor,
        avg_critic,
        avg_kl,
        avg_dkl,
        avg_dkl_loss,
    )


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
) -> Path:

    args.rl_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        args.rl_checkpoint_dir
        / f"rl_epoch_{epoch}.pt"
    )

    torch.save(
        {
            "epoch":
                epoch,

            "seed":
                args.seed,

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

            "config":
                serializable_config(
                    args
                ),
        },
        path,
    )

    print(
        f"RL checkpoint saved: {path}",
        flush=True,
    )

    return path


# ============================================================
# Replay-buffer saving
# ============================================================

def save_replay_buffer(
    buffer,
    epoch: int,
    args: argparse.Namespace,
) -> Path:

    args.replay_buffer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        args.replay_buffer_dir
        / f"replay_buffer_epoch_{epoch}.pkl"
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
        f"Rollout buffer saved: {path}",
        flush=True,
    )

    return path


# ============================================================
# Uncertainty-state loading
# ============================================================

def load_uncertainty_stats(
    path: Path,
    resume: bool,
) -> UncertaintyStats:

    stats = UncertaintyStats()

    if not resume:
        return stats

    if not path.exists():

        print(
            "WARNING: uncertainty statistics file "
            f"not found during resume: {path}"
        )

        return stats

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            "Uncertainty statistics JSON must contain a list."
        )

    stats.data = data

    print(
        f"Loaded {len(stats.data)} "
        f"uncertainty records."
    )

    return stats


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train ALBERTA with PPO against a historical "
            "Atomic Chess league."
        )
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    parser.add_argument(
        "--bc-checkpoint-dir",
        type=Path,
        default=DEFAULT_BC_CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--rl-checkpoint-dir",
        type=Path,
        default=DEFAULT_RL_CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--league-dir",
        type=Path,
        default=DEFAULT_LEAGUE_DIR,
    )

    parser.add_argument(
        "--replay-buffer-dir",
        type=Path,
        default=DEFAULT_REPLAY_BUFFER_DIR,
    )

    parser.add_argument(
        "--uncertainty-stats",
        type=Path,
        default=DEFAULT_UNCERTAINTY_STATS,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "RL checkpoint from which to resume. "
            "The next epoch is inferred automatically."
        ),
    )

    # --------------------------------------------------------
    # Reproducibility / device
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "'auto', 'cpu', 'cuda', 'cuda:0', ..."
        ),
    )

    # --------------------------------------------------------
    # Architecture / BC
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

    parser.add_argument(
        "--bc-init-epoch",
        type=int,
        default=DEFAULT_BC_INIT_EPOCH,
    )

    parser.add_argument(
        "--bc-reference-epoch",
        type=int,
        default=DEFAULT_BC_REFERENCE_EPOCH,
        help=(
            "BC policy used for opening prior and DKL "
            "regularization."
        ),
    )

    parser.add_argument(
        "--bc-anchor-epochs",
        type=int,
        nargs="+",
        default=DEFAULT_BC_ANCHOR_EPOCHS,
        help=(
            "BC checkpoints kept permanently as league "
            "opponents. Their random critics are excluded "
            "from U(s)."
        ),
    )

    # --------------------------------------------------------
    # Training horizon
    # --------------------------------------------------------

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=DEFAULT_START_EPOCH,
        help=(
            "First epoch for a fresh run. Ignored when "
            "--resume is supplied."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_RL_EPOCHS,
        help=(
            "Number of RL epochs to run in this invocation."
        ),
    )

    parser.add_argument(
        "--games-per-epoch",
        type=int,
        default=DEFAULT_GAMES_PER_EPOCH,
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--value-coef",
        type=float,
        default=DEFAULT_VALUE_COEF,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--sgd-epochs",
        type=int,
        default=DEFAULT_SGD_EPOCHS,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=DEFAULT_GAE_LAMBDA,
    )

    parser.add_argument(
        "--ppo-clip",
        type=float,
        default=DEFAULT_PPO_CLIP,
    )

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=DEFAULT_ENTROPY_COEF,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )

    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=DEFAULT_BUFFER_CAPACITY,
    )

    # --------------------------------------------------------
    # Policy used during rollout
    # --------------------------------------------------------

    parser.add_argument(
        "--temperature-selfplay",
        type=float,
        default=DEFAULT_TEMPERATURE_SELFPLAY,
    )

    parser.add_argument(
        "--opening-prior-strength",
        type=float,
        default=DEFAULT_OPENING_PRIOR_STRENGTH,
    )

    parser.add_argument(
        "--opening-prior-plies",
        type=int,
        default=DEFAULT_OPENING_PRIOR_PLIES,
    )

    # --------------------------------------------------------
    # League
    # --------------------------------------------------------

    parser.add_argument(
        "--league-max-agents",
        type=int,
        default=DEFAULT_LEAGUE_MAX_AGENTS,
    )

    # --------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    parser.add_argument(
        "--selfplay-batch-size",
        type=int,
        default=DEFAULT_SELFPLAY_BATCH_SIZE,
    )

    parser.add_argument(
        "--uncertainty-batch-size",
        type=int,
        default=DEFAULT_UNCERTAINTY_BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
    )

    parser.add_argument(
        "--save-buffer-every",
        type=int,
        default=DEFAULT_SAVE_BUFFER_EVERY,
        help=(
            "Save rollout buffer every N epochs. "
            "Use 0 to disable."
        ),
    )

    # --------------------------------------------------------
    # DKL
    # --------------------------------------------------------

    parser.add_argument(
        "--rl-total-epochs",
        type=int,
        default=DEFAULT_RL_TOTAL_EPOCHS,
    )

    parser.add_argument(
        "--dkl-fit-epoch-stride",
        type=float,
        default=DEFAULT_DKL_FIT_EPOCH_STRIDE,
    )

    parser.add_argument(
        "--dkl-inf",
        type=float,
        default=DEFAULT_DKL_INF,
    )

    parser.add_argument(
        "--dkl-decay-per-fit-unit",
        type=float,
        default=DEFAULT_DKL_DECAY_PER_FIT_UNIT,
    )

    parser.add_argument(
        "--dkl-alpha",
        type=float,
        default=DEFAULT_DKL_ALPHA,
    )

    parser.add_argument(
        "--lambda-dkl",
        type=float,
        default=DEFAULT_LAMBDA_DKL,
    )

    return parser.parse_args()


# ============================================================
# Validation
# ============================================================

def validate_args(
    args: argparse.Namespace,
) -> None:

    if args.seed < 0:

        raise ValueError(
            "--seed must be non-negative."
        )

    if args.resume is None:

        if args.start_epoch < 1:

            raise ValueError(
                "--start-epoch must be >= 1 because the "
                "DKL schedule is undefined at epoch 0."
            )

    if args.epochs <= 0:

        raise ValueError(
            "--epochs must be greater than zero."
        )

    if args.games_per_epoch <= 0:

        raise ValueError(
            "--games-per-epoch must be greater than zero."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.sgd_epochs <= 0:

        raise ValueError(
            "--sgd-epochs must be greater than zero."
        )

    if args.buffer_capacity <= 0:

        raise ValueError(
            "--buffer-capacity must be greater than zero."
        )

    if args.num_workers <= 0:

        raise ValueError(
            "--num-workers must be greater than zero."
        )

    if args.selfplay_batch_size <= 0:

        raise ValueError(
            "--selfplay-batch-size must be greater than zero."
        )

    if args.uncertainty_batch_size <= 0:

        raise ValueError(
            "--uncertainty-batch-size must be greater than zero."
        )

    if args.temperature_selfplay <= 0.0:

        raise ValueError(
            "--temperature-selfplay must be > 0 for PPO "
            "rollout training."
        )

    if args.opening_prior_strength < 0.0:

        raise ValueError(
            "--opening-prior-strength cannot be negative."
        )

    if args.opening_prior_plies < 0:

        raise ValueError(
            "--opening-prior-plies cannot be negative."
        )

    if args.league_max_agents <= 0:

        raise ValueError(
            "--league-max-agents must be greater than zero."
        )

    if len(
        set(
            args.bc_anchor_epochs
        )
    ) != len(
        args.bc_anchor_epochs
    ):

        raise ValueError(
            "--bc-anchor-epochs contains duplicates."
        )

    if (
        len(
            args.bc_anchor_epochs
        )
        > args.league_max_agents
    ):

        raise ValueError(
            "Number of BC anchors exceeds league capacity."
        )

    if args.checkpoint_every <= 0:

        raise ValueError(
            "--checkpoint-every must be greater than zero."
        )

    if args.save_buffer_every < 0:

        raise ValueError(
            "--save-buffer-every cannot be negative."
        )

    if args.dkl_fit_epoch_stride <= 0.0:

        raise ValueError(
            "--dkl-fit-epoch-stride must be > 0."
        )

    if args.rl_total_epochs <= 0:

        raise ValueError(
            "--rl-total-epochs must be > 0."
        )

    if args.grad_clip <= 0.0:

        raise ValueError(
            "--grad-clip must be > 0."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    validate_args(
        args
    )

    device = resolve_device(
        args.device
    )

    # ========================================================
    # Create output directories
    # ========================================================

    args.rl_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.league_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.replay_buffer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.uncertainty_stats.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Master reproducibility seed
    #
    # IMPORTANT: called before any ActorCritic construction,
    # because the value head is randomly initialized.
    # ========================================================

    seed_everything(
        args.seed
    )

    print()
    print("=" * 70)
    print("ALBERTA - LEAGUE PPO TRAINING")
    print("=" * 70)

    print(
        f"Project root:           "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Device:                 "
        f"{device}"
    )

    print(
        f"Master seed:            "
        f"{args.seed}"
    )

    print(
        f"Games / epoch:          "
        f"{args.games_per_epoch}"
    )

    print(
        f"Self-play temperature:  "
        f"{args.temperature_selfplay}"
    )

    print(
        f"Workers:                "
        f"{args.num_workers}"
    )

    if device.type == "cuda":

        print(
            f"GPU:                    "
            f"{torch.cuda.get_device_name(device)}"
        )

    # ========================================================
    # Current model + optimizer + league
    # ========================================================

    (
        model,
        optimizer,
        league,
        start_epoch,
    ) = initialize_training_state(
        args=args,
        device=device,
    )

    final_epoch = (
        start_epoch
        + args.epochs
        - 1
    )

    # ========================================================
    # Frozen BC reference policy
    #
    # Its random value head is irrelevant: only the policy head
    # is used for the opening prior and DKL regularization.
    # ========================================================

    bc_model = load_bc_actor_critic(
        epoch=args.bc_reference_epoch,
        args=args,
        device=device,
        evaluation=True,
    )

    # ========================================================
    # Rollout buffer
    # ========================================================

    buffer = ReplayBuffer(
        capacity=args.buffer_capacity
    )

    # ========================================================
    # Uncertainty statistics
    # ========================================================

    stats = load_uncertainty_stats(
        path=args.uncertainty_stats,
        resume=(
            args.resume
            is not None
        ),
    )

    # ========================================================
    # Shared CPU models
    # ========================================================

    print()
    print(
        "Preparing shared CPU models...",
        flush=True,
    )

    shared_current_model = (
        prepare_shared_model(
            model
        )
    )

    shared_bc_model = (
        prepare_shared_model(
            bc_model
        )
    )

    shared_league_models = {}

    for (
        name,
        league_model,
    ) in league.agents.items():

        shared_league_models[
            name
        ] = prepare_shared_model(
            league_model
        )

    # --------------------------------------------------------
    # Preallocate shared slots for snapshots created during
    # this invocation. Workers are created only once.
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        final_epoch + 1,
    ):

        name = (
            f"league_epoch_{epoch:03d}"
        )

        if name in shared_league_models:
            continue

        placeholder = (
            copy.deepcopy(
                model
            ).to(
                "cpu"
            )
        )

        placeholder.eval()

        placeholder.share_memory()

        shared_league_models[
            name
        ] = placeholder

    print(
        f"Shared league slots: "
        f"{len(shared_league_models)}",
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

    print()
    print(
        "Initial league registry:",
        flush=True,
    )

    for name in league_registry:

        print(
            f"  - {name}",
            flush=True,
        )

    try:

        with ctx.Pool(
            processes=args.num_workers,
            initializer=_init_selfplay_worker,
            initargs=(
                shared_current_model,
                shared_league_models,
                league_registry,
                shared_bc_model,
                args.temperature_selfplay,
                args.opening_prior_strength,
                args.opening_prior_plies,
            ),
        ) as pool:

            # =================================================
            # RL loop
            # =================================================

            for epoch in range(
                start_epoch,
                final_epoch + 1,
            ):

                print()
                print("=" * 70)
                print(
                    f"EPOCH {epoch}"
                )
                print("=" * 70)

                print(
                    f"League opponents:       "
                    f"{len(league)}"
                )

                print(
                    f"Uncertainty critics:    "
                    f"{len(league.uncertainty_names())}"
                )

                print(
                    "U critics:              "
                    + (
                        ", ".join(
                            league.uncertainty_names()
                        )
                        if league.uncertainty_names()
                        else "none"
                    )
                )

                wins = 0

                losses = 0

                draws = 0

                # =============================================
                # Self-play
                # =============================================

                games = collect_games_parallel(
                    pool=pool,
                    model=model,
                    league=league,
                    n_games=args.games_per_epoch,
                    stats=stats,
                    epoch=epoch,
                    args=args,
                    device=device,
                )

                # =============================================
                # Build on-policy rollout buffer
                # =============================================

                for game in games:

                    trajectory = (
                        game[
                            "trajectory"
                        ]
                    )

                    result = (
                        game[
                            "result"
                        ]
                    )

                    current_white = (
                        game[
                            "current_white"
                        ]
                    )

                    # -----------------------------------------
                    # Match result from learner perspective
                    # -----------------------------------------

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

                    # -----------------------------------------
                    # Sparse terminal reward
                    # -----------------------------------------

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

                    # -----------------------------------------
                    # GAE
                    # -----------------------------------------

                    (
                        advantages,
                        returns,
                    ) = compute_gae(
                        trajectory=trajectory,
                        rewards=rewards,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                    )

                    # -----------------------------------------
                    # Rollout buffer
                    # -----------------------------------------

                    for (
                        step,
                        advantage,
                        return_,
                    ) in zip(
                        trajectory,
                        advantages,
                        returns,
                    ):

                        buffer.add(
                            fen=step[
                                "fen"
                            ],
                            action=step[
                                "action"
                            ],
                            legal_moves=step[
                                "legal_moves"
                            ],
                            return_=return_,
                            value=step[
                                "value"
                            ],
                            old_log_prob=step[
                                "old_log_prob"
                            ],
                            advantage=advantage,
                            ply=step[
                                "ply"
                            ],
                            game_result=result,
                        )

                # =============================================
                # Epoch self-play statistics
                # =============================================

                score_rate = (
                    wins
                    + 0.5
                    * draws
                ) / len(
                    games
                )

                print()
                print(
                    f"Rollout buffer size: "
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

                # =============================================
                # Reproducible PPO RNG stream
                # =============================================

                ppo_seed = derive_seed(
                    master_seed=args.seed,
                    epoch=epoch,
                    stream=2,
                )

                seed_everything(
                    ppo_seed
                )

                print(
                    f"PPO seed: {ppo_seed}",
                    flush=True,
                )

                # =============================================
                # PPO
                # =============================================

                (
                    loss,
                    actor_loss,
                    critic_loss,
                    approx_kl,
                    dkl,
                    dkl_loss,
                ) = train_epoch(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    bc_model=bc_model,
                    epoch=epoch,
                    args=args,
                    device=device,
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

                # =============================================
                # Save rollout buffer before clearing
                # =============================================

                if (
                    args.save_buffer_every > 0
                    and epoch
                    % args.save_buffer_every
                    == 0
                ):

                    save_replay_buffer(
                        buffer=buffer,
                        epoch=epoch,
                        args=args,
                    )

                # =============================================
                # PPO remains on-policy
                # =============================================

                buffer.clear()

                print(
                    "Rollout buffer cleared after PPO update.",
                    flush=True,
                )

                # =============================================
                # Uncertainty statistics
                # =============================================

                stats.save(
                    args.uncertainty_stats
                )

                print(
                    f"Uncertainty JSON updated: "
                    f"{len(stats)} positions",
                    flush=True,
                )

                # =============================================
                # RL checkpoint
                # =============================================

                should_save_checkpoint = (
                    epoch
                    % args.checkpoint_every
                    == 0
                    or epoch
                    == final_epoch
                )

                if should_save_checkpoint:

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=loss,
                        args=args,
                    )

                # =============================================
                # Historical league snapshot
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
                    args.league_dir
                    / f"{agent_name}.pt"
                )

                torch.save(
                    {
                        "epoch":
                            epoch,

                        "seed":
                            args.seed,

                        "model_state_dict":
                            snapshot.state_dict(),

                        "config":
                            serializable_config(
                                args
                            ),
                    },
                    snapshot_path,
                )

                print(
                    f"League snapshot saved: "
                    f"{snapshot_path}",
                    flush=True,
                )

                # =============================================
                # Update shared snapshot slot
                # =============================================

                if (
                    agent_name
                    not in shared_league_models
                ):

                    raise RuntimeError(
                        "Missing shared model slot for "
                        f"{agent_name}"
                    )

                copy_model_state_to_shared(
                    shared_model=(
                        shared_league_models[
                            agent_name
                        ]
                    ),
                    source_model=snapshot,
                )

                shared_league_models[
                    agent_name
                ].eval()

                # =============================================
                # Active league registry
                # =============================================

                league_registry[:] = (
                    league.names()
                )

                print(
                    "Updated league registry:",
                    list(
                        league_registry
                    ),
                    flush=True,
                )

                # =============================================
                # Update shared current learner
                # =============================================

                copy_model_state_to_shared(
                    shared_model=shared_current_model,
                    source_model=model,
                )

                shared_current_model.eval()

                # =============================================
                # Epoch summary
                # =============================================

                print()
                print(
                    f"===== Epoch {epoch} summary =====",
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
                    f"{get_dkl_lambda(epoch, args, device).item():.6e}",
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
                    f"U critics: "
                    f"{len(league.uncertainty_names())}",
                    flush=True,
                )

    finally:

        manager.shutdown()

    print()
    print(
        "RL training finished.",
        flush=True,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()