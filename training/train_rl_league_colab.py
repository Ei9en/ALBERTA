# ============================================================
# Train_RL.py
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import time
import chess
import chess.variant
import copy
import multiprocessing as mp
import random

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
from src.rl.uncertainty_stats import UncertaintyStats
from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX


# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/ALBERTA"
)


# ============================================================
# Temperature
# ============================================================

TEMPERATURE_SELFPLAY = 1


# ============================================================
# Reprise RL
# ============================================================

START_EPOCH = 1

RESUME_RL = False

RESUME_EPOCH = START_EPOCH - 1

CHECKPOINT_EPOCH = START_EPOCH - 1


LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league_rl"
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

ENTROPY_COEF = 0.05


# ============================================================
# DKL regularization
# ============================================================

RL_TOTAL_EPOCHS = 60
DKL_FIT_EPOCH_STRIDE = 10.0

DKL_INF = 1.8681333083457523
DKL_DECAY_PER_FIT_UNIT = 0.2635650124178356

# Alpha = fraction de réduction du drift naturel.
# alpha=0.50 => KL cible = 50% de KL_naturel.

DKL_ALPHA = 0.50
LAMBDA_DKL = 0.048



def get_dkl_lambda(epoch):

    natural_dkl = get_natural_dkl(epoch)

    reference_dkl = get_natural_dkl(
        RL_TOTAL_EPOCHS/2
    )

    return (
        LAMBDA_DKL
        * reference_dkl
        / natural_dkl
    )
    


def get_natural_dkl(epoch):
    e = torch.as_tensor(
        epoch,
        dtype=torch.float32,
        device=DEVICE,
    )

    fit_time = e / DKL_FIT_EPOCH_STRIDE

    return DKL_INF * (
        -torch.expm1(
            -DKL_DECAY_PER_FIT_UNIT
            * fit_time
        )
    )


# ============================================================
# League
# ============================================================

LEAGUE_MAX_AGENTS = 12

LEAGUE_END_EPOCH = START_EPOCH - 1

LEAGUE_START_EPOCH = LEAGUE_END_EPOCH - 9


# ============================================================
# Model Loading
# ============================================================

def load_bc_agent(epoch):

    path = (
        PROJECT_ROOT
        / "checkpoints"
        / "bc_epoch"
        / f"bc_epoch_{epoch}.pt"
    )

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    bc_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = ActorCritic(
        bc_model
    )

    model.to(DEVICE)

    model.eval()

    return model


def load_league_agent(epoch):

    name = (
        f"league_epoch_{epoch:03d}"
    )

    path = (
        LEAGUE_DIR
        / f"{name}.pt"
    )

    if not path.exists():

        print(
            f"WARNING: missing league snapshot: "
            f"{path}"
        )

        return None

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        base_model
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# RL checkpoint
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / f"rl_epoch_{CHECKPOINT_EPOCH}.pt"
)


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
            "Initializing RL from BC7"
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
            "Initial policy loaded from BC7."
        )

    # ========================================================
    # League initiale
    # ========================================================

    league = League(
        max_agents=LEAGUE_MAX_AGENTS
    )

    bc6 = load_bc_agent(6)

    league.add_agent(
        "bc_epoch_6",
        bc6,
    )

    b7 = load_bc_agent(7)

    league.add_agent(
        "bc_epoch_7",
        b7,
    )

    # ========================================================
    # Reprise : charger les 10 dernières snapshots
    # ========================================================

    if RESUME_RL:

        start_epoch = max(
            1,
            LEAGUE_START_EPOCH,
        )

        for epoch in range(
            start_epoch,
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
# Self-play globals
# ============================================================

_WORKER_CURRENT_MODEL = None

_WORKER_LEAGUE_MODELS = None

_WORKER_CURRENT_AGENT = None

_WORKER_LEAGUE_AGENTS = None

_WORKER_LEAGUE_REGISTRY = None


# ============================================================
# Préparation modèle partagé
# ============================================================

def _prepare_shared_model(model):

    cpu_model = copy.deepcopy(
        model
    ).to("cpu")

    cpu_model.eval()

    cpu_model.share_memory()

    return cpu_model


# ============================================================
# Initialisation worker
# ============================================================

def _init_selfplay_worker(
    current_model,
    league_models,
    league_registry,
    bc_model,
):

    global _WORKER_CURRENT_MODEL
    global _WORKER_LEAGUE_MODELS
    global _WORKER_CURRENT_AGENT
    global _WORKER_LEAGUE_AGENTS
    global _WORKER_LEAGUE_REGISTRY

    torch.set_num_threads(1)

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

    _WORKER_CURRENT_AGENT = PPOAgent(
        _WORKER_CURRENT_MODEL,
        deterministic=False,
        temperature=TEMPERATURE_SELFPLAY,
        device="cpu",
        bc_model=bc_model,
        opening_prior_strength=1.0,
        opening_prior_plies=6,
    )

    _WORKER_LEAGUE_AGENTS = {}

    for name in _WORKER_LEAGUE_REGISTRY:

        if name not in _WORKER_LEAGUE_MODELS:
            continue

        model = (
            _WORKER_LEAGUE_MODELS[name]
        )

        model.eval()

        _WORKER_LEAGUE_AGENTS[name] = (
            PPOAgent(
                model,
                deterministic=False,
                temperature=TEMPERATURE_SELFPLAY,
                device="cpu",
            )
        )


# ============================================================
# Worker self-play
# ============================================================

def _selfplay_worker(worker_args):

    (
        n_games,
        worker_id,
        batch_size,
    ) = worker_args

    global _WORKER_CURRENT_AGENT
    global _WORKER_LEAGUE_MODELS
    global _WORKER_LEAGUE_AGENTS
    global _WORKER_LEAGUE_REGISTRY

    # ========================================================
    # Active league
    # ========================================================

    active_names = list(
        _WORKER_LEAGUE_REGISTRY
    )

    for name in active_names:

        if name not in _WORKER_LEAGUE_AGENTS:

            if name not in _WORKER_LEAGUE_MODELS:
                continue

            model = (
                _WORKER_LEAGUE_MODELS[name]
            )

            model.eval()

            _WORKER_LEAGUE_AGENTS[name] = (
                PPOAgent(
                    model,
                    deterministic=False,
                    temperature=TEMPERATURE_SELFPLAY,
                    device="cpu",
                )
            )

    # ========================================================
    # Random seed
    # ========================================================

    seed = (
        1000003
        + worker_id * 7919
        + random.randrange(
            100000000
        )
    )

    random.seed(seed)

    torch.manual_seed(seed)

    # ========================================================
    # Agents
    # ========================================================

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
            "Aucun adversaire disponible "
            "dans la league."
        )

    # ========================================================
    # Initialize games
    # ========================================================

    active_games = []

    completed_games = []

    for i in range(n_games):

        opponent_name = random.choice(
            opponent_names
        )

        opponent_agent = (
            league_agents[
                opponent_name
            ]
        )

        if i % 2 == 0:

            white_agent = current_agent

            black_agent = opponent_agent

            current_is_white = True

        else:

            white_agent = opponent_agent

            black_agent = current_agent

            current_is_white = False

        active_games.append(
            {
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

                board = game["board"]

                agent = (
                    game["white"]
                    if board.turn
                    else game["black"]
                )

                games_by_agent.setdefault(
                    agent,
                    [],
                ).append(game)

            for (
                agent,
                agent_games,
            ) in games_by_agent.items():

                for start in range(
                    0,
                    len(agent_games),
                    batch_size,
                ):

                    batch_games = agent_games[
                        start:start + batch_size
                    ]

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

                        board = game["board"]

                        if agent is current_agent:

                            game[
                                "trajectory"
                            ].append(
                                {
                                    "fen":
                                        board.fen(),

                                    "action":
                                        info["action"],

                                    "player":
                                        board.turn,

                                    "value":
                                        info["value"],

                                    "entropy":
                                        info["entropy"],

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
                            info["move"]
                        )

            still_active = []

            for game in active_games:

                board = game["board"]

                if board.is_game_over():

                    completed_games.append(
                        {
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

            active_games = still_active

    return completed_games


# ============================================================
# Collecte parallèle
# ============================================================

def collect_games_parallel(
    pool,
    shared_current_model,
    shared_league_models,
    model,
    league,
    n_games,
    stats,
    num_workers=12,
    batch_size=256,
):

    selfplay_start = time.perf_counter()

    if len(league) == 0:

        raise RuntimeError(
            "La league est vide."
        )

    if n_games <= 0:

        return []

    num_workers = min(
        num_workers,
        n_games,
    )

    games_per_task = max(
        12,
        n_games // (
            num_workers * 4
        ),
    )

    worker_args = []

    remaining = n_games

    task_id = 0

    while remaining > 0:

        task_games = min(
            games_per_task,
            remaining,
        )

        worker_args.append(
            (
                task_games,
                task_id,
                batch_size,
            )
        )

        remaining -= task_games

        task_id += 1

    completed_games = []

    progress = tqdm(
        total=n_games,
        desc="League self-play",
    )

    for worker_games in pool.imap_unordered(
        _selfplay_worker,
        worker_args,
        chunksize=1,
    ):

        completed_games.extend(
            worker_games
        )

        progress.update(
            len(worker_games)
        )

    progress.close()

    if len(completed_games) != n_games:

        raise RuntimeError(
            f"Nombre de parties incorrect : "
            f"{len(completed_games)} / "
            f"{n_games}"
        )

    selfplay_time = (
        time.perf_counter()
        - selfplay_start
    )

    total_positions = sum(
        len(game["trajectory"])
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
    # Calcul U + H + HU
    # ========================================================

    uncertainty_start = (
        time.perf_counter()
    )

    all_steps = []

    for game in completed_games:

        result = game["result"]

        for step in game["trajectory"]:

            step["_game_result"] = result

            all_steps.append(
                step
            )

    if all_steps:

        boards = [
            chess.variant.AtomicBoard(
                step["fen"]
            )
            for step in all_steps
        ]

        U_BATCH_SIZE = 256

        uncertainties = []

        with torch.no_grad():

            for start in range(
                0,
                len(boards),
                U_BATCH_SIZE,
            ):

                batch_boards = boards[
                    start:start + U_BATCH_SIZE
                ]

                x = encode_boards(
                    batch_boards
                ).to(DEVICE)

                batch_uncertainties = (
                    league.uncertainty_batch(
                        x,
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
            U,
        ) in zip(
            all_steps,
            uncertainties,
        ):

            H = step.get(
                "entropy",
                0.0,
            )

            HU = H * U

            step["uncertainty"] = U

            step["HU"] = HU

            stats.add(
                step["fen"],
                step["action"],
                H,
                U,
                HU,
                step["_game_result"],
            )

    uncertainty_time = (
        time.perf_counter()
        - uncertainty_start
    )

    print(
        f"U computation time: "
        f"{uncertainty_time:.2f}s "
        f"({uncertainty_time / max(len(all_steps), 1) * 1000:.2f}ms/position)"
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
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
):

    n = len(trajectory)

    if n == 0:

        return [], []

    values = [
        float(step["value"])
        for step in trajectory
    ]

    advantages = [
        0.0
    ] * n

    gae = 0.0

    for t in reversed(
        range(n)
    ):

        if t == n - 1:

            next_value = 0.0

        else:

            next_value = values[t + 1]

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

        advantages[t] = gae

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
    epoch,
):

    model.train()

    bc_model.eval()

    # ========================================================
    # Freeze BatchNorm
    # ========================================================

    for module in model.modules():

        if isinstance(
            module,
            torch.nn.BatchNorm2d,
        ):

            module.eval()

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

    TRAIN_STEPS = (
        len(buffer)
        // BATCH_SIZE
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

    # ========================================================
    # NEW: DKL gradient diagnostic
    # ========================================================

    total_dkl_grad_norm = 0.0

    total_adv_mean = 0.0
    total_adv_std = 0.0

    total_return_mean = 0.0
    total_return_std = 0.0

    total_value_mean = 0.0
    total_value_std = 0.0

    total_explained_variance = 0.0

    # ========================================================
    # DKL diagnostics
    # ========================================================

    total_dkl = 0.0

    total_dkl_loss = 0.0

    lambda_dkl = get_dkl_lambda(epoch)


    total_updates = (
        TRAIN_STEPS
        * SGD_EPOCHS
    )

    progress = tqdm(
        total=total_updates,
        desc="PPO Training",
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

            # =================================================
            # Encode boards
            # =================================================

            boards = [
                chess.variant.AtomicBoard(
                    s["fen"]
                )
                for s in batch
            ]

            x = encode_boards(
                boards
            ).to(DEVICE)

            # =================================================
            # Forward RL
            # =================================================

            policy, values = model(x)

            # =================================================
            # Forward BC
            # =================================================

            with torch.no_grad():

                bc_policy, _ = bc_model(x)

            # =================================================
            # Ply
            # =================================================

            plys = torch.tensor(
                [
                    s["ply"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            )

            # =================================================
            # BC prior décroissant
            # =================================================

            BC_PRIOR_STRENGTH = 1.0
            BC_PRIOR_PLIES = 6.0

            bc_weight = (
                BC_PRIOR_STRENGTH
                *
                torch.clamp(
                    1.0 - plys / BC_PRIOR_PLIES,
                    min=0.0,
                )
            )

            # =================================================
            # Returns
            # =================================================

            returns = torch.tensor(
                [
                    s["return"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            ).unsqueeze(1)

            # =================================================
            # Advantages RAW
            # =================================================

            raw_advantages = torch.tensor(
                [
                    s["advantage"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
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
            # Old log probabilities
            # =================================================

            old_log_probs = torch.tensor(
                [
                    s["old_log_prob"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            )

            # =================================================
            # Legal mask
            # =================================================

            legal_mask = torch.zeros(
                (
                    len(batch),
                    policy.shape[1],
                ),
                dtype=torch.bool,
                device=DEVICE,
            )

            actions = torch.tensor(
                [
                    s["action"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.long,
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
            # Masquage des coups illégaux
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

            # =================================================
            # Distributions RL / BC
            # =================================================

            rl_log_probs = F.log_softmax(
                legal_policy,
                dim=1,
            )

            bc_log_probs = F.log_softmax(
                legal_bc_policy,
                dim=1,
            )

            # =================================================
            # DKL(RL || BC), calcul pur par position
            # =================================================

            safe_rl_log_probs = rl_log_probs.masked_fill(
            ~legal_mask,
            0.0,
            )

            safe_bc_log_probs = bc_log_probs.masked_fill(
                ~legal_mask,
                0.0,
            )

            rl_probs_for_dkl = torch.exp(rl_log_probs)

            dkl_per_position = (
                rl_probs_for_dkl
                * (
                    safe_rl_log_probs
                    - safe_bc_log_probs
                )
            ).sum(dim=1)

            delta_dkl = dkl_per_position.mean()

            # =================================================
            # Budget de divergence relatif au PPO sans DKL
            # =================================================

            natural_dkl = get_natural_dkl(epoch)

            target_dkl = (
                (1.0 - DKL_ALPHA)
                * natural_dkl
            )

            dkl_error = (
                delta_dkl
                - target_dkl
            )

            # =================================================
            # Rappel unilatéral vers BC au-delà du budget
            # =================================================

            excess_dkl = torch.relu(
                dkl_error
            )

            dkl_loss = (
                0.5
                * lambda_dkl
                * excess_dkl.pow(2)
            )

            # =================================================
            # NEW: Gradient de L_DKL
            #
            # || ∇θ L_DKL ||_2
            #
            # θ = paramètres de la policy RL
            #
            # On conserve le graphe pour permettre ensuite
            # le backward() de la loss totale.
            # =================================================

            dkl_gradients = torch.autograd.grad(
                dkl_loss,
                model.policy.parameters(),
                retain_graph=True,
                allow_unused=True,
            )

            dkl_grad_sq = 0.0

            for grad in dkl_gradients:

                if grad is None:
                    continue

                dkl_grad_sq += (
                    grad.detach().norm(2).item() ** 2
                )

            dkl_grad_norm = (
                dkl_grad_sq ** 0.5
            )

            # =================================================
            # BC prior
            # =================================================

            combined_log_probs = (
                rl_log_probs
                +
                bc_weight.unsqueeze(1)
                *
                safe_bc_log_probs
            )

            # =================================================
            # Renormalisation
            # =================================================

            combined_log_probs = F.log_softmax(
                combined_log_probs,
                dim=1,
            )

            # =================================================
            # Température
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

            # =================================================
            # Log-prob du coup joué
            # =================================================

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

            # =================================================
            # Policy KL
            # =================================================

            approx_kl = (
                (
                    ratio
                    - 1.0
                    - log_ratio
                ).mean()
            )

            # =================================================
            # Clip fraction
            # =================================================

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
                    1 - PPO_CLIP,
                    1 + PPO_CLIP,
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
            # Critic
            # =================================================

            values_old = torch.tensor(
                [s["value"] for s in batch],
                device=DEVICE,
                dtype=torch.float32,
            ).unsqueeze(1)

            values_clipped = values_old + (values - values_old).clamp(-PPO_CLIP, PPO_CLIP)
            critic_loss = F.mse_loss(values_clipped, returns)

            # =================================================
            # Critic diagnostics
            # =================================================

            value_flat = (
                values.squeeze(1)
            )

            return_flat = (
                returns.squeeze(1)
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
                return_flat - value_flat,
                unbiased=False,
            )

            explained_variance = (
                1.0
                -
                residual_variance
                /
                (
                    return_variance
                    + 1e-8
                )
            )

            # =================================================
            # Total loss
            #
            # L =
            #     L_actor
            #     + VALUE_COEF * L_critic
            #     - ENTROPY_COEF * H
            #     + L_DKL
            # =================================================

            loss = (
                actor_loss                          # PPO policy
                + VALUE_COEF * critic_loss          # Value (avec clipping optionnel)
                - ENTROPY_COEF * entropy            # Exploration bonus
                + dkl_loss                          # DKL(RL || BC)², modulée par lambda(epoch)/2
            )

            # =================================================
            # Pure PPO actor gradient
            # =================================================

            actor_gradients = torch.autograd.grad(
                actor_loss,
                model.policy.parameters(),
                retain_graph=True,
                allow_unused=True,
            )

            actor_grad_sq = 0.0

            for grad in actor_gradients:

                if grad is None:
                    continue

                actor_grad_sq += (
                    grad.detach().norm(2).item() ** 2
                )

            actor_grad_norm = (
                actor_grad_sq ** 0.5
            )

            # =================================================
            # Pure critic gradient
            # =================================================

            critic_gradients = torch.autograd.grad(
                critic_loss,
                model.value.parameters(),
                retain_graph=True,
                allow_unused=True,
            )

            critic_grad_sq = 0.0

            for grad in critic_gradients:

                if grad is None:
                    continue

                critic_grad_sq += (
                    grad.detach().norm(2).item() ** 2
                )

            critic_grad_norm = (
                critic_grad_sq ** 0.5
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
            # Accumulate
            # =================================================

            total_loss += loss.item()

            total_actor += actor_loss.item()

            total_critic += critic_loss.item()

            total_kl += approx_kl.item()

            total_entropy += entropy.item()

            total_clip_fraction += (
                clip_fraction.item()
            )

            total_actor_grad_norm += (
                actor_grad_norm
            )

            total_critic_grad_norm += (
                critic_grad_norm
            )

            # =================================================
            # NEW: DKL gradient accumulation
            # =================================================

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

            # =================================================
            # DKL diagnostics
            # =================================================

            total_dkl += (
                delta_dkl.item()
            )

            total_dkl_loss += (
                dkl_loss.item()
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

    # ========================================================
    # NEW: Average DKL gradient norm
    # ========================================================

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

    # ========================================================
    # Diagnostics
    # ========================================================

    print()

    print(
        "======================================"
    )

    print(
        "PPO DIAGNOSTICS"
    )

    print(
        "======================================"
    )

    print(
        f"Epoch:                 "
        f"{epoch}"
    )

    print(
        f"Self-play temperature: "
        f"{TEMPERATURE_SELFPLAY:.2f}"
    )

    print(
        f"BC prior strength:     "
        f"{BC_PRIOR_STRENGTH:.2f}"
    )

    print(
        f"BC prior decay ply:    "
        f"{BC_PRIOR_PLIES:.1f}"
    )

    print(
        f"Advantage mean:        "
        f"{avg_adv_mean:+.6f}"
    )

    print(
        f"Advantage std:         "
        f"{avg_adv_std:.6f}"
    )

    print(
        f"Return mean:           "
        f"{avg_return_mean:+.6f}"
    )

    print(
        f"Return std:            "
        f"{avg_return_std:.6f}"
    )

    print(
        f"Value mean:            "
        f"{avg_value_mean:+.6f}"
    )

    print(
        f"Value std:             "
        f"{avg_value_std:.6f}"
    )

    print(
        f"Critic MSE:            "
        f"{avg_critic:.6f}"
    )

    print(
        f"Explained variance:    "
        f"{avg_explained_variance:+.6f}"
    )

    print(
        f"Actor gradient norm:   "
        f"{avg_actor_grad_norm:.6e}"
    )

    print(
        f"Critic gradient norm:  "
        f"{avg_critic_grad_norm:.6e}"
    )

    print(
        f"DKL gradient norm:     "
        f"{avg_dkl_grad_norm:.6e}"
    )

    print(
        f"Policy KL:             "
        f"{avg_kl:.6e}"
    )

    print(
        f"Clip fraction:         "
        f"{avg_clip_fraction:.2%}"
    )

    print(
        f"Entropy:               "
        f"{avg_entropy:.6f}"
    )

    print(
        "--------------------------------------"
    )

    print(
        f"DKL(RL || BC):         "
        f"{avg_dkl:.6e}"
    )

    print(
        f"DKL lambda:            "
        f"{lambda_dkl.item():.6e}"
    )

    print(
        f"DKL loss:              "
        f"{avg_dkl_loss:.6e}"
    )

    print(
        "======================================"
    )

    return (
        avg_loss,
        avg_actor,
        avg_critic,
        avg_kl,
        avg_dkl,
        avg_dkl_loss,
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
        / "rl_epoch"
        / f"rl_epoch_{epoch}.pt"
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
# Replay Buffer
# ============================================================

def save_replay_buffer(
    buffer,
    epoch,
):

    path = (
        PROJECT_ROOT
        / "checkpoints"
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
        "Replay buffer saved:",
        path,
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load model + optimizer + league + bc_model
    # ========================================================

    model, optimizer, league = (
        load_model()
    )

    bc_model = load_bc_agent(7)

    bc_model_selfplay = copy.deepcopy(
        bc_model
    ).to("cpu")

    bc_model_selfplay.eval()

    # ========================================================
    # Replay buffer
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
        # RL loop
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
                f"===== Epoch {epoch} =====",
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
                    compute_gae(
                        trajectory,
                        rewards,
                        gamma=GAMMA,
                        gae_lambda=GAE_LAMBDA,
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
                    )

            # =================================================
            # Stats
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

            # =================================================
            # PPO
            # =================================================

            (
                loss,
                actor_loss,
                critic_loss,
                approx_kl,
                dkl,
                dkl_loss
            ) = train_epoch(
                model,
                optimizer,
                buffer,
                bc_model,
                epoch,
            )

            print(
                f"Loss={loss:.4f} "
                f"| Actor={actor_loss:.4f} "
                f"| Critic={critic_loss:.4f} "
                f"| KL={approx_kl:.6f} "
                f"| DKL(RL||BC)={dkl:.6f}",
                f"| DKL loss={dkl_loss:.6e}",
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
                "Replay buffer cleared after PPO update.",
                flush=True,
            )

            # =================================================
            # Uncertainty stats
            # =================================================

            stats_path = (
                PROJECT_ROOT
                / "checkpoints"
                / "uncertainty_stats.json"
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
                    / "rl_best.pt",
                )

                print(
                    "New best checkpoint saved.",
                    flush=True,
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
                f"{get_dkl_lambda(epoch).item():.6e}",
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

    manager.shutdown()

    print(
        "\nRL training finished.",
        flush=True,
    )


# ============================================================

if __name__ == "__main__":

    main()