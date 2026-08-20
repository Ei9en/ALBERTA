# Train_RL.py


### Imports ###

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
from src.selfplay.game import SelfPlayGame

from src.rl.replay_buffer import ReplayBuffer
from src.rl.uncertainty_stats import UncertaintyStats

from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX


### Constants ###

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/ALBERTA"
)


# ============================================================
# Reprise RL
# ============================================================

START_EPOCH = 24

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

LR = 1e-4

GAMES_PER_EPOCH = 900

RL_EPOCHS = 10

CHECKPOINT_EVERY = 1

VALUE_COEF = 0.1

BATCH_SIZE = 4096

SGD_EPOCHS = 3

GAMMA = 0.99

GAE_LAMBDA = 0.95


# ============================================================
# League
# ============================================================

LEAGUE_MAX_AGENTS = 22

LEAGUE_START_EPOCH = 4

LEAGUE_END_EPOCH = 23


# ============================================================
# Model Loading
# ============================================================

def load_bc_agent(
    epoch,
):

    path = (
        PROJECT_ROOT
        / "checkpoints"
        / "bc_epoch"
        / f"bc_v2_5_epoch_{epoch}.pt"
    )


    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
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


def load_league_agent(
    epoch,
):

    name = (
        f"league_epoch_{epoch:03d}"
    )


    path = (
        LEAGUE_DIR
        / f"{name}.pt"
    )


    if not path.exists():

        raise FileNotFoundError(
            f"League snapshot not found: {path}"
        )


    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )


    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
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

CHECKPOINT_EPOCH = 20

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / f"rl_epoch_{CHECKPOINT_EPOCH}.pt"
)


def load_model():

    print()
    print("======================================")
    print("Loading RL checkpoint")
    print("======================================")
    print(
        f"Checkpoint: {CHECKPOINT}"
    )


    #
    # Architecture
    #

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )


    model = ActorCritic(
        bc_model
    )


    model = model.to(DEVICE)


    #
    # Charger checkpoint RL
    #

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    #
    # Optimizer
    #

    optimizer = Adam(
        model.parameters(),
        lr=LR,
    )


    #
    # Reprendre l'état optimizer
    #

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


    #
    # League
    #

    league = League(
        max_agents=LEAGUE_MAX_AGENTS
    )


    #
    # BC baselines
    #

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


    #
    # Snapshots RL 4 -> 23
    #

    for epoch in range(
        LEAGUE_START_EPOCH,
        LEAGUE_END_EPOCH + 1,
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
            channels=64,
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
        f"League loaded: "
        f"{len(league)} agents"
    )


    print(
        "League members:"
    )


    for name in league.names():

        print(
            f"  {name}"
        )


    return model, optimizer, league


### Self-play Collection ###


_WORKER_CURRENT_MODEL = None

_WORKER_LEAGUE_MODELS = None

_WORKER_CURRENT_AGENT = None

_WORKER_LEAGUE_AGENTS = None

_WORKER_LEAGUE_REGISTRY = None


# ============================================================
# Préparation modèle partagé
# ============================================================

def _prepare_shared_model(
    model,
):

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
        temperature=0.75,
        device="cpu",
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
                temperature=0.75,
                device="cpu",
            )
        )


# ============================================================
# Worker self-play
# ============================================================

def _selfplay_worker(
    worker_args,
):

    (
        n_games,
        worker_id,
        batch_size,
    ) = worker_args


    global _WORKER_CURRENT_AGENT
    global _WORKER_LEAGUE_MODELS
    global _WORKER_LEAGUE_AGENTS
    global _WORKER_LEAGUE_REGISTRY


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
                    temperature=0.75,
                    device="cpu",
                )
            )


    seed = (
        1000003
        + worker_id * 7919
        + random.randrange(
            100000000
        )
    )


    random.seed(seed)

    torch.manual_seed(seed)


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


    print(
        f"[WORKER {worker_id}] DONE "
        f"({len(completed_games)} games)",
        flush=True,
    )


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


    #
    # ========================================================
    # Calcul U
    # ========================================================
    #

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


        x = encode_boards(
            boards
        ).to(DEVICE)


        uncertainties = (
            league.uncertainty_batch(
                x,
                current_model=model,
            )
        )


        for (
            step,
            U,
        ) in zip(
            all_steps,
            uncertainties,
        ):

            U = U.item()


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


    return completed_games


### Fixed Evaluation ###

def evaluate_against_agent(
    model,
    opponent,
    n_games=100,
):

    model.eval()

    opponent.eval()


    wins = 0

    losses = 0

    draws = 0

    total_positions = 0


    start_time = time.time()


    with torch.no_grad():

        for i in tqdm(
            range(n_games),
            desc="Evaluation",
            leave=False,
        ):

            current_agent = PPOAgent(
                model,
                deterministic=False,
                temperature=0.75,
                device=DEVICE,
            )


            opponent_agent = PPOAgent(
                opponent,
                deterministic=False,
                temperature=0.75,
                device=DEVICE,
            )


            if i % 2 == 0:

                white_agent = current_agent

                black_agent = opponent_agent

                current_is_white = True

            else:

                white_agent = opponent_agent

                black_agent = current_agent

                current_is_white = False


            game = SelfPlayGame(
                white_agent,
                black_agent,
            )


            trajectory, result = (
                game.play()
            )


            total_positions += len(
                trajectory
            )


            if result == "1-0":

                if current_is_white:

                    wins += 1

                else:

                    losses += 1


            elif result == "0-1":

                if current_is_white:

                    losses += 1

                else:

                    wins += 1


            else:

                draws += 1


    elapsed = (
        time.time()
        - start_time
    )


    winrate = (
        wins / n_games
    )


    print(
        f"Evaluation time: "
        f"{elapsed:.2f}s "
        f"({elapsed / n_games:.2f}s/game)"
    )


    print(
        f"Evaluation positions: "
        f"{total_positions} "
        f"({total_positions / n_games:.1f}/game)"
    )


    print(
        f"Evaluation: "
        f"W={wins} "
        f"L={losses} "
        f"D={draws} "
        f"Win rate={winrate:.1%}"
    )


    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": winrate,
        "positions": total_positions,
        "time": elapsed,
    }


### GAE ###

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


### Training ###

def train_epoch(
    model,
    optimizer,
    buffer,
):

    model.train()


    #
    # Freeze BatchNorm
    #

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
        )


    TRAIN_STEPS = (
        len(buffer)
        // BATCH_SIZE
    )


    PPO_CLIP = 0.2

    ENTROPY_COEF = 0.01


    total_loss = 0.0

    total_actor = 0.0

    total_critic = 0.0


    total_updates = (
        TRAIN_STEPS
        * SGD_EPOCHS
    )


    progress = tqdm(
        total=total_updates,
        desc="PPO Training",
    )


    for _ in range(
        SGD_EPOCHS
    ):

        for _ in range(
            TRAIN_STEPS
        ):

            batch = buffer.sample(
                BATCH_SIZE
            )


            #
            # Encode boards
            #

            boards = [
                chess.variant.AtomicBoard(
                    s["fen"]
                )
                for s in batch
            ]


            x = encode_boards(
                boards
            ).to(DEVICE)


            #
            # Forward
            #

            policy, values = model(x)


            #
            # Returns
            #

            returns = torch.tensor(
                [
                    s["return"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            ).unsqueeze(1)


            #
            # Advantages
            #

            advantages = torch.tensor(
                [
                    s["advantage"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            )


            #
            # Normalisation
            #

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-8
            )


            #
            # Old log probabilities
            #

            old_log_probs = torch.tensor(
                [
                    s["old_log_prob"]
                    for s in batch
                ],
                device=DEVICE,
                dtype=torch.float32,
            )


            #
            # Legal mask
            #

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


            #
            # Mask logits
            #

            legal_logits = (
                policy.masked_fill(
                    ~legal_mask,
                    float("-inf"),
                )
            )


            #
            # Même température que self-play
            #

            ppo_logits = (
                legal_logits / 0.75
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


            #
            # PPO ratio
            #

            ratio = torch.exp(
                selected_log_probs
                - old_log_probs
            )


            #
            # PPO objective
            #

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


            #
            # Entropy
            #

            probs = torch.softmax(
                ppo_logits,
                dim=1,
            )


            entropy = -(
                probs
                *
                log_probs.masked_fill(
                    ~legal_mask,
                    0.0,
                )
            ).sum(
                dim=1
            ).mean()


            #
            # Critic
            #

            critic_loss = F.mse_loss(
                values,
                returns,
            )


            #
            # Total loss
            #

            loss = (
                actor_loss
                +
                VALUE_COEF
                * critic_loss
                -
                ENTROPY_COEF
                * entropy
            )


            optimizer.zero_grad()


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )


            optimizer.step()


            total_loss += (
                loss.item()
            )

            total_actor += (
                actor_loss.item()
            )

            total_critic += (
                critic_loss.item()
            )


            progress.update(1)


    progress.close()


    return (
        total_loss / total_updates,
        total_actor / total_updates,
        total_critic / total_updates,
    )


### Checkpoints ###

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


### Saving Replay Buffer ###

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

    #
    # Load model + optimizer + league
    #

    model, optimizer, league = (
        load_model()
    )


    #
    # Replay buffer
    #

    buffer = ReplayBuffer(
        capacity=100000
    )


    stats = UncertaintyStats()


    best_loss = None


    #
    # Parallel self-play
    #

    NUM_WORKERS = 12

    SELFPLAY_BATCH_SIZE = 256


    #
    # Shared current model
    #

    print(
        "\nPreparing shared CPU models...",
        flush=True,
    )


    shared_current_model = (
        _prepare_shared_model(
            model
        )
    )


    #
    # Shared league models
    #

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


    #
    # Préparer futurs slots
    #

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


    #
    # Multiprocessing
    #

    ctx = mp.get_context(
        "spawn"
    )


    manager = ctx.Manager()


    #
    # Registry
    #

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


    #
    # Pool
    #

    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=_init_selfplay_worker,
        initargs=(
            shared_current_model,
            shared_league_models,
            league_registry,
        ),
    ) as pool:

        #
        # RL loop
        #

        for epoch in range(
            START_EPOCH,
            START_EPOCH + RL_EPOCHS,
        ):

            print(
                f"\n======================================",
                flush=True,
            )


            print(
                f"===== Epoch {epoch} =====",
                flush=True,
            )


            print(
                f"======================================",
                flush=True,
            )


            wins = 0

            losses = 0

            draws = 0


            #
            # Self-play
            #

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


            #
            # Construction replay buffer
            #

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


                #
                # Résultat
                #

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


                #
                # Rewards
                #

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


                #
                # GAE
                #

                advantages, returns = (
                    compute_gae(
                        trajectory,
                        rewards,
                        gamma=GAMMA,
                        gae_lambda=GAE_LAMBDA,
                    )
                )


                #
                # Replay
                #

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
                    )


            #
            # Stats
            #

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


            #
            # PPO
            #

            loss, actor_loss, critic_loss = (
                train_epoch(
                    model,
                    optimizer,
                    buffer,
                )
            )


            print(
                f"Loss={loss:.4f} "
                f"| Actor={actor_loss:.4f} "
                f"| Critic={critic_loss:.4f}",
                flush=True,
            )


            #
            # Replay buffer sauvegarde
            #

            if epoch % 5 == 0:

                save_replay_buffer(
                    buffer,
                    epoch,
                )

            #
            # On-policy
            #

            buffer.clear()


            print(
                "Replay buffer cleared after PPO update.",
                flush=True,
            )

            #
            # Uncertainty stats
            #

            stats.save(
                PROJECT_ROOT
                / "checkpoints"
                / "uncertainty_stats.json"
            )


            #
            # RL checkpoint
            #

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


            #
            # Snapshot league
            #

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


            #
            # Sauvegarde
            #

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


            #
            # Shared snapshot
            #

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


            #
            # Registry
            #

            league_registry[:] = (
                league.names()
            )


            print(
                "Updated league registry:",
                list(league_registry),
                flush=True,
            )


            #
            # Shared current model
            #

            for (
                key,
                value,
            ) in model.state_dict().items():

                shared_current_model.state_dict()[
                    key
                ].copy_(
                    value.detach().cpu()
                )


            #
            # Best checkpoint
            #

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


            #
            # Summary
            #

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