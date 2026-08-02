# Train_RL.py

### Imports ###

import pickle
import time
import chess
import chess.variant

from src.selfplay.league import League
import copy

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.encoding import encode_boards

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame

from src.rl.compute_returns import compute_returns
from src.rl.replay_buffer import ReplayBuffer
from src.rl.uncertainty_stats import UncertaintyStats

from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX

### Constants ###

PROJECT_ROOT = Path("/content/drive/MyDrive/ALBERTA")

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_v2_5_epoch_5.pt" # Agent courant
)

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

LR = 5e-5

GAMES_PER_EPOCH = 600

RL_EPOCHS = 20

CHECKPOINT_EVERY = 1

VALUE_COEF = 0.1

BATCH_SIZE = 2048

SGD_EPOCHS = 1 # Nombre de passages complets sur le replay buffer pendant un epoch RL.
               # Plus élevé = plus d'updates par collecte de parties, mais risque de sur-apprentissage
               # sur les anciennes expériences.

def model_checksum(model):
    total = 0.0

    with torch.no_grad():
        for param in model.parameters():
            total += param.detach().float().sum().item()

    return total

### Model Loading ###

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

def load_model():

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    bc_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = ActorCritic(bc_model)

    model = model.to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=LR,
    )

    #
    # Initialisation de la league
    #

    league = League()


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

    return model, optimizer, league

### Self-play Collection ###
# Batched version

# ============================================================
# Parallel self-play
# ============================================================

import multiprocessing as mp
import random


#
# Modèles accessibles par les workers.
#
_WORKER_CURRENT_MODEL = None
_WORKER_LEAGUE_MODELS = None
_WORKER_CURRENT_AGENT = None
_WORKER_LEAGUE_AGENTS = None


# ============================================================
# Préparation des modèles CPU partagés
# ============================================================

def _prepare_shared_model(model):
    """
    Crée une copie CPU du modèle et place ses paramètres
    en mémoire partagée.

    Une seule copie physique est ainsi utilisée par les
    différents processus workers.
    """

    cpu_model = copy.deepcopy(
        model
    ).to("cpu")

    cpu_model.eval()

    cpu_model.share_memory()

    return cpu_model


# ============================================================
# Initialisation d'un worker
# ============================================================

def _init_selfplay_worker(
    current_model,
    league_models,
    league_registry,
):
    """
    Initialise un worker multiprocessing.

    Les modèles sont en mémoire CPU partagée.
    league_registry contient les noms des modèles
    actuellement disponibles.
    """

    global _WORKER_CURRENT_MODEL
    global _WORKER_LEAGUE_MODELS
    global _WORKER_CURRENT_AGENT
    global _WORKER_LEAGUE_AGENTS
    global _WORKER_LEAGUE_REGISTRY

    print(
        "[WORKER INIT] done",
        flush=True
    )

    #
    # Un seul thread PyTorch par worker.
    #
    torch.set_num_threads(1)

    #
    # Modèle courant
    #
    _WORKER_CURRENT_MODEL = current_model
    _WORKER_CURRENT_MODEL.eval()

    #
    # Modèles league
    #
    _WORKER_LEAGUE_MODELS = league_models

    for model in _WORKER_LEAGUE_MODELS.values():
        model.eval()

    #
    # Registry
    #
    _WORKER_LEAGUE_REGISTRY = league_registry

    #
    # Agent courant
    #
    _WORKER_CURRENT_AGENT = ActorCriticAgent(
        _WORKER_CURRENT_MODEL,
        deterministic=False,
        temperature=0.75,
        device="cpu",
    )

    #
    # Agents league
    #
    _WORKER_LEAGUE_AGENTS = {}

    for name, model in _WORKER_LEAGUE_MODELS.items():

        _WORKER_LEAGUE_AGENTS[name] = ActorCriticAgent(
            model,
            deterministic=False,
            temperature=0.75,
            device="cpu",
        )

    print(
        f"[WORKER INIT] "
        f"{len(_WORKER_LEAGUE_AGENTS)} league agents ready",
        flush=True
    )

# ============================================================
# Worker : self-play
# ============================================================

def _selfplay_worker(
    worker_args,
):
    """
    Exécute les parties attribuées à un worker.

    Retourne exactement le même format que
    collect_games_batched().
    """

    (
        n_games,
        worker_id,
        batch_size,
    ) = worker_args

    print(
        f"[WORKER {worker_id}] START "
        f"({n_games} games)",
        flush=True
    )

    global _WORKER_CURRENT_MODEL
    global _WORKER_CURRENT_AGENT
    global _WORKER_LEAGUE_MODELS
    global _WORKER_LEAGUE_AGENTS
    global _WORKER_LEAGUE_REGISTRY

    #
    # ========================================================
    # Synchronisation légère de la league
    # ========================================================
    #

    for name in _WORKER_LEAGUE_REGISTRY:

        if name not in _WORKER_LEAGUE_AGENTS:

            model = _WORKER_LEAGUE_MODELS[name]

            model.eval()

            _WORKER_LEAGUE_AGENTS[name] = ActorCriticAgent(
                model,
                deterministic=False,
                temperature=0.75,
                device="cpu",
            )

    #
    # ========================================================
    # Seed
    # ========================================================
    #

    seed = (
        1000003
        + worker_id * 7919
        + random.randrange(100000000)
    )

    random.seed(seed)
    torch.manual_seed(seed)

    current_agent = _WORKER_CURRENT_AGENT

    #
    # ========================================================
    # Checksum
    # ========================================================
    #

    model_checksum = sum(
        p.detach().float().sum().item()
        for p in _WORKER_CURRENT_MODEL.parameters()
    )

    print(
        f"[WORKER {worker_id}] "
        f"current model checksum = "
        f"{model_checksum:.10f}",
        flush=True
    )

    league_agents = _WORKER_LEAGUE_AGENTS

    opponent_names = list(
        league_agents.keys()
    )

    #
    # ========================================================
    # Initialisation des parties
    # ========================================================
    #

    active_games = []
    completed_games = []

    for i in range(n_games):

        opponent_name = random.choice(
            opponent_names
        )

        opponent_agent = (
            league_agents[opponent_name]
        )

        #
        # Alterner les couleurs.
        #
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

    #
    # ========================================================
    # Self-play
    # ========================================================
    #

    with torch.no_grad():

        while active_games:

            model_games = []
            opponent_games = []

            for game in active_games:

                board = game["board"]

                agent = (
                    game["white"]
                    if board.turn
                    else game["black"]
                )

                if agent is current_agent:

                    model_games.append(game)

                else:

                    opponent_games.append(game)

            #
            # =================================================
            # Coups du modèle courant
            # =================================================
            #

            for start in range(
                0,
                len(model_games),
                batch_size,
            ):

                batch_games = model_games[
                    start:start + batch_size
                ]

                if not batch_games:
                    continue

                boards = [
                    game["board"]
                    for game in batch_games
                ]

                infos = current_agent.choose_moves(
                    boards
                )

                for game, info in zip(
                    batch_games,
                    infos,
                ):

                    board = game["board"]

                    game["trajectory"].append(
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

            #
            # =================================================
            # Coups des adversaires
            # =================================================
            #

            for game in opponent_games:

                board = game["board"]

                agent = (
                    game["white"]
                    if board.turn
                    else game["black"]
                )

                info = agent.choose_move(
                    board
                )

                game["trajectory"].append(
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

            #
            # =================================================
            # Vérification
            # =================================================
            #

            still_active = []

            for game in active_games:

                board = game["board"]

                if board.is_game_over():

                    completed_games.append(
                        {
                            "trajectory":
                                game["trajectory"],

                            "result":
                                board.result(),

                            "current_white":
                                game[
                                    "current_white"
                                ],
                        }
                    )

                else:

                    still_active.append(game)

            active_games = still_active

    print(
        f"[WORKER {worker_id}] DONE "
        f"({len(completed_games)} games)",
        flush=True
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

    #
    # ========================================================
    # Nombre de workers réellement utilisés
    # ========================================================
    #

    num_workers = min(
        num_workers,
        n_games,
    )

    #
    # ========================================================
    # Répartition en petits lots
    #
    # IMPORTANT :
    #
    # On ne donne plus 50 parties d'un coup à chaque worker.
    #
    # Les processus restent vivants et conservent leurs modèles,
    # mais plusieurs tâches successives leur sont distribuées.
    #
    # Cela permet à tqdm de recevoir régulièrement des résultats.
    # ========================================================
    #

    games_per_task = max(
        5,
        n_games // (num_workers * 4),
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

    #
    # ========================================================
    # Self-play
    # ========================================================
    #

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

    #
    # ========================================================
    # Vérification
    # ========================================================
    #

    if len(completed_games) != n_games:

        raise RuntimeError(
            f"Nombre de parties incorrect : "
            f"{len(completed_games)} / {n_games}"
        )

    #
    # ========================================================
    # Statistiques self-play
    # ========================================================
    #

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

        for step, U in zip(
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
    """
    Évalue le modèle courant contre un snapshot fixe.

    - n_games parties
    - couleurs alternées
    - aucun impact sur replay buffer / league
    """

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

            current_agent = ActorCriticAgent(
                model,
                deterministic=False,
                temperature=0.75,
                device=DEVICE,
            )

            opponent_agent = ActorCriticAgent(
                opponent,
                deterministic=False,
                temperature=0.75,
                device=DEVICE,
            )

            #
            # Alterner les couleurs
            #
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

            trajectory, result = game.play()

            total_positions += len(trajectory)

            #
            # Résultat du modèle courant
            #
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

    elapsed = time.time() - start_time

    winrate = wins / n_games

    print(
        f"Evaluation time: {elapsed:.2f}s "
        f"({elapsed / n_games:.2f}s/game)"
    )

    print(
        f"Evaluation positions: {total_positions} "
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

### Training ###

def train_epoch(
    model,
    optimizer,
    buffer,
):

    model.train()

    #
    # Freeze BatchNorm statistics
    #
    for module in model.modules():

        if isinstance(
            module,
            torch.nn.BatchNorm2d
        ):
            module.eval()


    if len(buffer) < BATCH_SIZE:

        print(
            "Replay buffer too small for training."
        )

        return 0.0, 0.0, 0.0


    TRAIN_STEPS = (
        len(buffer)
        // BATCH_SIZE
    )


    total_loss = 0.0
    total_actor = 0.0
    total_critic = 0.0


    total_updates = (
        TRAIN_STEPS
        * SGD_EPOCHS
    )


    #
    # =========================
    # Diagnostics
    # =========================
    #

    max_advantage = 0.0
    mean_abs_advantage = 0.0

    max_log_prob = 0.0
    mean_abs_log_prob = 0.0

    diagnostic_samples = 0


    progress = tqdm(
        total=total_updates,
        desc="Training",
    )


    #
    # =========================
    # SGD epochs
    # =========================
    #

    for epoch in range(
        SGD_EPOCHS
    ):

        for update in range(
            TRAIN_STEPS
        ):

            batch = buffer.sample(
                BATCH_SIZE
            )


            #
            # =========================
            # Boards
            # =========================
            #

            boards = [
                chess.variant.AtomicBoard(
                    step["fen"]
                )
                for step in batch
            ]


            #
            # =========================
            # Vectorized encoding
            # =========================
            #

            x = encode_boards(
                boards
            ).to(DEVICE)


            #
            # =========================
            # Forward pass
            # =========================
            #

            optimizer.zero_grad()

            policy, value = model(x)


            #
            # =========================
            # Targets
            # =========================
            #

            target = torch.tensor(
                [
                    step["return"]
                    for step in batch
                ],
                dtype=torch.float32,
                device=DEVICE,
            ).view(-1, 1)


            #
            # =========================
            # Advantage
            # =========================
            #

            raw_advantage = (
                target
                - value.detach()
            )


            #
            # Diagnostics
            # =========================
            #

            batch_mean_abs_advantage = (
                raw_advantage
                .abs()
                .mean()
                .item()
            )

            batch_max_advantage = (
                raw_advantage
                .abs()
                .max()
                .item()
            )


            mean_abs_advantage += (
                batch_mean_abs_advantage
                * BATCH_SIZE
            )

            max_advantage = max(
                max_advantage,
                batch_max_advantage,
            )


            #
            # =========================
            # Legal move mask
            # =========================
            #

            legal_mask = torch.zeros(
                (
                    BATCH_SIZE,
                    policy.shape[1],
                ),
                dtype=torch.bool,
                device=DEVICE,
            )


            actions = torch.empty(
                BATCH_SIZE,
                dtype=torch.long,
                device=DEVICE,
            )


            for i, step in enumerate(batch):

                legal_indices = [
                    ACTION_TO_INDEX[m]
                    for m in step["legal_moves"]
                ]

                legal_mask[
                    i,
                    legal_indices
                ] = True

                actions[i] = step["action"]


            #
            # =========================
            # Mask illegal moves
            # =========================
            #

            legal_logits = policy.masked_fill(
                ~legal_mask,
                float("-inf"),
            )


            #
            # =========================
            # Log probabilities
            # =========================
            #

            log_probs = F.log_softmax(
                legal_logits,
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
            # =========================
            # Log probability diagnostics
            # =========================
            #

            batch_mean_abs_log_prob = (
                selected_log_probs
                .abs()
                .mean()
                .item()
            )

            batch_max_log_prob = (
                selected_log_probs
                .abs()
                .max()
                .item()
            )


            mean_abs_log_prob += (
                batch_mean_abs_log_prob
                * BATCH_SIZE
            )

            max_log_prob = max(
                max_log_prob,
                batch_max_log_prob,
            )


            diagnostic_samples += (
                BATCH_SIZE
            )


            #
            # =========================
            # Advantage clipping
            # =========================
            #

            advantage = torch.clamp(
                raw_advantage,
                -5,
                5,
            ).squeeze(1)


            #
            # =========================
            # Actor loss
            # =========================
            #

            actor_loss = (
                -selected_log_probs
                * advantage
            ).mean()


            #
            # =========================
            # Critic loss
            # =========================
            #

            critic_loss = F.mse_loss(
                value,
                target,
            )


            #
            # =========================
            # Total loss
            # =========================
            #

            loss = (
                actor_loss
                +
                VALUE_COEF
                * critic_loss
            )


            #
            # =========================
            # Backpropagation
            # =========================
            #

            loss.backward()


            #
            # =========================
            # Gradient clipping
            # =========================
            #

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
            )


            #
            # =========================
            # First update diagnostics
            # =========================
            #

            if (
                epoch == 0
                and update == 0
            ):

                print(
                    "\nRL diagnostics:"
                )

                print(
                    f"Raw advantage:"
                    f" mean|A|="
                    f"{batch_mean_abs_advantage:.4f}"
                    f" | max|A|="
                    f"{batch_max_advantage:.4f}"
                )

                print(
                    f"Selected logπ:"
                    f" mean|logπ|="
                    f"{batch_mean_abs_log_prob:.4f}"
                    f" | max|logπ|="
                    f"{batch_max_log_prob:.4f}"
                )

                print(
                    f"Gradient norm "
                    f"before clipping: "
                    f"{grad_norm.item():.4f}"
                )


            #
            # =========================
            # Optimizer
            # =========================
            #

            optimizer.step()


            #
            # =========================
            # Statistics
            # =========================
            #

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


    #
    # =========================
    # Final diagnostics
    # =========================
    #

    print(
        "\nRL diagnostics:"
        f" mean|A|="
        f"{mean_abs_advantage / diagnostic_samples:.4f}"
        f" | max|A|="
        f"{max_advantage:.4f}"
        f" | mean|logπ|="
        f"{mean_abs_log_prob / diagnostic_samples:.4f}"
        f" | max|logπ|="
        f"{max_log_prob:.4f}"
    )


    return (
        total_loss
        / total_updates,

        total_actor
        / total_updates,

        total_critic
        / total_updates,
    )

### Checkpoints ###

def save_checkpoint(model, optimizer, epoch, loss):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        PROJECT_ROOT
        / "checkpoints"
        / "rl_epoch"
        / f"rl_epoch_{epoch}.pt",
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
        path
    )

# Main Loop

def main():

    model, optimizer, league = load_model()

    buffer = ReplayBuffer(
        capacity=100000
    )

    stats = UncertaintyStats()

    best_loss = None

    NUM_WORKERS = 12
    SELFPLAY_BATCH_SIZE = 256

    #
    # ========================================================
    # Multiprocessing context
    # ========================================================
    #

    ctx = mp.get_context(
        "spawn"
    )

    manager = ctx.Manager()

    #
    # ========================================================
    # Modèles CPU partagés — créés UNE SEULE FOIS
    # ========================================================
    #

    print(
        "Preparing shared CPU models..."
    )

    #
    # Modèle courant
    #
    shared_current_model, _ = (
        _prepare_shared_model(
            model
        )
    )

    #
    # Modèles league
    #
    shared_league_models = {}

    for name, league_model in (
        league.agents.items()
    ):

        shared_league_models[name], _ = (
            _prepare_shared_model(
                league_model
            )
        )

    #
    # Registry partagé
    #
    league_registry = manager.list(
        league.agents.keys()
    )

    print(
        f"Shared models ready: "
        f"{len(shared_league_models)} league agents"
    )

    #
    # ========================================================
    # Pool — créé UNE SEULE FOIS
    # ========================================================
    #

    print(
        f"Starting {NUM_WORKERS} "
        f"self-play workers..."
    )

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
        # ====================================================
        # RL loop
        # ====================================================
        #

        for epoch in range(RL_EPOCHS):

            print(
                f"\n===== Epoch {epoch} ====="
            )

            wins = 0
            losses = 0
            draws = 0

            #
            # =================================================
            # Self-play
            # =================================================
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
            # =================================================
            # Replay buffer
            # =================================================
            #

            for game in games:

                trajectory = game["trajectory"]
                result = game["result"]
                current_white = game["current_white"]

                if result == "1-0":

                    if current_white:
                        wins += 1
                    else:
                        losses += 1

                    white_reward = 1.0
                    black_reward = -1.0

                elif result == "0-1":

                    if current_white:
                        losses += 1
                    else:
                        wins += 1

                    white_reward = -1.0
                    black_reward = 1.0

                else:

                    draws += 1

                    white_reward = 0.0
                    black_reward = 0.0

                rewards = []

                for step in trajectory:

                    if step["player"]:
                        rewards.append(
                            white_reward
                        )
                    else:
                        rewards.append(
                            black_reward
                        )

                returns = compute_returns(
                    rewards,
                    gamma=0.99,
                )

                returns = torch.as_tensor(
                    returns,
                    dtype=torch.float32,
                )

                returns = (
                    returns - returns.mean()
                ) / (
                    returns.std() + 1e-8
                )

                for step, ret in zip(
                    trajectory,
                    returns,
                ):

                    buffer.add(
                        step["fen"],
                        step["action"],
                        step["legal_moves"],
                        ret,
                    )

            #
            # =================================================
            # Score
            # =================================================
            #

            selfplay_score_rate = (
                wins
                + 0.5 * draws
            ) / GAMES_PER_EPOCH

            print(
                f"Replay buffer size: {len(buffer)}"
            )

            print(
                f"Results: W={wins} "
                f"L={losses} "
                f"D={draws} "
                f"Score rate={selfplay_score_rate:.1%}"
            )

            #
            # =================================================
            # Training
            # =================================================
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
                f"| Critic={critic_loss:.4f}"
            )

            #
            # =================================================
            # Sauvegardes
            # =================================================
            #

            if epoch % 5 == 0:

                print(
                    ">>> BEFORE replay save"
                )

                save_replay_buffer(
                    buffer,
                    epoch,
                )

            print(
                ">>> BEFORE stats save"
            )

            stats.save(
                PROJECT_ROOT
                / "checkpoints"
                / "uncertainty_stats.json"
            )

            if epoch % CHECKPOINT_EVERY == 0:

                print(
                    ">>> BEFORE RL checkpoint"
                )

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    loss,
                )

            #
            # =================================================
            # League snapshot
            # =================================================
            #

            print(
                ">>> BEFORE league deepcopy"
            )

            snapshot = copy.deepcopy(
                model
            ).to(DEVICE)

            snapshot.eval()

            agent_name = (
                f"league_epoch_{epoch:03d}"
            )

            league.add_agent(
                agent_name,
                snapshot,
            )

            #
            # =================================================
            # Ajouter le nouveau snapshot à la mémoire
            # partagée
            # =================================================
            #

            print(
                ">>> Preparing shared league snapshot"
            )

            shared_snapshot, _ = (
                _prepare_shared_model(
                    snapshot
                )
            )

            shared_league_models[
                agent_name
            ] = shared_snapshot

            #
            # Ajouter son nom au registry partagé.
            #
            # IMPORTANT :
            # le registry est partagé entre les workers.
            #
            #

            if agent_name not in league_registry:

                league_registry.append(
                    agent_name
                )

            print(
                f">>> Shared league updated: "
                f"{agent_name}"
            )

            #
            # =================================================
            # Sauvegarde league
            # =================================================
            #

            print(
                ">>> BEFORE league save"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        snapshot.state_dict(),
                },
                PROJECT_ROOT
                / "checkpoints"
                / "league"
                / f"{agent_name}.pt"
            )

            #
            # =================================================
            # Mise à jour du modèle courant partagé
            #
            # IMPORTANT :
            # les workers possèdent déjà ce même objet.
            # On ne recrée donc PAS le pool.
            #
            # =================================================
            #

            print(
                ">>> Updating shared current model"
            )

            current_state = (
                model.state_dict()
            )

            shared_state = (
                shared_current_model.state_dict()
            )

            for key in current_state:

                shared_state[key].copy_(
                    current_state[key]
                    .detach()
                    .cpu()
                )

            print(
                ">>> Shared current model updated"
            )

            #
            # =================================================
            # Best checkpoint
            # =================================================
            #

            if (
                best_loss is None
                or loss < best_loss
            ):

                best_loss = loss

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "loss": loss,
                    },
                    PROJECT_ROOT
                    / "checkpoints"
                    / "rl_best.pt",
                )

                print(
                    "New best checkpoint saved."
                )

            #
            # =================================================
            # Summary
            # =================================================
            #

            print(
                f"\n===== Epoch {epoch} summary ====="
            )

            print(
                f"Self-play: "
                f"{wins}W / {losses}L / {draws}D "
                f"({selfplay_score_rate:.1%})"
            )

    #
    # ========================================================
    # Manager
    # ========================================================
    #

    manager.shutdown()


if __name__ == "__main__":
    main()