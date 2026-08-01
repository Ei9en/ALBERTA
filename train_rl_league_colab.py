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

from src.encoding import encode_fen
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

LR = 1e-5

GAMES_PER_EPOCH = 150

RL_EPOCHS = 10

CHECKPOINT_EVERY = 5

VALUE_COEF = 0.1

BATCH_SIZE = 1024

SGD_EPOCHS = 1 # Nombre de passages complets sur le replay buffer pendant un epoch RL.
               # Plus élevé = plus d'updates par collecte de parties, mais risque de sur-apprentissage
               # sur les anciennes expériences.

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

def collect_games(
    model,
    league,
    n_games,
    stats,
):

    model.eval()

    games = []

    #
    # Timer self-play
    #
    selfplay_start = time.perf_counter()

    with torch.no_grad():

        #
        # =========================
        # Self-play
        # =========================
        #

        for i in tqdm(
            range(n_games),
            desc="League self-play",
        ):

            opponent_name, opponent = (
                league.sample_opponent()
            )

            #
            # Alterner les couleurs
            #
            if i % 2 == 0:

                white_agent = ActorCriticAgent(
                    model,
                    deterministic=False,
                    temperature=0.75,
                    device=DEVICE,
                )

                black_agent = ActorCriticAgent(
                    opponent,
                    deterministic=False,
                    temperature=0.75,
                    device=DEVICE,
                )

                current_is_white = True

            else:

                white_agent = ActorCriticAgent(
                    opponent,
                    deterministic=False,
                    temperature=0.75,
                    device=DEVICE,
                )

                black_agent = ActorCriticAgent(
                    model,
                    deterministic=False,
                    temperature=0.75,
                    device=DEVICE,
                )

                current_is_white = False


            game = SelfPlayGame(
                white_agent,
                black_agent,
            )

            trajectory, result = game.play()

            #
            # Garder les informations nécessaires.
            #
            games.append(
                {
                    "trajectory": trajectory,
                    "result": result,
                    "current_white": current_is_white,
                }
            )


    selfplay_time = (
        time.perf_counter()
        - selfplay_start
    )

    total_positions = sum(
        len(game["trajectory"])
        for game in games
    )

    print(
        f"Self-play time: {selfplay_time:.2f}s "
        f"({selfplay_time / n_games:.2f}s/game)"
    )

    print(
        f"Self-play positions: {total_positions} "
        f"({total_positions / n_games:.1f}/game)"
    )


    #
    # =========================
    # Calcul U global
    # =========================
    #

    uncertainty_start = time.perf_counter()

    all_steps = []

    for game in games:

        result = game["result"]

        for step in game["trajectory"]:

            step["_game_result"] = result

            all_steps.append(step)


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


    return games

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

    total_loss = 0
    total_actor = 0
    total_critic = 0

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


    for epoch in range(
        SGD_EPOCHS
    ):

        for update in range(
            TRAIN_STEPS
        ):

            batch = buffer.sample(
                BATCH_SIZE
            )


            optimizer.zero_grad()


            loss = 0

            actor_loss_sum = 0

            critic_loss_sum = 0


            for step in batch:

                #
                # =========================
                # Encoding
                # =========================
                #

                x = (
                    encode_fen(
                        step["fen"]
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )


                #
                # =========================
                # Forward
                # =========================
                #

                policy, value = model(x)


                target = torch.tensor(
                    [[step["return"]]],
                    device=DEVICE,
                )


                #
                # =========================
                # Raw advantage
                # =========================
                #

                raw_advantage = (
                    target
                    - value.detach()
                )


                raw_advantage_value = (
                    raw_advantage.item()
                )


                max_advantage = max(
                    max_advantage,
                    abs(
                        raw_advantage_value
                    ),
                )


                mean_abs_advantage += abs(
                    raw_advantage_value
                )


                diagnostic_samples += 1


                #
                # =========================
                # Legal moves
                # =========================
                #

                legal_indices = [
                    ACTION_TO_INDEX[m]
                    for m in step[
                        "legal_moves"
                    ]
                ]


                legal_logits = policy[
                    0,
                    legal_indices
                ]


                log_probs = F.log_softmax(
                    legal_logits,
                    dim=0,
                )


                action_position = (
                    legal_indices.index(
                        step["action"]
                    )
                )


                #
                # =========================
                # Selected log probability
                # =========================
                #

                selected_log_prob = (
                    log_probs[
                        action_position
                    ].item()
                )


                max_log_prob = max(
                    max_log_prob,
                    abs(
                        selected_log_prob
                    ),
                )


                mean_abs_log_prob += abs(
                    selected_log_prob
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
                )


                #
                # =========================
                # Actor loss
                # =========================
                #

                actor_loss = (
                    -log_probs[
                        action_position
                    ]
                    *
                    advantage.squeeze()
                )


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

                loss += (
                    actor_loss
                    +
                    VALUE_COEF
                    * critic_loss
                )


                actor_loss_sum += (
                    actor_loss.item()
                )

                critic_loss_sum += (
                    critic_loss.item()
                )


            #
            # =========================
            # Mean batch loss
            # =========================
            #

            loss /= BATCH_SIZE


            #
            # =========================
            # Backpropagation
            # =========================
            #

            loss.backward()


            #
            # =========================
            # Gradient diagnostics
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
            # Afficher uniquement
            # le premier update
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
                    f"{mean_abs_advantage / diagnostic_samples:.4f}"
                    f" | max|A|="
                    f"{max_advantage:.4f}"
                )

                print(
                    f"Selected logπ:"
                    f" mean|logπ|="
                    f"{mean_abs_log_prob / diagnostic_samples:.4f}"
                    f" | max|logπ|="
                    f"{max_log_prob:.4f}"
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
                actor_loss_sum
                / BATCH_SIZE
            )

            total_critic += (
                critic_loss_sum
                / BATCH_SIZE
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

### Main Loop ###

def main():

    model, optimizer, league = load_model()

    buffer = ReplayBuffer(
        capacity=20000
    )

    stats = UncertaintyStats()

    best_loss = None

    #
    # =========================
    # Fixed evaluation agents
    # =========================
    #

    bc4 = league.agents["bc_epoch_4"]
    bc5 = league.agents["bc_epoch_5"]

    #
    # =========================
    # RL loop
    # =========================
    #

    for epoch in range(RL_EPOCHS):

        print(
            f"\n===== Epoch {epoch} ====="
        )

        #
        # IMPORTANT :
        # On repart uniquement avec les données
        # collectées pendant cet epoch.
        #
        # Cela rend l'entraînement beaucoup plus
        # proche d'un régime on-policy.
        #
        buffer.clear()

        wins = 0
        losses = 0
        draws = 0

        #
        # =========================
        # Self-play collection
        # =========================
        #

        games = collect_games(
            model,
            league,
            GAMES_PER_EPOCH,
            stats,
        )

        #
        # =========================
        # Replay buffer
        # =========================
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

            #
            # Reward du point de vue du joueur
            # ayant joué chaque position.
            #
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
        # =========================
        # Self-play score
        # Draw = 0.5
        # =========================
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
        # =========================
        # Training
        # =========================
        #

        loss, actor_loss, critic_loss = train_epoch(
            model,
            optimizer,
            buffer,
        )

        print(
            f"Loss={loss:.4f} "
            f"| Actor={actor_loss:.4f} "
            f"| Critic={critic_loss:.4f}"
        )

        #
        # =========================
        # Fixed evaluation
        # =========================
        #

        print(
            "\n--- Evaluation vs BC4 ---"
        )

        evaluation_bc4 = evaluate_against_agent(
            model,
            bc4,
            n_games=100,
        )

        print(
            "\n--- Evaluation vs BC5 ---"
        )

        evaluation_bc5 = evaluate_against_agent(
            model,
            bc5,
            n_games=100,
        )

        #
        # =========================
        # Evaluation scores
        # Draw = 0.5
        # =========================
        #

        bc4_score_rate = (
            evaluation_bc4["wins"]
            + 0.5 * evaluation_bc4["draws"]
        ) / 100

        bc5_score_rate = (
            evaluation_bc5["wins"]
            + 0.5 * evaluation_bc5["draws"]
        ) / 100

        #
        # =========================
        # Save replay buffer
        # =========================
        #

        if epoch % 5 == 0:

            save_replay_buffer(
                buffer,
                epoch,
            )

        #
        # =========================
        # Save active learning stats
        # =========================
        #

        stats.save(
            PROJECT_ROOT
            / "checkpoints"
            / "uncertainty_stats.json"
        )

        #
        # =========================
        # RL checkpoint
        # =========================
        #

        if epoch % CHECKPOINT_EVERY == 0:

            save_checkpoint(
                model,
                optimizer,
                epoch,
                loss,
            )

        #
        # =========================
        # League snapshot
        # =========================
        #

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

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": snapshot.state_dict(),
            },
            PROJECT_ROOT
            / "checkpoints"
            / "league"
            / f"{agent_name}.pt"
        )

        #
        # =========================
        # Best checkpoint
        # =========================
        #

        if (
            best_loss is None
            or loss < best_loss
        ):

            best_loss = loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
        # =========================
        # Evaluation summary
        # =========================
        #

        print(
            f"\n===== Epoch {epoch} summary ====="
        )

        print(
            f"Self-play: "
            f"{wins}W / {losses}L / {draws}D "
            f"({selfplay_score_rate:.1%})"
        )

        print(
            f"vs BC4: "
            f"{evaluation_bc4['wins']}W / "
            f"{evaluation_bc4['losses']}L / "
            f"{evaluation_bc4['draws']}D "
            f"({bc4_score_rate:.1%})"
        )

        print(
            f"vs BC5: "
            f"{evaluation_bc5['wins']}W / "
            f"{evaluation_bc5['losses']}L / "
            f"{evaluation_bc5['draws']}D "
            f"({bc5_score_rate:.1%})"
        )


if __name__ == "__main__":
    main()