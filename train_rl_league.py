# Train_RL.py

### Imports ###

from src.selfplay.league import League
import copy

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.encoding import encode_fen

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame

from src.rl.compute_returns import compute_returns
from src.rl.replay_buffer import ReplayBuffer

from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX

### Constants ###

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bc_v2_5_epoch_3.pt"

DEVICE = "cpu"

LR = 5e-5

GAMES_PER_EPOCH = 20

RL_EPOCHS = 5

CHECKPOINT_EVERY = 2

VALUE_COEF = 0.1

BATCH_SIZE = 128

SGD_EPOCHS = 5 # Nombre de passages complets sur le replay buffer pendant un epoch RL.
               # Plus élevé = plus d'updates par collecte de parties, mais risque de sur-apprentissage
               # sur les anciennes expériences.

### Model Loading ###

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

    model.to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=LR,
    )

    #
    # Initialisation de la league
    #
    league = League()

    snapshot = copy.deepcopy(model)
    snapshot.eval()

    league.add_agent(
        "league_bc_init",
        snapshot
    )

    return model, optimizer, league

### Self-play Collection ###

def collect_games(model, league, n_games):

    model.eval()

    games = []

    with torch.no_grad():

        for i in tqdm(range(n_games), desc="League self-play"):

            opponent_name, opponent = league.sample_opponent()

            #
            # Alterner les couleurs
            #
            if i % 2 == 0:

                white_agent = ActorCriticAgent(
                    model,
                    deterministic=False,
                    temperature=0.75,
                )

                black_agent = ActorCriticAgent(
                    opponent,
                    deterministic=False,
                    temperature=0.75,
                )

                current_is_white = True

            else:

                white_agent = ActorCriticAgent(
                    opponent,
                    deterministic=False,
                    temperature=0.75,
                )

                black_agent = ActorCriticAgent(
                    model,
                    deterministic=False,
                    temperature=0.75,
                )

                current_is_white = False


            game = SelfPlayGame(
                white_agent,
                black_agent,
            )

            trajectory, result = game.play()

            #
            # Ne garder que les coups du modèle courant
            #

            filtered = []

            for step in trajectory:

                if step["player"] == current_is_white:
                    filtered.append(step)

            games.append(
                {
                    "trajectory": filtered,
                    "result": result,
                    "current_white": current_is_white,
                }
            )

    return games

### Training ###

def train_epoch(
    model,
    optimizer,
    buffer,
):

    model.train()

    TRAIN_STEPS = len(buffer) // BATCH_SIZE

    total_loss = 0
    total_actor = 0
    total_critic = 0

    total_updates = TRAIN_STEPS * SGD_EPOCHS


    progress = tqdm(
        total=total_updates,
        desc="Training",
    )


    for epoch in range(SGD_EPOCHS):

        for _ in range(TRAIN_STEPS):

            batch = buffer.sample(
                BATCH_SIZE
            )


            optimizer.zero_grad()


            loss = 0
            actor_loss_sum = 0
            critic_loss_sum = 0


            for step in batch:

                x = (
                    encode_fen(step["fen"])
                    .unsqueeze(0)
                    .to(DEVICE)
                )


                policy, value = model(x)


                target = torch.tensor(
                    [[step["return"]]],
                    device=DEVICE,
                )


                advantage = (
                    target
                    - value.detach()
                )


                legal_indices = [
                    ACTION_TO_INDEX[m]
                    for m in step["legal_moves"]
                ]


                legal_logits = policy[
                    0,
                    legal_indices
                ]


                log_probs = F.log_softmax(
                    legal_logits,
                    dim=0,
                )


                action_position = legal_indices.index(
                    step["action"]
                )


                actor_loss = (
                    -log_probs[action_position]
                    *
                    advantage.squeeze()
                )


                critic_loss = F.mse_loss(
                    value,
                    target,
                )


                loss += (
                    actor_loss
                    +
                    VALUE_COEF * critic_loss
                )


                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()


            loss /= BATCH_SIZE


            loss.backward()

            optimizer.step()


            total_loss += loss.item()
            total_actor += actor_loss_sum / BATCH_SIZE
            total_critic += critic_loss_sum / BATCH_SIZE


            progress.update(1)


    progress.close()


    return (
        total_loss / total_updates,
        total_actor / total_updates,
        total_critic / total_updates,
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
        PROJECT_ROOT / f"checkpoints/rl_epoch_{epoch}.pt",
    )

### Main Loop ###

def main():

    model, optimizer, league = load_model()

    buffer = ReplayBuffer(
        capacity=50000
    )

    best_loss = None


    for epoch in range(RL_EPOCHS):

        print(f"\n===== Epoch {epoch} =====")


        games = collect_games(
            model,
            league,
            GAMES_PER_EPOCH,
        )


        #
        # Ajout dans replay buffer
        #

        for game in games:

            trajectory = game["trajectory"]
            result = game["result"]
            current_white = game["current_white"]


            if result == "1-0":

                reward = (
                    1.0
                    if current_white
                    else -1.0
                )

            elif result == "0-1":

                reward = (
                    -1.0
                    if current_white
                    else 1.0
                )

            else:

                reward = 0.0


            rewards = [
                reward
                for _ in trajectory
            ]


            returns = compute_returns(
                rewards,
                gamma=0.99,
            )


            returns = torch.tensor(
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


        print(
            f"Replay buffer size: {len(buffer)}"
        )


        #
        # Training mini-batch
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
        # Checkpoint
        #

        if epoch % CHECKPOINT_EVERY == 0:

            save_checkpoint(
                model,
                optimizer,
                epoch,
                loss,
            )


        #
        # Ajout snapshot league
        #

        snapshot = copy.deepcopy(model)
        snapshot.eval()


        agent_name = (
            f"league_epoch_{epoch:03d}"
        )


        league.add_agent(
            agent_name,
            snapshot
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
        # Best checkpoint
        #

        if best_loss is None or loss < best_loss:

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


if __name__ == "__main__":
    main()