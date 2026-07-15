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
from src.selfplay.match import play_match

from src.rl.compute_returns import compute_returns

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

VALUE_COEF = 0.25 # variable

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

                trajectory, result = play_match(
                    model,
                    opponent,
                )

                current_is_white = True

            else:

                trajectory, result = play_match(
                    opponent,
                    model,
                )

                current_is_white = False

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

def train_epoch(model, optimizer, games):

    model.train()

    optimizer.zero_grad()

    total_loss = torch.tensor(
        0.0,
        device=DEVICE,
    )

    total_actor_loss = 0.0
    total_critic_loss = 0.0

    total_positions = sum(
        len(game["trajectory"])
        for game in games
    )

    progress = tqdm(
        total=total_positions,
        desc="Training",
        leave=False,
    )

    for game in games:

        trajectory = game["trajectory"]
        result = game["result"]
        current_white = game["current_white"]

        #
        # Rewards
        #

        if result == "1-0":
            reward = 1.0 if current_white else -1.0

        elif result == "0-1":
            reward = -1.0 if current_white else 1.0

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

        for i, step in enumerate(trajectory):

            x = (
                encode_fen(step["fen"])
                .unsqueeze(0)
                .to(DEVICE)
            )

            policy, value = model(x)

            target = (
                returns[i]
                .reshape(1, 1)
                .to(DEVICE)
            )

            advantage = (
                target - value.detach()
            )

            advantage = torch.clamp(
                advantage,
                -1,
                1,
            )

            #
            # Coups légaux
            #
            legal_indices = [
                ACTION_TO_INDEX[move]
                for move in step["legal_moves"]
            ]

            legal_logits = policy[0, legal_indices]

            legal_log_probs = F.log_softmax(
                legal_logits,
                dim=0,
            )

            assert step["action"] in legal_indices

            action_position = legal_indices.index(
                step["action"]
            )

            #
            # Actor
            #
            actor_loss = (
                -legal_log_probs[action_position]
                * advantage.squeeze()
            )

            #
            # Critic
            #
            critic_loss = F.mse_loss(
                value,
                target,
            )

            #
            # Entropy
            #
            entropy = -torch.sum(
                torch.exp(legal_log_probs)
                * legal_log_probs
            )

            entropy = entropy / len(legal_indices)

            #
            # Total
            #
            loss = (
                actor_loss
                + VALUE_COEF * critic_loss
                - 0.01 * entropy
            )

            total_loss += loss

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

            progress.update(1)

    progress.close()

    total_loss = total_loss / total_positions

    total_loss.backward()

    optimizer.step()

    actor_avg = (
        total_actor_loss
        / total_positions
    )

    critic_avg = (
        total_critic_loss
        / total_positions
    )

    return (
        total_loss.item(),
        actor_avg,
        critic_avg,
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

    best_loss = None

    for epoch in range(RL_EPOCHS):

        print(f"\n===== Epoch {epoch} =====")

        games = collect_games(
            model,
            league,
            GAMES_PER_EPOCH,
        )

        #
        # Self-play statistics
        #
        wins = 0
        losses = 0
        draw = 0

        for game in games:

            result = game["result"]
            current_white = game["current_white"]

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
                draw += 1


        print(
            f"League result : W={wins} L={losses} D={draw}"
        )

        #
        # Training
        #
        loss, actor_avg, critic_avg = train_epoch(
            model,
            optimizer,
            games,
        )

        print(
            f"Loss={loss:.4f} "
            f"| Actor={actor_avg:.4f} "
            f"| Critic={critic_avg:.4f}"
        )

        #
        # Epoch checkpoint
        #
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                loss,
            )

        snapshot = copy.deepcopy(model)
        snapshot.eval()

        agent_name = f"league_epoch_{epoch:03d}"

        league.add_agent(
            agent_name,
            snapshot
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": snapshot.state_dict(),
            },
            PROJECT_ROOT / "checkpoints" / "league" / f"{agent_name}.pt"
        )

        #
        # Best checkpoint (temporary criterion)
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
                PROJECT_ROOT / "checkpoints" / "rl_best.pt",
            )

            print("New best checkpoint saved.")


if __name__ == "__main__":

    main()