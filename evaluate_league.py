from pathlib import Path
from itertools import combinations
from tqdm import tqdm

import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame

from src.actions_space import ACTIONS

import json
from datetime import datetime

### Constants ###

PROJECT_ROOT = Path(__file__).resolve().parent

DEVICE = "cpu"

LEAGUE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "league"
)

BC_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_v2_5_epoch_3.pt"
)

GAMES_PER_MATCH = 30


### Model building ###

def build_actor_critic():

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    model = ActorCritic(
        bc_model
    )

    model.to(DEVICE)

    return model


### Loading ###

def load_bc_checkpoint(path):

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



def load_rl_checkpoint(path):

    model = build_actor_critic()

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model



def load_league():

    agents = {}

    #
    # BC initial
    #

    agents["bc_init"] = load_bc_checkpoint(
        BC_CHECKPOINT
    )


    #
    # RL snapshots
    #

    for path in sorted(
        LEAGUE_DIR.glob("*.pt")
    ):

        name = path.stem

        agents[name] = load_rl_checkpoint(
            path
        )


    return agents



### Match evaluation ###

def evaluate_match(
    white_model,
    black_model,
    n_games,
):

    white_wins = 0
    black_wins = 0
    draws = 0


    for _ in tqdm(
        range(n_games),
        leave=False,
    ):

        white_agent = ActorCriticAgent(
            white_model,
            deterministic=False,
            temperature=0.5,
        )

        black_agent = ActorCriticAgent(
            black_model,
            deterministic=False,
            temperature=0.5,
        )


        game = SelfPlayGame(
            white_agent,
            black_agent,
        )

        _, result = game.play()


        if result == "1-0":

            white_wins += 1

        elif result == "0-1":

            black_wins += 1

        else:

            draws += 1


    return (
        white_wins,
        black_wins,
        draws,
    )



### Evaluation ###

def main():

    agents = load_league()


    print(
        "\n===== League Members ====="
    )

    for name in agents:

        print(
            "-",
            name
        )


    names = list(
        agents.keys()
    )

    elos = {
        name: 1200.0
        for name in names
    }

    K = 32

    results = {}

    print(
        "\n===== Evaluation ====="
    )


    for a, b in combinations(
        names,
        2,
    ):

        print(
            f"\n{a} vs {b}"
        )


        #
        # a white
        #

        a_wins_1, b_wins_1, draws_1 = evaluate_match(
            agents[a],
            agents[b],
            GAMES_PER_MATCH // 2,
        )


        #
        # b white
        #

        b_wins_2, a_wins_2, draws_2 = evaluate_match(
            agents[b],
            agents[a],
            GAMES_PER_MATCH // 2,
        )


        results[(a, b)] = {

            "a_win":
                a_wins_1 + a_wins_2,

            "b_win":
                b_wins_1 + b_wins_2,

            "draw":
                draws_1 + draws_2,
        }

        games = (
            results[(a, b)]["a_win"]
            + results[(a, b)]["b_win"]
            + results[(a, b)]["draw"]
        )

        score_a = (
            results[(a, b)]["a_win"]
            + 0.5 * results[(a, b)]["draw"]
        ) / games

        score_b = 1.0 - score_a

        expected_a = 1.0 / (
            1.0
            + 10 ** ((elos[b] - elos[a]) / 400)
        )

        expected_b = 1.0 - expected_a

        elos[a] += K * (score_a - expected_a)
        elos[b] += K * (score_b - expected_b)

        print(
            f"{a}: "
            f"{results[(a,b)]['a_win']} | "
            f"{b}: "
            f"{results[(a,b)]['b_win']} | "
            f"D: "
            f"{results[(a,b)]['draw']}"
        )


    #
    # Simple leaderboard
    #

    print(
        "\n===== Scoreboard ====="
    )


    scores = {
        name: 0
        for name in names
    }


    for (a, b), result in results.items():

        scores[a] += (
            result["a_win"]
            + 0.5 * result["draw"]
        )

        scores[b] += (
            result["b_win"]
            + 0.5 * result["draw"]
        )


    for name, score in sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"{name:25s} {score:.1f}"
        )

    print("\n===== Elo =====")

    for name, elo in sorted(
        elos.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"{name:25s} {elo:.1f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "elos": elos,
        "results": {
            f"{a} vs {b}": result
            for (a, b), result in results.items()
        }
    }

    with open(
        PROJECT_ROOT
        / f"league_eval_{timestamp}.json",
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )


if __name__ == "__main__":

    main()