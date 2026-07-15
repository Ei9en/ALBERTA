from pathlib import Path
from itertools import combinations
from tqdm import tqdm

import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.selfplay.match import play_match

from src.actions_space import ACTIONS


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

GAMES_PER_MATCH = 10


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

        _, result = play_match(
            white_model,
            black_model,
            deterministic=True,
        )


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



if __name__ == "__main__":

    main()