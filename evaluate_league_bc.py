from pathlib import Path
from itertools import combinations
from tqdm import tqdm

import json
from datetime import datetime


from lichess_bot.atomic_engine.bc_bot_stochastic import BCBotStochastic
from src.selfplay.game import SelfPlayGame


### Constants ###

PROJECT_ROOT = Path(__file__).resolve().parent


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)


CHECKPOINT_PATTERN = (
    "bc_v2_5_epoch_*.pt"
)


TEMPERATURE = 0.5


GAMES_PER_MATCH = 20


K = 32



### Loading ###

def load_bc_league():

    bots = {}


    checkpoints = sorted(
        CHECKPOINT_DIR.glob(
            CHECKPOINT_PATTERN
        )
    )


    if not checkpoints:

        raise RuntimeError(
            "No BC checkpoints found"
        )


    for path in checkpoints:

        name = path.stem

        bots[name] = BCBotStochastic(
            checkpoint=path,
            temperature=TEMPERATURE,
        )


    return bots




### Match evaluation ###

def evaluate_match(
    white_bot,
    black_bot,
    n_games,
):

    white_wins = 0
    black_wins = 0
    draws = 0


    for _ in tqdm(
        range(n_games),
        leave=False,
    ):


        game = SelfPlayGame(
            white_bot,
            black_bot,
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





### Elo update ###

def update_elo(
    elo_a,
    elo_b,
    score_a,
    K=32,
):


    expected_a = 1.0 / (
        1.0
        +
        10 ** (
            (elo_b - elo_a)
            /
            400
        )
    )


    expected_b = 1.0 - expected_a



    elo_a += K * (
        score_a
        -
        expected_a
    )


    elo_b += K * (
        (1-score_a)
        -
        expected_b
    )


    return elo_a, elo_b





### Main ###

def main():


    bots = load_bc_league()


    print(
        "\n===== League Members ====="
    )


    for name in bots:

        print(
            "-",
            name
        )



    names = list(
        bots.keys()
    )



    elos = {

        name: 1500.0

        for name in names

    }



    scores = {

        name: 0.0

        for name in names

    }



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



        half = GAMES_PER_MATCH // 2



        #
        # a white
        #

        a_w1, b_b1, d1 = evaluate_match(
            bots[a],
            bots[b],
            half,
        )



        #
        # b white
        #

        b_w2, a_b2, d2 = evaluate_match(
            bots[b],
            bots[a],
            half,
        )



        a_wins = (
            a_w1
            +
            a_b2
        )


        b_wins = (
            b_b1
            +
            b_w2
        )


        draws = (
            d1
            +
            d2
        )



        games = (
            a_wins
            +
            b_wins
            +
            draws
        )



        score_a = (
            a_wins
            +
            0.5 * draws
        ) / games



        elos[a], elos[b] = update_elo(
            elos[a],
            elos[b],
            score_a,
            K,
        )



        scores[a] += (
            a_wins
            +
            0.5 * draws
        )


        scores[b] += (
            b_wins
            +
            0.5 * draws
        )



        results[
            f"{a} vs {b}"
        ] = {

            "a_wins":
                a_wins,

            "b_wins":
                b_wins,

            "draws":
                draws,

        }



        print(
            f"{a}: {a_wins} | "
            f"{b}: {b_wins} | "
            f"D: {draws}"
        )





    ### Leaderboard ###


    print(
        "\n===== Scoreboard ====="
    )


    for name, score in sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"{name:30s} "
            f"{score:.1f}"
        )




    print(
        "\n===== Elo ====="
    )


    for name, elo in sorted(
        elos.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"{name:30s} "
            f"{elo:.1f}"
        )





    ### Save ###


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    output = {

        "temperature":
            TEMPERATURE,

        "games_per_match":
            GAMES_PER_MATCH,

        "elos":
            elos,

        "scores":
            scores,

        "results":
            results,

    }



    output_path = (
        PROJECT_ROOT
        /
        f"bc_league_eval_{timestamp}.json"
    )


    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )



    print(
        "\nSaved:",
        output_path
    )





if __name__ == "__main__":

    main()