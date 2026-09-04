from pathlib import Path
import sys
import csv
import itertools

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame
from src.actions_space import ACTIONS


# ============================================================
# CONSTANTS
# ============================================================

DEVICE = "cpu"


# ============================================================
# CHECKPOINT DIRECTORIES
# ============================================================

RL_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "test"

)

ORACLE_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "oracle_epoch"
)


# ============================================================
# OPTIONS
# ============================================================

# True  -> RL + Oracle
# False -> RL only
INCLUDE_ORACLE = True

GAMES_PER_MATCH = 50

TEMPERATURE = 2


# ============================================================
# ELO
# ============================================================

INITIAL_ELO = 1500.0
ELO_SCALE = 400.0


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CSV = (
    PROJECT_ROOT
    / "round_robin_results.csv"
)


# ============================================================
# MODEL
# ============================================================

def build_model():

    bc_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=32,
        blocks=4,
    )

    model = ActorCritic(
        bc_model
    )

    model.to(DEVICE)

    return model


def load_model(path):

    model = build_model()

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def get_checkpoints():

    checkpoints = []

    # ========================================================
    # RL
    # ========================================================

    if RL_CHECKPOINT_DIR.exists():

        paths = list(
            RL_CHECKPOINT_DIR.glob(
                "rl_epoch_*.pt"
            )
        )

        paths.sort(
            key=lambda p: int(
                p.stem.split("_")[-1]
            )
        )

        for path in paths:

            epoch = int(
                path.stem.split("_")[-1]
            )

            checkpoints.append(
                {
                    "name": f"RL{epoch}",
                    "path": path,
                    "type": "RL",
                    "epoch": epoch,
                }
            )

    # ========================================================
    # ORACLE
    # ========================================================

    if (
        INCLUDE_ORACLE
        and ORACLE_CHECKPOINT_DIR.exists()
    ):

        paths = list(
            ORACLE_CHECKPOINT_DIR.glob(
                "al_epoch_*.pt"
            )
        )

        paths.sort(
            key=lambda p: int(
                p.stem.split("_")[-1]
            )
        )

        for path in paths:

            epoch = int(
                path.stem.split("_")[-1]
            )

            checkpoints.append(
                {
                    "name": f"ORACLE{epoch}",
                    "path": path,
                    "type": "ORACLE",
                    "epoch": epoch,
                }
            )

    # --------------------------------------------------------
    # Sort globally by epoch, then type
    # --------------------------------------------------------

    checkpoints.sort(
        key=lambda x: (
            x["epoch"],
            0 if x["type"] == "RL" else 1,
        )
    )

    return checkpoints


# ============================================================
# MATCH
# ============================================================

def play_game(
    white_model,
    black_model,
):

    white_agent = ActorCriticAgent(
        white_model,
        deterministic=False,
        temperature=TEMPERATURE,
    )

    black_agent = ActorCriticAgent(
        black_model,
        deterministic=False,
        temperature=TEMPERATURE,
    )

    game = SelfPlayGame(
        white_agent,
        black_agent,
    )

    _, result = game.play()

    return result


# ============================================================
# ELO
# ============================================================

def expected_score(
    rating_a,
    rating_b,
):

    return 1.0 / (
        1.0
        + 10.0 ** (
            (rating_b - rating_a)
            / ELO_SCALE
        )
    )


def fit_global_elo(
    agents,
    match_results,
    iterations=5000,
    learning_rate=1.0,
):

    """
    Estimate global Elo ratings from the COMPLETE
    round-robin tournament.

    This is NOT sequential Elo.

    All match results are considered simultaneously.

    Therefore the final ratings do not depend on
    the order in which matches were played.

    Draws are treated as 0.5 points.

    Elo convention:
        baseline = 1500
        scale    = 400
    """

    ratings = {
        agent_id: INITIAL_ELO
        for agent_id in agents
    }

    # --------------------------------------------------------
    # Iterative optimization
    # --------------------------------------------------------

    for _ in range(iterations):

        gradients = {
            agent_id: 0.0
            for agent_id in agents
        }

        for match in match_results:

            a = match["agent_a"]
            b = match["agent_b"]

            a_wins = match["a_wins"]
            b_wins = match["b_wins"]
            draws = match["draws"]

            total = (
                a_wins
                + b_wins
                + draws
            )

            if total == 0:
                continue

            # Observed score for A
            observed_a = (
                a_wins
                + 0.5 * draws
            ) / total

            # Predicted score from Elo
            predicted_a = expected_score(
                ratings[a],
                ratings[b],
            )

            error = (
                observed_a
                - predicted_a
            )

            gradients[a] += (
                error * total
            )

            gradients[b] -= (
                error * total
            )

        # ----------------------------------------------------
        # Normalize global gradient
        # ----------------------------------------------------

        max_change = 0.0

        for agent_id in agents:

            # Number of games played by this agent
            games = 0

            for match in match_results:

                if (
                    match["agent_a"]
                    == agent_id
                ):

                    games += (
                        match["a_wins"]
                        + match["b_wins"]
                        + match["draws"]
                    )

                elif (
                    match["agent_b"]
                    == agent_id
                ):

                    games += (
                        match["a_wins"]
                        + match["b_wins"]
                        + match["draws"]
                    )

            if games == 0:
                continue

            change = (
                learning_rate
                * gradients[agent_id]
                / games
            )

            ratings[agent_id] += change

            max_change = max(
                max_change,
                abs(change),
            )

        if max_change < 1e-7:
            break

    # --------------------------------------------------------
    # Center around 1500
    # --------------------------------------------------------

    mean_rating = (
        sum(ratings.values())
        / len(ratings)
    )

    shift = (
        INITIAL_ELO
        - mean_rating
    )

    for agent_id in ratings:

        ratings[agent_id] += shift

    return ratings


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ALBERTA - ROUND ROBIN TOURNAMENT")
    print("=" * 80)

    print()
    print(
        f"RL directory     : "
        f"{RL_CHECKPOINT_DIR}"
    )

    print(
        f"Oracle directory : "
        f"{ORACLE_CHECKPOINT_DIR}"
    )

    print(
        f"Include Oracle   : "
        f"{INCLUDE_ORACLE}"
    )

    print(
        f"Games per match  : "
        f"{GAMES_PER_MATCH}"
    )

    print(
        f"Temperature      : "
        f"{TEMPERATURE}"
    )

    print(
        f"Device           : "
        f"{DEVICE}"
    )

    print()


    # ========================================================
    # DISCOVER
    # ========================================================

    checkpoints = get_checkpoints()

    if len(checkpoints) < 2:

        raise RuntimeError(
            "Need at least two checkpoints."
        )

    print("=" * 80)
    print("CHECKPOINTS")
    print("=" * 80)

    for checkpoint in checkpoints:

        print(
            f"{checkpoint['name']:<12} "
            f"{checkpoint['path'].name}"
        )

    print()

    num_agents = len(checkpoints)

    total_matches = (
        num_agents
        * (num_agents - 1)
        // 2
    )

    total_games = (
        total_matches
        * GAMES_PER_MATCH
    )

    print(
        f"Agents        : {num_agents}"
    )

    print(
        f"Matches       : {total_matches}"
    )

    print(
        f"Games/match   : {GAMES_PER_MATCH}"
    )

    print(
        f"Total games   : {total_games}"
    )

    print()


    # ========================================================
    # LOAD MODELS
    # ========================================================

    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    models = {}

    for checkpoint in checkpoints:

        name = checkpoint["name"]

        print(
            f"Loading {name}..."
        )

        models[name] = load_model(
            checkpoint["path"]
        )

    print()

    print(
        f"Loaded {len(models)} models."
    )

    print()


    # ========================================================
    # STATS
    # ========================================================

    stats = {}

    for checkpoint in checkpoints:

        name = checkpoint["name"]

        stats[name] = {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "score": 0.0,
        }


    match_results = []


    # ========================================================
    # TOURNAMENT
    # ========================================================

    print("=" * 80)
    print("TOURNAMENT")
    print("=" * 80)

    for match_index, (
        checkpoint_a,
        checkpoint_b,
    ) in enumerate(
        itertools.combinations(
            checkpoints,
            2,
        ),
        start=1,
    ):

        name_a = checkpoint_a["name"]
        name_b = checkpoint_b["name"]

        model_a = models[name_a]
        model_b = models[name_b]

        a_wins = 0
        b_wins = 0
        draws = 0

        print(
            f"[{match_index}/{total_matches}] "
            f"{name_a} vs {name_b}"
        )


        # ====================================================
        # A WHITE / B BLACK
        # ====================================================

        half = GAMES_PER_MATCH // 2

        for _ in range(half):

            result = play_game(
                model_a,
                model_b,
            )

            if result == "1-0":

                a_wins += 1

            elif result == "0-1":

                b_wins += 1

            else:

                draws += 1


        # ====================================================
        # B WHITE / A BLACK
        # ====================================================

        for _ in range(half):

            result = play_game(
                model_b,
                model_a,
            )

            if result == "1-0":

                b_wins += 1

            elif result == "0-1":

                a_wins += 1

            else:

                draws += 1


        # ====================================================
        # SCORES
        # ====================================================

        a_score = (
            a_wins
            + 0.5 * draws
        )

        b_score = (
            b_wins
            + 0.5 * draws
        )

        a_percentage = (
            a_score
            / GAMES_PER_MATCH
            * 100
        )

        b_percentage = (
            b_score
            / GAMES_PER_MATCH
            * 100
        )


        print(
            f"    {name_a:<12} "
            f"{a_wins:>3}W / "
            f"{b_wins:>3}L / "
            f"{draws:>3}D "
            f"-> {a_percentage:5.1f}%"
        )

        print(
            f"    {name_b:<12} "
            f"{b_wins:>3}W / "
            f"{a_wins:>3}L / "
            f"{draws:>3}D "
            f"-> {b_percentage:5.1f}%"
        )

        print()


        # ====================================================
        # GLOBAL STATS
        # ====================================================

        stats[name_a]["games"] += (
            GAMES_PER_MATCH
        )

        stats[name_a]["wins"] += a_wins
        stats[name_a]["losses"] += b_wins
        stats[name_a]["draws"] += draws
        stats[name_a]["score"] += a_score


        stats[name_b]["games"] += (
            GAMES_PER_MATCH
        )

        stats[name_b]["wins"] += b_wins
        stats[name_b]["losses"] += a_wins
        stats[name_b]["draws"] += draws
        stats[name_b]["score"] += b_score


        # ====================================================
        # SAVE MATCH
        # ====================================================

        match_results.append(
            {
                "agent_a": name_a,
                "agent_b": name_b,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "draws": draws,
                "a_score_pct": a_percentage,
                "b_score_pct": b_percentage,
            }
        )


    # ========================================================
    # GLOBAL ELO
    # ========================================================

    print()
    print("=" * 80)
    print("ESTIMATING GLOBAL ELO")
    print("=" * 80)

    agent_names = [
        checkpoint["name"]
        for checkpoint in checkpoints
    ]

    elo = fit_global_elo(
        agent_names,
        match_results,
    )


    # ========================================================
    # FINAL RANKING
    # ========================================================

    ranking = sorted(
        agent_names,
        key=lambda name: elo[name],
        reverse=True,
    )


    print()
    print("=" * 80)
    print("FINAL RANKING")
    print("=" * 80)

    print(
        f"{'Rank':<6}"
        f"{'Agent':<12}"
        f"{'Elo':<10}"
        f"{'Score':<10}"
        f"{'Wins':<8}"
        f"{'Losses':<8}"
        f"{'Draws':<8}"
    )

    print("-" * 80)


    for rank, name in enumerate(
        ranking,
        start=1,
    ):

        s = stats[name]

        score_pct = (
            s["score"]
            / s["games"]
            * 100
        )

        print(
            f"{rank:<6}"
            f"{name:<12}"
            f"{elo[name]:>7.0f}   "
            f"{score_pct:>6.1f}%   "
            f"{s['wins']:<8}"
            f"{s['losses']:<8}"
            f"{s['draws']:<8}"
        )


    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "agent_a",
                "agent_b",
                "a_wins",
                "b_wins",
                "draws",
                "a_score_pct",
                "b_score_pct",
            ]
        )

        for result in match_results:

            writer.writerow(
                [
                    result["agent_a"],
                    result["agent_b"],
                    result["a_wins"],
                    result["b_wins"],
                    result["draws"],
                    f"{result['a_score_pct']:.4f}",
                    f"{result['b_score_pct']:.4f}",
                ]
            )


    print()
    print(
        "Match results saved to:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print()
    print("=" * 80)
    print("TOURNAMENT FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()