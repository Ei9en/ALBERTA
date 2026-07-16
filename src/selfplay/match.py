from src.agents.actor_critic_agent import ActorCriticAgent
from src.selfplay.game import SelfPlayGame
import tqdm

def play_match(
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
            deterministic=True,
            temperature=0.0,
        )

        black_agent = ActorCriticAgent(
            black_model,
            deterministic=True,
            temperature=0.0,
        )

        game = SelfPlayGame(
            white_agent,
            black_agent,
        )

        trajectory, result = game.play()

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