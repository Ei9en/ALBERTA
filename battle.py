from pathlib import Path

import chess.variant

from lichess_bot.atomic_engine.bc_bot_stochastic import BCBotStochastic
from lichess_bot.atomic_engine.bc_bot import BCBot

PROJECT_ROOT = Path(__file__).resolve().parent


CHECKPOINT_1 = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_v2_5_epoch_7.pt"
)

CHECKPOINT_2 = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_v2_5_epoch_6.pt"
)


GAMES = 2



def play_game(
    white,
    black,
):

    board = chess.variant.AtomicBoard()


    while not board.is_game_over():

        agent = (
            white
            if board.turn
            else black
        )


        move = agent.choose_move(
            board
        )


        board.push(move)


    return board.result()



def main():

    bot1 = BCBot(
        CHECKPOINT_1
    )

    bot2 = BCBot(
        CHECKPOINT_2
    )


    score = {
        "bot1": 0,
        "bot2": 0,
        "draw": 0,
    }


    for i in range(GAMES):

        #
        # Alternance des couleurs
        #

        if i % 2 == 0:

            white = bot1
            black = bot2

        else:

            white = bot2
            black = bot1



        result = play_game(
            white,
            black,
        )


        if result == "1-0":

            if white == bot1:
                score["bot1"] += 1
            else:
                score["bot2"] += 1


        elif result == "0-1":

            if black == bot1:
                score["bot1"] += 1
            else:
                score["bot2"] += 1


        else:

            score["draw"] += 1


        print(
            f"{i+1}/{GAMES}: {result}"
        )



    print("\n===== RESULT =====")

    print(
        "Bot 1:",
        score["bot1"]
    )

    print(
        "Bot 2:",
        score["bot2"]
    )

    print(
        "Draws:",
        score["draw"]
    )



if __name__ == "__main__":
    main()