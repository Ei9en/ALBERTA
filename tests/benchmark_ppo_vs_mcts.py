import sys
import time
from pathlib import Path

import chess
import chess.variant
import torch
from tqdm import tqdm


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.agents.ppo_agent import PPOAgent
from src.agents.mcts_agent import MCTSAgent

from src.actions_space import ACTIONS


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_EPOCH = 36

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / f"rl_epoch_{CHECKPOINT_EPOCH}.pt"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TEMPERATURE = 0.75

MCTS_SIMULATIONS = [
    25,
    50,
    100,
]

GAMES_PER_MATCH = 100

C_PUCT = 1.5


# ============================================================
# Model
# ============================================================

def load_model():

    print()
    print("======================================")
    print("Loading model")
    print("======================================")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")

    model_base = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    model = ActorCritic(
        model_base
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    return model


# ============================================================
# Agent construction
# ============================================================

def make_ppo_agent(model):

    return PPOAgent(
        model,
        deterministic=False,
        temperature=TEMPERATURE,
        device=DEVICE,
    )


def make_mcts_agent(
    model,
    simulations,
):

    return MCTSAgent(
        model=model,
        simulations=simulations,
        c_puct=C_PUCT,
        device=DEVICE,
    )


# ============================================================
# Single game
# ============================================================

def play_game(
    ppo_agent,
    mcts_agent,
    ppo_is_white,
):

    board = chess.variant.AtomicBoard()

    trajectory_length = 0

    while not board.is_game_over():

        if board.turn:

            agent = (
                ppo_agent
                if ppo_is_white
                else mcts_agent
            )

        else:

            agent = (
                mcts_agent
                if ppo_is_white
                else ppo_agent
            )


        #
        # PPO interface
        #

        if agent is ppo_agent:

            info = agent.choose_move(
                board
            )

            move = info["move"]


        #
        # MCTS interface
        #

        else:

            info = agent.choose_move(
                board
            )

            move = info["move"]


        if move is None:

            raise RuntimeError(
                "Agent returned None "
                "before game termination."
            )


        if move not in board.legal_moves:

            raise RuntimeError(
                f"Illegal move returned: "
                f"{move.uci()}"
            )


        board.push(move)

        trajectory_length += 1


    result = board.result()

    return (
        result,
        trajectory_length,
    )


# ============================================================
# Match
# ============================================================

def run_match(
    model,
    simulations,
    n_games=GAMES_PER_MATCH,
):

    print()
    print("--------------------------------------")
    print(
        f"PPO vs MCTS "
        f"({simulations} simulations)"
    )
    print("--------------------------------------")

    wins = 0
    losses = 0
    draws = 0

    total_positions = 0

    start_time = time.perf_counter()


    for game_idx in tqdm(
        range(n_games),
        desc=f"{simulations} sims",
    ):

        ppo_agent = make_ppo_agent(
            model
        )

        mcts_agent = make_mcts_agent(
            model,
            simulations,
        )


        #
        # Alternate colors
        #

        ppo_is_white = (
            game_idx % 2 == 0
        )


        result, positions = play_game(
            ppo_agent,
            mcts_agent,
            ppo_is_white,
        )


        total_positions += positions


        #
        # Result from PPO perspective
        #

        if result == "1/2-1/2":

            draws += 1

        elif result == "1-0":

            if ppo_is_white:

                wins += 1

            else:

                losses += 1

        elif result == "0-1":

            if ppo_is_white:

                losses += 1

            else:

                wins += 1


    elapsed = (
        time.perf_counter()
        - start_time
    )


    score = (
        wins
        + 0.5 * draws
    ) / n_games


    print()
    print(
        f"PPO W={wins} "
        f"L={losses} "
        f"D={draws}"
    )

    print(
        f"PPO score: {score:.1%}"
    )

    print(
        f"Total positions: "
        f"{total_positions}"
    )

    print(
        f"Time: "
        f"{elapsed:.2f}s "
        f"({elapsed / n_games:.2f}s/game)"
    )

    print(
        f"Average game length: "
        f"{total_positions / n_games:.1f} positions"
    )


    return {
        "simulations": simulations,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "positions": total_positions,
        "time": elapsed,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("======================================")
    print("ALBERTA — PPO vs MCTS BENCHMARK")
    print("======================================")

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Games per match: "
        f"{GAMES_PER_MATCH}"
    )


    #
    # Load model once
    #

    model = load_model()


    #
    # Warm-up
    #

    print()
    print("======================================")
    print("Warm-up")
    print("======================================")

    board = chess.variant.AtomicBoard()

    ppo = make_ppo_agent(model)

    with torch.no_grad():

        ppo.choose_move(board)

    print("Warm-up complete.")


    #
    # Run matches
    #

    results = []


    for simulations in MCTS_SIMULATIONS:

        result = run_match(
            model,
            simulations,
            n_games=GAMES_PER_MATCH,
        )

        results.append(result)


    #
    # Summary
    #

    print()
    print("======================================")
    print("SUMMARY")
    print("======================================")

    print(
        "MCTS sims | PPO W | PPO L | Draw | "
        "PPO score | Time/game"
    )

    print(
        "------------------------------------------------"
    )


    for r in results:

        print(
            f"{r['simulations']:>9} | "
            f"{r['wins']:>5} | "
            f"{r['losses']:>5} | "
            f"{r['draws']:>4} | "
            f"{r['score']:.1%}     | "
            f"{r['time'] / GAMES_PER_MATCH:.2f}s"
        )


    print()
    print("======================================")
    print("INTERPRETATION")
    print("======================================")

    print(
        "PPO score > 50%  → PPO is stronger"
    )

    print(
        "PPO score < 50%  → MCTS is stronger"
    )

    print(
        "PPO score ≈ 50%  → roughly equivalent"
    )

    print()
    print("Benchmark complete.")


# ============================================================

if __name__ == "__main__":
    main()