import sys
import time
from pathlib import Path

import torch
import chess
import chess.variant


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(
    "/Users/tom/Desktop/Atomic"
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports
# ============================================================

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.actions_space import ACTIONS

from src.mcts.search import MCTS


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_36.pt"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Nombre de simulations testées
SIMULATIONS = [
    10,
    25,
    50,
    100,
    250,
    500,
]


C_PUCT = 1.5


# Nombre de répétitions par position
REPEATS = 3


# ============================================================
# Model
# ============================================================

def load_model():

    print()
    print("======================================")
    print("Loading model")
    print("======================================")

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Device: {DEVICE}"
    )


    base_model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )


    model = ActorCritic(
        base_model
    )


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.to(DEVICE)

    model.eval()


    print(
        f"Loaded epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )


    return model


# ============================================================
# Positions
# ============================================================

def get_positions():

    positions = []


    #
    # 1. Initial
    #

    positions.append(
        (
            "Initial",
            chess.variant.AtomicBoard(),
        )
    )


    #
    # 2. Après e4
    #

    board = chess.variant.AtomicBoard()

    board.push_uci("e2e4")

    positions.append(
        (
            "After e4",
            board,
        )
    )


    #
    # 3. Position plus développée
    #

    board = chess.variant.AtomicBoard()

    for move in [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
    ]:

        board.push_uci(move)


    positions.append(
        (
            "Opening",
            board,
        )
    )


    #
    # 4. Position Atomic tactique
    #

    board = chess.variant.AtomicBoard()

    for move in [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1c4",
        "g8f6",
    ]:

        board.push_uci(move)


    positions.append(
        (
            "Tactical",
            board,
        )
    )


    return positions


# ============================================================
# Benchmark
# ============================================================

def benchmark_position(
    model,
    name,
    board,
    simulations,
):

    times = []

    best_moves = []


    for _ in range(REPEATS):

        mcts = MCTS(
            model=model,
            simulations=simulations,
            c_puct=C_PUCT,
            device=DEVICE,
        )


        start = time.perf_counter()


        with torch.no_grad():

            result = mcts.search(
                board
            )


        elapsed = (
            time.perf_counter()
            - start
        )


        times.append(
            elapsed
        )


        if result["move"] is not None:

            best_moves.append(
                result["move"].uci()
            )

        else:

            best_moves.append(
                None
            )


    avg_time = (
        sum(times)
        / len(times)
    )


    time_per_simulation = (
        avg_time
        / simulations
    )


    simulations_per_second = (
        simulations
        / avg_time
    )


    return {
        "name": name,

        "simulations":
            simulations,

        "avg_time":
            avg_time,

        "time_per_simulation":
            time_per_simulation,

        "simulations_per_second":
            simulations_per_second,

        "best_moves":
            best_moves,
    }


# ============================================================
# Main benchmark
# ============================================================

def main():

    print()
    print("======================================")
    print("ALBERTA — MCTS BENCHMARK")
    print("======================================")

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Device: "
        f"{DEVICE}"
    )

    print(
        f"Repeats: "
        f"{REPEATS}"
    )


    model = load_model()


    positions = get_positions()


    results = []


    #
    # Warm-up
    #

    print()
    print("======================================")
    print("Warm-up")
    print("======================================")


    warmup_board = (
        chess.variant.AtomicBoard()
    )


    warmup_mcts = MCTS(
        model=model,
        simulations=10,
        c_puct=C_PUCT,
        device=DEVICE,
    )


    warmup_mcts.search(
        warmup_board
    )


    print("Warm-up complete.")


    #
    # Benchmark
    #

    print()
    print("======================================")
    print("Benchmark")
    print("======================================")


    for (
        position_name,
        board,
    ) in positions:

        print()
        print(
            f"Position: "
            f"{position_name}"
        )

        print(
            f"Legal moves: "
            f"{board.legal_moves.count()}"
        )


        for simulations in SIMULATIONS:

            result = benchmark_position(
                model,
                position_name,
                board,
                simulations,
            )


            results.append(
                result
            )


            print(
                f"  "
                f"{simulations:4d} sims | "
                f"{result['avg_time']:.3f}s | "
                f"{result['time_per_simulation'] * 1000:.2f} ms/sim | "
                f"{result['simulations_per_second']:.1f} sims/s | "
                f"best={result['best_moves']}"
            )


    #
    # Summary
    #

    print()
    print("======================================")
    print("SUMMARY")
    print("======================================")


    print(
        f"{'Position':<15}"
        f"{'Sims':>8}"
        f"{'Time':>12}"
        f"{'ms/sim':>12}"
        f"{'sims/s':>12}"
        f"{'Best move':>12}"
    )


    print(
        "-" * 75
    )


    for result in results:

        best_move = (
            result["best_moves"][0]
            if result["best_moves"]
            else None
        )


        print(
            f"{result['name']:<15}"
            f"{result['simulations']:>8}"
            f"{result['avg_time']:>12.3f}"
            f"{result['time_per_simulation'] * 1000:>12.2f}"
            f"{result['simulations_per_second']:>12.1f}"
            f"{str(best_move):>12}"
        )


    #
    # Scaling
    #

    print()
    print("======================================")
    print("SCALING")
    print("======================================")


    for position_name, _ in positions:

        position_results = [
            r
            for r in results
            if r["name"] == position_name
        ]


        if not position_results:

            continue


        first = position_results[0]

        last = position_results[-1]


        speed_ratio = (
            last["avg_time"]
            / first["avg_time"]
        )


        simulation_ratio = (
            last["simulations"]
            / first["simulations"]
        )


        efficiency = (
            speed_ratio
            / simulation_ratio
        )


        print()

        print(
            f"{position_name}:"
        )

        print(
            f"  Simulation ratio: "
            f"{simulation_ratio:.1f}x"
        )

        print(
            f"  Time ratio: "
            f"{speed_ratio:.2f}x"
        )

        print(
            f"  Scaling efficiency: "
            f"{efficiency:.3f}"
        )


    print()
    print("======================================")
    print("BENCHMARK COMPLETE")
    print("======================================")


if __name__ == "__main__":

    main()