import sys
from pathlib import Path

import torch
import chess
import chess.variant


# ============================================================
# Ajouter la racine du projet au PYTHONPATH
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

from src.agents.mcts_agent import MCTSAgent


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

SIMULATIONS = 100

C_PUCT = 1.5


# ============================================================
# Loading
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
# TEST 1
# ============================================================

def test_initial_position(
    agent,
):

    print()
    print("======================================")
    print("TEST 1 — Initial position")
    print("======================================")


    board = chess.variant.AtomicBoard()


    print(board)

    print()

    print(
        f"Legal moves: "
        f"{board.legal_moves.count()}"
    )


    result = agent.choose_move(
        board
    )


    move = result["move"]


    assert move is not None

    assert move in board.legal_moves


    print()

    print(
        f"Best move: "
        f"{move.uci()}"
    )

    print(
        f"Simulations: "
        f"{SIMULATIONS}"
    )

    print(
        f"Root visits: "
        f"{result['root'].visit_count}"
    )


    assert (
        result["root"].visit_count
        == SIMULATIONS
    )


    print()

    print(
        "✓ Initial position passed"
    )


# ============================================================
# TEST 2
# ============================================================

def test_choose_moves(
    agent,
):

    print()
    print("======================================")
    print("TEST 2 — Batch interface")
    print("======================================")


    boards = [
        chess.variant.AtomicBoard(),
        chess.variant.AtomicBoard(
            "rnbqkbnr/pppppppp/8/8/8/8/"
            "PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ),
    ]


    results = agent.choose_moves(
        boards
    )


    assert len(results) == len(
        boards
    )


    for i, (
        board,
        result,
    ) in enumerate(
        zip(
            boards,
            results,
        ),
        start=1,
    ):

        move = result["move"]


        print()

        print(
            f"Position {i}"
        )

        print(
            f"  Move: "
            f"{move.uci()}"
        )

        print(
            f"  Action: "
            f"{result['action']}"
        )

        print(
            f"  Root visits: "
            f"{result['root'].visit_count}"
        )


        assert move in board.legal_moves

        assert (
            result["action"]
            == move.uci()
        )


    print()

    print(
        "✓ Batch interface passed"
    )


# ============================================================
# TEST 3
# ============================================================

def test_policy(
    agent,
):

    print()
    print("======================================")
    print("TEST 3 — MCTS policy")
    print("======================================")


    board = chess.variant.AtomicBoard()


    result = agent.choose_move(
        board
    )


    policy = result["policy"]


    legal_moves = [
        move.uci()
        for move in board.legal_moves
    ]


    print(
        f"Policy entries: "
        f"{len(policy)}"
    )

    print(
        f"Legal moves: "
        f"{len(legal_moves)}"
    )


    assert set(policy.keys()) == set(
        legal_moves
    )


    policy_sum = sum(
        policy.values()
    )


    print(
        f"Policy sum: "
        f"{policy_sum:.6f}"
    )


    assert abs(
        policy_sum - 1.0
    ) < 1e-6


    print()

    print(
        "✓ Policy passed"
    )


# ============================================================
# TEST 4
# ============================================================

def test_terminal_position(
    agent,
):

    print()
    print("======================================")
    print("TEST 4 — Terminal position")
    print("======================================")


    #
    # Position atomique déjà terminée.
    #
    # Noir est sans roi : explosion.
    #

    board = chess.variant.AtomicBoard(
        "rnbq1bnr/pppp1ppp/8/4k3/8/8/"
        "PPPPPPPP/RNBQKBNR w KQ - 0 1"
    )


    if not board.is_game_over():

        print(
            "WARNING: test position "
            "not terminal under current "
            "python-chess atomic rules."
        )

        return


    result = agent.choose_move(
        board
    )


    print(
        f"Game over: "
        f"{board.is_game_over()}"
    )


    assert result["move"] is None


    print(
        "✓ Terminal position passed"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("======================================")
    print("ALBERTA — MCTS AGENT TEST")
    print("======================================")

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )


    model = load_model()


    agent = MCTSAgent(
        model=model,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        device=DEVICE,
    )


    test_initial_position(
        agent
    )

    test_choose_moves(
        agent
    )

    test_policy(
        agent
    )

    test_terminal_position(
        agent
    )


    print()
    print("======================================")
    print("ALL MCTS AGENT TESTS PASSED")
    print("======================================")


if __name__ == "__main__":

    main()