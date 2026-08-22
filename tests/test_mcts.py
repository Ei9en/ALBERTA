# tests/test_mcts.py

import sys
from pathlib import Path
import time

import torch
import chess
import chess.variant


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

from src.mcts.search import MCTS

from src.actions_space import ACTIONS


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

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT}"
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
# TEST 1
# ============================================================

def test_initial_position(model):

    print()
    print("======================================")
    print("TEST 1 — Initial position")
    print("======================================")

    board = chess.variant.AtomicBoard()

    print(board)

    legal_moves = list(
        board.legal_moves
    )

    print()
    print(
        f"Legal moves: "
        f"{len(legal_moves)}"
    )

    assert len(legal_moves) > 0

    mcts = MCTS(
        model,
        simulations=SIMULATIONS,
        c_puct=1.5,
        device=DEVICE,
    )

    start = time.perf_counter()

    result = mcts.search(
        board
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    move = result["move"]

    root = result["root"]

    visits = result["visits"]

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
        f"{root.visit_count}"
    )

    print(
        f"Search time: "
        f"{elapsed:.3f}s"
    )

    print(
        f"Time / simulation: "
        f"{elapsed / SIMULATIONS * 1000:.2f} ms"
    )

    print()
    print("Top moves by visits:")

    sorted_visits = sorted(
        visits.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for action, count in sorted_visits[:10]:

        print(
            f"  {action:<10} "
            f"{count:>4} "
            f"({count / SIMULATIONS:.1%})"
        )

    #
    # Basic validity
    #

    assert move is not None

    assert move in legal_moves

    assert root.visit_count == SIMULATIONS

    assert len(root.children) == len(
        legal_moves
    )

    assert sum(visits.values()) == (
        SIMULATIONS
    )

    print()
    print(
        "✓ Initial position passed"
    )


# ============================================================
# TEST 2
# ============================================================

def test_multiple_positions(model):

    print()
    print("======================================")
    print("TEST 2 — Multiple positions")
    print("======================================")

    positions = [

        # Initial position
        chess.variant.AtomicBoard(),

        # After 1.Nf3
        chess.variant.AtomicBoard(
            "rnbqkbnr/"
            "pppppppp/"
            "8/8/8/6N1/"
            "PPPPPPPP/"
            "RNBQKB1R "
            "b KQkq - 1 1"
        ),

        # After 1.Nf3 d5 2.Ng5
        chess.variant.AtomicBoard(
            "rnbqkbnr/"
            "ppp1pppp/"
            "8/3p2N1/"
            "8/8/"
            "PPPPPPPP/"
            "RNBQKB1R "
            "b KQkq - 2 2"
        ),
    ]

    for i, board in enumerate(
        positions,
        start=1,
    ):

        print()
        print(
            f"Position {i}"
        )

        mcts = MCTS(
            model,
            simulations=SIMULATIONS,
            c_puct=1.5,
            device=DEVICE,
        )

        result = mcts.search(
            board
        )

        move = result["move"]

        root = result["root"]

        print(
            f"  Best move: "
            f"{move.uci()}"
        )

        print(
            f"  Root visits: "
            f"{root.visit_count}"
        )

        assert move is not None

        assert move in list(
            board.legal_moves
        )

        assert (
            root.visit_count
            == SIMULATIONS
        )

    print()
    print(
        "✓ Multiple positions passed"
    )


# ============================================================
# TEST 3
# ============================================================

def test_visit_distribution(model):

    print()
    print("======================================")
    print("TEST 3 — Visit distribution")
    print("======================================")

    board = chess.variant.AtomicBoard()

    mcts = MCTS(
        model,
        simulations=SIMULATIONS,
        c_puct=1.5,
        device=DEVICE,
    )

    result = mcts.search(
        board
    )

    root = result["root"]

    visits = result["visits"]

    print(
        f"Root visits: "
        f"{root.visit_count}"
    )

    print(
        f"Child visits: "
        f"{sum(visits.values())}"
    )

    #
    # Every legal move should have a
    # corresponding child.
    #

    legal_moves = {
        move.uci()
        for move in board.legal_moves
    }

    assert set(visits.keys()) == (
        legal_moves
    )

    #
    # The root must account for every
    # simulation.
    #

    assert root.visit_count == (
        SIMULATIONS
    )

    #
    # All child visits together must
    # account for all simulations.
    #

    assert sum(visits.values()) == (
        SIMULATIONS
    )

    #
    # It is NOT required that every child
    # is visited. PUCT can legitimately
    # leave low-prior moves at zero.
    #

    visited_children = sum(
        1
        for count in visits.values()
        if count > 0
    )

    print(
        f"Visited children: "
        f"{visited_children}/"
        f"{len(visits)}"
    )

    #
    # At least one child must be visited.
    #

    assert visited_children > 0

    #
    # Best move must correspond to the
    # highest visit count.
    #

    best_action = max(
        visits,
        key=visits.get,
    )

    assert result["move"].uci() == (
        best_action
    )

    #
    # Visit distribution
    #

    print()
    print("Visit distribution:")

    sorted_visits = sorted(
        visits.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for action, count in sorted_visits:

        if count > 0:

            print(
                f"  {action:<10} "
                f"{count:>4} "
                f"({count / SIMULATIONS:.1%})"
            )

    print()
    print(
        "✓ Visit distribution passed"
    )


# ============================================================
# TEST 4
# ============================================================

def test_policy(model):

    print()
    print("======================================")
    print("TEST 4 — MCTS policy")
    print("======================================")

    board = chess.variant.AtomicBoard()

    mcts = MCTS(
        model,
        simulations=SIMULATIONS,
        c_puct=1.5,
        device=DEVICE,
    )

    result = mcts.search(
        board
    )

    policy = result["policy"]

    visits = result["visits"]

    print(
        f"Policy entries: "
        f"{len(policy)}"
    )

    print(
        f"Legal moves: "
        f"{len(list(board.legal_moves))}"
    )

    #
    # Every legal move must appear.
    #

    assert len(policy) == len(
        list(board.legal_moves)
    )

    #
    # Policy must sum to approximately 1.
    #

    policy_sum = sum(
        policy.values()
    )

    print(
        f"Policy sum: "
        f"{policy_sum:.6f}"
    )

    assert abs(
        policy_sum - 1.0
    ) < 1e-5

    #
    # No negative probabilities.
    #

    assert all(
        p >= 0.0
        for p in policy.values()
    )

    #
    # Policy must correspond to visit
    # distribution.
    #

    total_visits = sum(
        visits.values()
    )

    for action in visits:

        expected = (
            visits[action]
            / total_visits
        )

        assert abs(
            policy[action]
            - expected
        ) < 1e-5

    #
    # Display top policy moves.
    #

    print()
    print("Top policy moves:")

    sorted_policy = sorted(
        policy.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for action, probability in (
        sorted_policy[:10]
    ):

        print(
            f"  {action:<10} "
            f"{probability:.3f}"
        )

    print()
    print(
        "✓ MCTS policy passed"
    )


# ============================================================
# TEST 5
# ============================================================

def test_terminal_position(model):

    print()
    print("======================================")
    print("TEST 5 — Terminal position")
    print("======================================")

    #
    # Atomic position where the game is
    # already over.
    #

    board = chess.variant.AtomicBoard(
        "4k3/"
        "8/"
        "8/"
        "8/"
        "8/"
        "8/"
        "8/"
        "4K3 "
        "w - - 0 1"
    )

    #
    # This particular position may be
    # interpreted as a draw depending on
    # Atomic rules. The important test is
    # that MCTS handles terminal states
    # without crashing.
    #

    print(
        f"Game over: "
        f"{board.is_game_over()}"
    )

    if board.is_game_over():

        mcts = MCTS(
            model,
            simulations=SIMULATIONS,
            c_puct=1.5,
            device=DEVICE,
        )

        result = mcts.search(
            board
        )

        assert result["move"] is None

        assert result["policy"] == {}

        assert result["root"].visit_count == 0

        print(
            "✓ Terminal position passed"
        )

    else:

        print(
            "⚠ Position was not terminal; "
            "terminal handling skipped"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("======================================")
    print("ALBERTA — MCTS TEST")
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

    test_initial_position(
        model
    )

    test_multiple_positions(
        model
    )

    test_visit_distribution(
        model
    )

    test_policy(
        model
    )

    test_terminal_position(
        model
    )

    print()
    print("======================================")
    print("ALL MCTS TESTS PASSED")
    print("======================================")


if __name__ == "__main__":

    main()