# tests/test_behavioral_mirror_bc.py

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import torch
import chess
import chess.variant

from train_rl_league_colab import ACTIONS

from src.models.resnet import ChessResNet


# ============================================================
# Configuration
# ============================================================

DEVICE = "cpu"

BC_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_v3_epoch_4.pt"
)

NUM_ACTIONS = 20160

NUM_TESTS = 100


# ============================================================
# Mirror
# ============================================================

def mirror_square(square):

    file = chess.square_file(square)
    rank = chess.square_rank(square)

    return chess.square(
        file,
        7 - rank,
    )


def mirror_and_swap_colors(board):

    mirrored = chess.variant.AtomicBoard.empty()

    for square, piece in board.piece_map().items():

        mirrored_square = mirror_square(
            square
        )

        mirrored_piece = chess.Piece(
            piece.piece_type,
            not piece.color,
        )

        mirrored.set_piece_at(
            mirrored_square,
            mirrored_piece,
        )

    mirrored.turn = not board.turn

    mirrored.clear_stack()

    return mirrored


# ============================================================
# Action mirror
# ============================================================

def mirror_uci(uci):

    if len(uci) < 4:

        raise ValueError(
            f"Invalid UCI action: {uci}"
        )

    from_square = chess.parse_square(
        uci[:2]
    )

    to_square = chess.parse_square(
        uci[2:4]
    )

    mirrored_from = mirror_square(
        from_square
    )

    mirrored_to = mirror_square(
        to_square
    )

    result = (
        chess.square_name(
            mirrored_from
        )
        +
        chess.square_name(
            mirrored_to
        )
    )

    # Promotion
    if len(uci) > 4:

        result += uci[4:]

    return result


# ============================================================
# Action map
# ============================================================

def build_action_maps():

    action_to_index = {
        str(action): index
        for index, action in enumerate(ACTIONS)
    }

    mirror_map = {}

    missing = []

    for index, action in enumerate(ACTIONS):

        action = str(action)

        mirrored_action = mirror_uci(
            action
        )

        if mirrored_action not in action_to_index:

            missing.append(
                (
                    action,
                    mirrored_action,
                )
            )

            continue

        mirror_map[index] = (
            action_to_index[
                mirrored_action
            ]
        )

    print(
        f"Actions loaded: {len(ACTIONS)}"
    )

    print(
        f"Mirrored actions found: "
        f"{len(mirror_map)}"
    )

    if missing:

        print(
            f"WARNING: "
            f"{len(missing)} actions have no mirror."
        )

        for action, mirrored in missing[:10]:

            print(
                f"  {action} -> {mirrored}"
            )

    if len(ACTIONS) != NUM_ACTIONS:

        raise RuntimeError(
            f"Expected {NUM_ACTIONS} actions, "
            f"got {len(ACTIONS)}"
        )

    return (
        action_to_index,
        mirror_map,
    )


# ============================================================
# Encoding
# ============================================================

PIECE_TO_CHANNEL = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def encode_board(board):

    planes = torch.zeros(
        (19, 8, 8),
        dtype=torch.float32,
    )

    for square, piece in board.piece_map().items():

        channel = PIECE_TO_CHANNEL[
            piece.symbol()
        ]

        row = (
            7
            - chess.square_rank(square)
        )

        col = chess.square_file(
            square
        )

        planes[
            channel,
            row,
            col,
        ] = 1.0

    # --------------------------------------------------------
    # Side to move
    # --------------------------------------------------------

    if board.turn:

        planes[12].fill_(1.0)

    # --------------------------------------------------------
    # Castling
    # --------------------------------------------------------

    if board.has_kingside_castling_rights(
        chess.WHITE
    ):

        planes[13].fill_(1.0)

    if board.has_queenside_castling_rights(
        chess.WHITE
    ):

        planes[14].fill_(1.0)

    if board.has_kingside_castling_rights(
        chess.BLACK
    ):

        planes[15].fill_(1.0)

    if board.has_queenside_castling_rights(
        chess.BLACK
    ):

        planes[16].fill_(1.0)

    # --------------------------------------------------------
    # En passant
    # --------------------------------------------------------

    if board.ep_square is not None:

        row = (
            7
            - chess.square_rank(
                board.ep_square
            )
        )

        col = chess.square_file(
            board.ep_square
        )

        planes[
            17,
            row,
            col,
        ] = 1.0

    # --------------------------------------------------------
    # Halfmove
    # --------------------------------------------------------

    planes[18].fill_(
        board.halfmove_clock / 100.0
    )

    return planes


# ============================================================
# Generate random position
# ============================================================

def generate_position():

    while True:

        board = chess.variant.AtomicBoard()

        moves = random.randint(
            8,
            30,
        )

        for _ in range(moves):

            if board.is_game_over():

                break

            legal_moves = list(
                board.legal_moves
            )

            if not legal_moves:

                break

            move = random.choice(
                legal_moves
            )

            board.push(move)

        legal_moves = list(
            board.legal_moves
        )

        if legal_moves:

            return board


# ============================================================
# Load BC
# ============================================================

def load_bc():

    print(
        f"Loading BC: {BC_CHECKPOINT}"
    )

    if not BC_CHECKPOINT.exists():

        print(
            "ERROR: BC checkpoint not found."
        )

        print(
            f"  {BC_CHECKPOINT}"
        )

        return None

    checkpoint = torch.load(
        BC_CHECKPOINT,
        map_location=DEVICE,
    )

    model = ChessResNet(
        num_actions=NUM_ACTIONS,
        channels=32,
        blocks=4,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"BC epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    return model


# ============================================================
# Policy
# ============================================================

@torch.no_grad()
def get_policy(
    model,
    board,
):

    x = encode_board(
        board
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    logits = model(x)

    return logits.squeeze(0)


# ============================================================
# Select best legal action
# ============================================================

def select_best_legal_action(
    logits,
    board,
    action_to_index,
):

    legal_indices = []
    legal_uci = []

    missing = []

    for move in board.legal_moves:

        uci = move.uci()

        if uci not in action_to_index:

            missing.append(
                uci
            )

            continue

        index = action_to_index[
            uci
        ]

        legal_indices.append(
            index
        )

        legal_uci.append(
            uci
        )

    if not legal_indices:

        print()
        print("=" * 70)
        print("ACTION SPACE MISMATCH")
        print("=" * 70)

        print()
        print("FEN:")
        print(board.fen())

        print()
        print("Turn:")

        print(
            "White"
            if board.turn
            else "Black"
        )

        print(
            f"python-chess legal moves : "
            f"{board.legal_moves.count()}"
        )

        print(
            f"Legal moves missing from ACTIONS : "
            f"{len(missing)}"
        )

        print()
        print("Missing moves:")

        for uci in missing[:100]:

            print(
                f"  {uci}"
            )

        print()
        print(
            f"ACTIONS size: "
            f"{len(action_to_index)}"
        )

        print()
        print("=" * 70)

        raise RuntimeError(
            "No legal actions found in ACTIONS."
        )

    legal_logits = logits[
        legal_indices
    ]

    best_position = torch.argmax(
        legal_logits
    ).item()

    return (
        legal_uci[
            best_position
        ],
        legal_indices[
            best_position
        ],
    )


# ============================================================
# Top-K legal actions
# ============================================================

def get_top_k_legal_actions(
    logits,
    board,
    action_to_index,
    k=5,
):

    candidates = []

    for move in board.legal_moves:

        uci = move.uci()

        if uci not in action_to_index:

            continue

        index = action_to_index[
            uci
        ]

        candidates.append(
            (
                logits[index].item(),
                uci,
                index,
            )
        )

    candidates.sort(
        reverse=True
    )

    return candidates[:k]


# ============================================================
# Test BC
# ============================================================

def test_model(
    model,
    name,
    positions,
    action_to_index,
    mirror_map,
):

    top1_correct = 0
    top5_contains = 0

    valid_tests = 0

    print()
    print(
        "----------------------------------------"
    )

    print(
        name
    )

    print(
        "----------------------------------------"
    )

    for i, board in enumerate(
        positions
    ):

        mirrored = (
            mirror_and_swap_colors(
                board
            )
        )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        original_logits = (
            get_policy(
                model,
                board,
            )
        )

        # ----------------------------------------------------
        # Mirrored
        # ----------------------------------------------------

        mirrored_logits = (
            get_policy(
                model,
                mirrored,
            )
        )

        # ----------------------------------------------------
        # Best original action
        # ----------------------------------------------------

        (
            original_action,
            original_index,
        ) = select_best_legal_action(
            original_logits,
            board,
            action_to_index,
        )

        # ----------------------------------------------------
        # Best mirrored action
        # ----------------------------------------------------

        (
            mirrored_action,
            mirrored_index,
        ) = select_best_legal_action(
            mirrored_logits,
            mirrored,
            action_to_index,
        )

        # ----------------------------------------------------
        # Expected mirrored action
        # ----------------------------------------------------

        expected_mirror_index = (
            mirror_map.get(
                original_index
            )
        )

        if expected_mirror_index is None:

            print(
                f"{i + 1:3d} | "
                f"SKIP: no mirror for "
                f"{original_action}"
            )

            continue

        expected_mirror_action = str(
            ACTIONS[
                expected_mirror_index
            ]
        )

        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        top1_match = (
            mirrored_index
            == expected_mirror_index
        )

        # ----------------------------------------------------
        # Top-5
        # ----------------------------------------------------

        mirrored_top5 = (
            get_top_k_legal_actions(
                mirrored_logits,
                mirrored,
                action_to_index,
                k=5,
            )
        )

        mirrored_top5_indices = {
            index
            for (
                _,
                _,
                index,
            )
            in mirrored_top5
        }

        top5_match = (
            expected_mirror_index
            in mirrored_top5_indices
        )

        if top1_match:

            top1_correct += 1

        if top5_match:

            top5_contains += 1

        valid_tests += 1

        side = (
            "w"
            if board.turn
            else "b"
        )

        print(
            f"{i + 1:3d} | "
            f"side={side} | "
            f"{original_action:6s} -> "
            f"expected={expected_mirror_action:6s} | "
            f"actual={mirrored_action:6s} | "
            f"top1="
            f"{'YES' if top1_match else 'NO '} | "
            f"top5="
            f"{'YES' if top5_match else 'NO '}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if valid_tests == 0:

        print()
        print(
            "No valid tests."
        )

        return

    top1_rate = (
        top1_correct
        / valid_tests
    )

    top5_rate = (
        top5_contains
        / valid_tests
    )

    print()

    print(
        f"{name} TOP-1 symmetry : "
        f"{top1_correct}/{valid_tests} "
        f"({top1_rate:.1%})"
    )

    print(
        f"{name} TOP-5 symmetry : "
        f"{top5_contains}/{valid_tests} "
        f"({top5_rate:.1%})"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "BC BEHAVIORAL MIRROR SYMMETRY"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    (
        action_to_index,
        mirror_map,
    ) = build_action_maps()

    # --------------------------------------------------------
    # BC
    # --------------------------------------------------------

    print()

    model = load_bc()

    if model is None:

        return

    # --------------------------------------------------------
    # Generate identical test positions
    # --------------------------------------------------------

    random.seed(42)

    positions = []

    for _ in range(
        NUM_TESTS
    ):

        positions.append(
            generate_position()
        )

    # --------------------------------------------------------
    # Test BC
    # --------------------------------------------------------

    test_model(
        model,
        "BC",
        positions,
        action_to_index,
        mirror_map,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "TEST COMPLETED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "TOP-1 = exact behavioral symmetry"
    )

    print(
        "TOP-5 = mirrored action remains "
        "among the five preferred legal actions"
    )

    print()


if __name__ == "__main__":

    main()