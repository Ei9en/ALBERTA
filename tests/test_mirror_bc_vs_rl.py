# tests/test_mirror_bc_rl.py

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

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic


# ============================================================
# Configuration
# ============================================================

DEVICE = "cpu"

BC5_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "bc_epoch"
    / "bc_v2_5_epoch_5.pt"
)

RL0_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_0.pt"
)

RL10_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10_baseline.pt"
)

NUM_TESTS = 10

NUM_ACTIONS = 20160


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

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        planes[
            channel,
            row,
            col,
        ] = 1.0

    # Side to move
    if board.turn:

        planes[12].fill_(1.0)

    # Castling
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

    # En passant
    if board.ep_square is not None:

        row = 7 - chess.square_rank(
            board.ep_square
        )

        col = chess.square_file(
            board.ep_square
        )

        planes[
            17,
            row,
            col,
        ] = 1.0

    # Halfmove clock
    planes[18].fill_(
        board.halfmove_clock / 100.0
    )

    return planes


# ============================================================
# Mirror + color swap
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
# Action transformation
# ============================================================

def mirror_uci(uci):

    """
    Transforme un coup UCI sous :

        vertical mirror
        +
        color swap

    Exemple :

        e2e4 -> e7e5

    Les couleurs ne sont pas explicitement encodées
    dans un coup UCI, donc seul le miroir géométrique
    des cases est nécessaire.
    """

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
        chess.square_name(mirrored_from)
        +
        chess.square_name(mirrored_to)
    )

    # Promotion éventuelle
    if len(uci) > 4:

        result += uci[4:]

    return result


# ============================================================
# Build action mirror map
# ============================================================

def build_action_mirror_map(actions):

    """
    Construit :

        original_action_index
            ->
        mirrored_action_index
    """

    action_to_index = {
        str(action): i
        for i, action in enumerate(actions)
    }

    mirror_map = []

    missing = []

    for action in actions:

        mirrored_action = mirror_uci(
            str(action)
        )

        if mirrored_action not in action_to_index:

            missing.append(
                (
                    str(action),
                    mirrored_action,
                )
            )

            mirror_map.append(
                None
            )

        else:

            mirror_map.append(
                action_to_index[
                    mirrored_action
                ]
            )

    if missing:

        print()
        print(
            "WARNING:"
        )

        print(
            f"{len(missing)} actions "
            f"have no mirrored action."
        )

        for original, mirrored in missing[:10]:

            print(
                f"  {original} -> {mirrored}"
            )

        print()

    return mirror_map


# ============================================================
# Random positions
# ============================================================

def generate_position():

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

    return board


# ============================================================
# Load BC5
# ============================================================

def load_bc5():

    print(
        "Loading BC5..."
    )

    checkpoint = torch.load(
        BC5_CHECKPOINT,
        map_location=DEVICE,
    )

    model = ChessResNet(
        num_actions=NUM_ACTIONS,
        channels=64,
        blocks=4,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"BC5 epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    return model


# ============================================================
# Load RL
# ============================================================

def load_rl(path):

    print(
        f"Loading RL: {path}"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    backbone = ChessResNet(
        num_actions=NUM_ACTIONS,
        channels=64,
        blocks=4,
    )

    model = ActorCritic(
        backbone
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"RL epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    return model


# ============================================================
# BC policy
# ============================================================

@torch.no_grad()
def get_bc_policy(
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

    policy = model(x)

    return policy.squeeze(0)


# ============================================================
# RL policy + value
# ============================================================

@torch.no_grad()
def rl_forward(
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

    policy, value = model(x)

    return (
        policy.squeeze(0),
        value.item(),
    )


# ============================================================
# Mirror policy
# ============================================================

def mirror_policy(
    policy,
    mirror_map,
):

    mirrored = torch.zeros_like(
        policy
    )

    for original_idx, mirrored_idx in enumerate(
        mirror_map
    ):

        if mirrored_idx is None:
            continue

        mirrored[
            mirrored_idx
        ] = policy[
            original_idx
        ]

    return mirrored


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )
    print(
        "BC5 vs RL0 vs RL10 — "
        "CORRECTED MIRROR SYMMETRY"
    )
    print(
        "=" * 70
    )
    print()

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    try:

        from train_rl_league_colab import ACTIONS

    except ImportError as e:

        print()
        print(
            "ERROR: impossible to import ACTIONS."
        )

        print(e)

        print()
        print(
            "The test requires the exact ACTIONS "
            "used by the model."
        )

        return

    print(
        f"Actions loaded: {len(ACTIONS)}"
    )

    if len(ACTIONS) != NUM_ACTIONS:

        raise RuntimeError(
            f"Expected {NUM_ACTIONS} actions, "
            f"but got {len(ACTIONS)}."
        )

    print(
        "Building action mirror map..."
    )

    mirror_map = (
        build_action_mirror_map(
            ACTIONS
        )
    )

    print(
        "Action mirror map ready."
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print()

    bc5 = load_bc5()

    print()

    rl0 = load_rl(
        RL0_CHECKPOINT
    )

    print()

    rl10 = load_rl(
        RL10_CHECKPOINT
    )

    print()

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    bc_policy_errors = []

    rl0_policy_errors = []
    rl0_value_errors = []

    rl10_policy_errors = []
    rl10_value_errors = []

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    for i in range(
        NUM_TESTS
    ):

        board = generate_position()

        mirrored = (
            mirror_and_swap_colors(
                board
            )
        )

        # ====================================================
        # BC5
        # ====================================================

        bc_original = get_bc_policy(
            bc5,
            board,
        )

        bc_mirrored_raw = get_bc_policy(
            bc5,
            mirrored,
        )

        bc_mirrored = mirror_policy(
            bc_mirrored_raw,
            mirror_map,
        )

        bc_policy_error = (
            torch.mean(
                torch.abs(
                    bc_original
                    - bc_mirrored
                )
            ).item()
        )

        # ====================================================
        # RL0
        # ====================================================

        (
            rl0_original,
            rl0_value,
        ) = rl_forward(
            rl0,
            board,
        )

        (
            rl0_mirrored_raw,
            rl0_value_mirror,
        ) = rl_forward(
            rl0,
            mirrored,
        )

        rl0_mirrored = mirror_policy(
            rl0_mirrored_raw,
            mirror_map,
        )

        rl0_policy_error = (
            torch.mean(
                torch.abs(
                    rl0_original
                    - rl0_mirrored
                )
            ).item()
        )

        rl0_value_error = abs(
            rl0_value
            + rl0_value_mirror
        )

        # ====================================================
        # RL10
        # ====================================================

        (
            rl10_original,
            rl10_value,
        ) = rl_forward(
            rl10,
            board,
        )

        (
            rl10_mirrored_raw,
            rl10_value_mirror,
        ) = rl_forward(
            rl10,
            mirrored,
        )

        rl10_mirrored = mirror_policy(
            rl10_mirrored_raw,
            mirror_map,
        )

        rl10_policy_error = (
            torch.mean(
                torch.abs(
                    rl10_original
                    - rl10_mirrored
                )
            ).item()
        )

        rl10_value_error = abs(
            rl10_value
            + rl10_value_mirror
        )

        # ====================================================
        # Store
        # ====================================================

        bc_policy_errors.append(
            bc_policy_error
        )

        rl0_policy_errors.append(
            rl0_policy_error
        )

        rl0_value_errors.append(
            rl0_value_error
        )

        rl10_policy_errors.append(
            rl10_policy_error
        )

        rl10_value_errors.append(
            rl10_value_error
        )

        # ====================================================
        # Print
        # ====================================================

        side = (
            "w"
            if board.turn
            else "b"
        )

        print(
            f"{i + 1:2d} | "
            f"side={side} | "
            f"BC policy={bc_policy_error:.6f} | "
            f"RL0 policy={rl0_policy_error:.6f} | "
            f"RL0 value={rl0_value_error:.6f} | "
            f"RL10 policy={rl10_policy_error:.6f} | "
            f"RL10 value={rl10_value_error:.6f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 70
    )

    print()

    print(
        f"BC5  policy mean error : "
        f"{sum(bc_policy_errors) / len(bc_policy_errors):.6f}"
    )

    print(
        f"RL0  policy mean error : "
        f"{sum(rl0_policy_errors) / len(rl0_policy_errors):.6f}"
    )

    print(
        f"RL0  value  mean error : "
        f"{sum(rl0_value_errors) / len(rl0_value_errors):.6f}"
    )

    print(
        f"RL10 policy mean error : "
        f"{sum(rl10_policy_errors) / len(rl10_policy_errors):.6f}"
    )

    print(
        f"RL10 value  mean error : "
        f"{sum(rl10_value_errors) / len(rl10_value_errors):.6f}"
    )

    print()
    print(
        "=" * 70
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  BC5  -> policy symmetry before RL"
    )

    print(
        "  RL0  -> policy after BC -> ActorCritic"
    )

    print(
        "  RL10 -> policy/value after PPO"
    )

    print()


if __name__ == "__main__":

    main()