# ============================================================
# tournament_bc.py
# ============================================================

from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch
import chess
import chess.variant

from src.models.resnet import ChessResNet
from src.encoding import encode_fen
from src.actions_space import ACTIONS, INDEX_TO_ACTION


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_DIR = Path(
    "checkpoints/bc_epoch"
)

TEMPERATURE = 1
CHANNELS = 32
BLOCKS = 4
SEED = 1337

N = 10

MAX_PLIES = 1000


# ============================================================
# Load model
# ============================================================

def load_model(path, device):

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model = ChessResNet(
        num_actions=len(ACTIONS),
        channels=CHANNELS,
        blocks=BLOCKS,
    ).to(device)

    assert checkpoint["actions"] == len(ACTIONS), (
        f"Action space mismatch in {path}"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# Select move
# ============================================================

# ============================================================
# Select move
# ============================================================

def select_move(
    model,
    board,
    device,
):

    x = encode_fen(
        board.fen()
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)[0]

    # --------------------------------------------------------
    # Mask illegal actions
    # --------------------------------------------------------

    legal = {
        move.uci()
        for move in board.legal_moves
    }

    masked_logits = torch.full_like(
        logits,
        -float("inf")
    )

    for idx, uci in INDEX_TO_ACTION.items():

        if uci in legal:

            masked_logits[idx] = logits[idx]

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if TEMPERATURE == 0:

        # Greedy / deterministic
        best_idx = (
            masked_logits.argmax()
        ).item()

    else:

        # ----------------------------------------------------
        # Softmax sampling
        #
        # T = 1:
        # distribution directly induced by the logits
        # ----------------------------------------------------

        probabilities = torch.softmax(
            masked_logits / TEMPERATURE,
            dim=0,
        )

        best_idx = torch.multinomial(
            probabilities,
            num_samples=1,
        ).item()

    return chess.Move.from_uci(
        INDEX_TO_ACTION[best_idx]
    )


# ============================================================
# Play one game
# ============================================================

def play_game(
    white_model,
    black_model,
    device,
):

    board = chess.variant.AtomicBoard()

    for _ in range(MAX_PLIES):

        if board.turn == chess.WHITE:

            model = white_model

        else:

            model = black_model

        move = select_move(
            model,
            board,
            device,
        )

        board.push(move)

        if board.is_game_over():

            return board.result()

    # --------------------------------------------------------
    # Maximum length reached
    # --------------------------------------------------------

    return "*"


# ============================================================
# Play a match
# ============================================================

# ============================================================
# Play a match
# ============================================================

def play_match(
    model_a,
    model_b,
    device,
):

    results = []

    for game in range(N):

        # ----------------------------------------------------
        # Alternate colors
        # ----------------------------------------------------

        if game % 2 == 0:

            # A = White
            # B = Black

            result = play_game(
                model_a,
                model_b,
                device,
            )

            results.append(
                (model_a, model_b, result)
            )

        else:

            # B = White
            # A = Black

            result = play_game(
                model_b,
                model_a,
                device,
            )

            results.append(
                (model_b, model_a, result)
            )

    return results


# ============================================================
# Update tournament score
# ============================================================

def update_score(
    scores,
    model_a,
    model_b,
    result,
):

    # result is from White's perspective
    #
    # 1-0 : White wins
    # 0-1 : Black wins
    # 1/2-1/2 : draw

    if result == "1-0":

        scores[model_a]["wins"] += 1
        scores[model_b]["losses"] += 1

    elif result == "0-1":

        scores[model_a]["losses"] += 1
        scores[model_b]["wins"] += 1

    elif result == "1/2-1/2":

        scores[model_a]["draws"] += 1
        scores[model_b]["draws"] += 1

    else:

        scores[model_a]["draws"] += 1
        scores[model_b]["draws"] += 1


# ============================================================
# Main
# ============================================================

def main():

    torch.manual_seed(SEED)

    # ========================================================
    # Device
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("BC ROUND-ROBIN TOURNAMENT")
    print("=" * 60)

    print()
    print("Device:", device)
    print("Temperature:", TEMPERATURE)
    print("Games per matchup:", N)
    print("Seed:", SEED)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # Find checkpoints
    # ========================================================

    checkpoints = sorted(
        CHECKPOINT_DIR.glob(
            "bc_epoch_*.pt"
        ),
        key=lambda p: int(
            p.stem.split("_")[-1]
        ),
    )

    if len(checkpoints) < 2:

        raise RuntimeError(
            "At least two BC checkpoints are required."
        )

    print()
    print(
        f"Found {len(checkpoints)} checkpoints."
    )

    # ========================================================
    # Load models
    # ========================================================

    models = {}

    for path in checkpoints:

        name = path.stem

        print(
            f"Loading {name}..."
        )

        models[name] = load_model(
            path,
            device,
        )

    # ========================================================
    # Initialize scores
    # ========================================================

    scores = {}

    for name in models:

        scores[name] = {

            "wins": 0,

            "draws": 0,

            "losses": 0,

            "points": 0.0,
        }

    total_games = 0

    # ========================================================
    # Round robin
    # ========================================================

    print()
    print("=" * 60)
    print("TOURNAMENT")
    print("=" * 60)

    for i in range(
        len(checkpoints)
    ):

        for j in range(
            i + 1,
            len(checkpoints)
        ):

            name_a = (
                checkpoints[i].stem
            )

            name_b = (
                checkpoints[j].stem
            )

            model_a = models[name_a]
            model_b = models[name_b]

            print()
            print(
                f"{name_a} vs {name_b}"
            )

            # ------------------------------------------------
            # Play N games
            # ------------------------------------------------

            results = play_match(
                model_a,
                model_b,
                device,
            )

            # ------------------------------------------------
            # Process results
            # ------------------------------------------------

            for (
                white_model,
                black_model,
                result,
            ) in results:

                # --------------------------------------------
                # Identify players
                # --------------------------------------------

                if white_model is model_a:

                    white_name = name_a
                    black_name = name_b

                else:

                    white_name = name_b
                    black_name = name_a

                # --------------------------------------------
                # Result
                # --------------------------------------------

                if result == "1-0":

                    scores[
                        white_name
                    ]["wins"] += 1

                    scores[
                        black_name
                    ]["losses"] += 1

                    scores[
                        white_name
                    ]["points"] += 1.0

                elif result == "0-1":

                    scores[
                        white_name
                    ]["losses"] += 1

                    scores[
                        black_name
                    ]["wins"] += 1

                    scores[
                        black_name
                    ]["points"] += 1.0

                else:

                    # ----------------------------------------
                    # Draw or unfinished game
                    # ----------------------------------------

                    scores[
                        white_name
                    ]["draws"] += 1

                    scores[
                        black_name
                    ]["draws"] += 1

                    scores[
                        white_name
                    ]["points"] += 0.5

                    scores[
                        black_name
                    ]["points"] += 0.5

                total_games += 1

            # ------------------------------------------------
            # Match summary
            # ------------------------------------------------

            a_points = (
                scores[name_a]["points"]
            )

            b_points = (
                scores[name_b]["points"]
            )

            # ------------------------------------------------
            # Count only this matchup
            # ------------------------------------------------

            matchup_a_wins = 0
            matchup_b_wins = 0
            matchup_draws = 0

            for (
                white_model,
                black_model,
                result,
            ) in results:

                if result == "1-0":

                    if white_model is model_a:
                        matchup_a_wins += 1
                    else:
                        matchup_b_wins += 1

                elif result == "0-1":

                    if black_model is model_a:
                        matchup_a_wins += 1
                    else:
                        matchup_b_wins += 1

                else:

                    matchup_draws += 1

            print(
                f"  {name_a}: "
                f"{matchup_a_wins}W "
                f"{matchup_draws}D "
                f"{matchup_b_wins}L"
            )

            print(
                f"  {name_b}: "
                f"{matchup_b_wins}W "
                f"{matchup_draws}D "
                f"{matchup_a_wins}L"
            )

    # ========================================================
    # Final ranking
    # ========================================================

    ranking = sorted(
        scores.items(),
        key=lambda item: (
            -item[1]["points"],
            -item[1]["wins"],
            item[1]["losses"],
        ),
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL RANKING")
    print("=" * 60)

    print()

    print(
        f"{'Rank':<6}"
        f"{'Model':<20}"
        f"{'W':>5}"
        f"{'D':>5}"
        f"{'L':>5}"
        f"{'Points':>10}"
    )

    print("-" * 60)

    for rank, (name, data) in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{name:<20}"
            f"{data['wins']:>5}"
            f"{data['draws']:>5}"
            f"{data['losses']:>5}"
            f"{data['points']:>10.1f}"
        )

    print()
    print(
        f"Total games: {total_games}"
    )

    print(
        f"Games per matchup: {N}"
    )

    print(
        f"Matchups: "
        f"{len(checkpoints) * (len(checkpoints) - 1) // 2}"
    )

    print("=" * 60)

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()