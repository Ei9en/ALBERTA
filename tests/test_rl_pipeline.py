# Test_RL.py

from pathlib import Path
import torch

from train_rl import (
    load_model,
    collect_games,
    train_epoch,
    save_checkpoint,
    GAMES_PER_EPOCH,
)

from src.models.actor_critic import ActorCritic


def main():

    print("=== RL PIPELINE TEST ===")


    #
    # 1) Chargement du modèle BC + ActorCritic
    #
    print("\n[1] Loading model...")

    model, optimizer = load_model()

    print("Model loaded.")


    #
    # 2) Une seule partie de self-play
    #
    print("\n[2] Self-play test...")

    games = collect_games(
        model,
        n_games=1,
    )

    assert len(games) == 1

    trajectory = games[0]["trajectory"]
    result = games[0]["result"]

    print(
        f"Game finished : {result}"
    )

    print(
        f"Positions collected : {len(trajectory)}"
    )

    assert len(trajectory) > 0


    #
    # 3) Test entraînement A2C
    #
    print("\n[3] Training step...")

    loss = train_epoch(
        model,
        optimizer,
        games,
    )

    print(
        f"Loss : {loss}"
    )

    assert loss == loss  # NaN check
    assert torch.isfinite(torch.tensor(loss))


    #
    # 4) Sauvegarde checkpoint
    #
    print("\n[4] Saving checkpoint...")

    save_checkpoint(
        model,
        optimizer,
        epoch=0,
        loss=loss,
    )

    checkpoint = Path(
        "checkpoints/rl_epoch_0.pt"
    )

    assert checkpoint.exists()

    print(
        "Checkpoint created."
    )


    #
    # 5) Reload du checkpoint
    #
    print("\n[5] Reload test...")

    data = torch.load(
        checkpoint,
        map_location="cpu",
    )

    assert "model_state_dict" in data
    assert "optimizer_state_dict" in data

    print(
        "Reload successful."
    )


    print(
        "\n=== RL PIPELINE OK ==="
    )


if __name__ == "__main__":
    main()