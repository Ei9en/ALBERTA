import time
import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.agents.actor_critic_agent import ActorCriticAgent

from src.selfplay.batch_selfplay import BatchedSelfPlayGame
from src.selfplay.game import SelfPlayGame

from src.actions_space import ACTIONS


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


CHECKPOINT = (
    "checkpoints/bc_epoch/bc_v2_5_epoch_5.pt"
)


def load_model():

    base = ChessResNet(
        num_actions=len(ACTIONS),
        channels=64,
        blocks=4,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    base.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = ActorCritic(base)

    model.to(DEVICE)
    model.eval()

    return model


def main():

    print("Device:", DEVICE)

    model = load_model()

    agent = ActorCriticAgent(
        model,
        deterministic=False,
        temperature=0.75,
        device=DEVICE,
    )

    #
    # =========================
    # Ancien self-play
    # =========================
    #

    print()
    print("=== OLD SELF-PLAY ===")

    start = time.perf_counter()

    old_results = []

    for i in range(4):

        game = SelfPlayGame(
            agent,
            agent,
        )

        trajectory, result = game.play()

        old_results.append(
            (
                len(trajectory),
                result,
            )
        )

    old_time = (
        time.perf_counter()
        - start
    )

    print(
        "Results:",
        old_results,
    )

    print(
        f"Time: {old_time:.3f} s"
    )


    #
    # =========================
    # Nouveau self-play
    # =========================
    #

    print()
    print("=== BATCHED SELF-PLAY ===")

    white_agents = [
        agent
        for _ in range(4)
    ]

    black_agents = [
        agent
        for _ in range(4)
    ]

    batched = BatchedSelfPlayGame(
        white_agents,
        black_agents,
    )

    start = time.perf_counter()

    new_results = batched.play()

    new_time = (
        time.perf_counter()
        - start
    )

    print(
        "Results:",
        [
            (
                len(trajectory),
                result,
            )
            for trajectory, result
            in new_results
        ],
    )

    print(
        f"Time: {new_time:.3f} s"
    )


    #
    # =========================
    # Comparaison
    # =========================
    #

    print()
    print("=== COMPARISON ===")

    print(
        f"Old:     {old_time:.3f} s"
    )

    print(
        f"Batched: {new_time:.3f} s"
    )

    if new_time > 0:

        print(
            f"Speedup: "
            f"{old_time / new_time:.2f}x"
        )


if __name__ == "__main__":
    main()