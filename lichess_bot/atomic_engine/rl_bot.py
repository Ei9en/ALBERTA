from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic
from src.actions_space import ACTIONS
from src.agents.ppo_agent import PPOAgent


DEFAULT_CHECKPOINT = (
    Path("/Users/tom/Desktop/Atomic")
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)


class RLBot:

    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        temperature=0.75,
        deterministic=False,
    ):

        self.device = torch.device("cpu")

        #
        # Backbone BC
        #

        bc_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=64,
            blocks=4,
        )


        #
        # Actor-Critic
        #

        self.model = ActorCritic(
            bc_model
        ).to(self.device)


        #
        # Checkpoint RL
        #

        checkpoint_data = torch.load(
            checkpoint,
            map_location=self.device,
        )


        self.model.load_state_dict(
            checkpoint_data["model_state_dict"]
        )


        self.model.eval()


        #
        # Agent PPO
        #

        self.agent = PPOAgent(
            self.model,
            device="cpu",
            deterministic=deterministic,
            temperature=temperature,
        )


        print(
            f"Loaded RL checkpoint: {checkpoint}"
        )


    @torch.no_grad()
    def choose_move(
        self,
        board,
    ):

        return self.agent.choose_move(
            board
        )