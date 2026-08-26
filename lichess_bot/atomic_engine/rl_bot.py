from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.encoding import encode_board

from src.models.resnet import ChessResNet
from src.models.actor_critic import ActorCritic

from src.actions_space import ACTIONS
from src.actions_space import ACTION_TO_INDEX

from src.agents.ppo_agent import PPOAgent


DEFAULT_CHECKPOINT = (
    Path("/Users/tom/Desktop/Atomic")
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_40.pt"
)


class RLBot:

    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        temperature=1.5,
        deterministic=False,
    ):

        self.device = torch.device("cpu")

        #
        # Backbone BC
        #

        bc_model = ChessResNet(
            num_actions=len(ACTIONS),
            channels=32,
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

    @torch.no_grad()
    def evaluate_policy(self, board):
        """
        Return the policy distribution over LEGAL moves.

        The distribution is the intrinsic policy:
        no temperature is applied.
        """

        x = encode_board(board)
        x = x.unsqueeze(0).to(self.device)

        policy, value = self.model(x)

        logits = policy[0]

        legal_moves = list(board.legal_moves)

        legal_uci = [
            move.uci()
            for move in legal_moves
        ]

        legal_indices = [
            ACTION_TO_INDEX[uci]
            for uci in legal_uci
        ]

        legal_logits = logits[
            legal_indices
        ]

        log_probs = torch.log_softmax(
            legal_logits,
            dim=0,
        )

        probs = torch.exp(
            log_probs
        )

        entropy = -(
            probs * log_probs
        ).sum().item()

        return {
            "moves": legal_uci,
            "probs": probs.cpu().tolist(),
            "log_probs": log_probs.cpu().tolist(),
            "entropy": entropy,
            "value": value.item(),
        }