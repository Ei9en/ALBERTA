import torch


def compute_returns(
    rewards,
    gamma=0.99,
):
    """
    Compute discounted returns.

    rewards:
        list of floats

    returns:
        tensor of same length
    """

    returns = []

    G = 0.0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(
            0,
            G
        )

    return torch.tensor(
        returns,
        dtype=torch.float32,
    )