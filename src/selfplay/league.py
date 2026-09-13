from __future__ import annotations

import random
from collections.abc import Iterable

import torch
from torch import nn


class League:
    """
    Historical league of actor-critic agents.

    The league serves two distinct purposes:

    1. opponent sampling for self-play;
    2. value-disagreement estimation for uncertainty U(s).

    Not every league member necessarily contributes to U(s).
    In particular, behavioral-cloning anchors may contain an
    untrained value head and can therefore be kept as opponents
    while being excluded from uncertainty estimation.

    League uncertainty is defined as the population variance:

        U(s) = Var_k[V_k(s)]

    over uncertainty-eligible historical agents and, optionally,
    the current model.
    """

    def __init__(
        self,
        max_agents: int = 22,
        protected_agents: Iterable[str] | None = None,
    ) -> None:

        if max_agents <= 0:
            raise ValueError(
                "max_agents must be greater than zero."
            )

        self.max_agents = max_agents

        # Agents protected from league pruning.
        self.protected_agents = set(
            protected_agents or []
        )

        # Python dictionaries preserve insertion order.
        # This allows deterministic FIFO-style pruning of
        # non-protected historical agents.
        self.agents: dict[
            str,
            nn.Module,
        ] = {}

        # Names of league members whose value heads are allowed
        # to contribute to U(s).
        self._uncertainty_agents: set[str] = set()


    # ========================================================
    # League management
    # ========================================================

    def add_agent(
        self,
        name: str,
        model: nn.Module,
        use_for_uncertainty: bool = True,
    ) -> None:
        """
        Add or replace a league member.

        Parameters
        ----------
        name
            Unique league-member name.

        model
            Actor-critic model used for inference.

            The caller is responsible for providing an
            independent snapshot when historical parameters
            must remain frozen.

        use_for_uncertainty
            Whether this model's value prediction contributes
            to league uncertainty U(s).

            This should be False for BC anchors whose value head
            was never trained.

        Notes
        -----
        When the league exceeds max_agents, the oldest
        non-protected member is removed.
        """

        model.eval()

        self.agents[
            name
        ] = model

        # ----------------------------------------------------
        # Uncertainty eligibility
        # ----------------------------------------------------

        if use_for_uncertainty:

            self._uncertainty_agents.add(
                name
            )

        else:

            self._uncertainty_agents.discard(
                name
            )

        # ----------------------------------------------------
        # Pruning
        # ----------------------------------------------------

        while len(self.agents) > self.max_agents:

            removable = [
                agent_name
                for agent_name in self.agents
                if agent_name
                not in self.protected_agents
            ]

            if not removable:

                raise RuntimeError(
                    "League exceeds max_agents, but all "
                    "registered agents are protected."
                )

            oldest = removable[
                0
            ]

            del self.agents[
                oldest
            ]

            self._uncertainty_agents.discard(
                oldest
            )


    # ========================================================
    # Opponent sampling
    # ========================================================

    def sample_opponent(
        self,
        exclude: str | None = None,
    ) -> tuple[str, nn.Module]:
        """
        Uniformly sample one eligible self-play opponent.

        All league members can be sampled as opponents,
        independently of whether they participate in U(s).

        Randomness is controlled by the caller's global Python
        RNG state. This class intentionally does not reseed it.
        """

        candidates = [
            (name, model)
            for name, model in self.agents.items()
            if name != exclude
        ]

        if not candidates:

            raise RuntimeError(
                "No eligible opponent available in league."
            )

        return random.choice(
            candidates
        )


    # ========================================================
    # Value predictions
    # ========================================================

    @torch.no_grad()
    def values(
        self,
        x: torch.Tensor,
    ) -> list[float]:
        """
        Return value predictions from all league members.

        This helper preserves the general league semantics and
        therefore includes every agent, including agents that
        may be excluded from U(s).

        The input must contain exactly one position.
        """

        if x.shape[0] != 1:

            raise ValueError(
                "values() expects a batch containing exactly "
                "one position."
            )

        values = []

        for model in self.agents.values():

            model.eval()

            _, value = model(
                x
            )

            values.append(
                value.item()
            )

        return values


    # ========================================================
    # Single-position uncertainty
    # ========================================================

    @torch.no_grad()
    def uncertainty(
        self,
        x: torch.Tensor,
        current_model: nn.Module | None = None,
    ) -> float:
        """
        Compute value-disagreement uncertainty for one state.

        Only historical agents explicitly marked with
        use_for_uncertainty=True participate.

        The current model is optionally included regardless of
        league membership.

        If fewer than two eligible value estimates are
        available, uncertainty is defined as 0.

        The training/evaluation mode of current_model is
        restored before returning.
        """

        if x.shape[0] != 1:

            raise ValueError(
                "uncertainty() expects a batch containing "
                "exactly one position."
            )

        values = []

        # ----------------------------------------------------
        # Eligible historical snapshots
        # ----------------------------------------------------

        for name, model in self.agents.items():

            if name not in self._uncertainty_agents:
                continue

            model.eval()

            _, value = model(
                x
            )

            values.append(
                value.item()
            )

        # ----------------------------------------------------
        # Current model
        # ----------------------------------------------------

        if current_model is not None:

            was_training = (
                current_model.training
            )

            current_model.eval()

            try:

                _, value = current_model(
                    x
                )

                values.append(
                    value.item()
                )

            finally:

                current_model.train(
                    was_training
                )

        # ----------------------------------------------------
        # At least two estimates are required for disagreement
        # ----------------------------------------------------

        if len(values) < 2:

            return 0.0

        value_tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device=x.device,
        )

        return torch.var(
            value_tensor,
            unbiased=False,
        ).item()


    # ========================================================
    # Batched uncertainty
    # ========================================================

    @torch.no_grad()
    def uncertainty_batch(
        self,
        x: torch.Tensor,
        current_model: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute league uncertainty for a batch of states.

        Parameters
        ----------
        x
            Encoded board batch with shape:

                [N, 19, 8, 8]

        current_model
            Current actor-critic model. Its value predictions
            are optionally added to the historical ensemble.

        Returns
        -------
        torch.Tensor
            Population variance across eligible value models,
            with shape:

                [N]

            If fewer than two eligible value estimates exist,
            a zero vector is returned.
        """

        all_values = []

        # ----------------------------------------------------
        # Eligible historical snapshots
        # ----------------------------------------------------

        for name, model in self.agents.items():

            if name not in self._uncertainty_agents:
                continue

            model.eval()

            _, values = model(
                x
            )

            values = values.squeeze(
                -1
            )

            all_values.append(
                values
            )

        # ----------------------------------------------------
        # Current model
        # ----------------------------------------------------

        if current_model is not None:

            was_training = (
                current_model.training
            )

            current_model.eval()

            try:

                _, values = current_model(
                    x
                )

                values = values.squeeze(
                    -1
                )

                all_values.append(
                    values
                )

            finally:

                current_model.train(
                    was_training
                )

        # ----------------------------------------------------
        # Not enough estimates for disagreement
        # ----------------------------------------------------

        if len(all_values) < 2:

            return torch.zeros(
                x.shape[0],
                dtype=torch.float32,
                device=x.device,
            )

        # ----------------------------------------------------
        # [agents, batch]
        # ----------------------------------------------------

        value_tensor = torch.stack(
            all_values,
            dim=0,
        )

        # ----------------------------------------------------
        # Population variance across value models
        # ----------------------------------------------------

        return torch.var(
            value_tensor,
            dim=0,
            unbiased=False,
        )


    # ========================================================
    # Metadata
    # ========================================================

    def uncertainty_names(
        self,
    ) -> list[str]:
        """
        Return league members currently contributing to U(s),
        in league insertion order.
        """

        return [
            name
            for name in self.agents
            if name in self._uncertainty_agents
        ]


    def names(
        self,
    ) -> list[str]:
        """
        Return all league-member names in insertion order.
        """

        return list(
            self.agents.keys()
        )


    def __len__(
        self,
    ) -> int:
        """
        Number of self-play opponents in the league.
        """

        return len(
            self.agents
        )