from __future__ import annotations

import json
from pathlib import Path

import torch


class UncertaintyStats:
    """
    Collector for state-level uncertainty statistics generated
    during self-play.

    Stored quantities:
        H  : policy entropy
        U  : league-based value disagreement
        HU : interaction H * U
    """

    def __init__(
        self,
    ) -> None:

        self.data: list[dict] = []


    # ========================================================
    # Uncertainty
    # ========================================================

    @torch.no_grad()
    def compute_uncertainty(
        self,
        x: torch.Tensor,
        league,
        current_model,
    ) -> float:
        """
        Compute league-based value uncertainty for one state.

        The canonical definition of U(s) is implemented by
        League.uncertainty(). This wrapper is retained for
        compatibility with the existing training pipeline.
        """

        return league.uncertainty(
            x=x,
            current_model=current_model,
        )


    # ========================================================
    # Add observation
    # ========================================================

    def add(
        self,
        fen: str,
        action: int,
        entropy: float,
        uncertainty: float,
        HU: float,
        result,
    ) -> None:
        """
        Store one state-level uncertainty observation.

        The existing JSON field names are intentionally kept
        unchanged for compatibility with downstream analysis.
        """

        self.data.append(
            {
                "fen":
                    fen,

                "actions":
                    action,

                "H":
                    entropy,

                "U":
                    uncertainty,

                "HU":
                    HU,

                "result":
                    result,
            }
        )


    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save all collected observations as JSON.
        """

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.data,
                f,
                indent=2,
                ensure_ascii=False,
            )


    # ========================================================
    # Length
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.data
        )