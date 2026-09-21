#!/usr/bin/env python3
"""Create the paper figure for ALBERTA's Pareto acquisition geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data" / "pareto").exists():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DEFAULT_INPUT = PROJECT_ROOT / "data" / "pareto" / "pareto_geometry_10.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "pareto_geometry.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ALBERTA response-space Pareto geometry."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gridsize", type=int, default=90)
    return parser.parse_args()


def load_geometry(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    data = np.load(path)

    required = (
        "predicted_delta_kl",
        "predicted_delta_v",
        "first_front_indices",
        "selected_indices",
    )
    for key in required:
        if key not in data:
            raise KeyError(f"Missing '{key}' in {path}")

    policy = np.asarray(data["predicted_delta_kl"], dtype=np.float64)
    value = np.asarray(data["predicted_delta_v"], dtype=np.float64)
    front = np.asarray(data["first_front_indices"], dtype=np.int64)
    selected = np.asarray(data["selected_indices"], dtype=np.int64)

    if policy.ndim != 1 or value.ndim != 1 or len(policy) != len(value):
        raise ValueError("Predicted response arrays must be 1D and equally sized.")

    n = len(policy)
    if np.any(front < 0) or np.any(front >= n):
        raise ValueError("First-front indices are out of range.")
    if np.any(selected < 0) or np.any(selected >= n):
        raise ValueError("Selected indices are out of range.")

    front_set = set(front.tolist())
    if not all(int(index) in front_set for index in selected):
        raise ValueError("Canonical selected set is not contained in the first front.")

    return policy, value, front, selected


def main() -> None:
    args = parse_args()
    policy, value, front, selected = load_geometry(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # (a) Full eligible candidate population
    # --------------------------------------------------------
    ax = axes[0]

    ax.hexbin(
        policy,
        value,
        gridsize=args.gridsize,
        mincnt=1,
        bins="log",
        linewidths=0.0,
        rasterized=True,
    )

    ax.scatter(
        policy[front],
        value[front],
        s=12,
        marker="o",
        alpha=0.75,
        label=f"First front (n={len(front):,})",
        zorder=3,
    )

    ax.scatter(
        policy[selected],
        value[selected],
        s=22,
        marker="x",
        linewidths=0.9,
        label=f"Selected (n={len(selected):,})",
        zorder=4,
    )

    ax.set_title("(a) Eligible candidate pool")
    ax.set_xlabel(r"Predicted policy response $\widehat{\Delta}_{\pi}$")
    ax.set_ylabel(r"Predicted value response $\widehat{\Delta}_{V}$")
    ax.grid(alpha=0.18, linewidth=0.4)
    ax.legend(loc="best", frameon=False)

    # --------------------------------------------------------
    # (b) First Pareto front and crowding selection
    # --------------------------------------------------------
    ax = axes[1]

    order = front[np.argsort(policy[front], kind="stable")]

    ax.plot(
        policy[order],
        value[order],
        linewidth=0.8,
        alpha=0.65,
        marker=".",
        markersize=2.5,
        label=f"First front (n={len(front):,})",
    )

    ax.scatter(
        policy[selected],
        value[selected],
        s=24,
        marker="x",
        linewidths=0.9,
        label=f"Crowding selection (n={len(selected):,})",
        zorder=3,
    )

    max_policy_index = int(front[np.argmax(policy[front])])
    max_value_index = int(front[np.argmax(value[front])])

    ax.annotate(
        r"max $\widehat{\Delta}_{\pi}$",
        xy=(policy[max_policy_index], value[max_policy_index]),
        xytext=(-8, 18),
        textcoords="offset points",
        ha="right",
        arrowprops={"arrowstyle": "->", "linewidth": 0.6},
    )

    ax.annotate(
        r"max $\widehat{\Delta}_{V}$",
        xy=(policy[max_value_index], value[max_value_index]),
        xytext=(10, -20),
        textcoords="offset points",
        ha="left",
        arrowprops={"arrowstyle": "->", "linewidth": 0.6},
    )

    ax.set_title("(b) First front and budget coverage")
    ax.set_xlabel(r"Predicted policy response $\widehat{\Delta}_{\pi}$")
    ax.set_ylabel(r"Predicted value response $\widehat{\Delta}_{V}$")
    ax.grid(alpha=0.18, linewidth=0.4)
    ax.legend(loc="best", frameon=False)

    fig.savefig(
        args.output,
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300,
    )

    print(f"Saved: {args.output}")
    print(f"Eligible candidates: {len(policy):,}")
    print(f"First front: {len(front):,}")
    print(f"Selected: {len(selected):,}")


if __name__ == "__main__":
    main()
