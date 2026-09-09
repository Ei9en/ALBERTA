#!/usr/bin/env python3

"""
ALBERTA - Queue Overlap Analysis
================================

Compare two Oracle queues and measure how much their selected
positions overlap.

Usage
-----

    python compare_queue_overlap.py

By default:

    Linear:
        checkpoints/queue/oracle_queue_1-10_AL.jsonl

    Non-linear:
        checkpoints/queue/oracle_queue_1-10_AL_nonlinear.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINEAR_QUEUE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_low.jsonl"
)

NONLINEAR_QUEUE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_AL_nonlinear.jsonl"
)


# ============================================================
# Load queue
# ============================================================

def load_queue(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Queue not found:\n{path}"
        )

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {e}"
                )

            if "query_id" not in item:
                raise ValueError(
                    f"Missing query_id in {path}, "
                    f"line {line_number}"
                )

            records.append(item)

    return records


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ALBERTA - QUEUE OVERLAP ANALYSIS")
    print("=" * 70)

    print()
    print("LINEAR QUEUE")
    print("-" * 70)
    print(LINEAR_QUEUE)

    print()
    print("NON-LINEAR QUEUE")
    print("-" * 70)
    print(NONLINEAR_QUEUE)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    linear_records = load_queue(
        LINEAR_QUEUE
    )

    nonlinear_records = load_queue(
        NONLINEAR_QUEUE
    )

    linear_ids = {
        item["query_id"]
        for item in linear_records
    }

    nonlinear_ids = {
        item["query_id"]
        for item in nonlinear_records
    }

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    n_linear = len(linear_ids)
    n_nonlinear = len(nonlinear_ids)

    overlap = (
        linear_ids
        & nonlinear_ids
    )

    only_linear = (
        linear_ids
        - nonlinear_ids
    )

    only_nonlinear = (
        nonlinear_ids
        - linear_ids
    )

    n_overlap = len(overlap)

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    overlap_linear = (
        100.0 * n_overlap / n_linear
        if n_linear
        else 0.0
    )

    overlap_nonlinear = (
        100.0 * n_overlap / n_nonlinear
        if n_nonlinear
        else 0.0
    )

    union = (
        linear_ids
        | nonlinear_ids
    )

    jaccard = (
        100.0 * n_overlap / len(union)
        if union
        else 0.0
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print()
    print(
        f"Linear positions      : {n_linear:,}"
    )

    print(
        f"Non-linear positions  : {n_nonlinear:,}"
    )

    print(
        f"Exact overlap          : {n_overlap:,}"
    )

    print(
        f"Overlap / linear      : "
        f"{overlap_linear:.2f}%"
    )

    print(
        f"Overlap / non-linear  : "
        f"{overlap_nonlinear:.2f}%"
    )

    print(
        f"Jaccard similarity    : "
        f"{jaccard:.2f}%"
    )

    print()
    print(
        f"Only linear           : "
        f"{len(only_linear):,}"
    )

    print(
        f"Only non-linear       : "
        f"{len(only_nonlinear):,}"
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    if len(linear_ids) != len(linear_records):

        print(
            "WARNING: duplicate query_id(s) "
            "in linear queue."
        )

    else:

        print(
            "Linear queue: no duplicate query_id."
        )

    if len(nonlinear_ids) != len(nonlinear_records):

        print(
            "WARNING: duplicate query_id(s) "
            "in non-linear queue."
        )

    else:

        print(
            "Non-linear queue: no duplicate query_id."
        )

    # --------------------------------------------------------
    # Different positions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIFFERENT POSITIONS")
    print("=" * 70)

    print()
    print(
        f"Positions only in linear queue: "
        f"{len(only_linear):,}"
    )

    for query_id in sorted(only_linear):

        print(
            f"  {query_id}"
        )

    print()
    print(
        f"Positions only in non-linear queue: "
        f"{len(only_nonlinear):,}"
    )

    for query_id in sorted(only_nonlinear):

        print(
            f"  {query_id}"
        )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if overlap_linear >= 90.0:

        print(
            "The non-linear acquisition selects a highly "
            "overlapping subset with the linear acquisition."
        )

        print(
            "The change in functional form therefore produces "
            "little effective change in the selected positions."
        )

    elif overlap_linear >= 70.0:

        print(
            "The non-linear acquisition remains substantially "
            "similar to the linear acquisition."
        )

    elif overlap_linear >= 50.0:

        print(
            "The two acquisition functions have moderate "
            "selection overlap."
        )

    else:

        print(
            "The non-linear acquisition produces a substantially "
            "different selected subset."
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()