import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from distribution_I import W_H, W_U, W_HU


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "uncertainty_stats_1-10.json"
)

QUEUE_FILE = (
    PROJECT_ROOT
    / "data"
    / "oracle_queue_1-10.jsonl"
)


# ============================================================
# Percentile rank normalization
# ============================================================

def percentile_rank(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    n = len(values)

    if n < 2:
        raise ValueError(
            "Not enough values for percentile rank."
        )


    order = np.argsort(
        values,
        kind="stable"
    )

    sorted_values = values[
        order
    ]

    ranks = np.empty(
        n,
        dtype=np.float64
    )


    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and
            sorted_values[end]
            ==
            sorted_values[start]
        ):
            end += 1


        rank = (
            (start + end - 1)
            /
            2.0
        )


        ranks[
            order[start:end]
        ] = (
            rank
            /
            (n - 1)
        )


        start = end


    return ranks



# ============================================================
# Extract side to move
# ============================================================

def extract_side(fens):

    sides = []

    for fen in fens:

        side = fen.split()[1]

        if side not in ("w", "b"):

            raise ValueError(
                f"Invalid FEN side: {fen}"
            )

        sides.append(side)


    return np.array(
        sides,
        dtype="<U1"
    )



# ============================================================
# Side aware normalization
# ============================================================

def normalize_side_aware(
    values,
    sides
):

    normalized = np.zeros_like(
        values,
        dtype=np.float64
    )


    for side in [
        "w",
        "b"
    ]:

        mask = (
            sides == side
        )


        normalized[
            mask
        ] = percentile_rank(
            values[
                mask
            ]
        )


    return normalized



# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Seed ALBERTA oracle queue using active learning threshold."
    )


    parser.add_argument(
        "--percentile",
        type=float,
        default=99.98,
        help=
        "Selection percentile (default: 99.98 = 0.02%% budget)"
    )


    args = parser.parse_args()



    print("=" * 70)
    print(
        "ALBERTA - SEED ORACLE QUEUE"
    )
    print("=" * 70)



    # --------------------------------------------------------
    # Load uncertainty data
    # --------------------------------------------------------

    print()
    print(
        "Loading uncertainty statistics..."
    )


    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    print(
        f"Positions loaded : {len(data):,}"
    )



    # --------------------------------------------------------
    # Extract signals
    # --------------------------------------------------------

    fens = np.array(
        [
            x["fen"]
            for x in data
        ],
        dtype=object
    )


    H = np.array(
        [
            x["H"]
            for x in data
        ],
        dtype=np.float64
    )


    U = np.array(
        [
            x["U"]
            for x in data
        ],
        dtype=np.float64
    )


    HU = np.array(
        [
            x["HU"]
            for x in data
        ],
        dtype=np.float64
    )


    sides = extract_side(
        fens
    )



    # --------------------------------------------------------
    # Compute active learning score
    # --------------------------------------------------------

    H_norm = normalize_side_aware(
        H,
        sides
    )


    U_norm = normalize_side_aware(
        U,
        sides
    )


    HU_norm = normalize_side_aware(
        HU,
        sides
    )


    I = (
        W_H * H_norm
        +
        W_U * U_norm
        +
        W_HU * HU_norm
    )



    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    threshold = np.percentile(
        I,
        args.percentile
    )


    selected_indices = np.where(
        I >= threshold
    )[0]

    selected_indices = selected_indices[
        np.argsort(
            I[selected_indices]
        )[::-1]
    ]


    print()
    print(
        "ACTIVE LEARNING SELECTION"
    )
    print("-" * 70)

    print(
        f"Percentile : P{args.percentile}"
    )

    print(
        f"Threshold  : {threshold:.9f}"
    )

    print(
        f"Selected   : {len(selected_indices):,}"
    )

    print(
        f"Budget     : "
        f"{100 * len(selected_indices) / len(I):.5f}%"
    )



    # --------------------------------------------------------
    # Load existing queue
    # --------------------------------------------------------

    if QUEUE_FILE.exists():

        with open(
            QUEUE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            existing = [
                json.loads(line)
                for line in f
            ]

    else:

        existing = []



    existing_ids = {
        x["query_id"]
        for x in existing
    }



    # --------------------------------------------------------
    # Append new positions
    # --------------------------------------------------------

    added = 0


    with open(
        QUEUE_FILE,
        "a",
        encoding="utf-8"
    ) as f:


        for idx in selected_indices:


            record = data[idx]


            query_id = uuid.uuid5(
                uuid.NAMESPACE_DNS,
                record["fen"]
            ).hex



            if query_id in existing_ids:

                continue



            item = {

                "query_id":
                    query_id,


                "fen":
                    record["fen"],


                "H":
                    float(H[idx]),


                "U":
                    float(U[idx]),


                "HU":
                    float(HU[idx]),


                "I":
                    float(I[idx]),


                "status":
                    "pending",


                "oracle_move":
                    None,


                "oracle_confidence":
                    None,


                "oracle_situation":
                    None,


                "created_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),


                "answered_at":
                    None
            }


            f.write(
                json.dumps(item)
                +
                "\n"
            )


            added += 1



    print()
    print("=" * 70)
    print(
        "QUEUE UPDATED"
    )
    print("=" * 70)

    print(
        f"Added : {added:,}"
    )

    print(
        f"File  : {QUEUE_FILE}"
    )



if __name__ == "__main__":

    main()