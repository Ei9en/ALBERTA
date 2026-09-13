from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import chess.variant
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import training.train_al as al

from src.actions_space import ACTIONS, ACTION_TO_INDEX
from src.encoding import encode_boards


# ============================================================
# Defaults
# ============================================================

DEFAULT_BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rl_epoch"
    / "rl_epoch_10.pt"
)

DEFAULT_TRAINED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "al_runs"
    / "random_1-10"
    / "checkpoints"
    / "al_epoch_30.pt"
)

DEFAULT_QUEUE = (
    PROJECT_ROOT
    / "checkpoints"
    / "queue"
    / "oracle_queue_1-10_random.jsonl"
)

DEFAULT_REFERENCE_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "selfplay_jsons"
    / "uncertainty_stats_1-10.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "trained_policy_shift"
)

DEFAULT_REFERENCE_POSITIONS = 1000
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 42

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EPS = 1e-12


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Compare a baseline ActorCritic checkpoint with a "
            "trained RL+Oracle checkpoint on annotated positions "
            "and an independent RL/self-play reference sample."
        )
    )

    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT,
    )

    parser.add_argument(
        "--trained-checkpoint",
        type=Path,
        default=DEFAULT_TRAINED_CHECKPOINT,
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help=(
            "Oracle queue used to define the annotated-position cohort."
        ),
    )

    parser.add_argument(
        "--reference-source",
        type=Path,
        default=DEFAULT_REFERENCE_SOURCE,
        help=(
            "JSON/JSONL file containing RL/self-play positions "
            "used as an independent reference distribution."
        ),
    )

    parser.add_argument(
        "--reference-positions",
        type=int,
        default=DEFAULT_REFERENCE_POSITIONS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    if hasattr(
        torch.backends,
        "cudnn",
    ):

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Generic helpers
# ============================================================

def safe_float(
    value,
) -> float:

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return np.nan

    if not math.isfinite(
        value
    ):

        return np.nan

    return value


def safe_mean(
    values,
) -> float:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def safe_normalize(
    probabilities,
) -> np.ndarray:

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    total = float(
        probabilities.sum()
    )

    if not math.isfinite(
        total
    ) or total <= 0.0:

        raise ValueError(
            "Probability vector has invalid sum."
        )

    return (
        probabilities
        / total
    )


def entropy(
    probabilities,
) -> float:

    probabilities = safe_normalize(
        probabilities
    )

    mask = (
        probabilities > 0.0
    )

    return float(
        -np.sum(
            probabilities[
                mask
            ]
            * np.log(
                probabilities[
                    mask
                ]
            )
        )
    )


def kl_divergence(
    p,
    q,
) -> float:

    p = safe_normalize(
        p
    )

    q = safe_normalize(
        q
    )

    return float(
        np.sum(
            p
            * (
                np.log(
                    p + EPS
                )
                - np.log(
                    q + EPS
                )
            )
        )
    )


def l1_distance(
    p,
    q,
) -> float:

    p = safe_normalize(
        p
    )

    q = safe_normalize(
        q
    )

    return float(
        np.sum(
            np.abs(
                p - q
            )
        )
    )


# ============================================================
# Checkpoint loading
# ============================================================

def load_actor_critic(
    checkpoint_path: Path,
    device: torch.device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    checkpoint_actions = checkpoint.get(
        "actions"
    )

    if (
        checkpoint_actions is not None
        and checkpoint_actions != len(ACTIONS)
    ):

        raise ValueError(
            "Action-space mismatch: "
            f"{checkpoint_actions} != {len(ACTIONS)}"
        )

    model = al.build_actor_critic(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# Oracle queue loading
# ============================================================

def load_annotations(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Oracle queue not found:\n{path}"
        )

    samples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                sample = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line {line_number} "
                    f"in:\n{path}"
                ) from exc

            if not isinstance(
                sample,
                dict,
            ):

                continue

            if sample.get(
                "status"
            ) == "discarded":

                continue

            fen = sample.get(
                "fen"
            )

            oracle_move = sample.get(
                "oracle_move"
            )

            confidence = sample.get(
                "oracle_confidence",
                sample.get(
                    "confidence"
                ),
            )

            situation = sample.get(
                "oracle_situation",
                sample.get(
                    "criticality"
                ),
            )

            reward = sample.get(
                "reward"
            )

            # ------------------------------------------------
            # This analysis can operate without reward, but it
            # requires an Oracle move to evaluate policy shift.
            # ------------------------------------------------

            if (
                not isinstance(
                    fen,
                    str,
                )
                or not fen
                or oracle_move is None
            ):

                continue

            if oracle_move not in ACTION_TO_INDEX:

                raise ValueError(
                    f"Unknown Oracle move at line "
                    f"{line_number}: {oracle_move}"
                )

            board = chess.variant.AtomicBoard(
                fen
            )

            legal_moves = {
                move.uci()
                for move in board.legal_moves
            }

            if oracle_move not in legal_moves:

                raise ValueError(
                    f"Illegal Oracle move at line "
                    f"{line_number}:\n"
                    f"FEN:  {fen}\n"
                    f"Move: {oracle_move}"
                )

            samples.append(
                {
                    "query_id":
                        sample.get(
                            "query_id"
                        ),

                    "fen":
                        fen,

                    "oracle_move":
                        oracle_move,

                    "oracle_confidence":
                        confidence,

                    "oracle_situation":
                        situation,

                    "reward":
                        safe_float(
                            reward
                        ),

                    "H":
                        safe_float(
                            sample.get(
                                "H"
                            )
                        ),

                    "U":
                        safe_float(
                            sample.get(
                                "U"
                            )
                        ),

                    "HU":
                        safe_float(
                            sample.get(
                                "HU"
                            )
                        ),

                    "score":
                        safe_float(
                            sample.get(
                                "score",
                                sample.get(
                                    "I"
                                ),
                            )
                        ),

                    "I_norm":
                        safe_float(
                            sample.get(
                                "I_norm"
                            )
                        ),
                }
            )

    if not samples:

        raise RuntimeError(
            "No usable Oracle annotations found."
        )

    return samples


# ============================================================
# Reference-position loading
# ============================================================

def load_reference_records(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Reference source not found:\n{path}"
        )

    if path.suffix.lower() == ".jsonl":

        records = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:

                    raise ValueError(
                        f"Invalid JSON at line {line_number} "
                        f"in:\n{path}"
                    ) from exc

                records.append(
                    record
                )

        return records

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if isinstance(
        data,
        list,
    ):

        return data

    if isinstance(
        data,
        dict,
    ):

        for key in (
            "data",
            "records",
            "positions",
            "samples",
            "stats",
            "uncertainty_stats",
        ):

            value = data.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                return value

    raise ValueError(
        f"Unsupported reference-source structure:\n{path}"
    )


def sample_reference_positions(
    records: list[dict],
    *,
    n_positions: int,
    seed: int,
    excluded_fens: set[str],
) -> list[str]:

    if n_positions <= 0:

        raise ValueError(
            "Number of reference positions must be positive."
        )

    unique_fens = []

    seen = set()

    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            continue

        fen = record.get(
            "fen"
        )

        if not isinstance(
            fen,
            str,
        ) or not fen:

            continue

        if fen in excluded_fens:
            continue

        if fen in seen:
            continue

        try:

            board = chess.variant.AtomicBoard(
                fen
            )

        except Exception:
            continue

        if board.is_game_over():
            continue

        if not any(
            board.legal_moves
        ):
            continue

        seen.add(
            fen
        )

        unique_fens.append(
            fen
        )

    if len(
        unique_fens
    ) < n_positions:

        raise RuntimeError(
            "Not enough independent reference FENs after "
            "deduplication/exclusion: "
            f"{len(unique_fens)} available, "
            f"{n_positions} requested."
        )

    rng = random.Random(
        seed
    )

    indices = rng.sample(
        range(
            len(
                unique_fens
            )
        ),
        n_positions,
    )

    return [
        unique_fens[
            index
        ]
        for index in indices
    ]


# ============================================================
# Policy/value evaluation
# ============================================================

@torch.no_grad()
def evaluate_batch(
    model,
    fens: list[str],
    device: torch.device,
) -> list[dict]:

    boards = [
        chess.variant.AtomicBoard(
            fen
        )
        for fen in fens
    ]

    encoded = encode_boards(
        boards
    ).to(
        device
    )

    logits, values = model(
        encoded
    )

    results = []

    for (
        index,
        board,
    ) in enumerate(
        boards
    ):

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:

            raise RuntimeError(
                f"Position has no legal moves:\n{fens[index]}"
            )

        legal_uci = [
            move.uci()
            for move in legal_moves
        ]

        legal_indices = torch.tensor(
            [
                ACTION_TO_INDEX[
                    move
                ]
                for move in legal_uci
            ],
            dtype=torch.long,
            device=device,
        )

        legal_logits = logits[
            index,
            legal_indices,
        ]

        legal_log_probs = F.log_softmax(
            legal_logits,
            dim=0,
        )

        legal_probs = torch.exp(
            legal_log_probs
        )

        results.append(
            {
                "moves":
                    legal_uci,

                "probs":
                    legal_probs
                    .detach()
                    .cpu()
                    .numpy(),

                "value":
                    float(
                        values[
                            index,
                            0,
                        ].item()
                    ),
            }
        )

    return results


# ============================================================
# Position comparison
# ============================================================

def compare_position_outputs(
    base_output: dict,
    trained_output: dict,
    *,
    oracle_move: str | None = None,
    reward: float = np.nan,
) -> dict:

    base_moves = base_output[
        "moves"
    ]

    trained_moves = trained_output[
        "moves"
    ]

    if set(
        base_moves
    ) != set(
        trained_moves
    ):

        raise RuntimeError(
            "Legal move support differs between models "
            "for the same board."
        )

    trained_lookup = {
        move:
            probability
        for (
            move,
            probability,
        ) in zip(
            trained_moves,
            trained_output[
                "probs"
            ],
        )
    }

    trained_vector = np.asarray(
        [
            trained_lookup[
                move
            ]
            for move in base_moves
        ],
        dtype=np.float64,
    )

    base_vector = np.asarray(
        base_output[
            "probs"
        ],
        dtype=np.float64,
    )

    base_vector = safe_normalize(
        base_vector
    )

    trained_vector = safe_normalize(
        trained_vector
    )

    # ========================================================
    # General policy shift
    # ========================================================

    kl = kl_divergence(
        base_vector,
        trained_vector,
    )

    l1 = l1_distance(
        base_vector,
        trained_vector,
    )

    entropy_base = entropy(
        base_vector
    )

    entropy_trained = entropy(
        trained_vector
    )

    base_best_index = int(
        np.argmax(
            base_vector
        )
    )

    trained_best_index = int(
        np.argmax(
            trained_vector
        )
    )

    base_best_move = base_moves[
        base_best_index
    ]

    trained_best_move = base_moves[
        trained_best_index
    ]

    value_base = float(
        base_output[
            "value"
        ]
    )

    value_trained = float(
        trained_output[
            "value"
        ]
    )

    result = {
        "kl_base_to_trained":
            kl,

        "l1_policy_distance":
            l1,

        "entropy_base":
            entropy_base,

        "entropy_trained":
            entropy_trained,

        "delta_entropy":
            (
                entropy_trained
                - entropy_base
            ),

        "base_best_move":
            base_best_move,

        "trained_best_move":
            trained_best_move,

        "best_move_changed":
            (
                base_best_move
                != trained_best_move
            ),

        "value_base":
            value_base,

        "value_trained":
            value_trained,

        "delta_value":
            (
                value_trained
                - value_base
            ),
    }

    # ========================================================
    # Oracle-specific shift
    # ========================================================

    if oracle_move is not None:

        if oracle_move not in base_moves:

            raise ValueError(
                f"Oracle move {oracle_move} is not legal."
            )

        oracle_index = base_moves.index(
            oracle_move
        )

        oracle_probability_base = float(
            base_vector[
                oracle_index
            ]
        )

        oracle_probability_trained = float(
            trained_vector[
                oracle_index
            ]
        )

        oracle_rank_base = (
            1
            + int(
                np.sum(
                    base_vector
                    > oracle_probability_base
                )
            )
        )

        oracle_rank_trained = (
            1
            + int(
                np.sum(
                    trained_vector
                    > oracle_probability_trained
                )
            )
        )

        result.update(
            {
                "oracle_probability_base":
                    oracle_probability_base,

                "oracle_probability_trained":
                    oracle_probability_trained,

                "delta_oracle_probability":
                    (
                        oracle_probability_trained
                        - oracle_probability_base
                    ),

                "oracle_rank_base":
                    oracle_rank_base,

                "oracle_rank_trained":
                    oracle_rank_trained,

                "delta_oracle_rank":
                    (
                        oracle_rank_trained
                        - oracle_rank_base
                    ),

                "oracle_best_base":
                    (
                        base_best_move
                        == oracle_move
                    ),

                "oracle_best_trained":
                    (
                        trained_best_move
                        == oracle_move
                    ),
            }
        )

    # ========================================================
    # Oracle value-target shift
    # ========================================================

    if math.isfinite(
        reward
    ):

        value_error_base = abs(
            value_base
            - reward
        )

        value_error_trained = abs(
            value_trained
            - reward
        )

        result.update(
            {
                "value_error_base":
                    value_error_base,

                "value_error_trained":
                    value_error_trained,

                "value_error_reduction":
                    (
                        value_error_base
                        - value_error_trained
                    ),
            }
        )

    return result


# ============================================================
# Cohort evaluation
# ============================================================

def evaluate_cohort(
    *,
    base_model,
    trained_model,
    records: list[dict],
    device: torch.device,
    batch_size: int,
    annotated: bool,
) -> pd.DataFrame:

    results = []

    for start in range(
        0,
        len(
            records
        ),
        batch_size,
    ):

        batch = records[
            start:
            start + batch_size
        ]

        fens = [
            record[
                "fen"
            ]
            for record in batch
        ]

        base_outputs = evaluate_batch(
            base_model,
            fens,
            device,
        )

        trained_outputs = evaluate_batch(
            trained_model,
            fens,
            device,
        )

        for (
            record,
            base_output,
            trained_output,
        ) in zip(
            batch,
            base_outputs,
            trained_outputs,
        ):

            oracle_move = (
                record.get(
                    "oracle_move"
                )
                if annotated
                else None
            )

            reward = (
                safe_float(
                    record.get(
                        "reward"
                    )
                )
                if annotated
                else np.nan
            )

            comparison = compare_position_outputs(
                base_output,
                trained_output,
                oracle_move=oracle_move,
                reward=reward,
            )

            row = {
                "fen":
                    record[
                        "fen"
                    ],

                "cohort":
                    (
                        "annotated"
                        if annotated
                        else "reference"
                    ),
            }

            if annotated:

                row.update(
                    {
                        "query_id":
                            record.get(
                                "query_id"
                            ),

                        "oracle_move":
                            record.get(
                                "oracle_move"
                            ),

                        "oracle_confidence":
                            record.get(
                                "oracle_confidence"
                            ),

                        "oracle_situation":
                            record.get(
                                "oracle_situation"
                            ),

                        "reward":
                            reward,

                        "H":
                            record.get(
                                "H",
                                np.nan,
                            ),

                        "U":
                            record.get(
                                "U",
                                np.nan,
                            ),

                        "HU":
                            record.get(
                                "HU",
                                np.nan,
                            ),

                        "score":
                            record.get(
                                "score",
                                np.nan,
                            ),

                        "I_norm":
                            record.get(
                                "I_norm",
                                np.nan,
                            ),
                    }
                )

            row.update(
                comparison
            )

            results.append(
                row
            )

        print(
            f"Compared "
            f"{min(start + batch_size, len(records)):,}/"
            f"{len(records):,}",
            flush=True,
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# Report helpers
# ============================================================

def summarize_numeric(
    df: pd.DataFrame,
    column: str,
) -> dict:

    if column not in df.columns:

        return {}

    values = pd.to_numeric(
        df[
            column
        ],
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if len(
        values
    ) == 0:

        return {}

    return {
        "n":
            int(
                len(
                    values
                )
            ),

        "mean":
            float(
                values.mean()
            ),

        "median":
            float(
                values.median()
            ),

        "std":
            float(
                values.std(
                    ddof=0
                )
            ),

        "p05":
            float(
                values.quantile(
                    0.05
                )
            ),

        "p95":
            float(
                values.quantile(
                    0.95
                )
            ),

        "p99":
            float(
                values.quantile(
                    0.99
                )
            ),

        "min":
            float(
                values.min()
            ),

        "max":
            float(
                values.max()
            ),
    }


def add_numeric_summary(
    lines: list[str],
    df: pd.DataFrame,
    column: str,
) -> None:

    summary = summarize_numeric(
        df,
        column,
    )

    if not summary:

        return

    lines.append(
        f"{column}:"
    )

    lines.append(
        f"  n      = {summary['n']:,}"
    )

    lines.append(
        f"  mean   = {summary['mean']:+.8f}"
    )

    lines.append(
        f"  median = {summary['median']:+.8f}"
    )

    lines.append(
        f"  std    = {summary['std']:.8f}"
    )

    lines.append(
        f"  p05    = {summary['p05']:+.8f}"
    )

    lines.append(
        f"  p95    = {summary['p95']:+.8f}"
    )

    lines.append(
        f"  p99    = {summary['p99']:+.8f}"
    )

    lines.append(
        f"  min    = {summary['min']:+.8f}"
    )

    lines.append(
        f"  max    = {summary['max']:+.8f}"
    )

    lines.append()


def build_report(
    *,
    annotated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    base_checkpoint: Path,
    trained_checkpoint: Path,
    queue_path: Path,
    reference_source: Path,
    seed: int,
) -> str:

    lines = []

    def add(
        text="",
    ) -> None:

        lines.append(
            str(
                text
            )
        )

    add(
        "=" * 80
    )

    add(
        "ALBERTA - TRAINED POLICY SHIFT ANALYSIS"
    )

    add(
        "=" * 80
    )

    add()

    add(
        f"Base checkpoint:    {base_checkpoint}"
    )

    add(
        f"Trained checkpoint: {trained_checkpoint}"
    )

    add(
        f"Oracle queue:       {queue_path}"
    )

    add(
        f"Reference source:   {reference_source}"
    )

    add(
        f"Seed:               {seed}"
    )

    add()

    # ========================================================
    # Cohort sizes
    # ========================================================

    add(
        "COHORTS"
    )

    add(
        "-" * 80
    )

    add(
        f"Annotated positions: {len(annotated_df):,}"
    )

    add(
        f"Reference positions: {len(reference_df):,}"
    )

    add()

    # ========================================================
    # General shift
    # ========================================================

    for label, df in (
        (
            "ANNOTATED POSITIONS",
            annotated_df,
        ),
        (
            "REFERENCE RL/SELF-PLAY POSITIONS",
            reference_df,
        ),
    ):

        add(
            label
        )

        add(
            "-" * 80
        )

        for column in (
            "kl_base_to_trained",
            "l1_policy_distance",
            "delta_entropy",
            "delta_value",
        ):

            add_numeric_summary(
                lines,
                df,
                column,
            )

        if len(
            df
        ):

            best_changed = float(
                df[
                    "best_move_changed"
                ].astype(
                    float
                ).mean()
            )

            add(
                "Best move changed: "
                f"{best_changed:.2%}"
            )

            add()

    # ========================================================
    # Oracle-specific response
    # ========================================================

    add(
        "ORACLE-SPECIFIC RESPONSE"
    )

    add(
        "-" * 80
    )

    for column in (
        "oracle_probability_base",
        "oracle_probability_trained",
        "delta_oracle_probability",
        "oracle_rank_base",
        "oracle_rank_trained",
        "delta_oracle_rank",
        "value_error_reduction",
    ):

        add_numeric_summary(
            lines,
            annotated_df,
            column,
        )

    if (
        "oracle_best_base"
        in annotated_df.columns
        and len(
            annotated_df
        )
    ):

        base_best = float(
            annotated_df[
                "oracle_best_base"
            ].astype(
                float
            ).mean()
        )

        trained_best = float(
            annotated_df[
                "oracle_best_trained"
            ].astype(
                float
            ).mean()
        )

        add(
            f"Oracle move top-1, base:    {base_best:.2%}"
        )

        add(
            f"Oracle move top-1, trained: {trained_best:.2%}"
        )

        add()

    # ========================================================
    # Breakdown by confidence
    # ========================================================

    add(
        "BREAKDOWN BY ORACLE CONFIDENCE"
    )

    add(
        "-" * 80
    )

    if "oracle_confidence" in annotated_df.columns:

        for confidence in (
            "high",
            "medium",
            "low",
        ):

            subset = annotated_df[
                annotated_df[
                    "oracle_confidence"
                ]
                == confidence
            ]

            if len(
                subset
            ) == 0:

                continue

            add(
                f"{confidence:8s} | "
                f"n={len(subset):4d} | "
                f"mean ΔP="
                f"{safe_mean(subset['delta_oracle_probability']):+.6f} | "
                f"mean Δrank="
                f"{safe_mean(subset['delta_oracle_rank']):+.3f}"
            )

    add()

    # ========================================================
    # Breakdown by situation
    # ========================================================

    add(
        "BREAKDOWN BY ORACLE SITUATION"
    )

    add(
        "-" * 80
    )

    if "oracle_situation" in annotated_df.columns:

        for situation in (
            "critical",
            "non_critical",
            "outcome_independent",
        ):

            subset = annotated_df[
                annotated_df[
                    "oracle_situation"
                ]
                == situation
            ]

            if len(
                subset
            ) == 0:

                continue

            add(
                f"{situation:20s} | "
                f"n={len(subset):4d} | "
                f"mean ΔP="
                f"{safe_mean(subset['delta_oracle_probability']):+.6f} | "
                f"mean Δrank="
                f"{safe_mean(subset['delta_oracle_rank']):+.3f}"
            )

    add()

    # ========================================================
    # Interpretation
    # ========================================================

    add(
        "INTERPRETATION"
    )

    add(
        "-" * 80
    )

    add(
        "This analysis measures the realized displacement "
        "between a baseline ActorCritic checkpoint and a fully "
        "trained RL+Oracle checkpoint."
    )

    add()

    add(
        "Annotated positions quantify direct changes on states "
        "that received Oracle supervision."
    )

    add(
        "Reference positions are sampled independently from an "
        "RL/self-play state distribution and therefore measure "
        "how much the learned change diffuses beyond the "
        "annotated set."
    )

    add()

    add(
        "KL and L1 measure policy displacement, while "
        "best-move changes measure decision-level displacement."
    )

    add(
        "Oracle probability/rank changes measure whether the "
        "trained policy moved toward the annotated action."
    )

    add(
        "Value-error reduction measures whether the trained "
        "critic moved toward the Oracle value target."
    )

    add()

    add(
        "This analysis describes realized model change. "
        "It does not by itself establish that the change "
        "improves downstream playing strength."
    )

    add()

    add(
        "For downstream utility, tournament evaluation remains "
        "the primary performance measurement."
    )

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.reference_positions <= 0:

        raise ValueError(
            "--reference-positions must be positive."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be positive."
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("ALBERTA - TRAINED POLICY SHIFT ANALYSIS")
    print("=" * 80)

    print(
        f"Device:             {device}"
    )

    print(
        f"Base checkpoint:    {args.base_checkpoint}"
    )

    print(
        f"Trained checkpoint: {args.trained_checkpoint}"
    )

    print(
        f"Oracle queue:       {args.queue}"
    )

    print(
        f"Reference source:   {args.reference_source}"
    )

    # ========================================================
    # Models
    # ========================================================

    (
        base_model,
        base_checkpoint,
    ) = load_actor_critic(
        args.base_checkpoint,
        device,
    )

    (
        trained_model,
        trained_checkpoint,
    ) = load_actor_critic(
        args.trained_checkpoint,
        device,
    )

    print()
    print(
        f"Base epoch: "
        f"{base_checkpoint.get('epoch', '?')}"
    )

    print(
        f"Trained epoch: "
        f"{trained_checkpoint.get('epoch', '?')}"
    )

    # ========================================================
    # Annotated cohort
    # ========================================================

    annotations = load_annotations(
        args.queue
    )

    annotated_fens = {
        record[
            "fen"
        ]
        for record in annotations
    }

    print()
    print(
        f"Annotated positions: "
        f"{len(annotations):,}"
    )

    # ========================================================
    # Reference cohort
    # ========================================================

    reference_records = load_reference_records(
        args.reference_source
    )

    reference_fens = sample_reference_positions(
        reference_records,
        n_positions=args.reference_positions,
        seed=args.seed,
        excluded_fens=annotated_fens,
    )

    reference_records_normalized = [
        {
            "fen":
                fen,
        }
        for fen in reference_fens
    ]

    print(
        f"Reference positions: "
        f"{len(reference_records_normalized):,}"
    )

    # ========================================================
    # Evaluate annotated cohort
    # ========================================================

    print()
    print("=" * 80)
    print("ANNOTATED COHORT")
    print("=" * 80)

    annotated_df = evaluate_cohort(
        base_model=base_model,
        trained_model=trained_model,
        records=annotations,
        device=device,
        batch_size=args.batch_size,
        annotated=True,
    )

    # ========================================================
    # Evaluate reference cohort
    # ========================================================

    print()
    print("=" * 80)
    print("REFERENCE COHORT")
    print("=" * 80)

    reference_df = evaluate_cohort(
        base_model=base_model,
        trained_model=trained_model,
        records=reference_records_normalized,
        device=device,
        batch_size=args.batch_size,
        annotated=False,
    )

    # ========================================================
    # Save CSV
    # ========================================================

    annotated_path = (
        args.output_dir
        / "annotated_positions.csv"
    )

    reference_path = (
        args.output_dir
        / "reference_positions.csv"
    )

    annotated_df.to_csv(
        annotated_path,
        index=False,
    )

    reference_df.to_csv(
        reference_path,
        index=False,
    )

    # ========================================================
    # Report
    # ========================================================

    report = build_report(
        annotated_df=annotated_df,
        reference_df=reference_df,
        base_checkpoint=args.base_checkpoint,
        trained_checkpoint=args.trained_checkpoint,
        queue_path=args.queue,
        reference_source=args.reference_source,
        seed=args.seed,
    )

    report_path = (
        args.output_dir
        / "trained_policy_shift_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for label, df in (
        (
            "ANNOTATED",
            annotated_df,
        ),
        (
            "REFERENCE",
            reference_df,
        ),
    ):

        print()
        print(
            label
        )

        print(
            "-" * 80
        )

        print(
            "Mean KL: "
            f"{safe_mean(df['kl_base_to_trained']):.6f}"
        )

        print(
            "Mean L1: "
            f"{safe_mean(df['l1_policy_distance']):.6f}"
        )

        print(
            "Best move changed: "
            f"{df['best_move_changed'].astype(float).mean():.2%}"
        )

        print(
            "Mean ΔV: "
            f"{safe_mean(df['delta_value']):+.6f}"
        )

    print()
    print(
        "ANNOTATED ORACLE RESPONSE"
    )

    print(
        "-" * 80
    )

    print(
        "Mean ΔP(Oracle): "
        f"{safe_mean(annotated_df['delta_oracle_probability']):+.6f}"
    )

    print(
        "Mean Δrank(Oracle): "
        f"{safe_mean(annotated_df['delta_oracle_rank']):+.3f}"
    )

    if (
        "value_error_reduction"
        in annotated_df.columns
    ):

        print(
            "Mean value-error reduction: "
            f"{safe_mean(annotated_df['value_error_reduction']):+.6f}"
        )

    print()
    print(
        f"Annotated CSV: {annotated_path}"
    )

    print(
        f"Reference CSV: {reference_path}"
    )

    print(
        f"Report:        {report_path}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()