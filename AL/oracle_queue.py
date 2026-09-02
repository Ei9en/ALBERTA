from __future__ import annotations

import json
import uuid

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ============================================================
# Data model
# ============================================================

@dataclass
class OracleQuery:

    query_id: str

    fen: str

    # Active learning raw signals
    H: float
    U: float
    HU: float

    # Active learning score
    score: float

    # Optional metadata
    threshold: Optional[float] = None

    model: str = "historical_json"
    epoch: int = -1

    game_id: int = -1
    ply: int = -1

    created_at: str = ""

    status: str = "pending"

    # --------------------------------------------------------
    # Oracle annotation
    # --------------------------------------------------------

    oracle_move: Optional[str] = None
    oracle_confidence: Optional[str] = None
    oracle_situation: Optional[str] = None

    # --------------------------------------------------------
    # Reward annotation
    #
    # +1 = good outcome for side to move
    #  0 = neutral / unclear
    # -1 = bad outcome for side to move
    # --------------------------------------------------------

    reward: Optional[float] = None

    answered_at: Optional[str] = None


# ============================================================
# Queue
# ============================================================

class OracleQueue:

    VALID_CONFIDENCE = {
        "low",
        "medium",
        "high",
    }

    VALID_SITUATION = {
        "critical",
        "non_critical",
        "outcome_independent",
    }

    VALID_REWARDS = {
        -1,
        0,
        1,
    }

    def __init__(
        self,
        path: str | Path = "data/oracle_queue.jsonl",
    ):

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.touch(
            exist_ok=True,
        )

    # ========================================================
    # Utils
    # ========================================================

    @staticmethod
    def _timestamp():

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================
    # Serialization
    # ========================================================

    @staticmethod
    def _serialize(
        q: OracleQuery,
    ) -> dict:

        return asdict(q)

    @staticmethod
    def _deserialize(
        data: dict,
    ) -> OracleQuery:

        # ----------------------------------------------------
        # Backward compatibility:
        # old queue used "I" instead of "score"
        # ----------------------------------------------------

        if "I" in data and "score" not in data:

            data["score"] = data.pop("I")

        # ----------------------------------------------------
        # Backward compatibility:
        # old entries have no reward field
        # ----------------------------------------------------

        if "reward" not in data:

            data["reward"] = None

        return OracleQuery(
            **data
        )

    # ========================================================
    # IO
    # ========================================================

    def _read_all(
        self,
    ) -> list[OracleQuery]:

        queries = []

        if not self.path.exists():

            return queries

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:

                    continue

                try:

                    data = json.loads(
                        line
                    )

                except json.JSONDecodeError as e:

                    raise RuntimeError(
                        f"Invalid JSONL at line {line_number}: {line}"
                    ) from e

                queries.append(
                    self._deserialize(data)
                )

        return queries

    def _write_all(
        self,
        queries: list[OracleQuery],
    ):

        tmp = self.path.with_suffix(
            ".tmp"
        )

        with open(
            tmp,
            "w",
            encoding="utf-8",
        ) as f:

            for q in queries:

                f.write(
                    json.dumps(
                        self._serialize(q),
                        ensure_ascii=False,
                    )
                )

                f.write("\n")

        tmp.replace(
            self.path
        )

    # ========================================================
    # Add
    # ========================================================

    def add(
        self,
        fen: str,
        H: float,
        U: float,
        HU: float,
        score: float,
        threshold: float,
        model: str,
        epoch: int,
        game_id: int,
        ply: int,
    ) -> OracleQuery:

        q = OracleQuery(

            query_id=uuid.uuid4().hex,

            fen=fen,

            H=H,
            U=U,
            HU=HU,

            score=score,
            threshold=threshold,

            model=model,
            epoch=epoch,

            game_id=game_id,
            ply=ply,

            created_at=self._timestamp(),
        )

        with open(
            self.path,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    self._serialize(q),
                    ensure_ascii=False,
                )
            )

            f.write("\n")

        return q

    # ========================================================
    # Retrieval
    # ========================================================

    def get(
        self,
        query_id: str,
    ) -> Optional[OracleQuery]:

        for q in self._read_all():

            if q.query_id == query_id:

                return q

        return None

    def pending(
        self,
        reward_mode: bool = False,
        oracle_mode: bool = False,
    ):
        """
        Return queries that still require annotation.

        reward_mode:
            Include queries whose reward is missing.

        oracle_mode:
            Include queries whose classical Oracle annotation
            is incomplete.

        If both modes are enabled, a query is returned if either
        annotation type is incomplete.

        Discarded queries are ignored.
        """

        queries = self._read_all()

        result = []

        for q in queries:

            if q.status == "discarded":
                continue

            reward_missing = (
                reward_mode
                and q.reward is None
            )

            oracle_missing = (
                oracle_mode
                and (
                    q.oracle_move is None
                    or q.oracle_confidence is None
                    or q.oracle_situation is None
                )
            )

            if reward_missing or oracle_missing:

                result.append(q)

        return result

    # ========================================================
    # Next annotation
    # ========================================================

    def next(
        self,
        reward_mode: bool = False,
        oracle_mode: bool = False,
    ):
        """
        Return the next query requiring annotation.
        """

        pending = self.pending(
            reward_mode=reward_mode,
            oracle_mode=oracle_mode,
        )

        if not pending:
            return None

        return pending[0]

    # ========================================================
    # Answer
    # ========================================================

    def answer(
        self,
        query_id: str,
        reward: Optional[int] = None,
        oracle_move: Optional[str] = None,
        confidence: Optional[str] = None,
        situation: Optional[str] = None,
    ):

        # ====================================================
        # Determine annotation type
        # ====================================================

        reward_given = reward is not None

        oracle_given = (
            oracle_move is not None
            or confidence is not None
            or situation is not None
        )

        if not reward_given and not oracle_given:

            raise ValueError(
                "No annotation provided."
            )

        # ====================================================
        # Validate reward
        # ====================================================

        if reward_given:

            if reward not in self.VALID_REWARDS:

                raise ValueError(
                    f"Invalid reward: {reward}. "
                    f"Expected one of {sorted(self.VALID_REWARDS)}."
                )

        # ====================================================
        # Validate Oracle annotation
        # ====================================================

        if oracle_given:

            if oracle_move is None:

                raise ValueError(
                    "Oracle move is required."
                )

            if confidence is None:

                raise ValueError(
                    "Oracle confidence is required."
                )

            if situation is None:

                raise ValueError(
                    "Oracle situation is required."
                )

            if confidence not in self.VALID_CONFIDENCE:

                raise ValueError(
                    f"Invalid confidence: {confidence}"
                )

            if situation not in self.VALID_SITUATION:

                raise ValueError(
                    f"Invalid situation: {situation}"
                )

        # ====================================================
        # Update queue
        # ====================================================

        queries = self._read_all()

        for q in queries:

            if q.query_id != query_id:

                continue

            if q.status == "discarded":

                raise ValueError(
                    "Cannot annotate a discarded query."
                )

            # ------------------------------------------------
            # Prevent accidental duplicate reward
            # ------------------------------------------------

            if reward_given and q.reward is not None:

                raise ValueError(
                    "Reward already annotated."
                )

            # ------------------------------------------------
            # Prevent accidental duplicate Oracle annotation
            # ------------------------------------------------

            if oracle_given:

                if (
                    q.oracle_move is not None
                    or q.oracle_confidence is not None
                    or q.oracle_situation is not None
                ):

                    raise ValueError(
                        "Oracle annotation already exists."
                    )

            # ------------------------------------------------
            # Apply reward
            # ------------------------------------------------

            if reward_given:

                q.reward = reward

            # ------------------------------------------------
            # Apply Oracle annotation
            # ------------------------------------------------

            if oracle_given:

                q.oracle_move = oracle_move
                q.oracle_confidence = confidence
                q.oracle_situation = situation

            # ------------------------------------------------
            # Determine whether the query is now complete
            # ------------------------------------------------

            oracle_complete = (
                q.oracle_move is not None
                and q.oracle_confidence is not None
                and q.oracle_situation is not None
            )

            reward_complete = (
                q.reward is not None
            )

            # A query is fully answered when at least the
            # requested annotation has been provided.
            #
            # For backward compatibility, status remains
            # "answered" once an annotation has been made.

            if reward_given or oracle_given:

                q.status = "answered"

                q.answered_at = self._timestamp()

            self._write_all(
                queries
            )

            return q

        raise KeyError(
            f"Unknown query id: {query_id}"
        )

    # ========================================================
    # Discard
    # ========================================================

    def discard(
        self,
        query_id: str,
    ):

        queries = self._read_all()

        for q in queries:

            if q.query_id != query_id:

                continue

            q.status = "discarded"

            self._write_all(
                queries
            )

            return q

        raise KeyError(
            f"Unknown query id: {query_id}"
        )

    # ========================================================
    # Statistics
    # ========================================================

    def stats(self):

        queries = self._read_all()

        return {

            "total": len(queries),

            "pending": sum(
                q.status == "pending"
                for q in queries
            ),

            "answered": sum(
                q.status == "answered"
                for q in queries
            ),

            "discarded": sum(
                q.status == "discarded"
                for q in queries
            ),

            "rewarded": sum(
                q.reward is not None
                for q in queries
            ),

            "oracle_annotated": sum(
                q.oracle_move is not None
                for q in queries
            ),

        }