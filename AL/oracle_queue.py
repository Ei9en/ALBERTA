from __future__ import annotations

import json
import uuid

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Data model
# ============================================================

@dataclass
class OracleQuery:

    query_id: str

    fen: str

    # --------------------------------------------------------
    # Active-learning raw signals
    # --------------------------------------------------------

    H: float
    U: float
    HU: float

    # --------------------------------------------------------
    # Active-learning acquisition score
    # --------------------------------------------------------

    score: float

    # Min-max normalized acquisition score.
    # Used for representation / diagnostics only.
    I_norm: float | None = None

    # Optional selection threshold.
    threshold: float | None = None

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    model: str = "historical_json"
    epoch: int = -1

    game_id: int = -1
    ply: int = -1

    created_at: str = ""

    # Lifecycle:
    #
    # pending   -> no annotation
    # partial   -> only one annotation family is complete
    # answered  -> reward + Oracle annotation complete
    # discarded -> ignored
    status: str = "pending"

    # --------------------------------------------------------
    # Oracle annotation
    # --------------------------------------------------------

    oracle_move: str | None = None
    oracle_confidence: str | None = None
    oracle_situation: str | None = None

    # --------------------------------------------------------
    # Reward annotation
    #
    # +1 = good outcome for side to move
    #  0 = neutral / unclear
    # -1 = bad outcome for side to move
    # --------------------------------------------------------

    reward: float | None = None

    answered_at: str | None = None


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

    VALID_STATUS = {
        "pending",
        "partial",
        "answered",
        "discarded",
    }

    def __init__(
        self,
        path: str | Path = "data/oracle_queue.jsonl",
    ) -> None:

        self.path = Path(
            path
        )

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
    def _timestamp() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def fen_to_query_id(
        fen: str,
    ) -> str:
        """
        Return the deterministic ALBERTA query identifier
        associated with a FEN.

        This must remain identical to the convention used by
        seed_oracle_queue.py and tools/fen_to_query_id.py.
        """

        return uuid.uuid5(
            uuid.NAMESPACE_DNS,
            fen,
        ).hex

    @staticmethod
    def _oracle_complete(
        q: OracleQuery,
    ) -> bool:

        return (
            q.oracle_move is not None
            and q.oracle_confidence is not None
            and q.oracle_situation is not None
        )

    @staticmethod
    def _oracle_started(
        q: OracleQuery,
    ) -> bool:

        return (
            q.oracle_move is not None
            or q.oracle_confidence is not None
            or q.oracle_situation is not None
        )

    @classmethod
    def _refresh_status(
        cls,
        q: OracleQuery,
    ) -> None:
        """
        Recompute queue lifecycle status from annotations.

        Discarded entries remain discarded.
        """

        if q.status == "discarded":
            return

        reward_complete = (
            q.reward is not None
        )

        oracle_complete = (
            cls._oracle_complete(
                q
            )
        )

        oracle_started = (
            cls._oracle_started(
                q
            )
        )

        if (
            reward_complete
            and oracle_complete
        ):

            q.status = "answered"

        elif (
            reward_complete
            or oracle_started
        ):

            q.status = "partial"

        else:

            q.status = "pending"

    # ========================================================
    # Serialization
    # ========================================================

    @staticmethod
    def _serialize(
        q: OracleQuery,
    ) -> dict:

        return asdict(
            q
        )

    @classmethod
    def _deserialize(
        cls,
        data: dict,
    ) -> OracleQuery:

        data = dict(
            data
        )

        # ----------------------------------------------------
        # Backward compatibility:
        # old queue used "I" instead of "score"
        # ----------------------------------------------------

        if (
            "I" in data
            and "score" not in data
        ):

            data[
                "score"
            ] = data.pop(
                "I"
            )

        elif "I" in data:

            data.pop(
                "I"
            )

        # ----------------------------------------------------
        # Backward compatibility:
        # old entries may not have I_norm
        # ----------------------------------------------------

        if "I_norm" not in data:

            data[
                "I_norm"
            ] = None

        # ----------------------------------------------------
        # Backward compatibility:
        # old entries may not have threshold
        # ----------------------------------------------------

        if "threshold" not in data:

            data[
                "threshold"
            ] = None

        # ----------------------------------------------------
        # Backward compatibility:
        # old entries may not have metadata
        # ----------------------------------------------------

        if "model" not in data:

            data[
                "model"
            ] = "historical_json"

        if "epoch" not in data:

            data[
                "epoch"
            ] = -1

        if "game_id" not in data:

            data[
                "game_id"
            ] = -1

        if "ply" not in data:

            data[
                "ply"
            ] = -1

        if "created_at" not in data:

            data[
                "created_at"
            ] = ""

        # ----------------------------------------------------
        # Backward compatibility:
        # old entries may not have reward
        # ----------------------------------------------------

        if "reward" not in data:

            data[
                "reward"
            ] = None

        # ----------------------------------------------------
        # Backward compatibility:
        # previous buggy version used "oracle_criticality".
        #
        # Canonical field:
        #
        #     oracle_situation
        # ----------------------------------------------------

        if (
            "oracle_criticality" in data
            and "oracle_situation" not in data
        ):

            data[
                "oracle_situation"
            ] = data.pop(
                "oracle_criticality"
            )

        elif "oracle_criticality" in data:

            data.pop(
                "oracle_criticality"
            )

        # ----------------------------------------------------
        # Oracle fields
        # ----------------------------------------------------

        if "oracle_move" not in data:

            data[
                "oracle_move"
            ] = None

        if "oracle_confidence" not in data:

            data[
                "oracle_confidence"
            ] = None

        if "oracle_situation" not in data:

            data[
                "oracle_situation"
            ] = None

        if "answered_at" not in data:

            data[
                "answered_at"
            ] = None

        if "status" not in data:

            data[
                "status"
            ] = "pending"

        q = OracleQuery(
            **data
        )

        # ----------------------------------------------------
        # Normalize historical status.
        #
        # Old versions marked an entry "answered" after only
        # one annotation family. Status is now derived from
        # the actual stored fields.
        # ----------------------------------------------------

        cls._refresh_status(
            q
        )

        return q

    # ========================================================
    # IO
    # ========================================================

    def _read_all(
        self,
    ) -> list[OracleQuery]:

        queries = []

        with self.path.open(
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

                except json.JSONDecodeError as exc:

                    raise RuntimeError(
                        f"Invalid JSONL in {self.path} "
                        f"at line {line_number}: {line}"
                    ) from exc

                queries.append(
                    self._deserialize(
                        data
                    )
                )

        return queries

    def _write_all(
        self,
        queries: list[OracleQuery],
    ) -> None:

        tmp = self.path.with_name(
            self.path.name
            + ".tmp"
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as f:

            for q in queries:

                f.write(
                    json.dumps(
                        self._serialize(
                            q
                        ),
                        ensure_ascii=False,
                    )
                )

                f.write(
                    "\n"
                )

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
        I_norm: float | None = None,
        threshold: float | None = None,
        model: str = "historical_json",
        epoch: int = -1,
        game_id: int = -1,
        ply: int = -1,
    ) -> OracleQuery:

        query_id = self.fen_to_query_id(
            fen
        )

        # ----------------------------------------------------
        # Prevent duplicate FENs inside the same queue.
        # ----------------------------------------------------

        if self.get(
            query_id
        ) is not None:

            raise ValueError(
                "Query already exists in this queue: "
                f"{query_id}"
            )

        q = OracleQuery(
            query_id=query_id,

            fen=fen,

            H=float(
                H
            ),

            U=float(
                U
            ),

            HU=float(
                HU
            ),

            score=float(
                score
            ),

            I_norm=(
                None
                if I_norm is None
                else float(
                    I_norm
                )
            ),

            threshold=(
                None
                if threshold is None
                else float(
                    threshold
                )
            ),

            model=model,
            epoch=int(
                epoch
            ),

            game_id=int(
                game_id
            ),

            ply=int(
                ply
            ),

            created_at=self._timestamp(),

            status="pending",
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    self._serialize(
                        q
                    ),
                    ensure_ascii=False,
                )
            )

            f.write(
                "\n"
            )

        return q

    # ========================================================
    # Retrieval
    # ========================================================

    def get(
        self,
        query_id: str,
    ) -> OracleQuery | None:

        for q in self._read_all():

            if q.query_id == query_id:

                return q

        return None

    # ========================================================
    # Pending
    # ========================================================

    def pending(
        self,
        reward_mode: bool = False,
        oracle_mode: bool = False,
    ) -> list[OracleQuery]:
        """
        Return queries that still require annotation.

        reward_mode:
            Include queries whose reward annotation is missing.

        oracle_mode:
            Include queries whose Oracle annotation is incomplete.

        If both modes are enabled, a query is returned whenever
        either requested annotation type remains incomplete.

        Discarded queries are always ignored.
        """

        if (
            not reward_mode
            and not oracle_mode
        ):

            return []

        result = []

        for q in self._read_all():

            if q.status == "discarded":

                continue

            reward_missing = (
                reward_mode
                and q.reward is None
            )

            oracle_missing = (
                oracle_mode
                and not self._oracle_complete(
                    q
                )
            )

            if (
                reward_missing
                or oracle_missing
            ):

                result.append(
                    q
                )

        return result

    # ========================================================
    # Next annotation
    # ========================================================

    def next(
        self,
        reward_mode: bool = False,
        oracle_mode: bool = False,
    ) -> OracleQuery | None:

        pending = self.pending(
            reward_mode=reward_mode,
            oracle_mode=oracle_mode,
        )

        if not pending:

            return None

        return pending[
            0
        ]

    # ========================================================
    # Answer
    # ========================================================

    def answer(
        self,
        query_id: str,
        reward: int | None = None,
        oracle_move: str | None = None,
        confidence: str | None = None,
        situation: str | None = None,
    ) -> OracleQuery:
        """
        Add a reward annotation and/or an Oracle annotation.

        A complete Oracle annotation requires all three fields:

            oracle_move
            confidence
            situation

        Reward and Oracle annotations may be supplied either
        together or in separate calls.
        """

        # ====================================================
        # Determine annotation families supplied
        # ========================================================

        reward_given = (
            reward is not None
        )

        oracle_given = (
            oracle_move is not None
            or confidence is not None
            or situation is not None
        )

        if (
            not reward_given
            and not oracle_given
        ):

            raise ValueError(
                "No annotation provided."
            )

        # ====================================================
        # Validate reward
        # ========================================================

        if (
            reward_given
            and reward not in self.VALID_REWARDS
        ):

            raise ValueError(
                f"Invalid reward: {reward}. "
                f"Expected one of "
                f"{sorted(self.VALID_REWARDS)}."
            )

        # ====================================================
        # Validate Oracle annotation
        # ========================================================

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
                    "Decision situation annotation is required."
                )

            if confidence not in self.VALID_CONFIDENCE:

                raise ValueError(
                    f"Invalid confidence: {confidence}. "
                    f"Expected one of "
                    f"{sorted(self.VALID_CONFIDENCE)}."
                )

            if situation not in self.VALID_SITUATION:

                raise ValueError(
                    f"Invalid situation: {situation}. "
                    f"Expected one of "
                    f"{sorted(self.VALID_SITUATION)}."
                )

        # ====================================================
        # Update queue
        # ========================================================

        queries = self._read_all()

        for q in queries:

            if q.query_id != query_id:

                continue

            if q.status == "discarded":

                raise ValueError(
                    "Cannot annotate a discarded query."
                )

            # ------------------------------------------------
            # Prevent duplicate reward annotation
            # ------------------------------------------------

            if (
                reward_given
                and q.reward is not None
            ):

                raise ValueError(
                    "Reward already annotated."
                )

            # ------------------------------------------------
            # Prevent duplicate / partial Oracle overwrite
            # ------------------------------------------------

            if (
                oracle_given
                and self._oracle_started(
                    q
                )
            ):

                raise ValueError(
                    "Oracle annotation already exists."
                )

            # ------------------------------------------------
            # Apply reward
            # ------------------------------------------------

            if reward_given:

                q.reward = float(
                    reward
                )

            # ------------------------------------------------
            # Apply Oracle annotation
            # ------------------------------------------------

            if oracle_given:

                q.oracle_move = (
                    oracle_move
                )

                q.oracle_confidence = (
                    confidence
                )

                q.oracle_situation = (
                    situation
                )

            # ------------------------------------------------
            # Recompute lifecycle status
            # ------------------------------------------------

            previous_status = (
                q.status
            )

            self._refresh_status(
                q
            )

            # answered_at records completion of the complete
            # annotation record, not a partial annotation.
            if (
                q.status == "answered"
                and previous_status != "answered"
            ):

                q.answered_at = (
                    self._timestamp()
                )

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
    ) -> OracleQuery:

        queries = self._read_all()

        for q in queries:

            if q.query_id != query_id:

                continue

            if q.status == "discarded":

                raise ValueError(
                    "Query is already discarded."
                )

            q.status = (
                "discarded"
            )

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

    def stats(
        self,
    ) -> dict[str, int]:

        queries = self._read_all()

        return {
            "total":
                len(
                    queries
                ),

            "pending":
                sum(
                    q.status == "pending"
                    for q in queries
                ),

            "partial":
                sum(
                    q.status == "partial"
                    for q in queries
                ),

            "answered":
                sum(
                    q.status == "answered"
                    for q in queries
                ),

            "discarded":
                sum(
                    q.status == "discarded"
                    for q in queries
                ),

            "rewarded":
                sum(
                    q.reward is not None
                    for q in queries
                ),

            "oracle_annotated":
                sum(
                    self._oracle_complete(
                        q
                    )
                    for q in queries
                ),
        }