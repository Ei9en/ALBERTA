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

    oracle_move: Optional[str] = None
    oracle_confidence: Optional[str] = None
    oracle_situation: Optional[str] = None

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
        "unique_move",
        "multiple_good",
        "everything_wins",
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

        # Backward compatibility: old queue used "I" instead of "score"
        if "I" in data and "score" not in data:
            data["score"] = data.pop("I")

        return OracleQuery(
            **data
        )



    # ========================================================
    # IO
    # ========================================================

    def _read_all(self) -> list[OracleQuery]:

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



    def pending(self):

        return [

            q

            for q in self._read_all()

            if q.status == "pending"

        ]



    def next(self):

        pending = self.pending()

        if not pending:
            return None

        return pending[0]



    # ========================================================
    # Answer
    # ========================================================

    def answer(
        self,
        query_id: str,
        oracle_move: str,
        confidence: str,
        situation: str,
    ):


        if confidence not in self.VALID_CONFIDENCE:

            raise ValueError(
                f"Invalid confidence: {confidence}"
            )


        if situation not in self.VALID_SITUATION:

            raise ValueError(
                f"Invalid situation: {situation}"
            )


        queries = self._read_all()


        for q in queries:

            if q.query_id != query_id:
                continue


            if q.status != "pending":

                raise ValueError(
                    "Query already answered"
                )


            q.oracle_move = oracle_move
            q.oracle_confidence = confidence
            q.oracle_situation = situation

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
        }