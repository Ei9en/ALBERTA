# oracle_queue.py

from __future__ import annotations

import json
import uuid

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ============================================================
# Oracle Query
# ============================================================

@dataclass
class OracleQuery:
    """
    One position submitted to the human oracle.
    """

    query_id: str

    fen: str

    H: float
    U: float
    HU: float

    score: float
    threshold: float

    model: str
    epoch: int

    game_id: int
    ply: int

    created_at: str

    status: str = "pending"

    oracle_move: Optional[str] = None

    answered_at: Optional[str] = None


# ============================================================
# Oracle Queue
# ============================================================

class OracleQueue:

    def __init__(
        self,
        path: str | Path = "data/oracle_queue.jsonl",
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        #
        # Create the file if it does not exist.
        #
        self.path.touch(
            exist_ok=True,
        )

    # ========================================================
    # Internal helpers
    # ========================================================

    def _read_all(self) -> list[OracleQuery]:
        """
        Load every query from the JSONL file.
        """

        queries = []

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                data = json.loads(line)

                queries.append(
                    OracleQuery(
                        **data
                    )
                )

        return queries

    def _rewrite(
        self,
        queries: list[OracleQuery],
    ) -> None:
        """
        Rewrite the complete queue.

        This is intentionally simple for now.
        """

        temporary_path = self.path.with_suffix(
            ".tmp"
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as f:

            for query in queries:

                f.write(
                    json.dumps(
                        asdict(query),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        temporary_path.replace(
            self.path
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

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
        """
        Add a new position to the oracle queue.
        """

        query = OracleQuery(
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
                    asdict(query),
                    ensure_ascii=False,
                )
                + "\n"
            )

        return query

    # ========================================================
    # Get
    # ========================================================

    def get(
        self,
        query_id: str,
    ) -> Optional[OracleQuery]:
        """
        Retrieve one query by ID.
        """

        queries = self._read_all()

        for query in queries:

            if query.query_id == query_id:

                return query

        return None

    # ========================================================
    # Pending
    # ========================================================

    def pending(
        self,
    ) -> list[OracleQuery]:
        """
        Return all unanswered queries.
        """

        return [
            query
            for query in self._read_all()
            if query.status == "pending"
        ]

    # ========================================================
    # Next
    # ========================================================

    def next(
        self,
    ) -> Optional[OracleQuery]:
        """
        Return the oldest pending query.
        """

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
    ) -> OracleQuery:
        """
        Register the human oracle's answer.
        """

        queries = self._read_all()

        for query in queries:

            if query.query_id != query_id:
                continue

            if query.status != "pending":

                raise ValueError(
                    f"Query {query_id} "
                    f"is already {query.status}."
                )

            query.oracle_move = oracle_move

            query.status = "answered"

            query.answered_at = self._timestamp()

            self._rewrite(
                queries
            )

            return query

        raise KeyError(
            f"Unknown query ID: {query_id}"
        )

    # ========================================================
    # Discard
    # ========================================================

    def discard(
        self,
        query_id: str,
    ) -> OracleQuery:
        """
        Mark a query as discarded.

        The position remains in the queue for traceability.
        """

        queries = self._read_all()

        for query in queries:

            if query.query_id != query_id:
                continue

            if query.status != "pending":

                raise ValueError(
                    f"Query {query_id} "
                    f"is already {query.status}."
                )

            query.status = "discarded"

            self._rewrite(
                queries
            )

            return query

        raise KeyError(
            f"Unknown query ID: {query_id}"
        )

    # ========================================================
    # Statistics
    # ========================================================

    def stats(self) -> dict:
        """
        Return queue statistics.
        """

        queries = self._read_all()

        total = len(queries)

        pending = sum(
            query.status == "pending"
            for query in queries
        )

        answered = sum(
            query.status == "answered"
            for query in queries
        )

        discarded = sum(
            query.status == "discarded"
            for query in queries
        )

        return {
            "total": total,
            "pending": pending,
            "answered": answered,
            "discarded": discarded,
        }


# ============================================================
# Test / demonstration
# ============================================================

if __name__ == "__main__":

    queue = OracleQueue()

    print(
        "=================================================="
    )
    print(
        "ALBERTA - ORACLE QUEUE"
    )
    print(
        "=================================================="
    )

    print(
        f"File: {queue.path}"
    )

    print()

    print(
        "Current queue:"
    )

    print(
        json.dumps(
            queue.stats(),
            indent=2,
        )
    )

    #
    # Create a demonstration query.
    #
    query = queue.add(
        fen="8/8/8/8/8/8/8/4K2k w - - 0 1",

        H=2.5,
        U=0.08,
        HU=0.20,

        score=0.981,
        threshold=0.964,

        model="rl_epoch_10",
        epoch=10,

        game_id=1,
        ply=42,
    )

    print()
    print(
        "Created query:"
    )

    print(
        json.dumps(
            asdict(query),
            indent=2,
        )
    )

    print()
    print(
        "Pending queries:"
    )

    for pending_query in queue.pending():

        print(
            pending_query.query_id,
            pending_query.status,
        )

    #
    # We deliberately do NOT automatically answer
    # the demonstration query.
    #
    print()
    print(
        "Queue ready."
    )