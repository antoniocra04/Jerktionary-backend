from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from app.domain.entities.explanation import CachedExplanation, Explanation
from app.infrastructure.db.sqlite import SQLiteDatabase


class SQLiteExplanationRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def get_by_normalized_term(self, normalized_term: str) -> CachedExplanation | None:
        cursor = await self._database.connection.execute(
            """
            SELECT *
            FROM explanations
            WHERE normalized_term = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (normalized_term,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return self._to_entity(row)

    async def save(
        self,
        *,
        normalized_term: str,
        original_term: str,
        context_hash: str,
        explanation: Explanation,
    ) -> CachedExplanation:
        cursor = await self._database.connection.execute(
            """
            INSERT INTO explanations (
                normalized_term,
                original_term,
                context_hash,
                title,
                short,
                example,
                why_important,
                source,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                normalized_term,
                original_term,
                context_hash,
                explanation.title,
                explanation.short,
                explanation.example,
                explanation.why_important,
                explanation.source,
            ),
        )
        await self._database.connection.commit()
        item_id = cursor.lastrowid
        await cursor.close()

        cursor = await self._database.connection.execute(
            "SELECT * FROM explanations WHERE id = ?",
            (item_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise RuntimeError("Saved explanation not found")
        return self._to_entity(row)

    @staticmethod
    def _to_entity(row: Any) -> CachedExplanation:
        raw_source = str(row["source"])
        if raw_source not in {"cache", "local_llm"}:
            raise ValueError(f"Unknown explanation source: {raw_source}")
        source: Literal["cache", "local_llm"] = (
            "cache" if raw_source == "cache" else "local_llm"
        )
        return CachedExplanation(
            id=int(row["id"]),
            normalized_term=str(row["normalized_term"]),
            original_term=str(row["original_term"]),
            context_hash=str(row["context_hash"]),
            title=str(row["title"]),
            short=str(row["short"]),
            example=str(row["example"]),
            why_important=str(row["why_important"]),
            source=source,
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )
