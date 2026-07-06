from __future__ import annotations

from pathlib import Path

import aiosqlite


class SQLiteDatabase:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection: aiosqlite.Connection | None = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("SQLite is not connected")
        return self._connection

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._path)
        self._connection.row_factory = aiosqlite.Row
        await self.connection.execute("PRAGMA journal_mode=WAL")
        await self.connection.execute("PRAGMA foreign_keys=ON")

    async def create_schema(self) -> None:
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized_term TEXT NOT NULL,
                original_term TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                short TEXT NOT NULL,
                example TEXT NOT NULL,
                why_important TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_explanations_normalized_term "
            "ON explanations(normalized_term)"
        )
        await self.connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

