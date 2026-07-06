from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.domain.entities.explanation import Explanation
from app.infrastructure.db.repositories.explanation_repository import SQLiteExplanationRepository
from app.infrastructure.db.sqlite import SQLiteDatabase
from app.services.explanation_service import ExplanationService


class FakeLlmProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def healthcheck(self) -> bool:
        return True

    async def explain(self, term: str, context: str) -> Explanation:
        self.calls += 1
        return Explanation(
            title=term.title(),
            short="Краткое объяснение",
            example=context or "Пример",
            why_important="Важно для понимания темы",
            source="local_llm",
        )

    async def explain_stream(self, term: str, context: str) -> AsyncIterator[dict[str, str]]:
        self.calls += 1
        yield {
            "title": term.title(),
            "short": "Краткое объяснение",
            "example": context or "Пример",
            "why_important": "Важно для понимания темы",
        }

    async def answer_stream(
        self, question: str, context: str, *, deep: bool = False
    ) -> AsyncIterator[dict[str, str]]:
        self.calls += 1
        yield {"answer": "Прямой ответ", "points": "- тезис", "example": "Пример"}


@pytest.mark.asyncio
async def test_explanation_service_streams_and_caches(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "cache.sqlite3")
    await database.connect()
    await database.create_schema()
    repository = SQLiteExplanationRepository(database)
    llm = FakeLlmProvider()
    service = ExplanationService(repository, llm, llm_enabled=True)

    snapshots = [
        snap
        async for snap in service.explain_stream(
            term="бинарный поиск",
            normalized_term="бинарный поиск",
            context="алгоритмы",
        )
    ]
    # streamed to completion and produced a final snapshot from the LLM
    assert snapshots[-1]["done"] is True
    assert snapshots[-1]["source"] == "local_llm"

    # a second call is served from cache without hitting the LLM again
    cached = [
        snap
        async for snap in service.explain_stream(
            term="бинарный поиск",
            normalized_term="бинарный поиск",
            context="другой контекст",
        )
    ]
    await database.close()
    assert cached[-1]["source"] == "cache"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_explanation_service_uses_cache(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "cache.sqlite3")
    await database.connect()
    await database.create_schema()
    repository = SQLiteExplanationRepository(database)
    llm = FakeLlmProvider()
    service = ExplanationService(repository, llm, llm_enabled=True)

    first = await service.explain(
        term="теория относительности",
        normalized_term="теория относительности",
        context="скорость света",
    )
    second = await service.explain(
        term="теории относительности",
        normalized_term="теория относительности",
        context="другой контекст",
    )

    await database.close()
    assert first.source == "local_llm"
    assert second.source == "cache"
    assert llm.calls == 1

