from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.core.errors import LlmUnavailableError
from app.services.answer_service import AnswerService


class FakeAnswerProvider:
    async def healthcheck(self) -> bool:
        return True

    async def explain_stream(self, term: str, context: str) -> AsyncIterator[dict[str, str]]:
        yield {"title": term, "short": "", "example": "", "why_important": ""}

    async def answer_stream(
        self,
        question: str,
        context: str,
        *,
        deep: bool = False,
        profile: str = "",
        meeting_context: str = "",
    ) -> AsyncIterator[dict[str, str]]:
        yield {"answer": "part", "points": "- a", "example": ""}
        yield {"answer": "Полный ответ", "points": "- a\n- b", "example": "Пример"}


@pytest.mark.asyncio
async def test_answer_service_streams_partials_then_final() -> None:
    service = AnswerService(FakeAnswerProvider(), llm_enabled=True)

    snapshots = [
        snap
        async for snap in service.answer_stream(question="что такое стек?", context="алгоритмы")
    ]

    assert snapshots[0]["done"] is False
    assert snapshots[-1]["done"] is True
    assert snapshots[-1]["answer"] == "Полный ответ"


@pytest.mark.asyncio
async def test_answer_service_raises_when_llm_disabled() -> None:
    service = AnswerService(FakeAnswerProvider(), llm_enabled=False)

    with pytest.raises(LlmUnavailableError):
        async for _ in service.answer_stream(question="q", context=""):
            pass
