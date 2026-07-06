from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.domain.entities.explanation import Explanation


class LlmProvider(Protocol):
    async def explain(self, term: str, context: str) -> Explanation:
        ...

    def explain_stream(self, term: str, context: str) -> AsyncIterator[dict[str, str]]:
        ...

    def answer_stream(
        self,
        question: str,
        context: str,
        *,
        deep: bool = False,
        profile: str = "",
        meeting_context: str = "",
    ) -> AsyncIterator[dict[str, str]]:
        ...

    async def healthcheck(self) -> bool:
        ...

