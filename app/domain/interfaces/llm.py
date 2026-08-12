from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from app.domain.entities.chat import ChatMessage
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

    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str = "",
        model: str = "",
        reasoning_effort: str = "",
        max_tokens: int = 0,
    ) -> AsyncIterator[str]:
        """Free-form multi-turn chat, yielding text deltas. Unlike explain/answer
        this carries no JSON contract, so the caller picks the model and reasoning
        effort per request."""
        ...

    async def healthcheck(self) -> bool:
        ...

