from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.core.errors import LlmResponseError, LlmUnavailableError
from app.domain.interfaces.llm import LlmProvider


class AnswerService:
    """Streams a spoken-style answer to a whole question detected in the conversation."""

    def __init__(self, llm_provider: LlmProvider, *, llm_enabled: bool) -> None:
        self._llm_provider = llm_provider
        self._llm_enabled = llm_enabled

    async def answer_stream(
        self,
        *,
        question: str,
        context: str,
        deep: bool = False,
        profile: str = "",
        meeting_context: str = "",
        provider: LlmProvider | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if provider is not None:
            active = provider
        elif self._llm_enabled:
            active = self._llm_provider
        else:
            raise LlmUnavailableError("Local LLM is unavailable")

        previous: dict[str, str] | None = None
        async for fields in active.answer_stream(
            question=question,
            context=context,
            deep=deep,
            profile=profile,
            meeting_context=meeting_context,
        ):
            if previous is not None:
                yield {**previous, "done": False}
            previous = fields

        if previous is None:
            raise LlmResponseError("Local LLM produced no output")

        yield {**previous, "done": True}
