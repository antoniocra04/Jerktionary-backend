from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any

from app.core.errors import LlmResponseError, LlmUnavailableError
from app.domain.entities.explanation import Explanation
from app.domain.interfaces.cache import ExplanationCacheRepository
from app.domain.interfaces.llm import LlmProvider


class ExplanationService:
    def __init__(
        self,
        repository: ExplanationCacheRepository,
        llm_provider: LlmProvider,
        *,
        llm_enabled: bool,
    ) -> None:
        self._repository = repository
        self._llm_provider = llm_provider
        self._llm_enabled = llm_enabled

    def _resolve(self, provider: LlmProvider | None) -> LlmProvider:
        if provider is not None:
            return provider
        if not self._llm_enabled:
            raise LlmUnavailableError("Local LLM is unavailable")
        return self._llm_provider

    async def explain(
        self,
        *,
        term: str,
        normalized_term: str,
        context: str,
        provider: LlmProvider | None = None,
    ) -> Explanation:
        cached = await self._repository.get_by_normalized_term(normalized_term)
        if cached is not None:
            return Explanation(
                title=cached.title,
                short=cached.short,
                example=cached.example,
                why_important=cached.why_important,
                source="cache",
            )

        active = self._resolve(provider)
        explanation = await active.explain(term=term, context=context)
        saved = await self._repository.save(
            normalized_term=normalized_term,
            original_term=term,
            context_hash=self._context_hash(context),
            explanation=explanation,
        )
        return Explanation(
            title=saved.title,
            short=saved.short,
            example=saved.example,
            why_important=saved.why_important,
            source="local_llm",
        )

    async def explain_stream(
        self,
        *,
        term: str,
        normalized_term: str,
        context: str,
        provider: LlmProvider | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        cached = await self._repository.get_by_normalized_term(normalized_term)
        if cached is not None:
            yield {
                "title": cached.title,
                "short": cached.short,
                "example": cached.example,
                "why_important": cached.why_important,
                "source": "cache",
                "done": True,
            }
            return

        active = self._resolve(provider)

        previous: dict[str, str] | None = None
        async for fields in active.explain_stream(term=term, context=context):
            if previous is not None:
                yield {**previous, "source": "local_llm", "done": False}
            previous = fields

        if previous is None:
            raise LlmResponseError("Local LLM produced no output")

        explanation = Explanation(
            title=previous["title"],
            short=previous["short"],
            example=previous["example"],
            why_important=previous["why_important"],
            source="local_llm",
        )
        await self._repository.save(
            normalized_term=normalized_term,
            original_term=term,
            context_hash=self._context_hash(context),
            explanation=explanation,
        )
        yield {**previous, "source": "local_llm", "done": True}

    @staticmethod
    def _context_hash(context: str) -> str:
        normalized = " ".join(context.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

