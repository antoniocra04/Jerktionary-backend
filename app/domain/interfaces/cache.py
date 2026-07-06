from __future__ import annotations

from typing import Protocol

from app.domain.entities.explanation import CachedExplanation, Explanation


class ExplanationCacheRepository(Protocol):
    async def get_by_normalized_term(self, normalized_term: str) -> CachedExplanation | None:
        ...

    async def save(
        self,
        *,
        normalized_term: str,
        original_term: str,
        context_hash: str,
        explanation: Explanation,
    ) -> CachedExplanation:
        ...

