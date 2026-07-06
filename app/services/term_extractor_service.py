from __future__ import annotations

from app.domain.entities.term import Term
from app.domain.interfaces.nlp import NlpTermExtractor


class TermExtractorService:
    def __init__(self, extractor: NlpTermExtractor) -> None:
        self._extractor = extractor

    async def extract(self, text: str) -> list[Term]:
        if not text.strip():
            return []
        terms = await self._extractor.extract(text)
        return sorted(terms, key=lambda item: (item.start, -item.confidence))

    def normalize(self, term: str) -> str:
        return self._extractor.normalize(term)

