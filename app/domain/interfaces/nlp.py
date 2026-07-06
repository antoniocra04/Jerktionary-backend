from __future__ import annotations

from typing import Protocol

from app.domain.entities.term import Term


class NlpTermExtractor(Protocol):
    async def extract(self, text: str) -> list[Term]:
        ...

    def normalize(self, term: str) -> str:
        ...

