from __future__ import annotations

from dataclasses import dataclass

from app.domain.entities.term import Term


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    text: str
    is_final: bool
    terms: list[Term]

