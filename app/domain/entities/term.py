from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Term:
    text: str
    normalized: str
    start: int
    end: int
    type: str
    confidence: float

