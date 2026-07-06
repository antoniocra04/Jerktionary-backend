from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class Explanation:
    title: str
    short: str
    example: str
    why_important: str
    source: Literal["cache", "local_llm"]


@dataclass(frozen=True, slots=True)
class CachedExplanation(Explanation):
    id: int
    normalized_term: str
    original_term: str
    context_hash: str
    created_at: datetime
    updated_at: datetime
