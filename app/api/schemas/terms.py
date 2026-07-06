from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.api.schemas.llm import LlmChoice


class TermItem(BaseModel):
    text: str
    normalized: str
    start: int
    end: int
    type: str = "concept"
    confidence: float = Field(ge=0.0, le=1.0)


class ExplainTermRequest(BaseModel):
    term: str = Field(min_length=1, max_length=200)
    context: str = Field(default="", max_length=2_000)
    llm: LlmChoice = Field(default_factory=LlmChoice)


class ExplainTermResponse(BaseModel):
    title: str
    short: str
    example: str
    why_important: str
    source: Literal["cache", "local_llm"]

