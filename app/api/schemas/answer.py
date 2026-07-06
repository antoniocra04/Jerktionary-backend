from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.schemas.llm import LlmChoice


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    context: str = Field(default="", max_length=2_000)
    deep: bool = False
    # Persistent "about me" from the user's settings: role, stack, experience.
    profile: str = Field(default="", max_length=1_000)
    # Free-form context for this particular meeting (position, company, topic).
    meeting_context: str = Field(default="", max_length=2_000)
    llm: LlmChoice = Field(default_factory=LlmChoice)
