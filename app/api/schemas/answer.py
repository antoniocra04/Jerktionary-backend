from __future__ import annotations

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from app.api.schemas.llm import LlmChoice

# Fields fed from live speech have no length guarantees: realtime partials carry no
# punctuation yet, so the frontend's question detector can hand over a run-on
# "sentence" far longer than any sane cap. Rejecting that with 422 kills the answer
# card mid-interview — instead over-long values are trimmed to these caps.
_KEEP_TAIL_CAPS = {"question": 1_000, "context": 2_000}
_KEEP_HEAD_CAPS = {"profile": 1_000, "meeting_context": 2_000}


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    context: str = Field(default="", max_length=2_000)
    deep: bool = False
    # Persistent "about me" from the user's settings: role, stack, experience.
    profile: str = Field(default="", max_length=1_000)
    # Free-form context for this particular meeting (position, company, topic).
    meeting_context: str = Field(default="", max_length=2_000)
    llm: LlmChoice = Field(default_factory=LlmChoice)

    @field_validator("question", "context", mode="before")
    @classmethod
    def _keep_tail(cls, value: object, info: ValidationInfo) -> object:
        # The transcript grows from the start, so the freshest speech — the actual
        # ask and the relevant conversation — is at the end.
        if isinstance(value, str) and info.field_name in _KEEP_TAIL_CAPS:
            return value[-_KEEP_TAIL_CAPS[info.field_name] :]
        return value

    @field_validator("profile", "meeting_context", mode="before")
    @classmethod
    def _keep_head(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, str) and info.field_name in _KEEP_HEAD_CAPS:
            return value[: _KEEP_HEAD_CAPS[info.field_name]]
        return value
