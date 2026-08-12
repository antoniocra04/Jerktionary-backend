from __future__ import annotations

import base64
import binascii
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.domain.entities.chat import ChatImage, ChatMessage

# Only formats every vision model in the catalog accepts. Anything else is
# rejected up front rather than turned into an opaque provider error.
ALLOWED_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

_DATA_URI = re.compile(r"^data:(?P<media_type>[-\w.+/]+);base64,(?P<data>.+)$", re.DOTALL)


class ChatImageIn(BaseModel):
    """An attachment as the client sends it: a base64 data: URI.

    Validated here so a malformed paste fails with a 422 naming the problem,
    instead of costing a round trip to the provider first.
    """

    # 8 MB of base64 is ~6 MB of image; more than any of these models can use.
    data_url: str = Field(min_length=1, max_length=8_000_000)

    @field_validator("data_url")
    @classmethod
    def _validate(cls, value: str) -> str:
        match = _DATA_URI.match(value.strip())
        if match is None:
            raise ValueError("expected a data:<media-type>;base64,<data> URI")
        media_type = match.group("media_type").lower()
        if media_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"unsupported image type {media_type}; "
                f"allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}"
            )
        try:
            base64.b64decode(match.group("data"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image payload is not valid base64") from exc
        return value

    def to_entity(self) -> ChatImage:
        match = _DATA_URI.match(self.data_url.strip())
        assert match is not None  # guaranteed by the validator
        return ChatImage(media_type=match.group("media_type").lower(), data=match.group("data"))


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=100_000)
    images: list[ChatImageIn] = Field(default_factory=list, max_length=8)

    def to_entity(self) -> ChatMessage:
        return ChatMessage(
            role=self.role,
            text=self.content,
            images=tuple(image.to_entity() for image in self.images),
        )


class ChatRequest(BaseModel):
    """One turn of a conversation. The whole history is resent every time — the
    backend keeps no chat state, so a client can edit or trim its own history."""

    messages: list[ChatMessageIn] = Field(min_length=1, max_length=100)
    system: str = Field(default="", max_length=10_000)
    # Empty means the model the backend was started with. There is no model list
    # endpoint: the client keeps its own, since providers disagree on /v1/models.
    model: str = Field(default="", max_length=200)
    # Must be one of the active provider's advertised levels; empty means "leave
    # the provider's own default alone".
    reasoning_effort: str = Field(default="", max_length=20)

    @field_validator("messages")
    @classmethod
    def _last_must_be_user(cls, value: list[ChatMessageIn]) -> list[ChatMessageIn]:
        if value[-1].role != "user":
            raise ValueError("the last message must be from the user")
        return value

    @field_validator("messages")
    @classmethod
    def _no_empty_turns(cls, value: list[ChatMessageIn]) -> list[ChatMessageIn]:
        for message in value:
            if not message.content.strip() and not message.images:
                raise ValueError("every message needs text or at least one image")
        return value


class ChatCapabilitiesResponse(BaseModel):
    """What the active provider can do, so the client can render only the controls
    that mean something — the reasoning picker disappears when the list is empty."""

    provider: str
    label: str
    default_model: str
    reasoning_levels: list[str]
    ready: bool
