from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ChatImage:
    """One attached image, already decoded out of its data: URI.

    Stored as raw base64 without the ``data:`` prefix because that is what the
    Anthropic and Ollama wire formats want; the OpenAI-compatible client puts the
    prefix back when it builds an ``image_url`` part.
    """

    media_type: str
    data: str


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    text: str
    images: tuple[ChatImage, ...] = field(default_factory=tuple)
