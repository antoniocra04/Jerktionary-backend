from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.api.schemas.terms import TermItem


class AsrChoice(BaseModel):
    """Per-connection ASR selection. ``local`` uses the on-box Whisper; ``api`` uses
    an OpenAI-compatible /audio/transcriptions endpoint with the supplied key."""

    provider: Literal["local", "api"] = "local"
    api_key: str = Field(default="", max_length=400)
    model: str = Field(default="", max_length=200)
    base_url: str = Field(default="", max_length=400)


class WsConfigMessage(BaseModel):
    """Optional first (text) message on /ws/audio, sent before any audio bytes."""

    type: Literal["config"]
    asr: AsrChoice = Field(default_factory=AsrChoice)


class TranscriptUpdateEvent(BaseModel):
    type: Literal["transcript_update"] = "transcript_update"
    text: str
    is_final: bool
    terms: list[TermItem] = []


class TermsUpdateEvent(BaseModel):
    type: Literal["terms_update"] = "terms_update"
    items: list[TermItem]

