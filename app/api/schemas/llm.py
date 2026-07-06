from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LlmChoice(BaseModel):
    """Per-request LLM selection sent by the client. ``local`` uses the on-box model;
    ``api`` uses a hosted provider with the supplied key. ``service`` picks the wire
    protocol: ``anthropic`` for the native Messages API, anything else (or empty) for
    OpenAI-compatible chat completions (OpenAI, Groq, Gemini, OpenRouter, DeepSeek…)."""

    provider: Literal["local", "api"] = "local"
    service: str = Field(default="", max_length=40)
    api_key: str = Field(default="", max_length=400)
    model: str = Field(default="", max_length=200)
    base_url: str = Field(default="", max_length=400)
