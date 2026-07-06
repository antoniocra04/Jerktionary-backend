from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import LlmResponseError
from app.domain.entities.explanation import Explanation
from app.infrastructure.llm.json_stream import (
    LlmAnswerResponse,
    LlmJsonResponse,
    extract_partial_fields,
)
from app.infrastructure.llm.prompts import (
    ANSWER_KEYS,
    EXPLAIN_KEYS,
    build_answer_prompt,
    build_explain_prompt,
)

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicLlmProvider:
    """Native Anthropic Messages API provider. Kept separate from the OpenAI-compatible
    path because Anthropic uses its own request shape, headers, and SSE stream format.
    Same JSON prompts, so the downstream streaming/parsing pipeline is identical."""

    def __init__(self, settings: Settings, *, api_key: str, model: str, base_url: str) -> None:
        self._settings = settings
        self._api_key = api_key
        self._model = model or DEFAULT_MODEL
        base = (base_url or "https://api.anthropic.com").rstrip("/")
        # Presets sometimes carry a trailing /v1 like OpenAI-compatible URLs do.
        if base.endswith("/v1"):
            base = base[: -len("/v1")].rstrip("/")
        self._base_url = base

    async def healthcheck(self) -> bool:
        return bool(self._api_key)

    async def explain(self, term: str, context: str) -> Explanation:
        prompt = build_explain_prompt(
            term=term, context=context, context_chars=self._settings.llm_context_chars
        )
        raw = await self._complete(prompt)
        try:
            parsed = LlmJsonResponse.model_validate_json(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmResponseError("External LLM returned invalid JSON") from exc
        return Explanation(
            title=parsed.title,
            short=parsed.short,
            example=parsed.example,
            why_important=parsed.why_important,
            source="local_llm",
        )

    async def explain_stream(self, term: str, context: str) -> AsyncIterator[dict[str, str]]:
        prompt = build_explain_prompt(
            term=term, context=context, context_chars=self._settings.llm_context_chars
        )
        accumulated = ""
        last_fields: dict[str, str] = {}
        async for delta in self._iter_content(prompt):
            accumulated += delta
            fields = extract_partial_fields(accumulated, EXPLAIN_KEYS)
            if fields and fields != last_fields:
                last_fields = fields
                yield fields

        try:
            parsed = LlmJsonResponse.model_validate_json(accumulated)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmResponseError("External LLM returned invalid JSON") from exc
        final = {
            "title": parsed.title,
            "short": parsed.short,
            "example": parsed.example,
            "why_important": parsed.why_important,
        }
        if final != last_fields:
            yield final

    async def answer_stream(
        self,
        question: str,
        context: str,
        *,
        deep: bool = False,
        profile: str = "",
        meeting_context: str = "",
    ) -> AsyncIterator[dict[str, str]]:
        prompt = build_answer_prompt(
            question=question,
            context=context,
            deep=deep,
            context_chars=self._settings.llm_context_chars,
            profile=profile,
            meeting_context=meeting_context,
        )
        accumulated = ""
        last_fields: dict[str, str] = {}
        async for delta in self._iter_content(prompt):
            accumulated += delta
            fields = extract_partial_fields(accumulated, ANSWER_KEYS)
            if fields and fields != last_fields:
                last_fields = fields
                yield fields

        try:
            parsed = LlmAnswerResponse.model_validate_json(accumulated)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmResponseError("External LLM returned invalid JSON") from exc
        final = {"answer": parsed.answer, "points": parsed.points, "example": parsed.example}
        if final != last_fields:
            yield final

    def _payload(self, prompt: str, *, stream: bool) -> dict[str, Any]:
        # No sampling params on purpose: current Claude models reject non-default
        # temperature/top_p; the JSON-only prompt already constrains the output.
        return {
            "model": self._model,
            "max_tokens": self._settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    async def _complete(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
                response = await client.post(
                    f"{self._base_url}/v1/messages",
                    json=self._payload(prompt, stream=False),
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()
                return "".join(
                    str(block.get("text", ""))
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                )
        except (httpx.HTTPError, KeyError) as exc:
            raise LlmResponseError(f"Anthropic request failed: {exc}") from exc

    async def _iter_content(self, prompt: str) -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/v1/messages",
                    json=self._payload(prompt, stream=True),
                    headers=self._headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        stripped = line.strip()
                        if not stripped.startswith("data:"):
                            continue
                        data = stripped[len("data:") :].strip()
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("type") == "error":
                            error = chunk.get("error") or {}
                            raise LlmResponseError(
                                f"Anthropic stream error: {error.get('message', 'unknown')}"
                            )
                        if chunk.get("type") != "content_block_delta":
                            continue
                        delta = chunk.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield str(delta["text"])
        except httpx.HTTPError as exc:
            raise LlmResponseError(f"Anthropic request failed: {exc}") from exc
