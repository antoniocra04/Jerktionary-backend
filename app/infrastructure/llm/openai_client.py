from __future__ import annotations

import json
from collections.abc import AsyncIterator

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


class OpenAiLlmProvider:
    """OpenAI-compatible chat-completions provider used when the user picks the
    "API key" mode in settings. Sends the same JSON prompts as the local model, so
    the streaming/parsing pipeline is identical."""

    def __init__(self, settings: Settings, *, api_key: str, model: str, base_url: str) -> None:
        self._settings = settings
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

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

    def _payload(self, prompt: str, *, stream: bool) -> dict[str, object]:
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
        }

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _complete(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=self._payload(prompt, stream=False),
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()
                raw = str(data["choices"][0]["message"]["content"])
                try:
                    json.loads(raw)
                except json.JSONDecodeError:
                    stripped = raw.strip()
                    if stripped.startswith("```"):
                        newline = stripped.find("\n")
                        if newline != -1:
                            stripped = stripped[newline + 1 :]
                    if stripped.endswith("```"):
                        stripped = stripped[: -3]
                    return stripped.strip()
                return raw
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LlmResponseError(f"External LLM request failed: {exc}") from exc

    async def _iter_content(self, prompt: str) -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=self._payload(prompt, stream=True),
                    headers=self._headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        stripped = line.strip()
                        if not stripped.startswith("data:"):
                            continue
                        data = stripped[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        content = (choices[0].get("delta") or {}).get("content")
                        if content:
                            yield content
        except httpx.HTTPError as exc:
            raise LlmResponseError(f"External LLM request failed: {exc}") from exc
