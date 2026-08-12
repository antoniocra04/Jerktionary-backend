from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import LlmResponseError
from app.domain.entities.chat import ChatMessage
from app.domain.entities.explanation import Explanation
from app.infrastructure.llm.json_stream import (
    LlmAnswerResponse,
    LlmJsonResponse,
    extract_partial_fields,
    parse_llm_json,
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

    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str,
        model: str,
        base_url: str,
        reasoning_effort: str = "",
    ) -> None:
        self._settings = settings
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._reasoning_effort = reasoning_effort

    async def healthcheck(self) -> bool:
        return bool(self._api_key)

    async def explain(self, term: str, context: str) -> Explanation:
        prompt = build_explain_prompt(
            term=term, context=context, context_chars=self._settings.llm_context_chars
        )
        raw = await self._complete(prompt)
        try:
            parsed = parse_llm_json(raw, LlmJsonResponse)
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
            parsed = parse_llm_json(accumulated, LlmJsonResponse)
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
            parsed = parse_llm_json(accumulated, LlmAnswerResponse)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmResponseError("External LLM returned invalid JSON") from exc
        final = {"answer": parsed.answer, "points": parsed.points, "example": parsed.example}
        if final != last_fields:
            yield final

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str = "",
        model: str = "",
        reasoning_effort: str = "",
        max_tokens: int = 0,
    ) -> AsyncIterator[str]:
        payload: dict[str, object] = {
            "model": model or self._model,
            "messages": self._chat_messages(messages, system=system),
            "stream": True,
            "temperature": self._settings.llm_temperature,
            "max_tokens": max_tokens or self._settings.chat_max_tokens,
        }
        # Free-form chat has no JSON to protect, so the catalog's suppression
        # default does not apply here: only an explicit per-request effort is sent.
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        async for delta in self._iter_chat_content(payload):
            yield delta

    @staticmethod
    def _chat_messages(
        messages: Sequence[ChatMessage], *, system: str
    ) -> list[dict[str, object]]:
        wire: list[dict[str, object]] = []
        if system:
            wire.append({"role": "system", "content": system})
        for message in messages:
            if not message.images:
                # Plain string content: some OpenAI-compatible endpoints reject the
                # parts array when there is nothing but text in it.
                wire.append({"role": message.role, "content": message.text})
                continue
            parts: list[dict[str, object]] = []
            if message.text:
                parts.append({"type": "text", "text": message.text})
            for image in message.images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.media_type};base64,{image.data}"
                        },
                    }
                )
            wire.append({"role": message.role, "content": parts})
        return wire

    def _payload(self, prompt: str, *, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
        }
        # Only sent when the provider catalog asks for it: endpoints that don't know
        # the field reject the whole request, so an empty value must stay absent.
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        return payload

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
                return str(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise LlmResponseError(f"External LLM request failed: {exc}") from exc

    async def _iter_chat_content(self, payload: dict[str, object]) -> AsyncIterator[str]:
        """Streams a prebuilt chat payload. Separate from `_iter_content` because
        chat runs on its own, much longer timeout: a reasoning model answering a
        real question routinely outlives the 30 s explain/answer budget."""
        try:
            async with httpx.AsyncClient(timeout=self._settings.chat_timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=payload,
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
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield str(content)
                        # Providers that expose thinking as its own field are asked
                        # for it by reasoning_effort but the chat shows answers, not
                        # chain-of-thought, so those deltas are dropped.
        except httpx.HTTPError as exc:
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
