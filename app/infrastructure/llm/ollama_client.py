from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger
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


class OllamaLlmProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self._settings.ollama_base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                names = {item.get("name") or item.get("model") for item in models}
                return self._settings.ollama_model in names
        except Exception:
            return False

    async def warmup(self) -> None:
        """Load the model into (V)RAM ahead of the first real request, so the first
        answer of a session doesn't pay multi-second model-load latency."""
        payload = self._build_payload("ok", stream=False)
        payload["options"]["num_predict"] = 1
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._settings.ollama_base_url}/api/generate", json=payload
            )
            response.raise_for_status()

    async def explain(self, term: str, context: str) -> Explanation:
        raw = await self._generate(term=term, context=context)
        try:
            parsed = self._parse_response(raw)
            return Explanation(
                title=parsed.title,
                short=parsed.short,
                example=parsed.example,
                why_important=parsed.why_important,
                source="local_llm",
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmResponseError("Local LLM returned invalid JSON") from exc

    async def explain_stream(self, term: str, context: str) -> AsyncIterator[dict[str, str]]:
        """Stream the explanation as it is generated, yielding progressively more
        complete ``{title, short, example, why_important}`` snapshots. The final
        yielded snapshot is validated against the strict schema."""
        prompt = build_explain_prompt(
            term=term, context=context, context_chars=self._settings.llm_context_chars
        )
        payload = self._build_payload(prompt, stream=True)

        accumulated = ""
        last_fields: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            async with client.stream(
                "POST", f"{self._settings.ollama_base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response")
                    if token:
                        accumulated += token
                        fields = extract_partial_fields(accumulated, EXPLAIN_KEYS)
                        if fields and fields != last_fields:
                            last_fields = fields
                            yield fields
                    if chunk.get("done"):
                        self._log_timings(chunk)

        try:
            parsed = self._parse_response(accumulated)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmResponseError("Local LLM returned invalid JSON") from exc
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
        """Stream a spoken-style answer to a whole question, yielding progressively
        complete ``{answer, points, example}`` snapshots (final one is validated)."""
        prompt = build_answer_prompt(
            question=question, context=context, deep=deep,
            context_chars=self._settings.llm_context_chars,
            profile=profile, meeting_context=meeting_context,
        )
        payload = self._build_payload(prompt, stream=True)

        accumulated = ""
        last_fields: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            async with client.stream(
                "POST", f"{self._settings.ollama_base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response")
                    if token:
                        accumulated += token
                        fields = extract_partial_fields(accumulated, ANSWER_KEYS)
                        if fields and fields != last_fields:
                            last_fields = fields
                            yield fields
                    if chunk.get("done"):
                        self._log_timings(chunk)

        try:
            parsed = LlmAnswerResponse.model_validate_json(accumulated)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmResponseError("Local LLM returned invalid JSON") from exc
        final = {"answer": parsed.answer, "points": parsed.points, "example": parsed.example}
        if final != last_fields:
            yield final

    async def _generate(self, *, term: str, context: str) -> str:
        prompt = build_explain_prompt(
            term=term, context=context, context_chars=self._settings.llm_context_chars
        )
        payload = self._build_payload(prompt, stream=False)

        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{self._settings.ollama_base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            self._log_timings(data)
            return str(data["response"])

    def _build_payload(self, prompt: str, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "prompt": prompt,
            "stream": stream,
            "format": "json",
            "keep_alive": self._settings.ollama_keep_alive,
            "think": self._settings.ollama_think,
            "options": {
                "temperature": self._settings.llm_temperature,
                "num_predict": self._settings.llm_max_tokens,
                "num_ctx": self._settings.ollama_num_ctx,
                "num_batch": self._settings.ollama_num_batch,
                "num_gpu": self._settings.ollama_num_gpu,
                "main_gpu": self._settings.ollama_main_gpu,
            },
        }
        if self._settings.ollama_num_thread is not None:
            payload["options"]["num_thread"] = self._settings.ollama_num_thread
        return payload

    @staticmethod
    def _parse_response(raw: str) -> LlmJsonResponse:
        return LlmJsonResponse.model_validate_json(raw)

    @staticmethod
    def _duration_ms(value: object) -> float:
        if not isinstance(value, int | float):
            return 0.0
        return float(value) / 1_000_000

    def _log_timings(self, data: dict[str, Any]) -> None:
        total_ms = self._duration_ms(data.get("total_duration"))
        load_ms = self._duration_ms(data.get("load_duration"))
        prompt_ms = self._duration_ms(data.get("prompt_eval_duration"))
        eval_ms = self._duration_ms(data.get("eval_duration"))
        eval_count = data.get("eval_count") or 0
        prompt_count = data.get("prompt_eval_count") or 0
        tokens_per_second = (float(eval_count) / eval_ms * 1000) if eval_ms else 0.0
        logger.info(
            "Ollama timing total={:.0f}ms load={:.0f}ms prompt={:.0f}ms "
            "eval={:.0f}ms prompt_tokens={} output_tokens={} tok/s={:.1f}",
            total_ms,
            load_ms,
            prompt_ms,
            eval_ms,
            prompt_count,
            eval_count,
            tokens_per_second,
        )

