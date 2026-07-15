from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError, field_validator

TModel = TypeVar("TModel", bound=BaseModel)

# Reasoning models emit <think>…</think> before the answer (tolerate an unclosed
# tag: max_tokens can cut the stream mid-thought).
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)\s*```", re.DOTALL)


def _strip_decorations(raw: str) -> str:
    """Drop non-JSON decoration: ``<think>`` blocks and markdown fences."""
    text = _THINK_RE.sub("", raw).strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    return text


def sanitize_llm_json(raw: str) -> str:
    """Cut the JSON object out of a raw completion.

    Providers that ignore (or don't support) ``response_format=json_object`` wrap
    the object in markdown fences, ``<think>`` blocks, or explanatory prose.
    Returns the outermost ``{...}`` slice; falls back to the cleaned text when no
    braces are present.
    """
    text = _strip_decorations(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_llm_json(raw: str, model_cls: type[TModel]) -> TModel:
    """Parse a completion into ``model_cls``, tolerating real-world LLM output.

    Tries progressively harder: the text as-is; the sanitized ``{...}`` slice;
    a lenient decode that allows literal newlines inside strings (models emit
    those in multi-line fields like ``points`` even though RFC JSON forbids
    them); finally salvages whatever fields already arrived when the response
    was truncated at the token limit — a shortened result beats a 502 mid-
    conversation. Logs the payload before giving up; callers translate the
    raised error into their provider-specific ``LlmResponseError``.
    """
    try:
        return model_cls.model_validate_json(raw)
    except (json.JSONDecodeError, ValidationError):
        pass
    sanitized = sanitize_llm_json(raw)
    try:
        return model_cls.model_validate(json.loads(sanitized, strict=False))
    except (json.JSONDecodeError, ValidationError):
        pass
    stripped = _strip_decorations(raw)
    fields = extract_partial_fields(stripped, tuple(model_cls.model_fields))
    if any(value.strip() for value in fields.values()):
        logger.warning(
            "LLM response is not valid JSON (likely truncated at the token limit); "
            "salvaged fields {} from {} chars",
            sorted(fields),
            len(raw),
        )
        return model_cls.model_validate(dict.fromkeys(model_cls.model_fields, "") | fields)
    logger.warning("Unparseable LLM response ({} chars): {!r}", len(raw), raw[:400])
    return model_cls.model_validate_json(sanitized)


def _coerce_to_str(value: Any) -> str:
    # Models occasionally emit a number/list for a field (e.g. "points": 10);
    # coerce so validation never fails on a type mismatch.
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


class LlmJsonResponse(BaseModel):
    title: str
    short: str
    example: str
    why_important: str

    @field_validator("*", mode="before")
    @classmethod
    def _as_str(cls, value: Any) -> str:
        return _coerce_to_str(value)


class LlmAnswerResponse(BaseModel):
    answer: str
    points: str
    example: str

    @field_validator("*", mode="before")
    @classmethod
    def _as_str(cls, value: Any) -> str:
        return _coerce_to_str(value)


def extract_partial_fields(raw: str, keys: tuple[str, ...]) -> dict[str, str]:
    """Best-effort extraction of string fields from an incomplete JSON object.

    Used while streaming: the model emits ``{"title":"...","short":"...`` token by
    token, so the buffer is not valid JSON yet. For each known key we read the string
    value up to the next unescaped quote (or the end of what arrived so far).
    """
    fields: dict[str, str] = {}
    for key in keys:
        match = re.search(rf'"{key}"\s*:\s*"', raw)
        if match is None:
            continue
        index = match.end()
        chars: list[str] = []
        while index < len(raw):
            char = raw[index]
            if char == "\\" and index + 1 < len(raw):
                chars.append(raw[index : index + 2])
                index += 2
                continue
            if char == '"':
                break
            chars.append(char)
            index += 1
        encoded = "".join(chars)
        for candidate in (encoded, encoded[:-1]):  # tolerate a dangling escape char
            try:
                fields[key] = json.loads(f'"{candidate}"')
                break
            except json.JSONDecodeError:
                continue
        else:
            fields[key] = encoded
    return fields
