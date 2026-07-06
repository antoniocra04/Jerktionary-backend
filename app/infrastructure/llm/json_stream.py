from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, field_validator


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
