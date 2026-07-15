from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.infrastructure.llm.json_stream import LlmJsonResponse, parse_llm_json, sanitize_llm_json

_PAYLOAD = '{"title": "Стек", "short": "LIFO-структура", "example": "", "why_important": ""}'


def test_plain_json_passes() -> None:
    parsed = parse_llm_json(_PAYLOAD, LlmJsonResponse)
    assert parsed.title == "Стек"


def test_markdown_fenced_json_passes() -> None:
    raw = f"```json\n{_PAYLOAD}\n```"
    assert parse_llm_json(raw, LlmJsonResponse).title == "Стек"


def test_think_block_and_prose_around_json_pass() -> None:
    raw = (
        "<think>Пользователь спросил про стек... надо ответить JSON.</think>\n"
        f"Вот ответ:\n{_PAYLOAD}\nНадеюсь, это поможет!"
    )
    assert parse_llm_json(raw, LlmJsonResponse).short == "LIFO-структура"


def test_unclosed_think_block_is_stripped() -> None:
    # max_tokens can cut the stream inside the reasoning block — everything after
    # <think> is reasoning, so there is no JSON to salvage; must raise, not hang.
    with pytest.raises(ValidationError):
        parse_llm_json("<think>бесконечное размышление без ответа", LlmJsonResponse)


def test_garbage_still_raises() -> None:
    with pytest.raises(ValidationError):
        parse_llm_json("Извините, я не могу ответить в формате JSON.", LlmJsonResponse)


def test_sanitize_keeps_outermost_object() -> None:
    raw = 'ответ: {"a": {"b": 1}} конец'
    assert sanitize_llm_json(raw) == '{"a": {"b": 1}}'
