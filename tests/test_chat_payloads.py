from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.core.config import Settings
from app.domain.entities.chat import ChatImage, ChatMessage
from app.infrastructure.llm.anthropic_client import AnthropicLlmProvider
from app.infrastructure.llm.openai_client import OpenAiLlmProvider

PNG = ChatImage(media_type="image/png", data="aGVsbG8=")


def _openai(**kwargs: str) -> OpenAiLlmProvider:
    return OpenAiLlmProvider(
        Settings(_env_file=None),
        api_key="k",
        model="m",
        base_url="https://example.test/v1",
        **kwargs,
    )


# --- OpenAI-compatible ---------------------------------------------------------


def test_openai_text_only_message_uses_plain_string_content() -> None:
    # Several OpenAI-compatible endpoints reject a parts array that holds nothing
    # but text, so a message without images must stay a plain string.
    wire = OpenAiLlmProvider._chat_messages([ChatMessage("user", "привет")], system="")
    assert wire == [{"role": "user", "content": "привет"}]


def test_openai_image_message_becomes_parts_with_data_uri() -> None:
    wire = OpenAiLlmProvider._chat_messages(
        [ChatMessage("user", "что тут?", (PNG,))], system=""
    )
    assert wire[0]["content"] == [
        {"type": "text", "text": "что тут?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ]


def test_openai_system_prompt_is_prepended_as_a_message() -> None:
    wire = OpenAiLlmProvider._chat_messages([ChatMessage("user", "hi")], system="be brief")
    assert wire[0] == {"role": "system", "content": "be brief"}
    assert len(wire) == 2


def test_openai_image_only_message_omits_the_text_part() -> None:
    wire = OpenAiLlmProvider._chat_messages([ChatMessage("user", "", (PNG,))], system="")
    assert wire[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    ]


# --- Anthropic -----------------------------------------------------------------


def test_anthropic_image_uses_base64_source_before_the_text() -> None:
    # Anthropic's own guidance: a question placed after its images is answered
    # noticeably better, so the order here is deliberate.
    wire = AnthropicLlmProvider._chat_messages([ChatMessage("user", "что тут?", (PNG,))])
    assert wire[0]["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
        },
        {"type": "text", "text": "что тут?"},
    ]


def test_anthropic_thinking_budget_scales_with_the_answer_budget() -> None:
    budget = AnthropicLlmProvider._thinking_budget("high", max_tokens=4096)
    assert budget == 3072
    assert AnthropicLlmProvider._thinking_budget("low", max_tokens=4096) == 1024


def test_anthropic_thinking_disabled_for_none_and_unknown_levels() -> None:
    for level in ("", "none", "minimal", "nonsense"):
        assert AnthropicLlmProvider._thinking_budget(level, max_tokens=4096) == 0


def test_anthropic_thinking_off_when_the_budget_would_be_rejected() -> None:
    # The API rejects budget_tokens below 1024; sending one fails the request.
    assert AnthropicLlmProvider._thinking_budget("low", max_tokens=1000) == 0


# --- reasoning_effort plumbing --------------------------------------------------


async def _captured_chat_payload(
    provider: OpenAiLlmProvider, **kwargs: str
) -> dict[str, object]:
    """Runs chat_stream with the HTTP layer swapped out, returning the payload it
    would have posted."""
    captured: dict[str, object] = {}

    async def fake_iter(payload: dict[str, object]) -> AsyncIterator[str]:
        captured.update(payload)
        yield "ok"

    provider._iter_chat_content = fake_iter  # type: ignore[method-assign]
    async for _ in provider.chat_stream([ChatMessage("user", "hi")], **kwargs):
        pass
    return captured


@pytest.mark.asyncio
async def test_chat_does_not_inherit_the_catalog_reasoning_suppression() -> None:
    # makora ships reasoning_effort="none" so explain/answer keep returning JSON.
    # Chat has no JSON to protect, and inheriting that default would silently pin
    # the new control to "off", so it must not leak in.
    payload = await _captured_chat_payload(_openai(reasoning_effort="none"))
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
async def test_chat_sends_the_requested_reasoning_effort() -> None:
    payload = await _captured_chat_payload(_openai(), reasoning_effort="high")
    assert payload["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_chat_model_override_beats_the_startup_model() -> None:
    assert (await _captured_chat_payload(_openai()))["model"] == "m"
    assert (await _captured_chat_payload(_openai(), model="other"))["model"] == "other"


@pytest.mark.asyncio
async def test_chat_uses_the_chat_token_budget_not_the_json_one() -> None:
    settings = Settings(_env_file=None)
    payload = await _captured_chat_payload(_openai())
    assert payload["max_tokens"] == settings.chat_max_tokens
    assert settings.chat_max_tokens > settings.llm_max_tokens
