from __future__ import annotations

from app.core.config import Settings
from app.core.providers import LLM_PROVIDERS
from app.infrastructure.llm.openai_client import OpenAiLlmProvider


def _provider(**kwargs: str) -> OpenAiLlmProvider:
    return OpenAiLlmProvider(
        Settings(_env_file=None),
        api_key="k",
        model="m",
        base_url="https://example.test/v1",
        **kwargs,
    )


def test_reasoning_effort_absent_by_default() -> None:
    # Providers that don't know the field reject the whole request, so it must not
    # be sent unless the catalog asks for it.
    payload = _provider()._payload("p", stream=False)
    assert "reasoning_effort" not in payload


def test_reasoning_effort_sent_when_configured() -> None:
    for stream in (False, True):
        payload = _provider(reasoning_effort="none")._payload("p", stream=stream)
        assert payload["reasoning_effort"] == "none"


def test_makora_disables_reasoning() -> None:
    # Makora's default effort writes raw chain-of-thought into message.content and
    # exhausts max_tokens before the JSON object starts, so every explain/answer
    # call ends in LLM_BAD_RESPONSE. Dropping this makes the provider unusable.
    assert LLM_PROVIDERS["makora"].reasoning_effort == "none"


def test_other_providers_leave_reasoning_effort_unset() -> None:
    for key, preset in LLM_PROVIDERS.items():
        if key != "makora":
            assert preset.reasoning_effort == "", key
