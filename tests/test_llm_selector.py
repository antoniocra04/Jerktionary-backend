from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import LlmUnavailableError
from app.infrastructure.llm.openai_client import OpenAiLlmProvider
from app.services.llm_selector import select_llm_provider


def test_local_provider_resolves_to_none() -> None:
    result = select_llm_provider(
        Settings(), provider="local", api_key="", model="", base_url=""
    )
    assert result is None


def test_api_provider_requires_key() -> None:
    with pytest.raises(LlmUnavailableError):
        select_llm_provider(Settings(), provider="api", api_key="   ", model="", base_url="")


def test_api_provider_builds_openai_client_with_defaults() -> None:
    settings = Settings()
    provider = select_llm_provider(
        settings, provider="api", api_key="sk-test", model="", base_url=""
    )
    assert isinstance(provider, OpenAiLlmProvider)
    assert provider._model == settings.openai_model
    assert provider._base_url == settings.openai_base_url.rstrip("/")
