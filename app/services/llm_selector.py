from __future__ import annotations

from app.core.config import Settings
from app.core.errors import LlmUnavailableError
from app.domain.interfaces.llm import LlmProvider
from app.infrastructure.llm.anthropic_client import AnthropicLlmProvider
from app.infrastructure.llm.openai_client import OpenAiLlmProvider


def select_llm_provider(
    settings: Settings,
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    service: str = "",
) -> LlmProvider | None:
    """Resolve the per-request LLM provider. Returns ``None`` for the local model
    (callers fall back to the on-box provider). ``service == "anthropic"`` selects the
    native Messages API client; everything else is treated as OpenAI-compatible
    (OpenAI, Groq, Gemini, OpenRouter, DeepSeek, custom base URLs). Raises when API
    mode is chosen without a key."""
    if provider != "api":
        return None

    key = api_key.strip()
    if not key:
        raise LlmUnavailableError("Для режима «API key» нужен ключ")

    if service.strip().lower() == "anthropic":
        return AnthropicLlmProvider(
            settings,
            api_key=key,
            model=model.strip(),
            base_url=base_url.strip(),
        )

    return OpenAiLlmProvider(
        settings,
        api_key=key,
        model=model.strip() or settings.openai_model,
        base_url=base_url.strip() or settings.openai_base_url,
    )
