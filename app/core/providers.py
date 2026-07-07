from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LlmProviderPreset:
    """Catalog entry for a selectable LLM provider.

    ``key`` is the stable identifier stored in ``.env`` (LLM_PROVIDER) and shown in
    the launcher menu. ``base_url`` is empty for the local Ollama provider (its URL
    lives in OLLAMA_BASE_URL). ``is_native_anthropic`` selects the native Messages
    API client instead of the OpenAI-compatible one.
    """

    key: str
    label: str
    base_url: str
    default_model: str
    is_native_anthropic: bool


# Canonical LLM provider catalog. The launcher lists these in order; startup wires
# the matching client. Keep in sync with the $LlmProviders table in
# scripts/backend.ps1 and the LLM_PROVIDER_* arrays in scripts/backend.sh.
LLM_PROVIDERS: dict[str, LlmProviderPreset] = {
    "ollama": LlmProviderPreset(
        key="ollama",
        label="Ollama (локально)",
        base_url="",
        default_model="qwen3:8b",
        is_native_anthropic=False,
    ),
    "openai": LlmProviderPreset(
        key="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.4-nano",
        is_native_anthropic=False,
    ),
    "anthropic": LlmProviderPreset(
        key="anthropic",
        label="Anthropic Claude",
        base_url="https://api.anthropic.com",
        default_model="claude-haiku-4-5",
        is_native_anthropic=True,
    ),
    "deepseek": LlmProviderPreset(
        key="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-v4-flash",
        is_native_anthropic=False,
    ),
    "zai": LlmProviderPreset(
        key="zai",
        label="ZAI (GLM)",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4.6",
        is_native_anthropic=False,
    ),
    "minimax": LlmProviderPreset(
        key="minimax",
        label="MiniMax",
        base_url="https://api.minimax.io/v1",
        default_model="MiniMax-M2.7-highspeed",
        is_native_anthropic=False,
    ),
    "gemini": LlmProviderPreset(
        key="gemini",
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-3.1-flash-lite",
        is_native_anthropic=False,
    ),
}
