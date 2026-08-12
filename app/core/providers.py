from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LlmProviderPreset:
    """Catalog entry for a selectable LLM provider.

    ``key`` is the stable identifier stored in ``.env`` (LLM_PROVIDER) and shown in
    the launcher menu. ``base_url`` is empty for the local Ollama provider (its URL
    lives in OLLAMA_BASE_URL). ``is_native_anthropic`` selects the native Messages
    API client instead of the OpenAI-compatible one.

    ``reasoning_effort`` is sent verbatim as the OpenAI-compatible request field of
    the same name; empty means "don't send it" (the correct choice for providers
    that would reject an unknown field). Set it for endpoints whose reasoning
    models otherwise stream chain-of-thought into ``message.content`` — see the
    makora entry below. It applies to the structured explain/answer calls, where
    thinking is never wanted; /api/chat lets the caller override it per request.

    ``reasoning_levels`` lists the efforts /api/chat may be asked for, in rising
    order; empty means the provider has no reasoning control and the client hides
    it. Only claim levels a provider is known to honour — an unknown field fails
    the whole request on strict endpoints.

    There is deliberately no image capability flag: vision is a property of the
    model, not the endpoint, and every provider here can route to both kinds. The
    client always allows attachments and a provider that can't accept them says so.
    """

    key: str
    label: str
    base_url: str
    default_model: str
    is_native_anthropic: bool
    reasoning_effort: str = ""
    reasoning_levels: tuple[str, ...] = ()


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
        # Ollama exposes thinking as a boolean `think`; "none" maps to off and any
        # other level to on, so only the two ends are offered.
        reasoning_levels=("none", "high"),
    ),
    "openai": LlmProviderPreset(
        key="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.4-nano",
        is_native_anthropic=False,
        reasoning_levels=("minimal", "low", "medium", "high"),
    ),
    "anthropic": LlmProviderPreset(
        key="anthropic",
        label="Anthropic Claude",
        base_url="https://api.anthropic.com",
        default_model="claude-haiku-4-5",
        is_native_anthropic=True,
        # Mapped to a `thinking` token budget rather than a string field.
        reasoning_levels=("none", "low", "medium", "high"),
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
        # The OpenAI-compatibility layer accepts reasoning_effort for 2.5+ models.
        reasoning_levels=("none", "low", "medium", "high"),
    ),
    "makora": LlmProviderPreset(
        key="makora",
        label="Makora",
        base_url="https://inference.makora.com/v1",
        default_model="deepseek-ai/DeepSeek-V4-Flash",
        is_native_anthropic=False,
        # Without this the endpoint defaults to full reasoning and writes the raw
        # chain-of-thought into message.content (no <think> markers, no separate
        # reasoning field), which burns the whole max_tokens budget before the JSON
        # object starts — finish_reason=length and nothing parseable. "none" keeps
        # content pure JSON and cuts the completion from 2048 tokens to ~230.
        # That default is right for explain/answer; /api/chat can ask for more,
        # since free-form chat has no JSON to corrupt.
        reasoning_effort="none",
        reasoning_levels=("none", "low", "medium", "high"),
    ),
}
