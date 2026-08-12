from __future__ import annotations

from app.core.config import Settings
from app.core.providers import LLM_PROVIDERS


def test_catalog_has_expected_providers() -> None:
    # Keys the launcher menus offer (mirrored in backend.ps1 $LlmProviders and
    # backend.sh LLM_PROVIDER_KEYS). Adding one without updating the launchers
    # would make it unselectable, so this guards the contract.
    assert set(LLM_PROVIDERS) == {
        "ollama",
        "openai",
        "anthropic",
        "deepseek",
        "zai",
        "minimax",
        "gemini",
        "makora",
    }


def test_only_ollama_has_empty_base_url() -> None:
    # The local provider delegates to OLLAMA_BASE_URL; every hosted provider must
    # ship a real endpoint so the startup path doesn't need a separate lookup.
    for key, preset in LLM_PROVIDERS.items():
        if key == "ollama":
            assert preset.base_url == ""
        else:
            assert preset.base_url.startswith("https://"), key
            assert preset.default_model, key


def test_anthropic_is_the_only_native_client() -> None:
    natives = [k for k, p in LLM_PROVIDERS.items() if p.is_native_anthropic]
    assert natives == ["anthropic"]


def test_default_settings_use_local_providers() -> None:
    # Out of the box (no .env / fresh checkout) the backend defaults to local
    # Ollama + local Whisper — the interactive launcher then rewrites .env.
    # Construct with _env_file=None so the test checks the class defaults, not
    # whatever happens to be in the developer's local .env right now.
    s = Settings(_env_file=None)
    assert s.llm_provider == "ollama"
    assert s.llm_api_key == ""
    assert s.whisper_provider == "local"
    assert s.whisper_api_key == ""


def test_listed_models_start_with_the_default() -> None:
    # The picker shows this order; the model the backend actually starts with
    # belongs at the top.
    for key, preset in LLM_PROVIDERS.items():
        if preset.models:
            assert preset.models[0] == preset.default_model, key


def test_makora_offers_a_multimodal_model() -> None:
    # Its default is text-only, so without a vision model in the list there is no
    # way to send an image without typing an exact id by hand.
    assert "moonshotai/Kimi-K3" in LLM_PROVIDERS["makora"].models


def test_reasoning_levels_are_a_subset_of_the_shared_vocabulary() -> None:
    # The client renders a label per level; an unmapped one would show raw.
    known = {"none", "minimal", "low", "medium", "high", "max", "enabled"}
    for key, preset in LLM_PROVIDERS.items():
        assert set(preset.reasoning_levels) <= known, key


def test_makora_does_not_claim_medium() -> None:
    # No makora model lists "medium"; offering it guarantees a 400.
    assert "medium" not in LLM_PROVIDERS["makora"].reasoning_levels
