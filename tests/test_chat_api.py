from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from app.api.schemas.chat import ChatRequest
from app.core.config import Settings
from app.core.errors import LlmUnavailableError
from app.core.state import AppState, Readiness, ServiceStatus
from app.domain.entities.chat import ChatMessage, ModelInfo
from app.main import create_app
from app.services.answer_service import AnswerService
from app.services.chat_service import ChatService
from app.services.explanation_service import ExplanationService
from app.services.term_extractor_service import TermExtractorService
from app.services.transcript_service import TranscriptService

PNG_URI = "data:image/png;base64,aGVsbG8="


class FakeChatProvider:
    def __init__(self, deltas: Sequence[str] = ("при", "вет")) -> None:
        self.deltas = deltas
        self.seen: dict[str, object] = {}

    async def healthcheck(self) -> bool:
        return True

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str = "",
        model: str = "",
        reasoning_effort: str = "",
        max_tokens: int = 0,
    ) -> AsyncIterator[str]:
        self.seen = {
            "messages": list(messages),
            "system": system,
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
        for delta in self.deltas:
            yield delta


def _state(chat_service: ChatService, *, llm_ready: bool = True) -> AppState:
    return AppState(
        readiness=Readiness(
            config=ServiceStatus(True),
            sqlite=ServiceStatus(True),
            whisper=ServiceStatus(True),
            natasha=ServiceStatus(True),
            llm=ServiceStatus(llm_ready, required=False),
            api=ServiceStatus(True),
            websocket=ServiceStatus(True),
        ),
        settings=Settings(_env_file=None),
        transcript_service=cast(TranscriptService, SimpleNamespace()),
        term_extractor_service=cast(TermExtractorService, SimpleNamespace()),
        explanation_service=cast(ExplanationService, SimpleNamespace()),
        answer_service=cast(AnswerService, SimpleNamespace()),
        chat_service=chat_service,
        resources=[],
    )


def _events(body: str) -> list[dict[str, object]]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


# --- schema --------------------------------------------------------------------


def test_request_rejects_a_history_not_ending_with_the_user() -> None:
    with pytest.raises(ValueError):
        ChatRequest(messages=[{"role": "assistant", "content": "hi"}])


def test_request_rejects_an_empty_turn() -> None:
    # A message with neither text nor an image wastes a round trip and confuses
    # providers that require non-empty content.
    with pytest.raises(ValueError):
        ChatRequest(messages=[{"role": "user", "content": "   "}])


def test_request_accepts_an_image_only_turn() -> None:
    request = ChatRequest(
        messages=[{"role": "user", "content": "", "images": [{"data_url": PNG_URI}]}]
    )
    assert request.messages[0].to_entity().images[0].media_type == "image/png"


def test_request_rejects_a_non_data_uri() -> None:
    with pytest.raises(ValueError):
        ChatRequest(
            messages=[
                {"role": "user", "content": "look", "images": [{"data_url": "https://x/y.png"}]}
            ]
        )


def test_request_rejects_an_unsupported_image_type() -> None:
    with pytest.raises(ValueError):
        ChatRequest(
            messages=[
                {
                    "role": "user",
                    "content": "look",
                    "images": [{"data_url": "data:image/tiff;base64,aGVsbG8="}],
                }
            ]
        )


def test_request_rejects_broken_base64() -> None:
    with pytest.raises(ValueError):
        ChatRequest(
            messages=[
                {
                    "role": "user",
                    "content": "look",
                    "images": [{"data_url": "data:image/png;base64,!!!not-base64!!!"}],
                }
            ]
        )


# --- service -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_rejects_an_unsupported_reasoning_level() -> None:
    # Forwarding an effort the provider doesn't know fails the whole upstream
    # request; silently dropping it would make the control look broken instead.
    service = ChatService(
        cast(object, FakeChatProvider()), llm_enabled=True, reasoning_levels=("none", "high")
    )
    with pytest.raises(Exception) as caught:
        async for _ in service.chat_stream(
            messages=[ChatMessage("user", "hi")], reasoning_effort="medium"
        ):
            pass
    assert "medium" in str(caught.value)


@pytest.mark.asyncio
async def test_service_raises_when_the_llm_is_disabled() -> None:
    service = ChatService(cast(object, FakeChatProvider()), llm_enabled=False)
    with pytest.raises(LlmUnavailableError):
        async for _ in service.chat_stream(messages=[ChatMessage("user", "hi")]):
            pass


# --- endpoints -----------------------------------------------------------------


def test_stream_emits_deltas_then_done() -> None:
    provider = FakeChatProvider(("при", "вет"))
    service = ChatService(cast(object, provider), llm_enabled=True, reasoning_levels=("high",))
    with TestClient(create_app(test_state=_state(service))) as client:
        response = client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "здравствуй"}]},
        )
    assert response.status_code == 200
    assert _events(response.text) == [{"delta": "при"}, {"delta": "вет"}, {"done": True}]


def test_stream_forwards_model_and_reasoning_and_images() -> None:
    provider = FakeChatProvider()
    service = ChatService(cast(object, provider), llm_enabled=True, reasoning_levels=("high",))
    with TestClient(create_app(test_state=_state(service))) as client:
        client.post(
            "/api/chat/stream",
            json={
                "messages": [
                    {"role": "user", "content": "что тут?", "images": [{"data_url": PNG_URI}]}
                ],
                "model": "some-model",
                "reasoning_effort": "high",
                "system": "будь краток",
            },
        )
    assert provider.seen["model"] == "some-model"
    assert provider.seen["reasoning_effort"] == "high"
    assert provider.seen["system"] == "будь краток"
    messages = cast(list[ChatMessage], provider.seen["messages"])
    assert messages[0].images[0].data == "aGVsbG8="


def test_stream_reports_an_unavailable_llm_as_an_event_not_a_500() -> None:
    # The 200 and SSE headers are flushed before the generator runs, so failures
    # can only reach the client as an event.
    service = ChatService(cast(object, FakeChatProvider()), llm_enabled=False)
    with TestClient(create_app(test_state=_state(service, llm_ready=False))) as client:
        response = client.post(
            "/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert response.status_code == 200
    assert _events(response.text) == [{"error": "LLM_UNAVAILABLE"}]


def test_capabilities_reports_levels_and_default_model() -> None:
    service = ChatService(
        cast(object, FakeChatProvider()), llm_enabled=True, reasoning_levels=("none", "high")
    )
    with TestClient(create_app(test_state=_state(service))) as client:
        body = client.get("/api/chat/capabilities").json()
    # Default settings select local Ollama, so the model comes from OLLAMA_MODEL.
    assert body["provider"] == "ollama"
    assert body["default_model"] == Settings(_env_file=None).ollama_model
    assert body["reasoning_levels"] == ["none", "high"]
    assert body["ready"] is True


def test_capabilities_empty_levels_mean_no_reasoning_control() -> None:
    service = ChatService(cast(object, FakeChatProvider()), llm_enabled=True)
    with TestClient(create_app(test_state=_state(service))) as client:
        body = client.get("/api/chat/capabilities").json()
    assert body["reasoning_levels"] == []


# --- per-model capabilities ----------------------------------------------------


class ModelAwareProvider(FakeChatProvider):
    """Reports different limits per model, as makora actually does."""

    CATALOG = {
        "text-only": ModelInfo(model="text-only", accepts_images=False,
                               reasoning_levels=("none", "high")),
        "vision": ModelInfo(model="vision", accepts_images=True,
                            reasoning_levels=("low", "max")),
    }

    async def model_info(self, model: str) -> ModelInfo | None:
        return self.CATALOG.get(model)


@pytest.mark.asyncio
async def test_levels_narrow_to_the_selected_model() -> None:
    # The provider-wide union would let through an effort the model rejects.
    service = ChatService(
        cast(object, ModelAwareProvider()),
        llm_enabled=True,
        reasoning_levels=("none", "low", "high", "max"),
    )
    assert await service.levels("vision") == ("low", "max")
    assert await service.levels("text-only") == ("none", "high")
    # Unknown models fall back to the provider-wide set.
    assert await service.levels("mystery") == ("none", "low", "high", "max")


@pytest.mark.asyncio
async def test_effort_valid_for_another_model_is_rejected() -> None:
    service = ChatService(
        cast(object, ModelAwareProvider()),
        llm_enabled=True,
        reasoning_levels=("none", "low", "high", "max"),
    )
    with pytest.raises(Exception) as caught:
        async for _ in service.chat_stream(
            messages=[ChatMessage("user", "hi")], model="vision", reasoning_effort="none"
        ):
            pass
    assert "none" in str(caught.value)


def test_capabilities_describe_the_requested_model() -> None:
    service = ChatService(
        cast(object, ModelAwareProvider()),
        llm_enabled=True,
        reasoning_levels=("none", "low", "high", "max"),
    )
    with TestClient(create_app(test_state=_state(service))) as client:
        vision = client.get("/api/chat/capabilities", params={"model": "vision"}).json()
        text = client.get("/api/chat/capabilities", params={"model": "text-only"}).json()

    assert vision["accepts_images"] is True
    assert vision["reasoning_levels"] == ["low", "max"]
    assert text["accepts_images"] is False
    assert text["model"] == "text-only"


def test_capabilities_leave_images_unknown_when_the_provider_is_silent() -> None:
    # A provider with no catalog must not be reported as refusing images — the
    # client would block attachments that would have worked.
    service = ChatService(cast(object, FakeChatProvider()), llm_enabled=True)
    with TestClient(create_app(test_state=_state(service))) as client:
        body = client.get("/api/chat/capabilities").json()
    assert body["accepts_images"] is None
