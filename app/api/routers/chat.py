from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.schemas.chat import ChatCapabilitiesResponse, ChatRequest
from app.core.dependencies import get_app_state
from app.core.errors import LlmResponseError, LlmUnavailableError
from app.core.providers import LLM_PROVIDERS
from app.core.state import AppState

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get(
    "/capabilities",
    summary="What the active chat provider supports",
    description=(
        "The provider chosen at backend startup, its default model, and the "
        "reasoning efforts it accepts. An empty `reasoning_levels` means the "
        "provider has no reasoning control and the client should hide it."
    ),
)
async def chat_capabilities(
    state: AppState = Depends(get_app_state),
) -> ChatCapabilitiesResponse:
    settings = state.settings
    key = settings.llm_provider.strip().lower() or "ollama"
    preset = LLM_PROVIDERS.get(key)
    if key == "ollama":
        default_model = settings.ollama_model
    else:
        default_model = settings.llm_api_model.strip() or (
            preset.default_model if preset else ""
        )
    return ChatCapabilitiesResponse(
        provider=key,
        label=preset.label if preset else key,
        default_model=default_model,
        reasoning_levels=list(state.chat_service.reasoning_levels),
        ready=state.readiness.llm.ready,
    )


@router.post(
    "/stream",
    summary="Chat with the active provider (streamed)",
    description=(
        "Server-Sent Events stream of a chat reply. Each `data:` line is a JSON "
        'object: `{"delta": "…"}` for text as it arrives, `{"done": true}` at the '
        'end, and `{"error": code}` on failure. The client sends the whole message '
        "history every turn; the backend keeps no conversation state."
    ),
)
async def chat_stream(
    request: ChatRequest,
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    messages = [message.to_entity() for message in request.messages]

    async def event_source() -> AsyncIterator[str]:
        try:
            async for delta in state.chat_service.chat_stream(
                messages=messages,
                system=request.system,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
            ):
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        # The 200 and SSE headers are already flushed once the generator runs, so
        # failures can't change the status code and never reach the handlers in
        # core.errors — without logging here the cause is invisible.
        except LlmUnavailableError as exc:
            logger.warning("chat stream: LLM_UNAVAILABLE: {}", exc.message)
            yield f"data: {json.dumps({'error': 'LLM_UNAVAILABLE'})}\n\n"
        except LlmResponseError as exc:
            logger.warning("chat stream: LLM_BAD_RESPONSE: {}", exc.message)
            yield f"data: {json.dumps({'error': 'LLM_BAD_RESPONSE'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
