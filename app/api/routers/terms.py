from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.schemas.errors import ErrorResponse
from app.api.schemas.terms import ExplainTermRequest, ExplainTermResponse
from app.core.dependencies import get_app_state
from app.core.errors import LlmResponseError, LlmUnavailableError
from app.core.state import AppState

router = APIRouter(prefix="/api/terms", tags=["terms"])


@router.post(
    "/explain",
    response_model=ExplainTermResponse,
    summary="Explain a term",
    description=(
        "Returns a short local explanation for a term. The backend uses SQLite cache first "
        "and falls back to the configured Ollama model when needed."
    ),
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def explain_term(
    request: ExplainTermRequest,
    state: AppState = Depends(get_app_state),
) -> ExplainTermResponse:
    normalized = state.term_extractor_service.normalize(request.term)
    # The LLM provider is chosen at backend startup (LLM_PROVIDER in .env); the
    # request no longer carries provider/key — pass None so the service uses the
    # startup-mounted provider.
    explanation = await state.explanation_service.explain(
        term=request.term,
        normalized_term=normalized,
        context=request.context,
        provider=None,
    )
    return ExplainTermResponse(
        title=explanation.title,
        short=explanation.short,
        example=explanation.example,
        why_important=explanation.why_important,
        source=explanation.source,
    )


@router.post(
    "/explain/stream",
    summary="Explain a term (streamed)",
    description=(
        "Server-Sent Events stream of progressively more complete explanation "
        "snapshots. Each `data:` line is a JSON object with title/short/example/"
        "why_important/source and a `done` flag; errors arrive as `{\"error\": code}`."
    ),
)
async def explain_term_stream(
    request: ExplainTermRequest,
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    normalized = state.term_extractor_service.normalize(request.term)

    async def event_source() -> AsyncIterator[str]:
        try:
            async for snapshot in state.explanation_service.explain_stream(
                term=request.term,
                normalized_term=normalized,
                context=request.context,
                provider=None,
            ):
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
        # See answer.py: the 200 is already on the wire, so core.errors never sees
        # this — without a log line the upstream cause is lost entirely.
        except LlmUnavailableError as exc:
            logger.warning("explain stream: LLM_UNAVAILABLE: {}", exc.message)
            yield f"data: {json.dumps({'error': 'LLM_UNAVAILABLE'})}\n\n"
        except LlmResponseError as exc:
            logger.warning("explain stream: LLM_BAD_RESPONSE: {}", exc.message)
            yield f"data: {json.dumps({'error': 'LLM_BAD_RESPONSE'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
