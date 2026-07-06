from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas.errors import ErrorResponse
from app.api.schemas.terms import ExplainTermRequest, ExplainTermResponse
from app.core.dependencies import get_app_state
from app.core.errors import LlmResponseError, LlmUnavailableError
from app.core.state import AppState
from app.services.llm_selector import select_llm_provider

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
    provider = select_llm_provider(
        state.settings,
        provider=request.llm.provider,
        api_key=request.llm.api_key,
        model=request.llm.model,
        base_url=request.llm.base_url,
        service=request.llm.service,
    )
    explanation = await state.explanation_service.explain(
        term=request.term,
        normalized_term=normalized,
        context=request.context,
        provider=provider,
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
            provider = select_llm_provider(
                state.settings,
                provider=request.llm.provider,
                api_key=request.llm.api_key,
                model=request.llm.model,
                base_url=request.llm.base_url,
                service=request.llm.service,
            )
            async for snapshot in state.explanation_service.explain_stream(
                term=request.term,
                normalized_term=normalized,
                context=request.context,
                provider=provider,
            ):
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
        except LlmUnavailableError:
            yield f"data: {json.dumps({'error': 'LLM_UNAVAILABLE'})}\n\n"
        except LlmResponseError:
            yield f"data: {json.dumps({'error': 'LLM_BAD_RESPONSE'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
