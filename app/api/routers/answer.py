from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.schemas.answer import AnswerRequest
from app.core.dependencies import get_app_state
from app.core.errors import LlmResponseError, LlmUnavailableError
from app.core.state import AppState

router = APIRouter(prefix="/api/answer", tags=["answer"])


@router.post(
    "/stream",
    summary="Answer a spoken question (streamed)",
    description=(
        "Server-Sent Events stream of a spoken-style answer to a whole question. Each "
        "`data:` line is a JSON object with answer/points/example and a `done` flag; "
        'errors arrive as `{"error": code}`.'
    ),
)
async def answer_stream(
    request: AnswerRequest,
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    async def event_source() -> AsyncIterator[str]:
        try:
            async for snapshot in state.answer_service.answer_stream(
                question=request.question,
                context=request.context,
                deep=request.deep,
                profile=request.profile,
                meeting_context=request.meeting_context,
                provider=None,
            ):
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
        # SSE headers (200) are already flushed by the time the generator runs, so a
        # failure here can't change the status code and never reaches the handlers in
        # core.errors — log it or the upstream cause (auth, rate limit, bad JSON) is
        # invisible and the access log shows a misleading 200.
        except LlmUnavailableError as exc:
            logger.warning("answer stream: LLM_UNAVAILABLE: {}", exc.message)
            yield f"data: {json.dumps({'error': 'LLM_UNAVAILABLE'})}\n\n"
        except LlmResponseError as exc:
            logger.warning("answer stream: LLM_BAD_RESPONSE: {}", exc.message)
            yield f"data: {json.dumps({'error': 'LLM_BAD_RESPONSE'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
