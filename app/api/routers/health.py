from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.errors import ErrorResponse
from app.api.schemas.health import ComponentStatus, HealthResponse, ReadinessResponse
from app.core.config import Settings, get_settings
from app.core.dependencies import get_app_state
from app.core.state import AppState

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns a lightweight process health response.",
)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description=(
        "Returns component readiness for SQLite, Whisper, Natasha, LLM, API, "
        "and WebSocket."
    ),
    responses={500: {"model": ErrorResponse}},
)
async def ready(state: AppState = Depends(get_app_state)) -> ReadinessResponse:
    components = {
        "whisper": state.readiness.whisper,
        "natasha": state.readiness.natasha,
        "cache": state.readiness.sqlite,
        "llm": state.readiness.llm,
        "api": state.readiness.api,
        "websocket": state.readiness.websocket,
    }
    return ReadinessResponse(
        ready=state.readiness.ready,
        components={
            name: ComponentStatus(ready=item.ready, required=item.required, details=item.details)
            for name, item in components.items()
        },
    )
