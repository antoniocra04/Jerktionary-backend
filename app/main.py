from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routers import answer, audio_ws, docs, health, terms
from app.core.config import Settings
from app.core.console import render_startup_status
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.startup import close_app_state, create_app_state
from app.core.state import AppState


def create_app(test_state: AppState | None = None) -> FastAPI:
    settings = Settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = test_state or await create_app_state(settings)
        app.state.app_state = state
        render_startup_status(state.readiness)
        try:
            yield
        finally:
            await close_app_state(state)

    tags_metadata = [
        {"name": "health", "description": "Health and readiness endpoints."},
        {"name": "terms", "description": "Term explanation endpoints backed by cache and Ollama."},
        {"name": "answer", "description": "Live spoken-answer streaming for detected questions."},
        {"name": "audio", "description": "Realtime audio WebSocket endpoint."},
        {"name": "documentation", "description": "Swagger/OpenAPI and WebSocket contract helpers."},
    ]
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Backend for realtime transcription, term extraction, and local LLM explanations. "
            "Swagger UI is available at /docs; /swagger redirects there."
        ),
        lifespan=lifespan,
        openapi_tags=tags_metadata,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,  # no auth in this app; safe with * origins
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(docs.router)
    app.include_router(health.router)
    app.include_router(terms.router)
    app.include_router(answer.router)
    app.include_router(audio_ws.router)

    @app.get("/swagger", include_in_schema=False)
    async def swagger_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
