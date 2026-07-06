from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.state import AppState, Readiness, ServiceStatus
from app.main import create_app
from app.services.answer_service import AnswerService
from app.services.explanation_service import ExplanationService
from app.services.term_extractor_service import TermExtractorService
from app.services.transcript_service import TranscriptService


def test_health_and_ready() -> None:
    state = AppState(
        readiness=Readiness(
            config=ServiceStatus(True),
            sqlite=ServiceStatus(True),
            whisper=ServiceStatus(True),
            natasha=ServiceStatus(True),
            llm=ServiceStatus(False, required=False, details="disabled"),
            api=ServiceStatus(True),
            websocket=ServiceStatus(True),
        ),
        settings=Settings(),
        transcript_service=cast(TranscriptService, SimpleNamespace()),
        term_extractor_service=cast(TermExtractorService, SimpleNamespace()),
        explanation_service=cast(ExplanationService, SimpleNamespace()),
        answer_service=cast(AnswerService, SimpleNamespace()),
        resources=[],
    )
    app = create_app(test_state=state)

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["components"]["llm"]["ready"] is False
    assert ready.json()["components"]["llm"]["required"] is False


def test_openapi_contains_documented_endpoints() -> None:
    state = AppState(
        readiness=Readiness(
            config=ServiceStatus(True),
            sqlite=ServiceStatus(True),
            whisper=ServiceStatus(True),
            natasha=ServiceStatus(True),
            llm=ServiceStatus(False, required=False, details="disabled"),
            api=ServiceStatus(True),
            websocket=ServiceStatus(True),
        ),
        settings=Settings(),
        transcript_service=cast(TranscriptService, SimpleNamespace()),
        term_extractor_service=cast(TermExtractorService, SimpleNamespace()),
        explanation_service=cast(ExplanationService, SimpleNamespace()),
        answer_service=cast(AnswerService, SimpleNamespace()),
        resources=[],
    )
    app = create_app(test_state=state)

    with TestClient(app) as client:
        openapi = client.get("/openapi.json")
        swagger = client.get("/swagger", follow_redirects=False)
        websocket_docs = client.get("/api/docs/websocket/audio")

    assert openapi.status_code == 200
    assert swagger.status_code in {307, 308}
    assert swagger.headers["location"] == "/docs"
    assert websocket_docs.status_code == 200
    assert websocket_docs.json()["path"] == "/ws/audio"

    paths = openapi.json()["paths"]
    assert "/health" in paths
    assert "/ready" in paths
    assert "/api/terms/explain" in paths
    assert "/api/docs" in paths
    assert "/api/docs/websocket/audio" in paths
