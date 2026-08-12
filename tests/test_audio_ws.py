from __future__ import annotations

from typing import cast

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.state import AppState, Readiness, ServiceStatus
from app.domain.entities.transcript import TranscriptUpdate
from app.main import create_app
from app.services.answer_service import AnswerService
from app.services.chat_service import ChatService
from app.services.explanation_service import ExplanationService
from app.services.term_extractor_service import TermExtractorService
from app.services.transcript_service import TranscriptService


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def process_audio(self, chunk: bytes) -> TranscriptUpdate | None:
        assert chunk == b"pcm"
        return TranscriptUpdate(text="hello", is_final=False, terms=[])

    async def close(self) -> None:
        self.closed = True


class FakeTranscriptService:
    def __init__(self) -> None:
        self.session = FakeSession()

    async def create_session(self) -> FakeSession:
        return self.session


def test_audio_websocket_uses_app_state_dependency() -> None:
    transcript_service = FakeTranscriptService()
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
        transcript_service=cast(TranscriptService, transcript_service),
        term_extractor_service=cast(TermExtractorService, None),
        explanation_service=cast(ExplanationService, None),
        answer_service=cast(AnswerService, None),
        chat_service=cast(ChatService, None),
        resources=[],
    )
    app = create_app(test_state=state)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            websocket.send_bytes(b"pcm")
            event = websocket.receive_json()

    assert event == {
        "type": "transcript_update",
        "text": "hello",
        "is_final": False,
        "terms": [],
    }
    assert transcript_service.session.closed is True
