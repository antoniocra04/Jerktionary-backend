from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.services.answer_service import AnswerService
from app.services.chat_service import ChatService
from app.services.explanation_service import ExplanationService
from app.services.term_extractor_service import TermExtractorService
from app.services.transcript_service import TranscriptService


@dataclass(slots=True)
class ServiceStatus:
    ready: bool
    required: bool = True
    details: str = ""


@dataclass(slots=True)
class Readiness:
    config: ServiceStatus
    sqlite: ServiceStatus
    whisper: ServiceStatus
    natasha: ServiceStatus
    llm: ServiceStatus
    api: ServiceStatus
    websocket: ServiceStatus

    def items(self) -> dict[str, ServiceStatus]:
        return {
            "Config loaded": self.config,
            "SQLite ready": self.sqlite,
            "Whisper loaded": self.whisper,
            "Natasha loaded": self.natasha,
            "LLM ready": self.llm,
            "API ready": self.api,
            "WebSocket ready": self.websocket,
        }

    @property
    def ready(self) -> bool:
        return all(item.ready for item in self.items().values() if item.required)


@dataclass(slots=True)
class AppState:
    readiness: Readiness
    settings: Settings
    transcript_service: TranscriptService
    term_extractor_service: TermExtractorService
    explanation_service: ExplanationService
    answer_service: AnswerService
    chat_service: ChatService
    resources: list[Any]

