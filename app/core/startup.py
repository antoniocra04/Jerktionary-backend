from __future__ import annotations

import asyncio

from loguru import logger

from app.core.config import Settings
from app.core.errors import StartupError
from app.core.state import AppState, Readiness, ServiceStatus
from app.infrastructure.asr.faster_whisper_asr import FasterWhisperAsrProvider
from app.infrastructure.db.repositories.explanation_repository import SQLiteExplanationRepository
from app.infrastructure.db.sqlite import SQLiteDatabase
from app.infrastructure.llm.ollama_client import OllamaLlmProvider
from app.infrastructure.nlp.natasha_extractor import NatashaTermExtractor
from app.services.answer_service import AnswerService
from app.services.asr_service import AsrService
from app.services.explanation_service import ExplanationService
from app.services.term_extractor_service import TermExtractorService
from app.services.transcript_service import TranscriptService


async def create_app_state(settings: Settings) -> AppState:
    resources: list[object] = []
    logger.info("Starting backend")

    sqlite = SQLiteDatabase(settings.sqlite_path)
    await sqlite.connect()
    await sqlite.create_schema()
    resources.append(sqlite)
    repository = SQLiteExplanationRepository(sqlite)

    # Local Whisper is optional: API-only users skip the multi-GB model load entirely.
    asr_service: AsrService | None = None
    whisper_status = ServiceStatus(
        False, required=False, details="disabled (WHISPER_ENABLED=false)"
    )
    if settings.whisper_enabled:
        try:
            asr_provider = FasterWhisperAsrProvider(settings)
            await asr_provider.load()
        except Exception as exc:
            logger.exception("Whisper startup failed")
            await _close_resources(resources)
            raise StartupError(f"Whisper failed to load: {exc}") from exc
        asr_service = AsrService(asr_provider)
        whisper_status = ServiceStatus(True, details=settings.whisper_model)

    try:
        nlp_extractor = NatashaTermExtractor(settings)
        await nlp_extractor.load()
    except Exception as exc:
        logger.exception("Natasha startup failed")
        await _close_resources(resources)
        raise StartupError(f"Natasha failed to load: {exc}") from exc

    llm_provider = OllamaLlmProvider(settings)
    llm_ready = False
    llm_details = "disabled"
    if settings.llm_enabled:
        llm_ready = await llm_provider.healthcheck()
        llm_details = f"model={settings.ollama_model}" if llm_ready else "Ollama unavailable"
    if llm_ready:
        _spawn_llm_warmup(llm_provider)

    term_service = TermExtractorService(nlp_extractor)
    transcript_service = TranscriptService(asr_service, term_service, settings)
    explanation_service = ExplanationService(repository, llm_provider, llm_enabled=llm_ready)
    answer_service = AnswerService(llm_provider, llm_enabled=llm_ready)

    readiness = Readiness(
        config=ServiceStatus(True, details="env loaded"),
        sqlite=ServiceStatus(True, details=str(settings.sqlite_path)),
        whisper=whisper_status,
        natasha=ServiceStatus(True, details="pipeline loaded"),
        llm=ServiceStatus(llm_ready, required=False, details=llm_details),
        api=ServiceStatus(True),
        websocket=ServiceStatus(True),
    )
    logger.info("Backend startup complete")
    return AppState(
        readiness=readiness,
        settings=settings,
        transcript_service=transcript_service,
        term_extractor_service=term_service,
        explanation_service=explanation_service,
        answer_service=answer_service,
        resources=resources,
    )


# Strong refs so fire-and-forget warmup tasks aren't garbage-collected mid-run.
_background_tasks: set[asyncio.Task[None]] = set()


def _spawn_llm_warmup(provider: OllamaLlmProvider) -> None:
    """Warm the model into (V)RAM in the background: without this, the first answer
    of a session pays the multi-second model-load latency."""

    async def _warmup() -> None:
        try:
            await provider.warmup()
            logger.info("Ollama model warmed up")
        except Exception as exc:
            logger.warning("Ollama warmup failed: {}", exc)

    task = asyncio.create_task(_warmup())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def close_app_state(state: AppState) -> None:
    await _close_resources(state.resources)
    logger.info("Backend stopped")


async def _close_resources(resources: list[object]) -> None:
    for resource in reversed(resources):
        close = getattr(resource, "close", None)
        if close is not None:
            await close()

