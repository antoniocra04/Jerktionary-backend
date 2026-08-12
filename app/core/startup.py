from __future__ import annotations

import asyncio

from loguru import logger

from app.core.config import Settings
from app.core.errors import StartupError
from app.core.providers import LLM_PROVIDERS
from app.core.state import AppState, Readiness, ServiceStatus
from app.domain.interfaces.llm import LlmProvider
from app.domain.interfaces.nlp import NlpTermExtractor
from app.infrastructure.asr.api_asr_provider import ApiAsrProvider
from app.infrastructure.asr.faster_whisper_asr import FasterWhisperAsrProvider
from app.infrastructure.asr.nemotron_asr import NemotronAsrProvider
from app.infrastructure.asr.yandex_asr import YandexAsrProvider
from app.infrastructure.db.repositories.explanation_repository import SQLiteExplanationRepository
from app.infrastructure.db.sqlite import SQLiteDatabase
from app.infrastructure.llm.anthropic_client import AnthropicLlmProvider
from app.infrastructure.llm.ollama_client import OllamaLlmProvider
from app.infrastructure.llm.openai_client import OpenAiLlmProvider
from app.infrastructure.nlp.natasha_extractor import NatashaTermExtractor
from app.infrastructure.nlp.spacy_extractor import SpacyTermExtractor
from app.services.answer_service import AnswerService
from app.services.asr_service import AsrService
from app.services.chat_service import ChatService
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

    # ASR provider is chosen at backend startup. "local" loads faster-whisper on
    # this box (multi-GB model, optional via WHISPER_ENABLED); "api" mounts the
    # OpenAI-compatible /audio/transcriptions client with key/base_url from .env;
    # "yandex" mounts SpeechKit v3 gRPC streaming (realtime partials/finals).
    asr_service: AsrService | None = None
    whisper_status = ServiceStatus(
        False, required=False, details="disabled (WHISPER_ENABLED=false)"
    )
    if settings.whisper_enabled:
        asr_provider_key = settings.whisper_provider.strip().lower()
        if asr_provider_key == "api":
            asr_service = AsrService(ApiAsrProvider(settings))
            model = settings.whisper_api_model.strip() or "whisper-1"
            whisper_status = ServiceStatus(True, details=f"api ({model})")
        elif asr_provider_key == "yandex":
            try:
                yandex_provider = YandexAsrProvider(settings)
                yandex_provider.load()
            except Exception as exc:
                logger.exception("Yandex SpeechKit startup failed")
                await _close_resources(resources)
                raise StartupError(f"Yandex SpeechKit failed to init: {exc}") from exc
            asr_service = AsrService(yandex_provider)
            whisper_status = ServiceStatus(True, details=f"yandex ({settings.yandex_stt_model})")
        elif settings.whisper_model.strip().lower().startswith("nemotron"):
            # Local NVIDIA streaming model (NeMo) picked in the local-model menu:
            # same "local" provider key, different runtime than faster-whisper.
            try:
                nemotron_provider = NemotronAsrProvider(settings)
                await nemotron_provider.load()
            except Exception as exc:
                logger.exception("Nemotron ASR startup failed")
                await _close_resources(resources)
                raise StartupError(f"Nemotron ASR failed to load: {exc}") from exc
            asr_service = AsrService(nemotron_provider)
            whisper_status = ServiceStatus(True, details=f"{settings.whisper_model} (streaming)")
        else:
            try:
                asr_provider = FasterWhisperAsrProvider(settings)
                await asr_provider.load()
            except Exception as exc:
                logger.exception("Whisper startup failed")
                await _close_resources(resources)
                raise StartupError(f"Whisper failed to load: {exc}") from exc
            asr_service = AsrService(asr_provider)
            whisper_status = ServiceStatus(True, details=settings.whisper_model)

    # Term-extraction morphology backend (NLP_BACKEND). Both share the same rule
    # layer; spaCy additionally handles English, Natasha is Russian only.
    nlp_backend = settings.nlp_backend.strip().lower()
    try:
        nlp_extractor: NlpTermExtractor
        if nlp_backend == "spacy":
            nlp_extractor = SpacyTermExtractor(settings)
        elif nlp_backend in ("", "natasha"):
            nlp_extractor = NatashaTermExtractor(settings)
        else:
            raise StartupError(f"Unknown NLP_BACKEND '{nlp_backend}'. Valid: natasha, spacy")
        await nlp_extractor.load()
    except Exception as exc:
        logger.exception("Term extractor startup failed")
        await _close_resources(resources)
        raise StartupError(f"Term extractor ({nlp_backend}) failed to load: {exc}") from exc

    # LLM provider is chosen at backend startup (LLM_PROVIDER in .env). "ollama"
    # uses the on-box model below; any other key mounts a hosted provider with
    # key/model/base_url from .env, resolved against the LLM_PROVIDERS catalog.
    llm_provider, llm_ready, llm_details = await _build_llm_provider(settings)

    term_service = TermExtractorService(nlp_extractor)
    transcript_service = TranscriptService(asr_service, term_service, settings)
    explanation_service = ExplanationService(repository, llm_provider, llm_enabled=llm_ready)
    answer_service = AnswerService(llm_provider, llm_enabled=llm_ready)
    # Chat reuses the startup provider; the catalog decides which reasoning
    # efforts it may be asked for (empty = no reasoning control in the client).
    chat_preset = LLM_PROVIDERS.get(settings.llm_provider.strip().lower() or "ollama")
    chat_service = ChatService(
        llm_provider,
        llm_enabled=llm_ready,
        reasoning_levels=chat_preset.reasoning_levels if chat_preset else (),
    )

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
        chat_service=chat_service,
        resources=resources,
    )


# Strong refs so fire-and-forget warmup tasks aren't garbage-collected mid-run.
_background_tasks: set[asyncio.Task[None]] = set()


async def _build_llm_provider(
    settings: Settings,
) -> tuple[LlmProvider, bool, str]:
    """Resolve the startup LLM provider from settings.llm_provider.

    Returns (provider, ready, details). "ollama" (and empty) build the on-box
    provider and healthcheck it; any other key builds the hosted provider from the
    LLM_PROVIDERS catalog using llm_api_key/model/base_url (falling back to the
    catalog defaults). Hosted providers are assumed ready — auth failures surface
    on the first real request, not at boot.
    """
    provider_key = settings.llm_provider.strip().lower()
    if not settings.llm_enabled:
        # Still construct an Ollama stub so the service objects have a provider
        # reference; the ready flag stays False and explain/answer raise.
        return OllamaLlmProvider(settings), False, "disabled"

    provider: LlmProvider
    if provider_key in ("", "ollama"):
        provider = OllamaLlmProvider(settings)
        ready = await provider.healthcheck()
        details = f"model={settings.ollama_model}" if ready else "Ollama unavailable"
        if ready:
            _spawn_llm_warmup(provider)
        return provider, ready, details

    preset = LLM_PROVIDERS.get(provider_key)
    if preset is None:
        raise StartupError(
            f"Unknown LLM_PROVIDER '{provider_key}'. "
            f"Valid: {', '.join(LLM_PROVIDERS)}"
        )

    api_key = settings.llm_api_key.strip()
    if not api_key:
        raise StartupError(
            f"LLM_PROVIDER={provider_key} requires LLM_API_KEY (it is empty in .env)"
        )
    base_url = settings.llm_api_base_url.strip() or preset.base_url
    model = settings.llm_api_model.strip() or preset.default_model

    if preset.is_native_anthropic:
        provider = AnthropicLlmProvider(settings, api_key=api_key, model=model, base_url=base_url)
    else:
        provider = OpenAiLlmProvider(
            settings,
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=preset.reasoning_effort,
        )
    return provider, True, f"{provider_key} ({model})"


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

