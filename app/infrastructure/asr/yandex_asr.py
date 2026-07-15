from __future__ import annotations

import asyncio
import time
from typing import Any, NamedTuple

from loguru import logger

from app.core.config import Settings
from app.core.errors import AsrApiError
from app.domain.interfaces.asr import AsrResult, AsrStream

# asr_language holds a bare Whisper code ("ru"); SpeechKit wants a BCP-47 tag.
# Codes not listed here fall back to SpeechKit's auto language detection.
_LANGUAGE_TAGS = {
    "ru": "ru-RU",
    "en": "en-US",
    "kk": "kk-KZ",
    "uz": "uz-UZ",
    "de": "de-DE",
    "fr": "fr-FR",
    "tr": "tr-TR",
    "he": "he-IL",
}


def _language_tag(code: str) -> str:
    code = code.strip()
    if not code or code.lower() == "auto":
        return "auto"
    if "-" in code:
        return code
    return _LANGUAGE_TAGS.get(code.lower(), "auto")


class _SpeechKitModules(NamedTuple):
    grpc: Any
    stt_pb2: Any
    stt_service_pb2_grpc: Any


def _import_speechkit() -> _SpeechKitModules:
    """Import grpc and the pre-generated SpeechKit v3 stubs from ``yandexcloud``.

    Deferred so the backend runs without the optional dependencies unless the
    Yandex provider is actually selected.
    """
    try:
        import grpc
        from yandex.cloud.ai.stt.v3 import stt_pb2, stt_service_pb2_grpc
    except ImportError as exc:
        raise RuntimeError(
            "Yandex SpeechKit support requires the optional dependencies; "
            'install them with: pip install -e ".[yandex]"'
        ) from exc
    return _SpeechKitModules(grpc, stt_pb2, stt_service_pb2_grpc)


class StreamingTranscript:
    """Folds SpeechKit v3 events into the committed-prefix + volatile-tail shape the
    rest of the pipeline expects (see ``AsrResult.committed_len``).

    Event flow per utterance: ``partial``* → ``final`` → ``final_refinement`` (the
    normalized text; we always enable normalization). Only refined text is committed
    because it is the last version SpeechKit ever sends for an utterance — so the
    committed prefix never changes retroactively, which ``TranscriptSession`` relies
    on to freeze terms. Raw finals wait in ``_pending_finals`` and are shown as part
    of the volatile tail; if the session ends before their refinement arrives, they
    are committed as-is.
    """

    def __init__(self) -> None:
        self._committed = ""
        self._pending_finals: list[str] = []
        self._partial = ""

    def handle_partial(self, text: str) -> None:
        self._partial = text.strip()

    def handle_final(self, text: str) -> None:
        text = text.strip()
        if text:
            self._pending_finals.append(text)
        self._partial = ""

    def handle_final_refinement(self, text: str) -> None:
        # The refinement replaces the raw final it refers to (FIFO), never adds to it.
        if self._pending_finals:
            self._pending_finals.pop(0)
        self._commit(text.strip())

    def finish_session(self) -> None:
        """The gRPC session ended: no more refinements are coming for what was
        already recognized, so promote everything volatile to committed."""
        for text in self._pending_finals:
            self._commit(text)
        self._pending_finals.clear()
        self._commit(self._partial)
        self._partial = ""

    def snapshot(self) -> tuple[str, int]:
        """Full transcript text and the length of its committed prefix."""
        parts = [p for p in (self._committed, *self._pending_finals, self._partial) if p]
        full = " ".join(parts)
        return full, min(len(self._committed), len(full))

    def _commit(self, text: str) -> None:
        if text:
            self._committed = f"{self._committed} {text}".strip()


class YandexAsrProvider:
    """Yandex SpeechKit v3 streaming recognition (gRPC) as a startup-time provider.

    Unlike the Whisper providers there is no buffer re-decoding: PCM chunks are
    pushed into a bidirectional gRPC stream and partial/final hypotheses come back
    in real time as SpeechKit produces them.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._modules: _SpeechKitModules | None = None

    def load(self) -> None:
        """Validate config and import the gRPC stack, so a missing key or missing
        optional dependency fails at startup instead of on the first connection."""
        if not self._settings.yandex_stt_api_key.strip():
            raise RuntimeError(
                "WHISPER_PROVIDER=yandex requires YANDEX_STT_API_KEY (it is empty in .env)"
            )
        self._modules = _import_speechkit()

    async def create_stream(self) -> AsrStream:
        modules = self._modules or _import_speechkit()
        return YandexStreamingAsrStream(self._settings, modules)


class YandexStreamingAsrStream:
    """One WebSocket connection's live SpeechKit session.

    ``append_pcm`` writes the chunk into the gRPC stream; a background task reads
    recognition events and folds them into ``StreamingTranscript``. Sessions are
    re-opened transparently when SpeechKit closes them (network hiccup, session
    duration limit) — committed text survives the reconnect. Auth/config errors are
    not retried: they surface once as ``AsrApiError``, like the API Whisper stream,
    instead of hammering the paid API on every mic chunk.
    """

    _RECONNECT_COOLDOWN_SECONDS = 2.0
    _FATAL_STATUS_NAMES = frozenset(
        {"UNAUTHENTICATED", "PERMISSION_DENIED", "INVALID_ARGUMENT", "NOT_FOUND"}
    )

    def __init__(self, settings: Settings, modules: _SpeechKitModules) -> None:
        self._settings = settings
        self._modules = modules
        self._transcript = StreamingTranscript()
        self._channel: Any | None = None
        self._call: Any | None = None
        self._reader: asyncio.Task[None] | None = None
        self._session_broken = False
        self._next_connect_at = 0.0
        self._fatal_error: str | None = None
        self._fatal_reported = False
        self._last_text = ""
        self._last_committed_len = 0
        self._last_emit_at = 0.0

    async def append_pcm(self, chunk: bytes) -> AsrResult | None:
        if not chunk:
            return None
        if self._fatal_error is not None:
            if self._fatal_reported:
                return None
            self._fatal_reported = True
            raise AsrApiError(self._fatal_error)

        await self._ensure_session()
        if self._call is not None:
            pb = self._modules.stt_pb2
            try:
                await self._call.write(pb.StreamingRequest(chunk=pb.AudioChunk(data=chunk)))
            except Exception as exc:
                logger.warning("SpeechKit write failed ({}); will reconnect", exc)
                await self._teardown_session()
        return self._emit()

    async def close(self) -> None:
        await self._teardown_session()

    async def _ensure_session(self) -> None:
        if self._session_broken:
            await self._teardown_session()
        if self._call is not None or self._fatal_error is not None:
            return
        now = time.monotonic()
        if now < self._next_connect_at:
            return
        self._next_connect_at = now + self._RECONNECT_COOLDOWN_SECONDS

        grpc = self._modules.grpc
        pb = self._modules.stt_pb2
        self._channel = grpc.aio.secure_channel(
            self._settings.yandex_stt_endpoint, grpc.ssl_channel_credentials()
        )
        stub = self._modules.stt_service_pb2_grpc.RecognizerStub(self._channel)
        api_key = self._settings.yandex_stt_api_key.strip()
        call = stub.RecognizeStreaming(metadata=(("authorization", f"Api-Key {api_key}"),))
        try:
            await call.write(pb.StreamingRequest(session_options=self._session_options()))
        except Exception as exc:
            logger.warning("SpeechKit session open failed ({}); will retry", exc)
            await self._close_transport(call)
            return
        self._call = call
        self._reader = asyncio.create_task(self._read_events(call))

    def _session_options(self) -> Any:
        pb = self._modules.stt_pb2
        model_options: dict[str, Any] = {
            "model": self._settings.yandex_stt_model,
            "audio_format": pb.AudioFormatOptions(
                raw_audio=pb.RawAudio(
                    audio_encoding=pb.RawAudio.LINEAR16_PCM,
                    sample_rate_hertz=self._settings.audio_sample_rate,
                    audio_channel_count=self._settings.audio_channels,
                )
            ),
            # Normalized (refined) finals are what gets committed permanently;
            # see StreamingTranscript.
            "text_normalization": pb.TextNormalizationOptions(
                text_normalization=pb.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                profanity_filter=False,
            ),
            "audio_processing_type": pb.RecognitionModelOptions.REAL_TIME,
        }
        language = _language_tag(self._settings.asr_language)
        if language != "auto":
            model_options["language_restriction"] = pb.LanguageRestrictionOptions(
                restriction_type=pb.LanguageRestrictionOptions.WHITELIST,
                language_code=[language],
            )
        return pb.StreamingOptions(recognition_model=pb.RecognitionModelOptions(**model_options))

    async def _read_events(self, call: Any) -> None:
        grpc = self._modules.grpc
        try:
            while True:
                response = await call.read()
                if response is grpc.aio.EOF:
                    logger.info("SpeechKit session closed by server; will reconnect")
                    break
                self._apply_event(response)
        except asyncio.CancelledError:
            raise
        except grpc.aio.AioRpcError as exc:
            code = exc.code().name
            if code in self._FATAL_STATUS_NAMES:
                # A key/model/config problem won't fix itself: report once, stop retrying.
                self._fatal_error = f"SpeechKit rejected the session ({code}): {exc.details()}"
                logger.warning("{}", self._fatal_error)
            else:
                logger.warning(
                    "SpeechKit session ended ({}): {}; will reconnect", code, exc.details()
                )
        except Exception as exc:
            logger.warning("SpeechKit reader failed ({}); will reconnect", exc)
        finally:
            # Signal append_pcm to tear down and (if not fatal) open a fresh session.
            self._session_broken = True

    def _apply_event(self, response: Any) -> None:
        kind = response.WhichOneof("Event")
        if kind == "partial":
            if response.partial.alternatives:
                self._transcript.handle_partial(response.partial.alternatives[0].text)
        elif kind == "final":
            if response.final.alternatives:
                self._transcript.handle_final(response.final.alternatives[0].text)
        elif kind == "final_refinement":
            alternatives = response.final_refinement.normalized_text.alternatives
            if alternatives:
                self._transcript.handle_final_refinement(alternatives[0].text)

    def _emit(self) -> AsrResult | None:
        text, committed_len = self._transcript.snapshot()
        if not text or text == self._last_text:
            return None
        now = time.monotonic()
        # Partials can change on every ~85 ms mic chunk; rate-limit transcript/NLP
        # updates. Newly committed text always goes out immediately so downstream
        # term freezing keeps up with finals.
        if (
            committed_len == self._last_committed_len
            and now - self._last_emit_at < self._settings.yandex_stt_emit_interval_seconds
        ):
            return None
        self._last_text = text
        self._last_committed_len = committed_len
        self._last_emit_at = now
        return AsrResult(text=text, is_final=False, committed_len=committed_len)

    async def _teardown_session(self) -> None:
        reader, self._reader = self._reader, None
        call, self._call = self._call, None
        if reader is not None:
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # reader errors are already logged inside _read_events
        await self._close_transport(call)
        # Whatever was recognized so far will get no more refinements — freeze it
        # so the transcript survives the reconnect instead of being re-partialled.
        self._transcript.finish_session()
        self._session_broken = False

    async def _close_transport(self, call: Any) -> None:
        if call is not None:
            try:
                call.cancel()
            except Exception:
                pass
        channel, self._channel = self._channel, None
        if channel is not None:
            try:
                await channel.close()
            except Exception:
                pass
