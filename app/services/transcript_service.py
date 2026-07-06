from __future__ import annotations

from dataclasses import replace

from app.api.schemas.transcript import AsrChoice
from app.core.config import Settings
from app.core.errors import AsrUnavailableError
from app.domain.entities.term import Term
from app.domain.entities.transcript import TranscriptUpdate
from app.domain.interfaces.asr import AsrStream
from app.infrastructure.asr.api_asr import ApiTranscriptionStream
from app.services.asr_service import AsrService
from app.services.term_extractor_service import TermExtractorService


class TranscriptService:
    def __init__(
        self,
        asr_service: AsrService | None,
        term_service: TermExtractorService,
        settings: Settings,
    ) -> None:
        self._asr_service = asr_service
        self._term_service = term_service
        self._settings = settings

    async def create_session(self, asr: AsrChoice | None = None) -> TranscriptSession:
        stream = await self._create_stream(asr)
        return TranscriptSession(stream, self._term_service)

    async def _create_stream(self, asr: AsrChoice | None) -> AsrStream:
        if asr is not None and asr.provider == "api":
            return ApiTranscriptionStream(
                self._settings,
                api_key=asr.api_key,
                model=asr.model,
                base_url=asr.base_url,
            )
        if self._asr_service is None:
            raise AsrUnavailableError(
                "Local Whisper is disabled (WHISPER_ENABLED=false); choose the API provider"
            )
        return await self._asr_service.create_stream()


class TranscriptSession:
    """One WebSocket connection's transcript state. Term extraction runs only on the
    volatile tail each cycle; terms for the already-committed prefix are computed once
    and reused, so cost stays flat instead of growing with the whole transcript."""

    def __init__(self, stream: AsrStream, term_service: TermExtractorService) -> None:
        self._stream = stream
        self._term_service = term_service
        self._stable_terms: list[Term] = []
        self._stable_len = 0

    async def process_audio(self, chunk: bytes) -> TranscriptUpdate | None:
        result = await self._stream.append_pcm(chunk)
        if result is None or not result.text.strip():
            return None
        terms = await self._extract_terms(result.text, result.committed_len)
        return TranscriptUpdate(text=result.text, is_final=result.is_final, terms=terms)

    async def _extract_terms(self, text: str, committed_len: int) -> list[Term]:
        committed_len = max(0, min(committed_len, len(text)))

        # Freeze terms for text that just became permanent — parse only that delta.
        if committed_len > self._stable_len:
            delta = text[self._stable_len : committed_len]
            for term in await self._term_service.extract(delta):
                self._stable_terms.append(_shift(term, self._stable_len))
            self._stable_len = committed_len

        # Re-parse only the still-changing tail every cycle.
        tail = text[committed_len:]
        tail_terms = (
            [_shift(term, committed_len) for term in await self._term_service.extract(tail)]
            if tail.strip()
            else []
        )
        return [*self._stable_terms, *tail_terms]

    async def close(self) -> None:
        await self._stream.close()


def _shift(term: Term, offset: int) -> Term:
    return replace(term, start=term.start + offset, end=term.end + offset)
