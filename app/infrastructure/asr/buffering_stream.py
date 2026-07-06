from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.core.config import Settings
from app.domain.interfaces.asr import AsrResult


class BufferingAsrStream(ABC):
    """Shared PCM buffering for streaming transcription: silence gating, transcribe
    throttling, and committing segments that scrolled past the live tail. Subclasses
    only provide `_decode` — locally via faster-whisper, or remotely via an API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._buffer = bytearray()
        self._last_text = ""
        self._committed_text = ""
        self._bytes_since_transcribe = 0
        self._trailing_silent = False

    @abstractmethod
    async def _decode(self) -> list[tuple[float, float, str]]:
        """Transcribe the current buffer into (start, end, text) segments."""

    @property
    def _transcribe_interval_seconds(self) -> float:
        return self._settings.asr_transcribe_interval_seconds

    async def append_pcm(self, chunk: bytes) -> AsrResult | None:
        if not chunk:
            return None

        if self._is_silent(chunk):
            # Keep a single chunk of trailing silence as a pause marker, but drop
            # any further sustained silence. Otherwise silence keeps filling the
            # 30 s window and slides earlier speech out, erasing the transcript from
            # the top while the user is simply not talking.
            if self._trailing_silent:
                return await self._maybe_flush()
            self._trailing_silent = True
        else:
            self._trailing_silent = False

        self._buffer.extend(chunk)
        self._bytes_since_transcribe += len(chunk)
        self._trim_buffer()
        if self._buffer_seconds < self._settings.asr_min_audio_seconds:
            return None

        # Throttle: only re-transcribe once enough new audio has arrived, instead
        # of on every mic chunk. Otherwise the decoder re-runs on the whole buffer
        # many times per second and the transcript falls behind real speech.
        min_new_bytes = int(self._bytes_per_second * self._transcribe_interval_seconds)
        if self._bytes_since_transcribe < min_new_bytes:
            return None

        return await self._run_transcription()

    async def _maybe_flush(self) -> AsrResult | None:
        """Transcribe once when speech stops so the tail of the last utterance is
        captured even if it is shorter than the throttle interval."""
        if self._bytes_since_transcribe <= 0:
            return None
        if self._buffer_seconds < self._settings.asr_min_audio_seconds:
            return None
        return await self._run_transcription()

    async def _run_transcription(self) -> AsrResult | None:
        self._bytes_since_transcribe = 0
        segments = await self._decode()
        if not segments:
            # Transient empty decode (e.g. only silence in the buffer): keep whatever
            # was last shown instead of wiping it.
            return None

        # Commit segments that have scrolled past the live tail: append their text to
        # the permanent transcript and drop their audio from the buffer. Everything
        # already committed stays on screen forever, so long speech never erases it.
        commit_before = self._buffer_seconds - self._settings.asr_commit_tail_seconds
        committed_parts: list[str] = []
        tail_parts: list[str] = []
        last_commit_end = 0.0
        for _start, end, seg_text in segments:
            if not seg_text:
                continue
            if end <= commit_before:
                committed_parts.append(seg_text)
                last_commit_end = end
            else:
                tail_parts.append(seg_text)

        if committed_parts:
            self._committed_text = " ".join(
                part for part in (self._committed_text, *committed_parts) if part
            )
            del self._buffer[: int(last_commit_end * self._bytes_per_second)]

        full = " ".join(part for part in (self._committed_text, *tail_parts) if part).strip()
        if not full or full == self._last_text:
            return None
        self._last_text = full
        committed_len = min(len(self._committed_text), len(full))
        return AsrResult(text=full, is_final=False, committed_len=committed_len)

    async def close(self) -> None:
        self._buffer.clear()
        self._committed_text = ""

    def _pcm_buffer(self) -> bytes:
        return bytes(self._buffer)

    def _is_silent(self, chunk: bytes) -> bool:
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            return True
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        return rms < self._settings.asr_silence_rms_threshold

    @property
    def _bytes_per_second(self) -> int:
        return (
            self._settings.audio_sample_rate
            * self._settings.audio_channels
            * self._settings.audio_sample_width_bytes
        )

    @property
    def _buffer_seconds(self) -> float:
        return len(self._buffer) / self._bytes_per_second

    def _trim_buffer(self) -> None:
        max_bytes = int(self._bytes_per_second * self._settings.asr_max_buffer_seconds)
        if len(self._buffer) > max_bytes:
            del self._buffer[: len(self._buffer) - max_bytes]
