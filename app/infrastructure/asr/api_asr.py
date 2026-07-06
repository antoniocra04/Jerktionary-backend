from __future__ import annotations

import io
import wave

import httpx
from loguru import logger

from app.core.config import Settings
from app.core.errors import AsrApiError
from app.infrastructure.asr.buffering_stream import BufferingAsrStream


class ApiTranscriptionStream(BufferingAsrStream):
    """OpenAI-compatible ``/audio/transcriptions`` transcription on top of the shared
    buffering pipeline. Lets users run without a local Whisper: OpenAI, Groq, or any
    compatible server. ``verbose_json`` supplies segment timestamps, so the commit
    logic behaves exactly like the local model."""

    def __init__(self, settings: Settings, *, api_key: str, model: str, base_url: str) -> None:
        super().__init__(settings)
        self._api_key = api_key
        self._model = model or "whisper-1"
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._rejected = False

    @property
    def _transcribe_interval_seconds(self) -> float:
        return self._settings.asr_api_transcribe_interval_seconds

    async def _decode(self) -> list[tuple[float, float, str]]:
        if self._rejected:
            return []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    data={
                        "model": self._model,
                        "language": self._settings.asr_language,
                        "response_format": "verbose_json",
                    },
                    files={"file": ("audio.wav", self._to_wav(), "audio/wav")},
                )
        except httpx.HTTPError as exc:
            # Transient network problem — keep the last shown text and retry later.
            logger.warning("API transcription request failed: {}", exc)
            return []

        if response.status_code >= 400:
            # A key/model/URL problem won't fix itself: surface it once, then stop
            # hammering the paid API on every interval.
            self._rejected = True
            raise AsrApiError(f"HTTP {response.status_code}: {response.text[:200]}")

        payload = response.json()
        segments = [
            (
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
                str(seg.get("text", "")).strip(),
            )
            for seg in payload.get("segments") or []
        ]
        if segments:
            return segments

        # Some models return plain text without segments — degrade to one segment
        # spanning the buffer (it stays in the live tail and never commits).
        text = str(payload.get("text", "")).strip()
        return [(0.0, self._buffer_seconds, text)] if text else []

    def _to_wav(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self._settings.audio_channels)
            wav.setsampwidth(self._settings.audio_sample_width_bytes)
            wav.setframerate(self._settings.audio_sample_rate)
            wav.writeframes(self._pcm_buffer())
        return buffer.getvalue()
