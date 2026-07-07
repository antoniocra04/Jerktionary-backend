from __future__ import annotations

from app.core.config import Settings
from app.domain.interfaces.asr import AsrStream
from app.infrastructure.asr.api_asr import ApiTranscriptionStream


class ApiAsrProvider:
    """Mounts the OpenAI-compatible Whisper API as a startup-time ``AsrProvider``.

    The provider/key/base_url are fixed at backend startup (chosen in the launcher
    and stored in ``.env``), so every WebSocket connection gets a fresh
    ``ApiTranscriptionStream`` built from the same settings — replacing the old
    per-connection ``AsrChoice`` routing in TranscriptService.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def create_stream(self) -> AsrStream:
        return ApiTranscriptionStream(
            self._settings,
            api_key=self._settings.whisper_api_key,
            model=self._settings.whisper_api_model,
            base_url=self._settings.whisper_api_base_url,
        )
