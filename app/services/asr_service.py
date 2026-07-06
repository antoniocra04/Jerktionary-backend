from __future__ import annotations

from app.domain.interfaces.asr import AsrProvider, AsrStream


class AsrService:
    def __init__(self, provider: AsrProvider) -> None:
        self._provider = provider

    async def create_stream(self) -> AsrStream:
        return await self._provider.create_stream()

