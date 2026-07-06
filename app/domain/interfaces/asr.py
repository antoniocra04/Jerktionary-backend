from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AsrResult:
    text: str
    is_final: bool
    # Length of the leading, already-committed part of ``text`` that will never
    # change again. Lets consumers parse only the volatile tail each cycle.
    committed_len: int = 0


class AsrStream(Protocol):
    async def append_pcm(self, chunk: bytes) -> AsrResult | None:
        ...

    async def close(self) -> None:
        ...


class AsrProvider(Protocol):
    async def create_stream(self) -> AsrStream:
        ...

