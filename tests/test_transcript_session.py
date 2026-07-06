from __future__ import annotations

import pytest

from app.domain.entities.term import Term
from app.domain.interfaces.asr import AsrResult
from app.services.transcript_service import TranscriptSession


class FakeStream:
    def __init__(self, results: list[AsrResult]) -> None:
        self._results = list(results)
        self.closed = False

    async def append_pcm(self, chunk: bytes) -> AsrResult | None:
        return self._results.pop(0) if self._results else None

    async def close(self) -> None:
        self.closed = True


class FakeTermService:
    """Emits a term for every occurrence of the marker word 'терм' in the text it
    is given, with offsets relative to that text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def extract(self, text: str) -> list[Term]:
        self.calls.append(text)
        terms: list[Term] = []
        start = text.find("терм")
        while start != -1:
            terms.append(
                Term(text="терм", normalized="терм", start=start, end=start + 4,
                     type="concept", confidence=0.9)
            )
            start = text.find("терм", start + 1)
        return terms


@pytest.mark.asyncio
async def test_incremental_extraction_offsets_and_reuse() -> None:
    stream = FakeStream(
        [
            AsrResult(text="один терм", is_final=False, committed_len=0),
            AsrResult(text="один терм два терм", is_final=False, committed_len=9),
            AsrResult(text="один терм два терм три", is_final=False, committed_len=9),
        ]
    )
    term_service = FakeTermService()
    session = TranscriptSession(stream, term_service)  # type: ignore[arg-type]

    first = await session.process_audio(b"a")
    second = await session.process_audio(b"b")
    third = await session.process_audio(b"c")

    assert first is not None and [t.start for t in first.terms] == [5]
    # committed "один терм" (term @5) frozen + tail " два терм" (term @14)
    assert second is not None and [t.start for t in second.terms] == [5, 14]
    assert second.text[14:18] == "терм"
    # nothing new committed on the third cycle → no re-parse of the committed prefix
    assert third is not None and [t.start for t in third.terms] == [5, 14]

    # the whole growing transcript is never parsed as one piece...
    assert all(("один" not in call) or ("три" not in call) for call in term_service.calls)
    # ...and once the prefix is committed, later cycles only parse the tail
    assert term_service.calls[-1] == " два терм три"
