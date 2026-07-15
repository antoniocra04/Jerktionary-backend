from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.infrastructure.asr.yandex_asr import (
    StreamingTranscript,
    YandexStreamingAsrStream,
    _language_tag,
)


def test_partial_then_final_then_refinement_commits_only_refined_text() -> None:
    t = StreamingTranscript()
    t.handle_partial("привет")
    t.handle_partial("привет мир")
    assert t.snapshot() == ("привет мир", 0)

    # The raw final stays volatile (its refinement is still coming)...
    t.handle_final("привет мир")
    assert t.snapshot() == ("привет мир", 0)

    # ...and only the normalized refinement becomes the permanent prefix.
    t.handle_final_refinement("Привет, мир!")
    text, committed_len = t.snapshot()
    assert text == "Привет, мир!"
    assert committed_len == len("Привет, мир!")


def test_committed_prefix_never_changes_across_utterances() -> None:
    t = StreamingTranscript()
    t.handle_final("раз два")
    t.handle_final_refinement("Раз, два.")
    t.handle_partial("три")
    text, committed_len = t.snapshot()
    assert text == "Раз, два. три"
    assert text[:committed_len] == "Раз, два."

    # A refinement never duplicates the raw final it replaces.
    t.handle_final("три четыре")
    t.handle_final_refinement("Три, четыре.")
    text, committed_len = t.snapshot()
    assert text == "Раз, два. Три, четыре."
    assert committed_len == len(text)


def test_finish_session_promotes_pending_finals_and_partial() -> None:
    # Session died before the refinement/final arrived: nothing may be lost.
    t = StreamingTranscript()
    t.handle_final("первая фраза")
    t.handle_partial("вторая")
    t.finish_session()
    text, committed_len = t.snapshot()
    assert text == "первая фраза вторая"
    assert committed_len == len(text)

    # A new session keeps appending after the frozen prefix.
    t.handle_partial("третья")
    assert t.snapshot() == ("первая фраза вторая третья", len("первая фраза вторая"))


def test_language_tag_mapping() -> None:
    assert _language_tag("ru") == "ru-RU"
    assert _language_tag("en") == "en-US"
    assert _language_tag("kk-KZ") == "kk-KZ"  # full tags pass through
    assert _language_tag("") == "auto"
    assert _language_tag("xx") == "auto"  # unknown codes fall back to detection


def _make_stream() -> YandexStreamingAsrStream:
    settings = Settings(_env_file=None, yandex_stt_api_key="test-key")
    # Dummy modules: _emit never touches gRPC, only the transcript state.
    return YandexStreamingAsrStream(settings, modules=None)  # type: ignore[arg-type]


def test_emit_throttles_partials_but_not_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _make_stream()
    clock = {"now": 100.0}
    monkeypatch.setattr("app.infrastructure.asr.yandex_asr.time.monotonic", lambda: clock["now"])
    transcript: Any = stream._transcript

    transcript.handle_partial("привет")
    first = stream._emit()
    assert first is not None and first.text == "привет" and first.committed_len == 0

    # A changed partial inside the throttle window is suppressed...
    clock["now"] += 0.1
    transcript.handle_partial("привет мир")
    assert stream._emit() is None

    # ...but a commit (final_refinement) goes out immediately.
    clock["now"] += 0.05
    transcript.handle_final("привет мир")
    transcript.handle_final_refinement("Привет, мир!")
    committed = stream._emit()
    assert committed is not None
    assert committed.committed_len == len("Привет, мир!")

    # Unchanged text never re-emits, even after the window passes.
    clock["now"] += 10.0
    assert stream._emit() is None
