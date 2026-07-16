from __future__ import annotations

from app.infrastructure.asr.lang_tags import language_tag
from app.infrastructure.asr.nemotron_asr import _MAX_VOLATILE_CHARS, commit_boundary


def test_commit_boundary_freezes_up_to_last_sentence_end() -> None:
    text = "Первое предложение. Второе, ещё не закончен"
    boundary = commit_boundary(text)
    assert text[:boundary] == "Первое предложение."


def test_commit_boundary_without_punctuation_caps_volatile_tail() -> None:
    text = "слово " * 100  # 600 chars, no sentence-final punctuation
    boundary = commit_boundary(text)
    assert len(text) - boundary == _MAX_VOLATILE_CHARS


def test_commit_boundary_is_monotonic_as_text_grows() -> None:
    # RNN-T streaming text is append-only; the frozen prefix must never shrink.
    previous = 0
    text = ""
    for sentence in ["Привет!", " Как дела?", " Расскажи про кэш.", " И ещё немного слов"]:
        text += sentence
        boundary = commit_boundary(text)
        assert boundary >= previous
        previous = boundary


def test_language_tag_shared_mapping() -> None:
    assert language_tag("ru") == "ru-RU"
    assert language_tag("auto") == "auto"
