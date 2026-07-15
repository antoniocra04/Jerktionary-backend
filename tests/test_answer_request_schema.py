from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas.answer import AnswerRequest


def test_overlong_question_is_trimmed_to_tail_instead_of_422() -> None:
    # Live-speech questions are unbounded (realtime partials have no punctuation, so
    # the frontend detector can hand over a giant run-on "sentence"). The actual ask
    # is the freshest speech — the tail.
    filler = "и вот я думаю про всякое разное " * 40
    request = AnswerRequest(question=filler + "что такое замыкание")
    assert len(request.question) == 1_000
    assert request.question.endswith("что такое замыкание")


def test_overlong_context_keeps_tail_and_profile_keeps_head() -> None:
    request = AnswerRequest(
        question="что такое стек?",
        context="старое " * 1000 + "самое свежее",
        profile="важное начало " + "хвост профиля " * 200,
        meeting_context="собеседование в банк " * 300,
    )
    assert len(request.context) == 2_000
    assert request.context.endswith("самое свежее")
    assert len(request.profile) == 1_000
    assert request.profile.startswith("важное начало")
    assert len(request.meeting_context) == 2_000


def test_normal_payload_passes_unchanged() -> None:
    request = AnswerRequest(
        question="что такое замыкание?",
        context="мы обсуждали javascript",
        deep=True,
        profile="frontend, react",
        meeting_context="собеседование",
    )
    assert request.question == "что такое замыкание?"
    assert request.context == "мы обсуждали javascript"


def test_empty_question_is_still_rejected() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(question="")
