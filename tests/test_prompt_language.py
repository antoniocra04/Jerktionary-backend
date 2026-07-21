from __future__ import annotations

from app.infrastructure.llm.prompts import build_answer_prompt, build_explain_prompt


def _answer_prompt(question: str) -> str:
    return build_answer_prompt(question=question, context="", deep=False, context_chars=500)


def test_russian_question_answered_in_russian() -> None:
    assert "на русском языке" in _answer_prompt("Что такое замыкание?")


def test_english_question_answered_in_english() -> None:
    assert "на английском (English) языке" in _answer_prompt("What is a closure?")


def test_russian_question_about_english_terms_stays_russian() -> None:
    # The Latin letters outnumber the Cyrillic ones here, so a majority-script
    # rule would wrongly switch to English — this is the app's core use case.
    prompt = _answer_prompt("Чем отличается dependency injection от service locator?")
    assert "на русском языке" in prompt


def test_question_without_letters_falls_back_to_russian() -> None:
    assert "на русском языке" in _answer_prompt("2 + 2 ?")


def test_explanation_follows_conversation_not_the_term() -> None:
    # An English term inside a Russian talk must still be explained in Russian.
    prompt = build_explain_prompt(
        term="dependency injection",
        context="Мы обсуждали, как правильно собирать зависимости в сервисах.",
        context_chars=500,
    )
    assert "термины на русском языке" in prompt


def test_explanation_in_english_conversation_is_english() -> None:
    prompt = build_explain_prompt(
        term="closure",
        context="We were discussing how functions capture their surrounding scope.",
        context_chars=500,
    )
    assert "термины на английском (English) языке" in prompt
