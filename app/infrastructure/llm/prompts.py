from __future__ import annotations

EXPLAIN_KEYS = ("title", "short", "example", "why_important")
ANSWER_KEYS = ("answer", "points", "example")


def _context_tail(context: str, context_chars: int) -> str:
    """The transcript grows from the start, so the relevant part of the conversation
    is at the end — keep the tail, not the head."""
    return context[-context_chars:] if context_chars > 0 else ""


def _reply_language(text: str) -> str:
    """Name of the language the model must reply in, inferred from ``text``.

    Any Cyrillic at all means Russian rather than whichever script has more
    letters: a Russian question about English terms ("Что такое dependency
    injection?") is mostly Latin by character count, so picking the majority
    script would flip it to English — while a genuinely English question
    contains no Cyrillic whatsoever.

    Deciding here instead of telling the model to "match the question" keeps it
    deterministic: these prompts are themselves written in Russian, and a small
    local model (qwen3:8b) otherwise tends to answer in the prompt's language.
    """
    if any("Ѐ" <= char <= "ӿ" for char in text):
        return "русском"
    if any("a" <= char.lower() <= "z" for char in text):
        return "английском (English)"
    # No letters to judge by (empty/among digits) — fall back to the app default.
    return "русском"


def build_explain_prompt(*, term: str, context: str, context_chars: int) -> str:
    context_excerpt = _context_tail(context, context_chars)
    # Follow the conversation's language, NOT the term's: an English term dropped
    # into a Russian talk ("dependency injection") still needs a Russian explanation.
    language = _reply_language(context_excerpt)
    return (
        "/no_think\n"
        f"Ты объясняешь термины на {language} языке для любознательного слушателя, "
        "который не эксперт. Не рассуждай вслух.\n"
        "Верни ТОЛЬКО валидный JSON. Без markdown, без комментариев, без текста вне JSON.\n"
        'JSON: {"title":"","short":"","example":"","why_important":""}\n'
        "Требования к полям:\n"
        "- title: сам термин, аккуратно оформленный.\n"
        "- short: содержательное объяснение из 3-5 предложений — что это, ключевая "
        "идея и как это работает, важные свойства. Пиши конкретно и по существу, "
        "не ограничивайся одной строкой.\n"
        "- example: один конкретный пример или типичный сценарий (1-2 предложения).\n"
        "- why_important: чем полезно и где применяется (1-2 предложения).\n"
        f"Термин: {term}\n"
        f"Контекст (как упомянуто, может быть шумным): {context_excerpt}"
    )


def build_answer_prompt(
    *,
    question: str,
    context: str,
    deep: bool,
    context_chars: int,
    profile: str = "",
    meeting_context: str = "",
) -> str:
    context_excerpt = _context_tail(context, context_chars)
    depth = (
        "answer: 1-2 предложения; points: 3-5 тезисов подробнее"
        if deep
        else "answer: одно предложение; points: 2-4 коротких тезиса"
    )
    personalization = ""
    if profile.strip():
        personalization += (
            "О человеке, за которого ты отвечаешь (говори от его лица, опирайся "
            f"на его опыт и стек): {profile.strip()}\n"
        )
    if meeting_context.strip():
        personalization += f"Контекст этой встречи: {meeting_context.strip()}\n"
    language = _reply_language(question)
    return (
        "/no_think\n"
        "Ты помогаешь человеку отвечать вслух на вопросы в живом разговоре "
        f"(например, на собеседовании). Отвечай на {language} языке — на том же, "
        "на котором задан вопрос — ОТ ПЕРВОГО ЛИЦА, как "
        "человек, который разобрался в теме и объясняет своими словами, а не "
        "зачитывает определение из учебника. Не рассуждай вслух.\n"
        "Стиль ответа:\n"
        "- Простые слова и разговорные связки («по сути», «грубо говоря», «то есть»), "
        "как в устной речи.\n"
        "- НЕ начинай с формального определения. Никакого канцелярита и книжных "
        "оборотов: «является», «представляет собой», «данный», «осуществляет» — "
        "запрещены.\n"
        "- Своё понимание, а не пересказ: сначала суть и зачем это нужно, детали потом.\n"
        "- Где уместно — опирайся на практику («на практике обычно…», «я обычно "
        "делаю так…»), особенно на опыт и стек человека, если они указаны.\n"
        "Верни ТОЛЬКО валидный JSON. Без markdown и текста вне JSON. "
        "ВСЕ значения полей — строки.\n"
        'JSON: {"answer":"","points":"","example":""}\n'
        "Требования к полям:\n"
        "- answer: прямой ответ своими словами, который можно сразу произнести.\n"
        "- points: ОДНА СТРОКА с тезисами для проговаривания, каждый тезис с новой "
        'строки, начинается с "- ". Тезисы тоже разговорные, а не конспект. '
        "Это текст, а не число и не список.\n"
        "- example: один короткий пример из практики (1 предложение), по возможности "
        "от первого лица.\n"
        f"Объём: {depth}.\n"
        f"{personalization}"
        f"Вопрос: {question}\n"
        f"Контекст разговора (может быть шумным): {context_excerpt}"
    )
