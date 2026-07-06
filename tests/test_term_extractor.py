from __future__ import annotations

from app.core.config import Settings
from app.infrastructure.nlp.natasha_extractor import NatashaTermExtractor, TokenInfo


def test_normalize_genitive_noun_phrase() -> None:
    extractor = NatashaTermExtractor(Settings())
    tokens = [
        TokenInfo("теории", "теория", 0, 6, "NOUN", {"Case": "Gen"}),
        TokenInfo(
            "относительности",
            "относительность",
            7,
            22,
            "NOUN",
            {"Case": "Gen"},
        ),
    ]

    assert extractor._normalize_tokens(tokens) == "теория относительности"


def test_normalize_mixed_technical_phrase() -> None:
    extractor = NatashaTermExtractor(Settings())
    tokens = [
        TokenInfo("индексы", "индекс", 0, 7, "NOUN", {"Case": "Nom"}),
        TokenInfo("PostgreSQL", "postgresql", 8, 18, "PROPN", {}),
    ]

    assert extractor._normalize_tokens(tokens) == "индекс PostgreSQL"


def _tok(text: str, pos: str = "PROPN") -> TokenInfo:
    return TokenInfo(text, text.lower(), 0, len(text), pos, {})


def test_is_tech_recognizes_technical_tokens() -> None:
    assert _tok("Python").is_tech
    assert _tok("API").is_tech
    assert _tok("ООП").is_tech
    assert _tok("C++").is_tech


def test_is_tech_ignores_plain_capitalized_words() -> None:
    # Sentence-initial capitalization must not turn ordinary words into terms.
    assert not _tok("Итак", "ADV").is_tech
    assert not _tok("Ну", "PART").is_tech
    assert not _tok("Компонент", "NOUN").is_tech


def test_is_adj_excludes_determiners() -> None:
    assert _tok("нейронный", "ADJ").is_adj
    assert not _tok("такое", "DET").is_adj


def test_candidate_spans_do_not_cross_punctuation() -> None:
    extractor = NatashaTermExtractor(Settings())
    text = "Whisper, Natasha"  # comma between two technical tokens
    tokens = [
        TokenInfo("Whisper", "whisper", 0, 7, "PROPN", {}),
        TokenInfo("Natasha", "natasha", 9, 16, "PROPN", {}),
    ]
    spans = extractor._candidate_spans(tokens, text)
    # only single-token spans, never a merged "Whisper Natasha"
    assert all(end - start == 1 for start, end, _ in spans)


def test_lexicon_catches_single_lowercase_term() -> None:
    extractor = NatashaTermExtractor(Settings())
    text = "полиморфизм"
    tokens = [TokenInfo("полиморфизм", "полиморфизм", 0, len(text), "NOUN", {})]
    spans = extractor._candidate_spans(tokens, text)
    assert (0, 1, 0.95) in spans


def test_capitalized_duplicates_not_glued() -> None:
    extractor = NatashaTermExtractor(Settings())
    text = "Полиморфизм Полиморфизм"
    tokens = [
        TokenInfo("Полиморфизм", "полиморфизм", 0, 11, "PROPN", {}),
        TokenInfo("Полиморфизм", "полиморфизм", 12, 23, "PROPN", {"Case": "Gen"}),
    ]
    spans = extractor._candidate_spans(tokens, text)
    # each capitalized word is its own lexicon term; the pair is never glued
    assert all(end - start == 1 for start, end, _ in spans)
    assert all(confidence == 0.95 for _, _, confidence in spans)
