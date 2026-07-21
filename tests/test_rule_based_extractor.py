from __future__ import annotations

from app.core.config import Settings
from app.infrastructure.nlp.rule_based_extractor import (
    RuleBasedTermExtractor,
    build_token,
    document_is_cyrillic,
)


def _extractor() -> RuleBasedTermExtractor:
    return RuleBasedTermExtractor(Settings())


def _tokens(text: str, tagged: list[tuple[str, str]], *, cyrillic: bool) -> list:
    """Build tokens for ``tagged`` (word, pos) pairs, locating each word in ``text``."""
    tokens = []
    cursor = 0
    for word, pos in tagged:
        start = text.index(word, cursor)
        cursor = start + len(word)
        tokens.append(
            build_token(
                text=word,
                lemma=word.lower(),
                start=start,
                end=cursor,
                pos=pos,
                feats={},
                document_is_cyrillic=cyrillic,
            )
        )
    return tokens


def test_document_language_is_decided_by_cyrillic_presence() -> None:
    # Latin outnumbers Cyrillic here, yet the sentence is Russian.
    assert document_is_cyrillic("паттерн dependency injection")
    assert not document_is_cyrillic("the dependency injection pattern")


def test_english_words_are_not_technical_inside_english_text() -> None:
    # The old "any Latin letter means technical" rule marked every English word as
    # a term; inside English prose the script carries no signal at all.
    [the] = _tokens("the", [("the", "DET")], cyrillic=False)
    assert not the.is_tech


def test_latin_token_is_technical_inside_russian_text() -> None:
    text = "паттерн observer"
    _, observer = _tokens(text, [("паттерн", "NOUN"), ("observer", "X")], cyrillic=True)
    assert observer.is_tech


def test_acronyms_and_camel_case_are_technical_in_any_language() -> None:
    text = "the OOP useState idea"
    _, oop, use_state, _ = _tokens(
        text,
        [("the", "DET"), ("OOP", "PROPN"), ("useState", "PROPN"), ("idea", "NOUN")],
        cyrillic=False,
    )
    assert oop.is_tech
    assert use_state.is_tech


def test_adjacent_foreign_tokens_merge_into_one_term() -> None:
    # Taggers label borrowed words X, so no noun rule can pair them up.
    text = "используем dependency injection тут"
    tokens = _tokens(
        text,
        [("используем", "VERB"), ("dependency", "X"), ("injection", "X"), ("тут", "ADV")],
        cyrillic=True,
    )
    spans = _extractor()._candidate_spans(tokens, text)
    assert (1, 3, 0.82) in spans


def test_adjacent_english_words_do_not_merge_inside_english_text() -> None:
    text = "we will discuss"
    tokens = _tokens(
        text,
        [("we", "PRON"), ("will", "AUX"), ("discuss", "VERB")],
        cyrillic=False,
    )
    assert _extractor()._candidate_spans(tokens, text) == []


def test_adjective_run_before_noun_forms_one_term() -> None:
    # Taggers that split hyphenated compounds turn ADJ+NOUN into ADJ+ADJ+NOUN.
    text = "объектно-ориентированное программирование"
    tokens = _tokens(
        text,
        [("объектно", "ADJ"), ("ориентированное", "ADJ"), ("программирование", "NOUN")],
        cyrillic=True,
    )
    spans = _extractor()._candidate_spans(tokens, text)
    assert (0, 3, 0.86) in spans


def test_hyphen_does_not_break_a_phrase_but_punctuation_does() -> None:
    hyphen_text = "объектно-ориентированное программирование"
    hyphen_tokens = _tokens(
        hyphen_text,
        [("объектно", "ADJ"), ("ориентированное", "ADJ"), ("программирование", "NOUN")],
        cyrillic=True,
    )
    assert any(
        end - start == 3
        for start, end, _ in _extractor()._candidate_spans(hyphen_tokens, hyphen_text)
    )

    comma_text = "Whisper, Natasha"
    comma_tokens = _tokens(comma_text, [("Whisper", "PROPN"), ("Natasha", "PROPN")], cyrillic=True)
    spans = _extractor()._candidate_spans(comma_tokens, comma_text)
    assert all(end - start == 1 for start, end, _ in spans)


def test_proper_noun_outscores_a_plain_noun() -> None:
    # Product names (Python, React) must clear the confidence threshold; generic
    # nouns must not.
    text = "the Kubernetes cluster"
    _, kubernetes, cluster = _tokens(
        text,
        [("the", "DET"), ("Kubernetes", "PROPN"), ("cluster", "NOUN")],
        cyrillic=False,
    )
    threshold = Settings().term_confidence_threshold
    spans = _extractor()._candidate_spans([kubernetes], "Kubernetes")
    assert any(confidence > threshold for _, _, confidence in spans)
    spans = _extractor()._candidate_spans([cluster], "cluster")
    assert all(confidence < threshold for _, _, confidence in spans)
