from __future__ import annotations

import re
from dataclasses import dataclass

import anyio

from app.core.config import Settings
from app.domain.entities.term import Term
from app.infrastructure.nlp.term_lexicon import TECH_TERM_LEXICON

LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
# All-caps token with at least one letter: ООП, LLM, API, ГОСТ. NOT a plain
# capitalized word like "Итак" (which has lowercase letters).
ACRONYM_RE = re.compile(r"^(?=.*[A-ZА-Я])[A-ZА-Я0-9]{2,}$")
# An uppercase letter in the middle of a word: useState, PostgreSQL, JavaScript.
# Language-independent, so it still marks identifiers inside English prose where
# "is this token foreign?" tells us nothing.
CAMEL_CASE_RE = re.compile(r"[a-z][A-Z]")
TECH_CHARS = frozenset("0123456789+#")

STOPWORDS = {
    "это",
    "как",
    "что",
    "для",
    "при",
    "про",
    "или",
    "если",
    "когда",
    "такой",
    "такая",
    "такие",
    "один",
    "весь",
    "мочь",
    "быть",
}


@dataclass(frozen=True, slots=True)
class TokenInfo:
    text: str
    lemma: str
    start: int
    end: int
    pos: str
    feats: dict[str, str]
    # Whether this token's script differs from the document's. In a Russian
    # transcript a Latin token is a strong technical signal (Python, useState),
    # but inside English prose every word is Latin, so the same rule would mark
    # "the" and "and" as terms. Parsers set this per document language; the
    # default keeps the Russian-document behaviour for directly built tokens.
    script_is_foreign: bool = True

    @property
    def lower(self) -> str:
        return self.text.lower()

    @property
    def is_noun(self) -> bool:
        return self.pos in {"NOUN", "PROPN"}

    @property
    def is_propn(self) -> bool:
        return self.pos == "PROPN"

    @property
    def is_adj(self) -> bool:
        # Only true adjectives — not DET/pronouns like "такое", "этот", "какой-нибудь",
        # which otherwise glue onto nouns and produce junk terms with bad boundaries.
        return self.pos == "ADJ"

    @property
    def is_prep(self) -> bool:
        return self.pos == "ADP"

    @property
    def is_genitive(self) -> bool:
        return self.feats.get("Case") == "Gen"

    @property
    def is_tech(self) -> bool:
        text = self.text
        return (
            (self.script_is_foreign and bool(LATIN_RE.search(text)))
            or bool(ACRONYM_RE.match(text))
            or bool(CAMEL_CASE_RE.search(text))
            or any(ch in TECH_CHARS for ch in text)
        )

    @property
    def is_titlecase(self) -> bool:
        # A lone capitalized Cyrillic word (sentence start or a standalone dictated
        # word that Whisper capitalized). Two of these in a row are almost always
        # separate utterances, not a phrase.
        text = self.text
        return (
            len(text) >= 2
            and text[0].isupper()
            and text[1:].islower()
            and not bool(LATIN_RE.search(text))
        )

    @property
    def in_lexicon(self) -> bool:
        return self.lemma in TECH_TERM_LEXICON or self.lower in TECH_TERM_LEXICON


class RuleBasedTermExtractor:
    """Grammar-driven term spotting shared by every morphology backend.

    Subclasses supply only ``load`` and ``_parse`` — the candidate rules, scoring,
    normalization and de-duplication below work purely off ``TokenInfo``, so
    swapping the morphology engine (Natasha, spaCy, …) does not touch them.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def load(self) -> None:
        raise NotImplementedError

    def _parse(self, text: str) -> list[TokenInfo]:
        raise NotImplementedError

    async def extract(self, text: str) -> list[Term]:
        # Morphological tagging is CPU-heavy; keep it off the event loop so it can't
        # stall the audio WebSocket while it runs.
        return await anyio.to_thread.run_sync(self._extract_sync, text)

    def _extract_sync(self, text: str) -> list[Term]:
        tokens = self._parse(text)
        candidates = self._candidate_spans(tokens, text)
        terms: list[Term] = []
        seen: set[tuple[int, int, str]] = set()

        for start_idx, end_idx, confidence in candidates:
            if confidence < self._settings.term_confidence_threshold:
                continue
            phrase_tokens = tokens[start_idx:end_idx]
            raw = text[phrase_tokens[0].start : phrase_tokens[-1].end]
            normalized = self._normalize_tokens(phrase_tokens)
            if not self._is_allowed(normalized, phrase_tokens):
                continue
            key = (phrase_tokens[0].start, phrase_tokens[-1].end, normalized)
            if key in seen:
                continue
            seen.add(key)
            terms.append(
                Term(
                    text=raw,
                    normalized=normalized,
                    start=phrase_tokens[0].start,
                    end=phrase_tokens[-1].end,
                    type="concept",
                    confidence=confidence,
                )
            )
        return self._deduplicate_nested(terms)

    def normalize(self, term: str) -> str:
        try:
            tokens = self._parse(term)
        except RuntimeError:
            return " ".join(term.strip().lower().split())
        if not tokens:
            return " ".join(term.strip().lower().split())
        return self._normalize_tokens(tokens)

    def _candidate_spans(self, tokens: list[TokenInfo], text: str) -> list[tuple[int, int, float]]:
        # tokens i and i+1 may be merged into a phrase only if nothing but whitespace
        # or a hyphen separates them in the source. Punctuation (commas, sentence
        # breaks) is dropped during parsing, so without this check words from
        # different clauses look adjacent and get glued into junk spans like
        # "WebSock. Ага, понятненько". Hyphens are allowed because taggers split
        # compounds ("объектно-ориентированное") into separate tokens, and dropping
        # the connection there would truncate the term to its tail.
        joinable = [
            text[tokens[i].end : tokens[i + 1].start].strip(" \t\n\r-") == ""
            for i in range(len(tokens) - 1)
        ]

        def contiguous(start: int, end: int) -> bool:
            return all(joinable[i] for i in range(start, end - 1))

        def is_glued(phrase: list[TokenInfo]) -> bool:
            # Two+ standalone capitalized words in a row are separate dictated
            # utterances Whisper capitalized, which the tagger mis-parses into a phrase
            # (e.g. "Полиморфизм Полиморфизм", "Наследование Полиморфизм"). Not a term.
            return sum(1 for item in phrase if item.is_titlecase) >= 2

        spans: list[tuple[int, int, float]] = []
        for index, token in enumerate(tokens):
            # Known technical terms are highlighted regardless of case/part of speech,
            # so single common nouns like "полиморфизм" are caught (grammar alone
            # cannot tell them from non-terms).
            if token.in_lexicon:
                spans.append((index, index + 1, 0.95))
            elif token.is_noun or token.is_tech:
                # Prefer the technical signal: a lone technical token/acronym (Python,
                # API, PostgreSQL) is a strong term even when tagged PROPN, while a
                # bare common noun is weak and should fall below the threshold.
                # A proper noun scores like a technical token — product and library
                # names (Python, React, Kubernetes) are exactly what we want to catch,
                # and in English prose no foreign-script signal is available to do it.
                # Titlecase Cyrillic is excluded: there PROPN usually just means the
                # word started a sentence.
                if token.is_tech or (token.is_propn and not token.is_titlecase):
                    spans.append((index, index + 1, 0.7))
                else:
                    spans.append((index, index + 1, 0.62))

            if index + 1 < len(tokens) and contiguous(index, index + 2):
                first, second = token, tokens[index + 1]
                if not is_glued([first, second]):
                    if first.is_adj and second.is_noun:
                        spans.append((index, index + 2, 0.86))
                    if first.is_noun and second.is_noun and second.is_genitive:
                        spans.append((index, index + 2, 0.9))
                    if first.is_tech and second.is_noun:
                        spans.append((index, index + 2, 0.82))
                    if first.is_noun and second.is_tech:
                        spans.append((index, index + 2, 0.82))
                    # Two adjacent foreign-script tokens inside otherwise Russian
                    # speech are nearly always one borrowed multi-word term
                    # ("dependency injection", "event loop"). Taggers label such
                    # tokens X, so neither noun rule above can catch the pair.
                    # Keyed on the script rather than on is_tech: within English
                    # prose no word is foreign, which stops "Today we"/"we will"
                    # from gluing into pairs.
                    if first.script_is_foreign and second.script_is_foreign:
                        spans.append((index, index + 2, 0.82))

            if index + 2 < len(tokens) and contiguous(index, index + 3):
                first, second, third = token, tokens[index + 1], tokens[index + 2]
                if not is_glued([first, second, third]):
                    if first.is_adj and second.is_noun and third.is_noun and third.is_genitive:
                        spans.append((index, index + 3, 0.94))
                    if first.is_noun and second.is_noun and third.is_tech:
                        spans.append((index, index + 3, 0.88))

            max_end = min(len(tokens), index + self._settings.term_max_words)
            for end in range(index + 3, max_end + 1):
                if not contiguous(index, end):
                    break
                phrase = tokens[index:end]
                # Keep boundaries tight: the whole span must be content words
                # (noun/adj/tech) and end on a noun or tech token, so we don't drag
                # in prepositions, verbs or pronouns and blow up the phrase.
                if not (phrase[-1].is_noun or phrase[-1].is_tech):
                    continue
                if not all(item.is_noun or item.is_adj or item.is_tech for item in phrase):
                    continue
                if is_glued(phrase):
                    continue
                # A run of adjectives closed by a noun is a plain noun phrase
                # ("объектно-ориентированное программирование"). Taggers that split
                # hyphenated compounds turn the two-token ADJ+NOUN rule above into a
                # three-token ADJ+ADJ+NOUN one, which would otherwise be dropped here
                # for containing no technical token.
                if all(item.is_adj for item in phrase[:-1]) and phrase[-1].is_noun:
                    spans.append((index, end, 0.86))
                if any(item.is_tech for item in phrase) and any(item.is_noun for item in phrase):
                    spans.append((index, end, 0.8))
        return spans

    def _normalize_tokens(self, tokens: list[TokenInfo]) -> str:
        parts: list[str] = []
        for index, token in enumerate(tokens):
            if token.is_tech:
                parts.append(token.text)
            elif token.is_prep:
                parts.append(token.lower)
            elif token.is_adj:
                parts.append(token.lemma)
            elif index == 0 and token.is_noun:
                parts.append(token.lemma)
            elif token.is_noun and token.is_genitive:
                parts.append(token.lower)
            else:
                parts.append(token.lemma if token.lemma else token.lower)
        return " ".join(parts)

    def _is_allowed(self, normalized: str, tokens: list[TokenInfo]) -> bool:
        if len(normalized) < self._settings.term_min_chars:
            return False
        if normalized in STOPWORDS:
            return False
        if len(tokens) == 1 and tokens[0].lower in STOPWORDS:
            return False
        # A single token must be a noun/technical token — unless it's a known term
        # from the lexicon, which the tagger sometimes mis-tags (e.g. "полиморфизм"
        # as ADJ).
        if len(tokens) == 1 and not (
            tokens[0].is_noun or tokens[0].is_tech or tokens[0].in_lexicon
        ):
            return False
        return True

    @staticmethod
    def _deduplicate_nested(terms: list[Term]) -> list[Term]:
        ordered = sorted(
            terms,
            key=lambda item: (item.start, -(item.end - item.start), -item.confidence),
        )
        result: list[Term] = []
        for term in ordered:
            nested = any(term.start >= kept.start and term.end <= kept.end for kept in result)
            if not nested:
                result.append(term)
        return sorted(result, key=lambda item: item.start)


def build_token(
    *,
    text: str,
    lemma: str,
    start: int,
    end: int,
    pos: str,
    feats: dict[str, str],
    document_is_cyrillic: bool,
) -> TokenInfo:
    """Assemble a ``TokenInfo``, deciding the foreign-script flag from the document.

    A Latin token counts as foreign only inside a Cyrillic document, and vice
    versa — that is what stops every English word from looking technical.
    """
    if document_is_cyrillic:
        script_is_foreign = bool(LATIN_RE.search(text))
    else:
        script_is_foreign = bool(CYRILLIC_RE.search(text))
    return TokenInfo(
        text=text,
        lemma=lemma,
        start=start,
        end=end,
        pos=pos,
        feats=feats,
        script_is_foreign=script_is_foreign,
    )


def document_is_cyrillic(text: str) -> bool:
    """Whether the document should be treated as Russian.

    Presence of Cyrillic decides it rather than which script has more letters: a
    Russian sentence about English terms ("паттерн dependency injection") is
    mostly Latin by character count, while English prose has no Cyrillic at all.
    """
    return bool(CYRILLIC_RE.search(text))


__all__ = [
    "ACRONYM_RE",
    "CAMEL_CASE_RE",
    "CYRILLIC_RE",
    "LATIN_RE",
    "STOPWORDS",
    "TECH_CHARS",
    "RuleBasedTermExtractor",
    "TokenInfo",
    "build_token",
    "document_is_cyrillic",
]
