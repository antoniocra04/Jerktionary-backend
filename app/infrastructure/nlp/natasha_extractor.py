from __future__ import annotations

import threading
from typing import Any

from app.core.config import Settings
from app.infrastructure.nlp.rule_based_extractor import (
    ACRONYM_RE,
    LATIN_RE,
    STOPWORDS,
    TECH_CHARS,
    RuleBasedTermExtractor,
    TokenInfo,
    build_token,
    document_is_cyrillic,
)

__all__ = [
    "ACRONYM_RE",
    "LATIN_RE",
    "STOPWORDS",
    "TECH_CHARS",
    "NatashaTermExtractor",
    "TokenInfo",
]


class NatashaTermExtractor(RuleBasedTermExtractor):
    """Natasha (Slovnet) morphology behind the shared rule layer. Russian only —
    Latin tokens get no part of speech and are judged by the technical heuristics."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._segmenter: Any | None = None
        self._morph_tagger: Any | None = None
        self._morph_vocab: Any | None = None
        self._doc_cls: Any | None = None
        # Natasha models are not safe to call from several threads at once; extract
        # runs in a worker thread, so guard the shared model access.
        self._lock = threading.Lock()

    async def load(self) -> None:
        from natasha import (  # type: ignore[import-untyped]
            Doc,
            MorphVocab,
            NewsEmbedding,
            NewsMorphTagger,
            Segmenter,
        )

        emb = NewsEmbedding()
        self._segmenter = Segmenter()
        self._morph_tagger = NewsMorphTagger(emb)
        self._morph_vocab = MorphVocab()
        self._doc_cls = Doc

    def _parse(self, text: str) -> list[TokenInfo]:
        if (
            self._segmenter is None
            or self._morph_tagger is None
            or self._morph_vocab is None
            or self._doc_cls is None
        ):
            raise RuntimeError("Natasha is not loaded")

        # Only segmentation + morphology + lemma are used downstream; dependency
        # syntax parsing (the slowest stage) was pure overhead and is intentionally
        # dropped.
        with self._lock:
            doc = self._doc_cls(text)
            doc.segment(self._segmenter)
            doc.tag_morph(self._morph_tagger)
            for token in doc.tokens:
                token.lemmatize(self._morph_vocab)

        # Natasha models Russian only, but the transcript is not guaranteed to be
        # Russian. Judging the script per document still matters: in English speech
        # every word is Latin, and treating those as foreign/technical would mark
        # the whole utterance as terms.
        is_cyrillic = document_is_cyrillic(text)

        result: list[TokenInfo] = []
        for token in doc.tokens:
            if not token.text.strip() or not any(ch.isalnum() for ch in token.text):
                continue
            result.append(
                build_token(
                    text=str(token.text),
                    lemma=str(token.lemma or token.text).lower(),
                    start=int(token.start),
                    end=int(token.stop),
                    pos=str(token.pos or ""),
                    feats=dict(token.feats or {}),
                    document_is_cyrillic=is_cyrillic,
                )
            )
        return result
