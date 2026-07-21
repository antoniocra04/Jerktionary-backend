from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from app.core.config import Settings
from app.infrastructure.nlp.rule_based_extractor import (
    RuleBasedTermExtractor,
    TokenInfo,
    build_token,
    document_is_cyrillic,
)

# Only tagging and lemmas feed the rule layer. The dependency parser and NER are
# the two slowest pipes and nothing downstream reads them, so they are excluded at
# load time rather than merely disabled — that keeps them out of memory too.
_UNUSED_PIPES = ("parser", "ner", "senter")


def _import_spacy() -> Any:
    """Deferred so the backend still starts when the optional models are absent."""
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy term extraction requires the optional dependencies; "
            'install them with: pip install -e ".[spacy]" '
            "&& python -m spacy download ru_core_news_md "
            "&& python -m spacy download en_core_web_md"
        ) from exc
    return spacy


class SpacyTermExtractor(RuleBasedTermExtractor):
    """spaCy morphology behind the shared rule layer, with per-text language routing.

    Unlike Natasha this covers English as a first-class language: English tokens get
    real parts of speech and lemmas, so ordinary words ("the", "and", "are") are
    rejected by the same grammar rules that already filter Russian ones, and English
    multi-word terms ("dependency injection") merge through the noun-phrase rules.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._ru: Any | None = None
        self._en: Any | None = None
        # spaCy pipelines are not re-entrant; extract runs in a worker thread.
        self._lock = threading.Lock()

    async def load(self) -> None:
        spacy = _import_spacy()
        try:
            self._ru = spacy.load(self._settings.spacy_model_ru, exclude=list(_UNUSED_PIPES))
            self._en = spacy.load(self._settings.spacy_model_en, exclude=list(_UNUSED_PIPES))
        except OSError as exc:
            raise RuntimeError(
                f"spaCy models not installed ({exc}). Install them with: "
                f"python -m spacy download {self._settings.spacy_model_ru} && "
                f"python -m spacy download {self._settings.spacy_model_en}"
            ) from exc
        logger.info(
            "spaCy loaded: {} + {}",
            self._settings.spacy_model_ru,
            self._settings.spacy_model_en,
        )

    def _parse(self, text: str) -> list[TokenInfo]:
        if self._ru is None or self._en is None:
            raise RuntimeError("spaCy is not loaded")

        is_cyrillic = document_is_cyrillic(text)
        nlp = self._ru if is_cyrillic else self._en

        with self._lock:
            doc = nlp(text)

        result: list[TokenInfo] = []
        for token in doc:
            if token.is_space or token.is_punct or not any(ch.isalnum() for ch in token.text):
                continue
            result.append(
                build_token(
                    text=token.text,
                    lemma=(token.lemma_ or token.text).lower(),
                    start=token.idx,
                    end=token.idx + len(token.text),
                    pos=token.pos_,
                    # morph is spaCy's UD feature bundle ({'Case': 'Gen', …}) — the
                    # rule layer reads Case=Gen from it exactly as it read Natasha's.
                    feats=token.morph.to_dict(),
                    document_is_cyrillic=is_cyrillic,
                )
            )
        return result
