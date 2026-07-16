from __future__ import annotations

# asr_language holds a bare Whisper code ("ru"); realtime providers (Yandex
# SpeechKit, NVIDIA Nemotron) want a BCP-47 locale tag. Codes not listed here
# fall back to the provider's automatic language detection.
_LANGUAGE_TAGS = {
    "ru": "ru-RU",
    "en": "en-US",
    "kk": "kk-KZ",
    "uz": "uz-UZ",
    "de": "de-DE",
    "fr": "fr-FR",
    "tr": "tr-TR",
    "he": "he-IL",
}


def language_tag(code: str) -> str:
    code = code.strip()
    if not code or code.lower() == "auto":
        return "auto"
    if "-" in code:
        return code
    return _LANGUAGE_TAGS.get(code.lower(), "auto")
