from __future__ import annotations

import sys

from app.core.config import Settings  # noqa: E402
from app.infrastructure.asr.hf_download import (  # noqa: E402
    prefetch_nemotron_weights,
    prefetch_whisper_weights,
)


def _preload_nemotron(model_name: str) -> int:
    """Pre-download the NVIDIA NeMo checkpoint (~2.4 GB) with a visible progress
    bar. Loading/validating the model happens at backend startup — importing the
    NeMo stack here would double the multi-second import cost."""
    model_id = f"nvidia/{model_name}"
    print(f"==> checking Nemotron model {model_id}", flush=True)
    print("==> downloading weights on first run — this can take several minutes", flush=True)
    prefetch_nemotron_weights(model_id)
    print("OK  Nemotron model ready", flush=True)
    return 0


def main() -> int:
    settings = Settings()
    provider = settings.whisper_provider.strip().lower()
    if not settings.whisper_enabled or provider != "local":
        # API/Yandex transcription (or disabled ASR) needs no multi-GB local model.
        reason = "WHISPER_ENABLED=false" if not settings.whisper_enabled else f"provider={provider}"
        print(f"==> skipping local Whisper preload ({reason})", flush=True)
        return 0
    if settings.whisper_model.strip().lower().startswith("nemotron"):
        return _preload_nemotron(settings.whisper_model.strip().lower())
    print(
        "==> checking Whisper model "
        f"{settings.whisper_model} on {settings.whisper_device}/{settings.whisper_compute_type}",
        flush=True,
    )
    print("==> downloading weights on first run — this can take several minutes", flush=True)

    # Pre-fetch weights with a visible progress bar (faster-whisper's own bar is
    # hard-disabled, so this is the only place the download is observable).
    prefetch_whisper_weights(settings.whisper_model)

    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    except Exception as exc:
        print("ERROR Whisper model failed to load on configured device.", flush=True)
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        print(
            "ERROR Check NVIDIA driver/CUDA runtime for faster-whisper, "
            "or set WHISPER_DEVICE=cpu in .env if you need a CPU fallback.",
            flush=True,
        )
        print(
            "ERROR On Windows with NVIDIA GPU, install CUDA runtime first: "
            "pip install -e '.[cuda]'",
            flush=True,
        )
        return 1

    print("OK  Whisper model ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
