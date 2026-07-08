from __future__ import annotations

import os

# huggingface_hub (used by faster-whisper to fetch model weights) disables its
# tqdm progress bar by default in recent versions, so a multi-hundred-MB
# download looks like a silent hang. FORCE it on (a plain assignment, not
# setdefault — the env var may already be set to "1" by the hub or the shell,
# and setdefault wouldn't override that). Must run before the hub is imported.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from app.core.config import Settings  # noqa: E402

# faster-whisper's model id → Hugging Face repo. large-v3-turbo is a special-case
# alias; everything else maps to the Systran/faster-whisper-<name> repo. We use
# this to pre-download weights with an explicit progress bar.
_WHISPER_REPO = {
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
    "tiny": "Systran/faster-whisper-tiny",
}


def _prefetch_weights(model_name: str) -> None:
    """Pre-download the model weights with a visible progress bar.

    faster-whisper's WhisperModel() downloads weights lazily via huggingface_hub,
    whose tqdm bar is unreliable in subprocess contexts (silent on macOS). Calling
    snapshot_download() first — with progress bars forced on — makes the download
    visible. If the repo id is unknown or the hub isn't installed, we silently
    fall back to faster-whisper's own (possibly silent) download.
    """
    repo = _WHISPER_REPO.get(model_name)
    if repo is None:
        return
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        return
    # allow_patterns limits the fetch to the model files (skip README, etc.).
    # If the repo is unavailable (404, network error, etc.), silently continue —
    # faster-whisper's WhisperModel() will download weights on its own.
    try:
        snapshot_download(repo, allow_patterns=["*.bin", "*.json", "*.txt", "*.model"])
    except Exception:
        pass


def main() -> int:
    settings = Settings()
    print(
        "==> checking Whisper model "
        f"{settings.whisper_model} on {settings.whisper_device}/{settings.whisper_compute_type}",
        flush=True,
    )
    print("==> downloading weights on first run — this can take several minutes", flush=True)

    _prefetch_weights(settings.whisper_model)

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
    raise SystemExit(main())
