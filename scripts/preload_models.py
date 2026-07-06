from __future__ import annotations

from app.core.config import Settings


def main() -> int:
    settings = Settings()
    print(
        "==> checking Whisper model "
        f"{settings.whisper_model} on {settings.whisper_device}/{settings.whisper_compute_type}",
        flush=True,
    )

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
        return 1

    print("OK  Whisper model ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
