from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Realtime Terms Backend"
    app_version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:5173", "http://localhost:5173"]
    )

    sqlite_path: Path = Field(default=Path("data/app.sqlite3"))

    # Set WHISPER_ENABLED=false to skip loading the local Whisper model entirely —
    # for users who transcribe via an API provider (or don't need ASR on this box).
    whisper_enabled: bool = True
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    audio_sample_rate: int = 16_000
    audio_channels: int = 1
    audio_sample_width_bytes: int = 2
    asr_min_audio_seconds: float = 1.5
    asr_max_buffer_seconds: float = 30.0
    # Minimum amount of *new* audio to accumulate before re-transcribing the buffer.
    # Prevents re-running Whisper on every ~85 ms mic chunk (which saturates the
    # WS loop and makes the transcript lag further and further behind real speech).
    asr_transcribe_interval_seconds: float = 0.6
    # Interval for API-based transcription: every cycle is a paid network round trip,
    # so poll less often than the local model.
    asr_api_transcribe_interval_seconds: float = 2.0
    # Mic chunks quieter than this RMS (0..1) count as silence and are not added to
    # the buffer. Without this, silence while you pause keeps filling the 30 s window
    # and slides earlier speech out, making the transcript erase itself from the top.
    asr_silence_rms_threshold: float = 0.01
    # Live tail kept for re-decoding/correction. Whisper segments older than this
    # (measured from the end of the buffer) are "committed" to the permanent
    # transcript and their audio dropped, so long continuous speech no longer slides
    # earlier text out of the window and erases it.
    asr_commit_tail_seconds: float = 10.0
    asr_language: str = "ru"
    asr_beam_size: int = 1
    # Domain vocabulary hint fed to Whisper as initial_prompt: technical terms listed
    # here are recognized far more reliably. Override per-domain via ASR_INITIAL_PROMPT.
    asr_initial_prompt: str = ""

    llm_enabled: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Keep in sync with .env.example: backend.cmd -PullLlm pulls OLLAMA_MODEL from
    # .env (falling back to this default), so a mismatch leaves LLM disabled.
    ollama_model: str = "qwen3:8b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512
    llm_timeout_seconds: float = 30.0
    # Defaults for the external (API-key) LLM provider when the request doesn't
    # specify its own model/base URL. OpenAI-compatible chat completions endpoint.
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_keep_alive: str = "30m"
    ollama_num_ctx: int = 2048
    ollama_num_batch: int = 1024
    ollama_num_gpu: int = 999
    ollama_main_gpu: int = 0
    ollama_num_thread: int | None = None
    ollama_think: bool = False
    llm_context_chars: int = 500

    # Provider selection chosen at backend startup (interactive launcher writes
    # these into .env). Keys map to app.core.providers.LLM_PROVIDERS. The local
    # Ollama path uses ollama_base_url/ollama_model above; the API path uses the
    # three llm_api_* fields together with the provider's catalog defaults.
    llm_provider: str = "ollama"
    llm_api_key: str = ""
    llm_api_model: str = ""
    llm_api_base_url: str = ""

    # Whisper provider chosen at startup: "local" (faster-whisper on this box) or
    # "api" (OpenAI-compatible /audio/transcriptions). The api fields mirror the
    # LLM ones — empty means "use the OpenAI Whisper default".
    whisper_provider: str = "local"
    whisper_api_key: str = ""
    whisper_api_model: str = ""
    whisper_api_base_url: str = ""

    term_min_chars: int = 3
    term_max_words: int = 5
    term_confidence_threshold: float = 0.68


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
