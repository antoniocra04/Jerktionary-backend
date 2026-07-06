from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anyio
import numpy as np

from app.core.config import Settings
from app.domain.interfaces.asr import AsrStream
from app.infrastructure.asr.buffering_stream import BufferingAsrStream


def _register_cuda_dll_directories() -> None:
    """Make CUDA runtime DLLs from the ``nvidia-*-cu12`` wheels loadable on Windows.

    CTranslate2 bundles cuDNN but not cuBLAS/cudart. Those ship as separate
    ``nvidia-cublas-cu12`` / ``nvidia-cuda-runtime-cu12`` wheels whose ``bin``
    directories are not on the DLL search path, so CTranslate2's own
    ``LoadLibrary("cublas64_12.dll")`` fails with "not found or cannot be loaded".
    Prepending them to ``PATH`` (honored by the standard DLL search order, unlike
    ``os.add_dll_directory``) fixes it. Must run before ``faster_whisper`` /
    ``ctranslate2`` is first imported.
    """
    if os.name != "nt":
        return
    try:
        import nvidia  # type: ignore[import-untyped]
    except ImportError:
        return

    bin_dirs: list[str] = []
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            if bin_dir.is_dir():
                bin_dirs.append(str(bin_dir))
                os.add_dll_directory(str(bin_dir))

    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


class FasterWhisperAsrProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None

    async def load(self) -> None:
        _register_cuda_dll_directories()
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        self._model = await anyio.to_thread.run_sync(
            lambda: WhisperModel(
                self._settings.whisper_model,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
        )

    async def create_stream(self) -> AsrStream:
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded")
        return BufferingWhisperStream(self._model, self._settings)


class BufferingWhisperStream(BufferingAsrStream):
    """Local faster-whisper decoding on top of the shared buffering pipeline."""

    def __init__(self, model: Any, settings: Settings) -> None:
        super().__init__(settings)
        self._model = model

    async def _decode(self) -> list[tuple[float, float, str]]:
        return await anyio.to_thread.run_sync(self._decode_segments)

    def _decode_segments(self) -> list[tuple[float, float, str]]:
        audio = np.frombuffer(self._pcm_buffer(), dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language=self._settings.asr_language,
            beam_size=self._settings.asr_beam_size,
            vad_filter=False,
            # The same buffer is re-decoded every cycle; feeding the previous decode
            # back in (the default) makes Whisper lock onto its own hallucinations.
            condition_on_previous_text=False,
            initial_prompt=self._settings.asr_initial_prompt or None,
        )
        return [(float(seg.start), float(seg.end), seg.text.strip()) for seg in segments]
