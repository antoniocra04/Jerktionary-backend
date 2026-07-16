from __future__ import annotations

from typing import Any

import anyio
import numpy as np
from loguru import logger

from app.core.config import Settings
from app.domain.interfaces.asr import AsrResult, AsrStream
from app.infrastructure.asr.lang_tags import language_tag

_HF_NAMESPACE = "nvidia"
_SENTENCE_END = (".", "!", "?", "…")
# Cap on the volatile tail so a long run without punctuation doesn't make every
# cycle re-parse unbounded text downstream.
_MAX_VOLATILE_CHARS = 240


def commit_boundary(text: str) -> int:
    """Length of the prefix that is safe to freeze for incremental term extraction.

    Greedy RNN-T streaming output is append-only (emitted tokens never change), but
    the current sentence is still growing, and freezing mid-sentence would split a
    term between two extraction passes. Freeze up to the last sentence-final
    punctuation mark (Nemotron punctuates natively), capped by _MAX_VOLATILE_CHARS.
    """
    boundary = 0
    for mark in _SENTENCE_END:
        index = text.rfind(mark)
        if index + 1 > boundary:
            boundary = index + 1
    if len(text) - boundary > _MAX_VOLATILE_CHARS:
        boundary = len(text) - _MAX_VOLATILE_CHARS
    return boundary


def _import_nemo() -> Any:
    """Deferred so the backend runs without the multi-GB optional dependency unless
    the Nemotron model is actually selected."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron ASR requires the optional dependencies; "
            'install them with: pip install -e ".[nemotron]"'
        ) from exc
    return nemo_asr


class NemotronAsrProvider:
    """NVIDIA Nemotron cache-aware streaming ASR (NeMo) as a local realtime provider.

    Unlike faster-whisper there is no buffer re-decoding: each mic chunk passes
    through the encoder exactly once with cached context, and the RNN-T decoder
    emits final tokens as speech arrives — with native punctuation and
    capitalization. Selected via WHISPER_PROVIDER=local + WHISPER_MODEL=nemotron-*.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None

    async def load(self) -> None:
        await anyio.to_thread.run_sync(self._load_model)

    def _load_model(self) -> None:
        nemo_asr = _import_nemo()
        import torch

        model_id = f"{_HF_NAMESPACE}/{self._settings.whisper_model.strip().lower()}"
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        model.eval()
        if self._settings.whisper_device.strip().lower() == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")

        # Multilingual checkpoints (nemotron-3.5-*) take the target language as an
        # inference prompt and prefix the output with a language tag.
        if hasattr(model, "set_inference_prompt"):
            model.set_inference_prompt(language_tag(self._settings.asr_language))
            decoding = getattr(model, "decoding", None)
            if decoding is not None and hasattr(decoding, "set_strip_lang_tags"):
                try:
                    decoding.set_strip_lang_tags(True)
                except TypeError:
                    # Older/newer NeMo may require the tag pattern explicitly; a
                    # visible tag in the transcript is cosmetic, not fatal.
                    logger.warning("Could not enable language-tag stripping")
        self._model = model

    async def create_stream(self) -> AsrStream:
        if self._model is None:
            raise RuntimeError("Nemotron model is not loaded")
        return NemotronStreamingAsrStream(self._model, self._settings)


class NemotronStreamingAsrStream:
    """One WebSocket connection's live NeMo streaming session (batch of 1).

    ``append_pcm`` feeds samples into NeMo's ``CacheAwareStreamingAudioBuffer``;
    whenever at least one full encoder chunk is ready it runs
    ``conformer_stream_step`` with the running cache state (mirrors NeMo's official
    cache-aware streaming example). The step returns the whole transcription so
    far, so the AsrResult carries the full text with a sentence-aware committed
    prefix (see ``commit_boundary``).
    """

    def __init__(self, model: Any, settings: Settings) -> None:
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        self._model = model
        self._settings = settings
        # per_feature normalization needs stats over the whole utterance — in
        # streaming that must be approximated online, chunk by chunk.
        normalize = str(model.cfg.preprocessor.get("normalize", ""))
        self._buffer = CacheAwareStreamingAudioBuffer(
            model=model, online_normalization=(normalize == "per_feature")
        )
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)
        self._stream_id = -1
        self._step = 0
        self._previous_hypotheses: Any | None = None
        self._pred_out_stream: Any | None = None
        self._text = ""
        self._committed_len = 0
        self._last_emitted = ""

    async def append_pcm(self, chunk: bytes) -> AsrResult | None:
        if not chunk:
            return None
        text = await anyio.to_thread.run_sync(self._process, chunk)
        if not text or text == self._last_emitted:
            return None
        self._last_emitted = text
        # Monotonic: the committed prefix must never move backwards.
        self._committed_len = max(self._committed_len, commit_boundary(text))
        return AsrResult(
            text=text,
            is_final=False,
            committed_len=min(self._committed_len, len(text)),
        )

    async def close(self) -> None:
        self._buffer.reset_buffer()

    def _process(self, chunk: bytes) -> str:
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        _, _, self._stream_id = self._buffer.append_audio(samples, stream_id=self._stream_id)
        # Drain every encoder chunk that is now complete; the buffer keeps the
        # incomplete remainder for the next mic chunk.
        for chunk_audio, chunk_lengths in self._buffer:
            self._stream_step(chunk_audio, chunk_lengths)
        return self._text

    def _stream_step(self, chunk_audio: Any, chunk_lengths: Any) -> None:
        import torch

        with torch.inference_mode():
            (
                self._pred_out_stream,
                transcribed_texts,
                self._cache_last_channel,
                self._cache_last_time,
                self._cache_last_channel_len,
                self._previous_hypotheses,
            ) = self._model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=self._buffer.is_buffer_empty(),
                previous_hypotheses=self._previous_hypotheses,
                previous_pred_out=self._pred_out_stream,
                drop_extra_pre_encoded=self._drop_extra_pre_encoded(),
                return_transcription=True,
            )
        self._step += 1
        if transcribed_texts:
            hypothesis = transcribed_texts[0]
            self._text = str(getattr(hypothesis, "text", hypothesis)).strip()

    def _drop_extra_pre_encoded(self) -> int:
        # First chunk carries its own pre-encode context; later chunks reuse the
        # cache and must drop the overlapping pre-encoded frames.
        if self._step == 0:
            return 0
        return int(self._model.encoder.streaming_cfg.drop_extra_pre_encoded)
