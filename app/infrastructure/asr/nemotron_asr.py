from __future__ import annotations

from typing import Any

import anyio
import numpy as np
from loguru import logger

from app.core.config import Settings
from app.domain.interfaces.asr import AsrResult, AsrStream
from app.infrastructure.asr.hf_download import prefetch_nemotron_weights
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
        # Pre-fetch the .nemo checkpoint with a visible progress bar before NeMo's
        # own (unreliable-in-process) download runs. If the backend is started
        # directly without the launcher's preload step, this is the only place the
        # ~2.4 GB download is observable. from_pretrained then reads from the cache.
        prefetch_nemotron_weights(model_id)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        model.eval()
        if self._settings.whisper_device.strip().lower() == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        self._apply_lookahead(model)

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

    def _apply_lookahead(self, model: Any) -> None:
        """Select the encoder's attention lookahead (NEMOTRON_LOOKAHEAD_MS).

        The checkpoint is multi-latency: it ships several supported [left, right]
        attention contexts (right = lookahead in encoder frames, 80 ms each) and
        defaults to the low-latency one. More lookahead markedly improves rare
        terms and punctuation, so pick the supported context matching the
        configured milliseconds; anything unsupported keeps the model default.
        """
        encoder = model.encoder
        supported = getattr(encoder, "att_context_size_all", None)
        if not supported:
            return
        stride_s = float(model.cfg.preprocessor.window_stride)  # mel frame, 0.01 s
        frame_ms = stride_s * 1000 * int(model.cfg.encoder.subsampling_factor)
        right = round(self._settings.nemotron_lookahead_ms / frame_ms)
        match = next((ctx for ctx in supported if int(ctx[1]) == right), None)
        if match is None:
            options = sorted(int(ctx[1] * frame_ms) for ctx in supported)
            logger.warning(
                "NEMOTRON_LOOKAHEAD_MS={} is not supported by {}; keeping default {}. "
                "Supported: {} ms",
                self._settings.nemotron_lookahead_ms,
                self._settings.whisper_model,
                encoder.att_context_size,
                options,
            )
            return
        # Also recomputes streaming_cfg (chunk/shift sizes) for the new context.
        encoder.set_default_att_context_size(list(match))
        logger.info(
            "Nemotron lookahead set to {} ms (att_context_size={})",
            int(right * frame_ms),
            list(match),
        )

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
        # per_feature normalization needs stats over the whole utterance— in
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
        # There is exactly one stream per WebSocket connection. NeMo's
        # CacheAwareStreamingAudioBuffer.append_audio returns stream_id=-1 on the
        # *first* append (it forgets to set stream_id=0 in the buffer-is-None
        # branch), so blindly trusting the return value makes the second append
        # open a NEW stream — growing the batch dim to 2 while the attention cache
        # stays batch=1, which crashes conformer_stream_step with "Sizes of
        # tensors must match except in dimension 1". Pin stream 0 and ignore the
        # bogus first-call return value.
        self._stream_id = 0
        self._stream_initialized = False
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
        # First append must pass stream_id=-1 so the buffer creates the stream;
        # NeMo returns -1 here instead of 0, so we don't store the return value.
        # Subsequent appends use the pinned stream_id=0 to append to that stream.
        append_id = -1 if not self._stream_initialized else self._stream_id
        self._buffer.append_audio(samples, stream_id=append_id)
        self._stream_initialized = True
        # Drain only COMPLETE encoder chunks. The buffer's own iterator yields as
        # soon as `sampling_frames` (8) frames are available but advances
        # buffer_idx by the FULL shift_size (32) regardless — fine for offline
        # files (only the very last chunk is short), fatal for a live mic: every
        # partial yield hands the encoder an undersized chunk that breaks the
        # chunked-attention alignment AND skips the not-yet-arrived frames the
        # pointer jumped over. In practice the transcript degrades within seconds
        # and the decoder then goes permanently silent. Gating on a full chunk
        # keeps buffer_idx <= streams_length, so every yielded chunk is complete;
        # the incomplete tail waits for the next mic chunk.
        while self._available_frames() >= self._full_chunk_frames():
            chunk_audio, chunk_lengths = next(iter(self._buffer))
            self._stream_step(chunk_audio, chunk_lengths)
        return self._text

    def _available_frames(self) -> int:
        return int(self._buffer.streams_length[self._stream_id]) - self._buffer.buffer_idx

    def _full_chunk_frames(self) -> int:
        # streaming_cfg.chunk_size is [first_chunk, other_chunks] for models whose
        # first chunk carries its own pre-encode context (no left cache yet).
        size = self._buffer.streaming_cfg.chunk_size
        if isinstance(size, (list, tuple)):
            return int(size[0]) if self._buffer.buffer_idx == 0 else int(size[1])
        return int(size)

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
                # keep_all_outputs MUST be False for continuous realtime streaming.
                # The official NeMo example passes buf.is_buffer_empty(), which is
                # only correct for OFFLINE file transcription: the buffer is fully
                # loaded upfront there, so is_buffer_empty() flips to False after the
                # first chunk and outputs get truncated to valid_out_len on the rest.
                # For a live mic the buffer keeps growing indefinitely (never "empty"),
                # so that flag stays True forever — the encoder then never truncates
                # (att_context_style="chunked_limited" only truncates when
                # keep_all_outputs is False), the attention cache grows without bound,
                # and decoding silently stalls after the first few words. Forcing
                # False keeps the cache capped at last_channel_cache_size.
                keep_all_outputs=False,
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
