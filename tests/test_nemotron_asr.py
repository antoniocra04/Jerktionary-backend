from __future__ import annotations

from app.infrastructure.asr.lang_tags import language_tag
from app.infrastructure.asr.nemotron_asr import (
    _MAX_VOLATILE_CHARS,
    NemotronStreamingAsrStream,
    commit_boundary,
)


class _FakeStreamingCfg:
    chunk_size = [25, 32]


class _FakeBuffer:
    """Mimics NeMo's CacheAwareStreamingAudioBuffer partial-yield semantics:
    the iterator yields as soon as >= 8 frames are buffered but advances
    buffer_idx by the FULL chunk size — the exact behavior _process must gate."""

    frames_per_append = 26  # what 250 ms of 16 kHz mic audio produces

    def __init__(self) -> None:
        self.streaming_cfg = _FakeStreamingCfg()
        self.buffer_idx = 0
        self.streams_length = [0]

    def append_audio(self, samples: object, stream_id: int) -> None:
        self.streams_length[0] += self.frames_per_append

    def _chunk_size(self) -> int:
        return self.streaming_cfg.chunk_size[0 if self.buffer_idx == 0 else 1]

    def __iter__(self):  # noqa: ANN204 - mirrors NeMo's untyped generator
        while self.buffer_idx < self.streams_length[0]:
            size = self._chunk_size()
            available = self.streams_length[0] - self.buffer_idx
            if available < 8:  # NeMo's sampling_frames floor
                return
            # NeMo advances the pointer BEFORE yielding — _process relies on that
            # when it pulls a single chunk via next(iter(buffer)).
            self.buffer_idx += size
            yield min(size, available), None


def test_process_only_drains_full_encoder_chunks() -> None:
    stream = object.__new__(NemotronStreamingAsrStream)
    buffer = _FakeBuffer()
    stream._buffer = buffer
    stream._stream_id = 0
    stream._stream_initialized = False
    stream._text = ""
    yielded: list[int] = []
    stream._stream_step = lambda chunk_audio, chunk_lengths: yielded.append(chunk_audio)

    for _ in range(20):
        stream._process(b"\x00\x00" * 4000)
        # The pointer must never run past the audio that has actually arrived.
        assert buffer.buffer_idx <= buffer.streams_length[0]

    # First chunk is the [0] size, every later chunk the full [1] size — never partial.
    assert yielded[0] == 25
    assert all(size == 32 for size in yielded[1:])
    assert len(yielded) > 1


def test_commit_boundary_freezes_up_to_last_sentence_end() -> None:
    text = "Первое предложение. Второе, ещё не закончен"
    boundary = commit_boundary(text)
    assert text[:boundary] == "Первое предложение."


def test_commit_boundary_without_punctuation_caps_volatile_tail() -> None:
    text = "слово " * 100  # 600 chars, no sentence-final punctuation
    boundary = commit_boundary(text)
    assert len(text) - boundary == _MAX_VOLATILE_CHARS


def test_commit_boundary_is_monotonic_as_text_grows() -> None:
    # RNN-T streaming text is append-only; the frozen prefix must never shrink.
    previous = 0
    text = ""
    for sentence in ["Привет!", " Как дела?", " Расскажи про кэш.", " И ещё немного слов"]:
        text += sentence
        boundary = commit_boundary(text)
        assert boundary >= previous
        previous = boundary


def test_language_tag_shared_mapping() -> None:
    assert language_tag("ru") == "ru-RU"
    assert language_tag("auto") == "auto"
