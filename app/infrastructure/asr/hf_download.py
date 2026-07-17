from __future__ import annotations

import logging
import os
from typing import Any

from loguru import logger
from tqdm.auto import tqdm as _BaseTqdm

# huggingface_hub (used by faster-whisper AND nemo_toolkit to fetch model weights)
# ships its own `tqdm` subclass and force-disables it when:
#   • HF_HUB_DISABLE_PROGRESS_BARS is set (the hub sets it to "1" in some envs);
#   • stdout/stderr is not a TTY (the case in a launched backend subprocess);
#   • the logging level is NOTSET.
# On top of that, faster-whisper's own `download_model` passes `tqdm_class=disabled_tqdm`,
# so its bars are permanently off regardless of env vars. The net effect is that a
# multi-hundred-MB / multi-GB first-run download looks like a silent hang.
#
# Fix: pass our OWN tqdm subclass (which does NOT derive from the hub's tqdm) to
# `snapshot_download(..., tqdm_class=...)`. The hub's progress-bar factory only
# injects `disable=True` into subclasses of its own tqdm — for any other class it
# calls `cls(**kwargs)` verbatim, so our bar is shown as long as the stream is a
# TTY. Forcing it on for non-TTY too would spam logs, so we keep auto-detection
# there and rely on the launcher (which IS a TTY) calling preload first.
#
# Must be importable without the optional [nemotron] deps — NeMo is only imported
# lazily inside NemotronAsrProvider.

# faster-whisper model alias → Hugging Face repo. Mirrors faster_whisper.utils._MODELS
# for the sizes the launcher menu offers. large-v3-turbo lives under mobiuslabsgmbh,
# NOT Systran (the old hardcoded mapping 404'd on the Hub).
_WHISPER_REPO = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v1": "Systran/faster-whisper-large-v1",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
}

# File types that make up a faster-whisper (CTranslate2) checkpoint. Matching the
# allow-list faster-whisper itself uses, so we don't pull READMEs/preview images.
_WHISPER_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
    "*.txt",
    "*.model",
]


class VisibleTqdm(_BaseTqdm):  # type: ignore[misc]
    """A tqdm progress bar that is always shown when attached to a terminal.

    Subclasses plain tqdm (NOT huggingface_hub's tqdm), so the hub's progress-bar
    factory treats it as a foreign class and does not force `disable=True`. The
    `disable=False` default overrides any per-instance `disable` the hub might
    forward, giving a visible bar on the launcher's TTY even when
    HF_HUB_DISABLE_PROGRESS_BARS=1.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Hard-disable off. The hub forwards `disable=<bool>` for its own subclass
        # path only, but some callers pass it explicitly — drop it so our bar wins.
        kwargs.pop("disable", None)
        kwargs["disable"] = False
        super().__init__(*args, **kwargs)


def _snapshot_download(repo_id: str, allow_patterns: list[str] | None) -> str | None:
    """Download a HF repo snapshot with a guaranteed-visible progress bar.

    Returns the local snapshot path, or None if the hub isn't installed / the repo
    is unreachable — callers fall back to the library's own (possibly silent)
    download in that case, exactly as before.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed; skipping pre-download for {}", repo_id)
        return None

    # Force the hub's global toggle off too (belt-and-suspenders for nested calls
    # that ignore tqdm_class, e.g. metadata fetches). A plain assignment, not
    # setdefault — the hub/shell may have already set it to "1".
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

    try:
        return snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            tqdm_class=VisibleTqdm,
        )
    except Exception as exc:
        # 404 / network error / gated repo — non-fatal. The downstream loader will
        # retry and raise a clearer error if the model is genuinely unavailable.
        logger.warning("pre-download failed for {} ({}); will download at load time", repo_id, exc)
        return None


def prefetch_whisper_weights(model_name: str) -> None:
    """Pre-download faster-whisper weights with a visible progress bar.

    faster-whisper's WhisperModel() routes through `download_model`, which hard-codes
    `tqdm_class=disabled_tqdm` — so its own bar is permanently invisible. Calling
    snapshot_download() first with our VisibleTqdm makes the multi-hundred-MB fetch
    observable; WhisperModel() then finds the files cached and skips re-downloading.
    """
    repo = _whisper_repo(model_name)
    if repo is None:
        return
    _snapshot_download(repo, _WHISPER_ALLOW_PATTERNS)


def prefetch_nemotron_weights(model_id: str) -> None:
    """Pre-download the NVIDIA NeMo checkpoint (~2.4 GB) with a visible progress bar.

    NeMo's `ASRModel.from_pretrained` downloads the `.nemo` archive via the Hub but,
    like faster-whisper, its progress reporting is unreliable in a launched process.
    Fetching just the `.nemo` file up front (skipping the duplicate safetensors copy,
    README, and PNG diagrams that together nearly double the download) gives a
    visible bar; from_pretrained then resolves the cached artifact without re-fetch.
    """
    # Only the .nemo checkpoint is consumed by nemo_toolkit at load time; the
    # safetensors mirror (another ~2.5 GB) is for non-NeMo inference paths.
    _snapshot_download(model_id, ["*.nemo"])


def _whisper_repo(model_name: str) -> str | None:
    """Resolve a faster-whisper size alias (or raw repo id) to a HF repo id."""
    name = model_name.strip().lower()
    if _WHISPER_REPO.get(name):
        return _WHISPER_REPO[name]
    # A literal "org/repo" id is passed straight through.
    if "/" in model_name:
        return model_name.strip()
    logger.warning("unknown Whisper model '{}'; skipping pre-download", model_name)
    return None


# Keep the hub's NOTSET-log-level check from suppressing our bar: the factory only
# forces disable when log_level == NOTSET, and NeMo/faster-whisper may not have
# configured logging yet at import time. Pinning the root logger level to INFO
# (the project default) ensures our bar is treated as eligible.
if logging.getLogger().level == logging.NOTSET:
    logging.getLogger().setLevel(logging.INFO)
