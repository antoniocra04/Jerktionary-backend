---
task: TASK-010
title: "Fix Windows backend crash: install CUDA DLLs when GPU is detected for local Whisper"
status: approved
gates: []
branch: "task/010-windows-cuda-dlls"
created: "2026-07-08"
spec_version: 1
---

# TASK-010 — Fix Windows backend crash with local Whisper (missing CUDA DLLs)

## User Story

As a Windows user with an NVIDIA GPU, I want the backend to start successfully
after I select local Whisper in the model picker, so that transcription works
out of the box without a cryptic DLL crash.

## Root Cause

```
backend.ps1: Run-Backend →
  Ensure-GpuSettings        → detects nvidia-smi, sets WHISPER_DEVICE=cuda ✓
  Ensure-Dependencies       → pip install -e "."   ← NO [cuda] extras!
  Select-Models             → user picks local Whisper
  Ensure-WhisperModel       → preload_models.py
    └─ WhisperModel(device="cuda")
         └─ CTranslate2 → LoadLibrary("cublas64_12.dll") → DLL NOT FOUND → CRASH
```

Fix: New `Install-CudaDeps` function installs `[cuda]` extras (cuBLAS + cudart wheels)
when `nvidia-smi` is detected on Windows. Plus warnings in the ASR provider and
error hints in `preload_models.py`.

## Acceptance Criteria

- [x] AC1: Windows+NVIDIA: after selecting local Whisper, CUDA extras installed, backend starts
- [x] AC2: Windows no GPU: `[cuda]` extras NOT installed (no unnecessary download)
- [x] AC3: CUDA DLLs installed but incompatible → `_register_cuda_dll_directories()` logs clear warning
- [x] AC4: `preload_models.py` error includes `pip install -e '.[cuda]'` hint
- [x] AC5: macOS/Linux `backend.sh` unchanged, no regression
- [x] AC6: `python -m pytest` passes (20 tests)

## Implementation Notes

- `5744f79` — Added `Install-CudaDeps` in `backend.ps1`, called in `Run-Backend` after `Ensure-GpuSettings`
- `d94c41f` — Added loguru warnings in `_register_cuda_dll_directories()` for missing/incompatible CUDA DLLs
- `18bf56f` — Added `pip install -e '.[cuda]'` hint to `preload_models.py` error message
- `bb13651` — Fixed mypy type ignore for optional `nvidia` import
- All tests pass (20/20), ruff clean, mypy clean
