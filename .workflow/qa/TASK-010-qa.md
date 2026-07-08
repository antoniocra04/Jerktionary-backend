# QA Report — TASK-010

- Date: 2026-07-08
- Branch: task/010-windows-cuda-dlls
- Commit: bb1365165735279df40df89b2e1c153de90ed726
- Verdict: **PASS**

## Acceptance Criteria Verification

### AC1: On Windows with NVIDIA GPU (nvidia-smi exits 0), after selecting local Whisper and restarting the backend, the CUDA extras are installed and `WhisperModel(device="cuda")` loads successfully. Backend starts without DLL errors.

- Status: **PASS** (code inspection — live test requires Windows GPU hardware per spec test strategy)
- Evidence: The `Install-CudaDeps` function in `scripts/backend.ps1` (lines 358–370) runs `pip install --upgrade --quiet -e ".[cuda]"` when `nvidia-smi` is found and exits 0. It is called from `Run-Backend` (line 771) after `Ensure-GpuSettings` and before `Ensure-Dependencies`, so the CUDA wheels are installed before any Python code imports faster-whisper/CTranslate2.

### AC2: On Windows without NVIDIA GPU (nvidia-smi absent or exits non-zero), the `[cuda]` extras are NOT installed. No unnecessary ~500MB download for CPU-only users.

- Status: **PASS** (code inspection)
- Evidence: `Install-CudaDeps` returns early when `nvidia-smi` is not found (`Get-Command` fails) or exits non-zero (`$LASTEXITCODE -ne 0`).

### AC3: If CUDA DLLs are installed but `WhisperModel` still fails (e.g. driver version mismatch, TCC mode), `_register_cuda_dll_directories()` logs a clear warning explaining that the CUDA runtime was found but may be incompatible, instead of silently returning.

- Status: **PASS** (code inspection)
- Evidence: `faster_whisper_asr.py` now has two `logger.warning` calls:
  - **ImportError path**: warns that CUDA runtime packages are not installed and suggests `pip install -e ".[cuda]"`
  - **Empty bin_dirs path**: warns that wheels are installed but no DLL directories found, runtime may be incompatible

### AC4: `preload_models.py` exit code 1 path includes a concrete hint: "Install CUDA runtime: pip install -e '.[cuda]'" alongside the existing "set WHISPER_DEVICE=cpu in .env" fallback, so the user knows BOTH options.

- Status: **PASS** (code inspection)
- Evidence: `preload_models.py` lines 75–79 add the CUDA install hint immediately after the existing CPU fallback hint.

### AC5: macOS and Linux launcher paths are unchanged. `backend.sh` still works as before — it already forces `WHISPER_DEVICE=cpu` on macOS and warns on Linux without nvidia-smi. No regression.

- Status: **PASS** (diff check)
- Command: `git diff main...HEAD -- scripts/backend.sh`
- Output: (empty — no diff)

### AC6: Existing backend tests (`python -m pytest`) still pass. No Python source changes that break existing test assumptions.

- Status: **PASS** (automated)
- Command: `python -m pytest`
- Output: 20/20 tests passed in 0.51s

## Full Test Suite
- Result: **PASS** — 20/20 tests passed

## Linter
- Result: **PASS** — "All checks passed!"

## Type Check
- Result: **PASS** — "Success: no issues found in 61 source files"

## Scope Check
- 3 files changed: `backend.ps1`, `faster_whisper_asr.py`, `preload_models.py` (+32/-1)
- Diff stays within spec scope: **YES**

## Definition of Done
- [x] All AC pass
- [x] `python -m pytest` passes (20/20)
- [x] `python -m ruff check .` passes
- [x] `python -m mypy app` passes
- [x] No secrets, debug code, or commented-out blocks in the diff
- [x] No regression on macOS/Linux launcher paths
