# Evidence Pack — TASK-010

- Date: 2026-07-08
- Branch: `task/010-windows-cuda-dlls`
- Spec: `.workflow/specs/TASK-010.md`

## Gate Verdicts

| Gate | Verdict | Report |
|------|---------|--------|
| QA | PASS | `.workflow/qa/TASK-010-qa.md` |
| Design | N/A (gates: []) | — |
| Security | PASS (0 findings) | `.workflow/security/TASK-010-sec.md` |

## Commits (4)

```
bb13651 fix(asr): correct mypy type ignore for optional nvidia import [TASK-010]
18bf56f feat(scripts): add CUDA install hint to preload_models error message [TASK-010]
d94c41f feat(asr): add loguru warnings when CUDA DLLs missing [TASK-010]
5744f79 feat(launcher): add Install-CudaDeps for Windows GPU detection [TASK-010]
```

## Diff Stat

```
3 files changed, 32 insertions(+), 1 deletion(-)
```

| File | +/- |
|------|-----|
| `scripts/backend.ps1` | +15 |
| `app/infrastructure/asr/faster_whisper_asr.py` | +12/-1 |
| `scripts/preload_models.py` | +5 |

## Test Output

```
python -m pytest       → 20 passed, 1 warning (pre-existing)
python -m ruff check . → All checks passed!
python -m mypy app     → Success: no issues found in 61 source files
```

## Root Cause Fixed

`Install-Dependencies` ran `pip install -e "."` without `[cuda]` extras. `Ensure-GpuSettings` unconditionally set `WHISPER_DEVICE=cuda` on Windows+NVIDIA. CTranslate2 tried `LoadLibrary("cublas64_12.dll")` — DLL not found → crash.

**Fix:** New `Install-CudaDeps` function installs `".[cuda]"` extras when `nvidia-smi` is detected. Additional warnings in `_register_cuda_dll_directories()` and error hint in `preload_models.py`.

## HITL (Human-in-the-Loop)

- [ ] AC1 — Manual verification on real Windows with NVIDIA GPU: run `backend.cmd run`, select local Whisper, confirm backend starts with Whisper ready
- [ ] AC2 — Manual verification on Windows without NVIDIA GPU: confirm `[cuda]` extras are NOT installed
