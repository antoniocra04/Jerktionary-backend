# Evidence Pack — TASK-011

- Date: 2026-07-08
- Branch: `task/011-lan-access`
- Spec: `.workflow/specs/TASK-011.md`

## Gate Verdicts

| Gate | Verdict | Report |
|------|---------|--------|
| QA | PASS | `.workflow/qa/TASK-011-qa.md` |
| Design | N/A (gates: []) | — |
| Security | PASS (3 findings, 0 critical) | `.workflow/security/TASK-011-sec.md` |

## Commits

### Backend
```
8eb42e1 feat(config): bind to 0.0.0.0, permissive CORS for LAN access [TASK-011]
```

### Frontend
```
a49956c fix(electron): allow any http:// URL for shell:openExternal [TASK-011]
```

## Diff Stat

### Backend
```
 .env.example        | 11 ++++++++++-
 app/core/config.py  |  4 ++--
 app/main.py         |  2 +-
 scripts/backend.ps1 |  2 +-
 scripts/backend.sh  |  2 +-
 5 files changed, 15 insertions(+), 6 deletions(-)
```

### Frontend
```
 electron/main/index.ts | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

## Test Output

```
Backend: python -m pytest       → 20 passed
Backend: python -m ruff check . → All checks passed!
Backend: python -m mypy app     → Success: no issues found in 61 source files
Frontend: npm test              → 41 passed (7 test files)
Frontend: npm run lint          → clean (0 errors)
```

## Root Cause Fixed

Three blockers prevented LAN access: (1) backend bound to `127.0.0.1` (loopback only), (2) CORS allowed only Vite dev-server origins, (3) Electron `shell:openExternal` only accepted `http://127.0.0.1:8000`.

**Fix:** Changed all defaults to `0.0.0.0`, CORS to `["*"]` with `allow_credentials=False`, Electron to allow any `http://` URL.

## HITL (Human-in-the-Loop)

- [ ] AC2 — Manual test: from another machine on LAN, `curl http://<backend-ip>:8000/health` → `{"status": "ok"}`
- [ ] AC3 — Manual test: from another machine's browser, open `http://<backend-ip>:8000/docs` → Swagger UI loads without CORS errors
- [ ] AC4 — Manual test: build frontend with `VITE_BACKEND_HTTP_URL=http://<backend-ip>:8000`, click a backend link → opens in browser
- [ ] Firewall — Run `netsh advfirewall firewall add rule name="Backend 8000" dir=in action=allow protocol=TCP localport=8000` on the backend machine
