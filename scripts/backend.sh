#!/usr/bin/env bash
# macOS / Linux launcher for the Realtime Terms Backend.
# Bash port of scripts/backend.ps1 (which stays the Windows path). Same step order
# in `run`, same .env keys, same interactive model picker, same flags where they
# make sense cross-platform.
#
# Usage:
#   bash backend.sh [run|install|test|lint|format] \
#       [-SkipModelSelect] [-SkipOllamaInstall] [-Host 127.0.0.1] [-Port 8000]
set -euo pipefail

# ── CLI ──────────────────────────────────────────────────────────────────────
COMMAND="run"
SKIP_MODEL_SELECT=0
SKIP_OLLAMA_INSTALL=0
HOST_ADDRESS="127.0.0.1"
PORT=8000

while [ $# -gt 0 ]; do
    case "$1" in
        run|install|test|lint|format) COMMAND="$1"; shift ;;
        -SkipModelSelect) SKIP_MODEL_SELECT=1; shift ;;
        -SkipOllamaInstall) SKIP_OLLAMA_INSTALL=1; shift ;;
        -Host) HOST_ADDRESS="${2:?}"; shift 2 ;;
        -Port) PORT="${2:?}"; shift 2 ;;
        -Dev|-PullLlm) shift ;;   # accepted for parity, no-op on Unix
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMARY_VENV_DIR="$ROOT/.venv"
FALLBACK_VENV_DIR="$HOME/.cache/RealtimeTermsBackend/.venv"
VENV_DIR="$PRIMARY_VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
ENV_FILE="$ROOT/.env"
ENV_EXAMPLE="$ROOT/.env.example"
DATA_DIR="$ROOT/data"

cd "$ROOT"

# ── Pretty output ────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RESET=$'\033[0m'
else
    C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RESET=""
fi

write_step() { printf '%s==> %s%s\n' "$C_CYAN" "$*" "$C_RESET"; }
write_ok()   { printf '%sOK   %s%s\n' "$C_GREEN" "$*" "$C_RESET"; }
write_warn() { printf '%sWARN %s%s\n' "$C_YELLOW" "$*" "$C_RESET"; }

# Run a command, fail the script (with context) if it exits non-zero.
invoke_checked() {
    "$@" || { echo "Command failed (exit $?): $*" >&2; exit 1; }
}

# ── Curated model catalogs (mirror backend.ps1) ──────────────────────────────
LLM_MODEL_NAMES=("qwen3.5:9b" "qwen3:8b" "qwen3.5:4b")
LLM_MODEL_SIZES=("~6.6 GB" "~5.0 GB" "~2.5 GB")
LLM_MODEL_NOTES=("Рекомендуется: лучший баланс для русского" "Стабильный официальный Qwen3" "Быстрее, слабее")

WHISPER_MODEL_NAMES=("large-v3-turbo" "large-v3" "medium" "small" "base" "tiny")
WHISPER_MODEL_SIZES=("~1.6 GB" "~3.0 GB" "~1.5 GB" "~480 MB" "~145 MB" "~75 MB")
WHISPER_MODEL_NOTES=("Рекомендуется: баланс скорости и качества" "Максимальное качество" "" "" "" "Самая быстрая")

# ── .env helpers (direct ports of Get-EnvValue/Set-EnvValue/Test-EnvEnabled) ─
get_env_value() {
    # echo the value of KEY in $ENV_FILE (quotes/whitespace stripped), or $2 default.
    local key="$1" default="${2:-}"
    [ -f "$ENV_FILE" ] || { printf '%s' "$default"; return; }
    local value
    # Match  ^<optional space>KEY<optional space>=<value>  ; strip surrounding quotes/spaces.
    value="$(grep -E "^[[:space:]]*$(printf '%s' "$key" | sed 's/[^A-Za-z0-9_]/\\&/g')[[:space:]]*=" "$ENV_FILE" 2>/dev/null | head -n1 || true)"
    [ -z "$value" ] && { printf '%s' "$default"; return; }
    value="${value#*=}"
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    printf '%s' "$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
}

set_env_value() {
    # Set KEY=VALUE in $ENV_FILE: replace existing line or append. No BOM (macOS
    # Python reads plain utf-8 — the PS version's UTF-8 BOM surfaces as \ufeff).
    local key="$1" value="$2"
    local tmp
    tmp="$(mktemp)"
    if [ -f "$ENV_FILE" ] && grep -Eq "^[[:space:]]*$(printf '%s' "$key" | sed 's/[^A-Za-z0-9_]/\\&/g')[[:space:]]*=" "$ENV_FILE"; then
        sed -E "s|^[[:space:]]*$(printf '%s' "$key" | sed 's/[^A-Za-z0-9_]/\\&/g')[[:space:]]*=.*|${key}=${value}|" "$ENV_FILE" > "$tmp"
    else
        { [ -f "$ENV_FILE" ] && cat "$ENV_FILE"; echo "${key}=${value}"; } > "$tmp"
    fi
    mv "$tmp" "$ENV_FILE"
}

test_env_enabled() {
    local v
    v="$(printf '%s' "$(get_env_value "$1" "true")" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        0|false|no|off) return 1 ;;
        *) return 0 ;;
    esac
}

# ── Python / venv ────────────────────────────────────────────────────────────
find_system_python() {
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
                command -v "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

ensure_system_python() {
    local found
    if found="$(find_system_python)"; then
        printf '%s' "$found"
        return 0
    fi
    write_warn "Python 3.11+ not found"
    case "$(uname -s)" in
        Darwin) echo "  Install it:  brew install python@3.11" >&2 ;;
        *)      echo "  Install it:  sudo apt install python3.11 python3.11-venv  (or your distro's equivalent)" >&2 ;;
    esac
    return 1
}

use_venv() {
    VENV_DIR="$1"
    PYTHON="$VENV_DIR/bin/python"
}

test_venv_pip() {
    [ -x "$PYTHON" ] || return 1
    "$PYTHON" -m pip --version >/dev/null 2>&1
}

ensure_venv() {
    use_venv "$PRIMARY_VENV_DIR"
    if test_venv_pip; then
        write_ok "venv exists"
        return
    fi
    if [ -e "$VENV_DIR" ]; then
        write_warn ".venv exists but pip is broken; using $FALLBACK_VENV_DIR"
        use_venv "$FALLBACK_VENV_DIR"
        if test_venv_pip; then
            write_ok "fallback venv exists"
            return
        fi
    fi
    local sys_python
    sys_python="$(ensure_system_python)" || exit 1
    write_step "creating venv at $VENV_DIR"
    invoke_checked "$sys_python" -m venv "$VENV_DIR"
    write_ok "venv created"
}

ensure_project_files() {
    if [ ! -f "$ENV_FILE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        write_ok ".env created from .env.example"
    else
        write_ok ".env exists"
    fi
    if [ ! -d "$DATA_DIR" ]; then
        mkdir -p "$DATA_DIR"
        write_ok "data directory created"
    else
        write_ok "data directory exists"
    fi
}

# On macOS there is no CUDA — force Whisper to CPU so startup doesn't fail on the
# cuda default from .env.example. On Linux we leave .env alone (CUDA-on-Linux is
# a valid setup the user may have configured).
ensure_gpu_settings() {
    case "$(uname -s)" in
        Darwin)
            write_warn "macOS detected: no CUDA, Whisper will run on CPU"
            if [ "$(get_env_value "WHISPER_DEVICE" "cuda")" != "cpu" ]; then
                set_env_value "WHISPER_DEVICE" "cpu"
                write_ok "WHISPER_DEVICE set to cpu"
            else
                write_ok "WHISPER_DEVICE=cpu"
            fi
            if [ "$(get_env_value "WHISPER_COMPUTE_TYPE" "float16")" != "int8" ]; then
                set_env_value "WHISPER_COMPUTE_TYPE" "int8"
                write_ok "WHISPER_COMPUTE_TYPE set to int8"
            else
                write_ok "WHISPER_COMPUTE_TYPE=int8"
            fi
            ;;
        *)
            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
                write_ok "NVIDIA GPU detected"
            else
                write_warn "nvidia-smi not found; CUDA startup may fail (set WHISPER_DEVICE=cpu in .env if needed)"
            fi
            ;;
    esac
}

test_import() {
    "$PYTHON" -c "import $1" >/dev/null 2>&1
}

install_dependencies() {
    write_step "upgrading pip"
    invoke_checked "$PYTHON" -m pip install --quiet --upgrade pip
    write_step "installing backend dependencies"
    invoke_checked "$PYTHON" -m pip install --quiet -e ".[dev]"
    write_ok "dependencies installed"
}

ensure_dependencies() {
    local needed=(fastapi uvicorn pydantic_settings aiosqlite)
    for mod in "${needed[@]}"; do
        if ! test_import "$mod"; then
            install_dependencies
            return
        fi
    done
    # The project package itself must be importable. An editable install records
    # the package path at install time, so a moved/copied project (or a venv made
    # elsewhere) leaves the third-party libs present but `app` missing — which
    # crashes preload_models.py. Reinstall editable to repoint it.
    if ! test_import "app"; then
        write_warn "project package 'app' is not importable in this venv; reinstalling editable"
        install_dependencies
        return
    fi
    write_ok "dependencies ready"
}

# ── Ollama ───────────────────────────────────────────────────────────────────
find_ollama() {
    local p
    if p="$(command -v ollama 2>/dev/null)" && [ -n "$p" ]; then
        printf '%s' "$p"; return 0
    fi
    for p in /usr/local/bin/ollama /opt/homebrew/bin/ollama \
             "/Applications/Ollama.app/Contents/Resources/ollama" \
             /usr/bin/ollama; do
        if [ -x "$p" ]; then printf '%s' "$p"; return 0; fi
    done
    return 1
}

test_ollama_model() {
    local model="$1" base_url
    base_url="$(get_env_value "OLLAMA_BASE_URL" "http://127.0.0.1:11434")"
    write_step "checking Ollama models at $base_url"
    curl -fsS --max-time 5 "$base_url/api/tags" 2>/dev/null \
        | grep -q "\"name\":\"$model\"\|\"model\":\"$model\""
}

wait_ollama_server() {
    local ollama="$1" base_url attempt
    base_url="$(get_env_value "OLLAMA_BASE_URL" "http://127.0.0.1:11434")"
    write_step "checking Ollama server $base_url"
    for attempt in 1 2 3; do
        if curl -fsS --max-time 5 "$base_url/api/tags" >/dev/null 2>&1; then
            write_ok "Ollama server responding"
            return 0
        fi
        if [ "$attempt" -eq 1 ]; then
            write_warn "Ollama server is not responding; starting ollama serve"
            "$ollama" serve >/dev/null 2>&1 &
        else
            write_warn "waiting for Ollama server ($attempt/3)"
        fi
        sleep 3
    done
    echo "Ollama server did not respond at $base_url. Open Ollama or run: ollama serve" >&2
    exit 1
}

warm_ollama_model() {
    local model="$1" base_url keep_alive num_ctx num_batch num_gpu think
    base_url="$(get_env_value "OLLAMA_BASE_URL" "http://127.0.0.1:11434")"
    keep_alive="$(get_env_value "OLLAMA_KEEP_ALIVE" "30m")"
    num_ctx="$(get_env_value "OLLAMA_NUM_CTX" "1024")"
    num_batch="$(get_env_value "OLLAMA_NUM_BATCH" "1024")"
    num_gpu="$(get_env_value "OLLAMA_NUM_GPU" "999")"
    think="$(printf '%s' "$(get_env_value "OLLAMA_THINK" "false")" | tr '[:upper:]' '[:lower:]')"
    case "$think" in 0|false|no|off) think=false ;; *) think=true ;; esac

    write_step "warming Ollama model $model; this can take a while"
    curl -fsS --max-time 600 "$base_url/api/generate" \
        -H 'Content-Type: application/json' \
        -d "$(cat <<EOF
{"model":"$model","prompt":"/no_think\nready","stream":false,"keep_alive":"$keep_alive",
 "options":{"num_predict":1,"temperature":0,"num_ctx":$num_ctx,"num_batch":$num_batch,"num_gpu":$num_gpu}}
EOF
)" >/dev/null
    write_ok "Ollama model warmed; keep_alive=$keep_alive"
}

check_ollama() {
    if ! test_env_enabled "LLM_ENABLED"; then
        write_warn "LLM is disabled in .env"
        return
    fi

    local ollama model
    if ! ollama="$(find_ollama)"; then
        if [ "$SKIP_OLLAMA_INSTALL" -eq 0 ]; then
            write_warn "Ollama not found. Install it:"
            case "$(uname -s)" in
                Darwin) echo "  brew install ollama" >&2 ;;
                *)      echo "  curl -fsSL https://ollama.com/install.sh | sh" >&2 ;;
            esac
            echo "  Then re-run. Backend will start, but the LLM will be disabled until Ollama is available." >&2
        else
            write_warn "Ollama not found (-SkipOllamaInstall). Backend will run; explain endpoint may return LLM_UNAVAILABLE."
        fi
        return
    fi

    write_ok "Ollama found"
    wait_ollama_server "$ollama"
    model="$(get_env_value "OLLAMA_MODEL" "qwen3:8b")"
    if test_ollama_model "$model"; then
        write_ok "Ollama model exists: $model"
    else
        write_step "pulling Ollama model $model; download progress is shown below"
        invoke_checked "$ollama" pull "$model"
        write_ok "Ollama model ready: $model"
    fi
    warm_ollama_model "$model"
}

ensure_whisper_model() {
    write_step "preloading Whisper model"
    invoke_checked "$PYTHON" "scripts/preload_models.py"
}

# ── Interactive model picker (mirror of Show-ModelMenu/Select-Models) ─────────
show_model_menu() {
    local title="$1" current="$2"
    # nameref into the caller's arrays — _n/_s/_t are the NAMES/SIZES/NOTES arrays.
    local -n _n=_menu_names _s=_menu_sizes _t=_menu_notes

    echo ""
    write_step "$title"
    echo "Доступные модели:"

    local i=1 default_idx=1 in_list=0
    local current_norm; current_norm="$(printf '%s' "$current" | tr '[:upper:]' '[:lower:]')"
    local idx=0
    for name in "${_n[@]}"; do
        local name_norm; name_norm="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
        printf '  [%d] %-18s %s' "$i" "$name" "${_s[$idx]}"
        [ -n "${_t[$idx]}" ] && printf '   • %s' "${_t[$idx]}"
        printf '\n'
        if [ "$name_norm" = "$current_norm" ] && [ -n "$current_norm" ]; then
            default_idx=$i; in_list=1
        fi
        i=$((i+1)); idx=$((idx+1))
    done

    local keep_current=0
    if [ -n "$current_norm" ] && [ "$in_list" -eq 0 ]; then
        keep_current=1
        printf '  [0] Оставить текущую: %s\n' "$current"
    fi

    local max_choice=${#_n[@]}
    local default_choice
    if [ "$keep_current" -eq 1 ]; then default_choice=0; else default_choice=$default_idx; fi

    while true; do
        read -r -p "Введите номер [$default_choice]: " input || { echo ""; echo "aborted" >&2; exit 130; }
        input="$(printf '%s' "$input" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        if [ -z "$input" ]; then
            if [ "$keep_current" -eq 1 ] && [ "$default_choice" -eq 0 ]; then
                printf '%s' "$current"; return
            fi
            printf '%s' "${_n[$((default_idx-1))]}"; return
        fi
        case "$input" in
            ''|*[!0-9]*) write_warn "Нужно число от 0 до $max_choice"; continue ;;
        esac
        if [ "$input" -eq 0 ] && [ "$keep_current" -eq 1 ]; then
            printf '%s' "$current"; return
        fi
        if [ "$input" -ge 1 ] && [ "$input" -le "$max_choice" ]; then
            printf '%s' "${_n[$((input-1))]}"; return
        fi
        write_warn "Нужно число от 0 до $max_choice"
    done
}

select_models() {
    if [ "$SKIP_MODEL_SELECT" -eq 1 ]; then
        write_ok "model selection skipped (-SkipModelSelect)"
        return
    fi
    echo ""
    write_step "Выбор моделей"

    if test_env_enabled "LLM_ENABLED"; then
        local current_llm choice
        current_llm="$(get_env_value "OLLAMA_MODEL" "")"
        _menu_names=("${LLM_MODEL_NAMES[@]}"); _menu_sizes=("${LLM_MODEL_SIZES[@]}"); _menu_notes=("${LLM_MODEL_NOTES[@]}")
        choice="$(show_model_menu "Выбор LLM-модели (Ollama)" "$current_llm")"
        if [ -n "$choice" ] && [ "$(printf '%s' "$choice" | tr '[:upper:]' '[:lower:]')" != "$(printf '%s' "$current_llm" | tr '[:upper:]' '[:lower:]')" ]; then
            set_env_value "OLLAMA_MODEL" "$choice"
            write_ok "OLLAMA_MODEL=$choice"
        else
            write_ok "LLM-модель без изменений: $current_llm"
        fi
    else
        write_warn "LLM disabled in .env — пропуск выбора LLM-модели"
    fi

    if test_env_enabled "WHISPER_ENABLED"; then
        local current_whisper choice
        current_whisper="$(get_env_value "WHISPER_MODEL" "")"
        _menu_names=("${WHISPER_MODEL_NAMES[@]}"); _menu_sizes=("${WHISPER_MODEL_SIZES[@]}"); _menu_notes=("${WHISPER_MODEL_NOTES[@]}")
        choice="$(show_model_menu "Выбор Whisper-модели (faster-whisper)" "$current_whisper")"
        if [ -n "$choice" ] && [ "$(printf '%s' "$choice" | tr '[:upper:]' '[:lower:]')" != "$(printf '%s' "$current_whisper" | tr '[:upper:]' '[:lower:]')" ]; then
            set_env_value "WHISPER_MODEL" "$choice"
            write_ok "WHISPER_MODEL=$choice"
        else
            write_ok "Whisper-модель без изменений: $current_whisper"
        fi
    else
        write_warn "WHISPER disabled in .env — пропуск выбора Whisper-модели"
    fi
    echo ""
}

# ── Commands ─────────────────────────────────────────────────────────────────
run_backend() {
    ensure_venv
    ensure_project_files
    ensure_gpu_settings
    ensure_dependencies
    select_models
    check_ollama
    ensure_whisper_model
    write_step "starting FastAPI http://$HOST_ADDRESS:$PORT"
    exec "$PYTHON" -m uvicorn app.main:app --host "$HOST_ADDRESS" --port "$PORT" --reload
}

run_install() {
    ensure_venv
    ensure_project_files
    install_dependencies
    check_ollama
}

run_tests() {
    ensure_venv; ensure_project_files; ensure_dependencies
    invoke_checked "$PYTHON" -m pytest
}

run_lint() {
    ensure_venv; ensure_project_files; ensure_dependencies
    invoke_checked "$PYTHON" -m ruff check .
    invoke_checked "$PYTHON" -m mypy app
}

run_format() {
    ensure_venv; ensure_project_files; ensure_dependencies
    invoke_checked "$PYTHON" -m black .
}

case "$COMMAND" in
    run)     run_backend ;;
    install) run_install ;;
    test)    run_tests ;;
    lint)    run_lint ;;
    format)  run_format ;;
esac
