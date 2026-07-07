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
SKIP_PROVIDER_SELECT=0
HOST_ADDRESS="127.0.0.1"
PORT=8000

while [ $# -gt 0 ]; do
    case "$1" in
        run|install|test|lint|format) COMMAND="$1"; shift ;;
        -SkipModelSelect) SKIP_MODEL_SELECT=1; shift ;;
        -SkipOllamaInstall) SKIP_OLLAMA_INSTALL=1; shift ;;
        -SkipProviderSelect) SKIP_PROVIDER_SELECT=1; shift ;;
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

# LLM provider catalog — API providers only (mirrors app/core/providers.py
# LLM_PROVIDERS minus "ollama"; the local branch of the launcher menu handles
# Ollama via the qwen model menu). Parallel arrays at the same index.
LLM_PROVIDER_KEYS=("openai" "anthropic" "deepseek" "zai" "minimax" "gemini")
LLM_PROVIDER_LABELS=("OpenAI" "Anthropic Claude" "DeepSeek" "ZAI (GLM)" "MiniMax" "Google Gemini")
LLM_PROVIDER_BASE_URLS=("https://api.openai.com/v1" "https://api.anthropic.com" "https://api.deepseek.com" "https://open.bigmodel.cn/api/paas/v4" "https://api.minimax.io/v1" "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_PROVIDER_DEFAULT_MODELS=("gpt-4o-mini" "claude-haiku-4-5" "deepseek-chat" "glm-4" "MiniMax-M2" "gemini-2.5-flash")

# ── .env helpers (direct ports of Get-EnvValue/Set-EnvValue/Test-EnvEnabled) ─
get_env_value() {
    # Print the value of KEY from $ENV_FILE (surrounding quotes/spaces stripped),
    # or $2 default. Pure-bash line scan — no sed/grep, so no GNU-vs-BSD/portable
    # quoting pain. CR (from a CRLF .env written on Windows) is stripped too.
    local key="$1" default="${2:-}"
    [ -f "$ENV_FILE" ] || { printf '%s' "$default"; return; }
    local line raw value
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        # <optional spaces>KEY<optional spaces>=<rest>
        case "$line" in
            *"$key"=*)
                raw="${line#"${line%%[![:space:]]*}"}"   # ltrim
                case "$raw" in
                    "$key"=*) ;;
                    *"$key"=*)
                        raw="${raw#*"$key"=}"
                        raw="${raw%"${raw##*[![:space:]]}"}"
                        case "$raw" in
                            "$key"=*) ;;
                            *) continue ;;
                        esac
                        ;;
                    *) continue ;;
                esac
                value="${raw#*=}"
                value="${value%$'\r'}"
                value="${value%\"}"; value="${value#\"}"
                value="${value%\'}"; value="${value#\'}"
                value="${value%"${value##*[![:space:]]}"}"   # rtrim
                value="${value#"${value%%[![:space:]]*}"}"   # ltrim
                printf '%s' "$value"
                return
                ;;
        esac
    done < "$ENV_FILE"
    printf '%s' "$default"
}

set_env_value() {
    # Set KEY=VALUE in $ENV_FILE: rewrite the file line-by-line, replacing the
    # existing KEY line or appending a new one. Pure bash — no sed (BSD sed on
    # macOS chokes on the substitution with CRLF lines), no BOM (macOS Python
    # reads plain utf-8). Preserves all other lines, comments, and blank lines.
    local key="$1" value="$2"
    local tmp found=0 line keyline
    tmp="$(mktemp)"
    keyline="${key}=${value}"
    if [ -f "$ENV_FILE" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            line="${line%$'\r'}"
            case "$line" in
                *"$key"=*)
                    # Is the trimmed prefix exactly KEY= ? (not a substring like
                    # OLLAMA_MODEL_KEY= matching OLLAMA_MODEL=)
                    local trimmed="${line#"${line%%[![:space:]]*}"}"
                    case "$trimmed" in
                        "$key"=*)
                            printf '%s\n' "$keyline" >> "$tmp"
                            found=1
                            ;;
                        *)
                            printf '%s\n' "$line" >> "$tmp"
                            ;;
                    esac
                    ;;
                *)
                    printf '%s\n' "$line" >> "$tmp"
                    ;;
            esac
        done < "$ENV_FILE"
    fi
    [ "$found" -eq 0 ] && printf '%s\n' "$keyline" >> "$tmp"
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
    invoke_checked "$PYTHON" -m pip install --upgrade pip
    write_step "installing backend dependencies (this can take several minutes on first run)"
    # No --quiet: on macOS faster-whisper/numpy/natasha pull large wheels, and a
    # silent multi-minute install looks like a hang. pip's own progress is the
    # clearest signal that something is actually downloading.
    invoke_checked "$PYTHON" -m pip install -e ".[dev]"
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
# macOS ships bash 3.2, which has no `local -n` (nameref, bash 4.3+). So the
# caller stashes the three parallel arrays into globals _menu_names / _menu_sizes
# / _menu_notes and we read them by name through eval — portable back to 3.2.
show_model_menu() {
    local title="$1" current="$2"

    # Snapshot the global arrays into locals (bash 3.2-safe array copy).
    local -a names sizes notes
    eval 'names=("${_menu_names[@]}")'
    eval 'sizes=("${_menu_sizes[@]}")'
    eval 'notes=("${_menu_notes[@]}")'

    echo ""
    write_step "$title"
    echo "Доступные модели:"

    local i=1 default_idx=1 in_list=0 idx=0
    local current_norm; current_norm="$(printf '%s' "$current" | tr '[:upper:]' '[:lower:]')"
    for name in "${names[@]}"; do
        local name_norm; name_norm="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
        printf '  [%d] %-18s %s' "$i" "$name" "${sizes[$idx]}"
        [ -n "${notes[$idx]}" ] && printf '   • %s' "${notes[$idx]}"
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

    local max_choice=${#names[@]}
    local default_choice
    if [ "$keep_current" -eq 1 ]; then default_choice=0; else default_choice=$default_idx; fi

    while true; do
        read -r -p "Введите номер [$default_choice]: " input || { echo ""; echo "aborted" >&2; exit 130; }
        input="$(printf '%s' "$input" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        if [ -z "$input" ]; then
            if [ "$keep_current" -eq 1 ] && [ "$default_choice" -eq 0 ]; then
                printf '%s' "$current"; return
            fi
            printf '%s' "${names[$((default_idx-1))]}"; return
        fi
        case "$input" in
            ''|*[!0-9]*) write_warn "Нужно число от 0 до $max_choice"; continue ;;
        esac
        if [ "$input" -eq 0 ] && [ "$keep_current" -eq 1 ]; then
            printf '%s' "$current"; return
        fi
        if [ "$input" -ge 1 ] && [ "$input" -le "$max_choice" ]; then
            printf '%s' "${names[$((input-1))]}"; return
        fi
        write_warn "Нужно число от 0 до $max_choice"
    done
}

select_models() {
    # For each model family the user is FIRST asked local-vs-API, then shown the
    # relevant submenu: local → model menu, API → deferred to select_providers.
    if [ "$SKIP_MODEL_SELECT" -eq 1 ]; then
        write_ok "model selection skipped (-SkipModelSelect)"
        return
    fi
    echo ""
    write_step "Выбор моделей"

    if test_env_enabled "LLM_ENABLED"; then
        local current_llm_provider llm_mode_default llm_mode_num
        current_llm_provider="$(get_env_value "LLM_PROVIDER" "ollama")"
        echo ""
        echo "LLM — как запускать:"
        echo "  [1] Локально (Ollama)"
        echo "  [2] Через API (ключ)"
        if [ "$current_llm_provider" != "ollama" ] && [ -n "$current_llm_provider" ]; then
            llm_mode_default=2
        else
            llm_mode_default=1
        fi
        while true; do
            read -r -p "Режим LLM [$llm_mode_default]: " llm_mode_num || { echo ""; exit 130; }
            llm_mode_num="$(printf '%s' "$llm_mode_num" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
            [ -z "$llm_mode_num" ] && llm_mode_num=$llm_mode_default
            case "$llm_mode_num" in 1|2) break ;; *) write_warn "Нужно 1 или 2" ;; esac
        done

        if [ "$llm_mode_num" -eq 1 ]; then
            set_env_value "LLM_PROVIDER" "ollama"
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
            write_ok "LLM: через API — провайдер и ключ спрашиваются далее"
        fi
    else
        write_warn "LLM disabled in .env — пропуск выбора LLM-модели"
    fi

    if test_env_enabled "WHISPER_ENABLED"; then
        local current_whisper_provider whisper_mode_default whisper_mode_num
        current_whisper_provider="$(get_env_value "WHISPER_PROVIDER" "local")"
        echo ""
        echo "Whisper — как запускать:"
        echo "  [1] Локально (faster-whisper)"
        echo "  [2] Через API (ключ)"
        if [ "$current_whisper_provider" = "api" ]; then whisper_mode_default=2; else whisper_mode_default=1; fi
        while true; do
            read -r -p "Режим Whisper [$whisper_mode_default]: " whisper_mode_num || { echo ""; exit 130; }
            whisper_mode_num="$(printf '%s' "$whisper_mode_num" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
            [ -z "$whisper_mode_num" ] && whisper_mode_num=$whisper_mode_default
            case "$whisper_mode_num" in 1|2) break ;; *) write_warn "Нужно 1 или 2" ;; esac
        done

        if [ "$whisper_mode_num" -eq 1 ]; then
            set_env_value "WHISPER_PROVIDER" "local"
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
            write_ok "Whisper: через API — провайдер и ключ спрашиваются далее"
        fi
    else
        write_warn "WHISPER disabled in .env — пропуск выбора Whisper-модели"
    fi
    echo ""
}

# Prompt for an API key. If $1 (current value) is non-empty, offer to keep it
# (Enter) or re-enter. Never echoes the stored value beyond a 4-char mask.
read_secret_with_keep() {
    local current="$1" prompt="$2" entered mask
    if [ -n "$current" ]; then
        mask="$(printf '%s' "$current" | cut -c1-4)$(printf '%*s' $(( ${#current} > 4 ? ${#current} - 4 : 0 )) '' | tr ' ' '*')"
        read -r -p "$prompt [Enter = оставить текущий ($mask)]: " entered || { echo ""; exit 130; }
        [ -z "$entered" ] && { printf '%s' "$current"; return; }
        printf '%s' "$entered"
    else
        read -r -p "$prompt: " entered || { echo ""; exit 130; }
        printf '%s' "$entered"
    fi
}

select_providers() {
    if [ "$SKIP_PROVIDER_SELECT" -eq 1 ]; then
        write_ok "provider selection skipped (-SkipProviderSelect)"
        return
    fi
    echo ""
    write_step "Выбор провайдеров"

    # ── LLM ──
    # Only prompt for the API provider/key when the user chose the API mode in
    # select_models (LLM_PROVIDER != ollama). Local mode is fully configured by
    # the qwen model menu there and needs nothing here.
    if test_env_enabled "LLM_ENABLED"; then
        local current_provider
        current_provider="$(get_env_value "LLM_PROVIDER" "ollama")"
        if [ "$current_provider" = "ollama" ] || [ -z "$current_provider" ]; then
            write_ok "LLM: локальный режим — выбор API-провайдера пропущен"
        else
            local default_idx i chosen_key chosen_idx count num key model cur_model
            echo ""
            echo "LLM-провайдер:"
            default_idx=1
            i=0
            for key in "${LLM_PROVIDER_KEYS[@]}"; do
                printf '  [%d] %-22s %s\n' "$((i+1))" "${LLM_PROVIDER_LABELS[$i]}" "${LLM_PROVIDER_DEFAULT_MODELS[$i]}"
                [ "$key" = "$current_provider" ] && default_idx=$((i+1))
                i=$((i+1))
            done
            count="${#LLM_PROVIDER_KEYS[@]}"
            while true; do
                read -r -p "LLM-провайдер [$default_idx]: " num || { echo ""; exit 130; }
                num="$(printf '%s' "$num" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
                [ -z "$num" ] && num=$default_idx
                case "$num" in *[!0-9]*|'') write_warn "Нужно число от 1 до $count"; continue ;; esac
                if [ "$num" -ge 1 ] && [ "$num" -le "$count" ]; then break; fi
                write_warn "Нужно число от 1 до $count"
            done
            chosen_key="${LLM_PROVIDER_KEYS[$((num-1))]}"
            chosen_idx=$((num-1))
            if [ "$chosen_key" != "$current_provider" ]; then
                set_env_value "LLM_PROVIDER" "$chosen_key"
                write_ok "LLM_PROVIDER=$chosen_key"
            else
                write_ok "LLM-провайдер без изменений: $chosen_key"
            fi

            key="$(read_secret_with_keep "$(get_env_value "LLM_API_KEY" "")" "LLM API key")"
            set_env_value "LLM_API_KEY" "$key"
            cur_model="$(get_env_value "LLM_API_MODEL" "")"
            read -r -p "Модель [Enter = ${LLM_PROVIDER_DEFAULT_MODELS[$chosen_idx]}]: " model || { echo ""; exit 130; }
            model="$(printf '%s' "$model" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
            [ -z "$model" ] && model="$cur_model"
            set_env_value "LLM_API_MODEL" "$model"
            write_ok "LLM_API_MODEL=$model"
        fi
    else
        write_warn "LLM disabled in .env — пропуск выбора LLM-провайдера"
    fi

    # ── Whisper ──
    if test_env_enabled "WHISPER_ENABLED"; then
        local current_w wdefault num wchoice
        current_w="$(get_env_value "WHISPER_PROVIDER" "local")"
        echo ""
        echo "Whisper-провайдер:"
        echo "  [1] Локально (faster-whisper)"
        echo "  [2] Через API (OpenAI/Groq/совместимый)"
        if [ "$current_w" = "api" ]; then wdefault=2; else wdefault=1; fi
        while true; do
            read -r -p "Whisper-провайдер [$wdefault]: " num || { echo ""; exit 130; }
            num="$(printf '%s' "$num" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
            [ -z "$num" ] && num=$wdefault
            case "$num" in 1|2) break ;; *) write_warn "Нужно 1 или 2" ;; esac
        done
        if [ "$num" -eq 2 ]; then wchoice="api"; else wchoice="local"; fi
        if [ "$wchoice" != "$current_w" ]; then
            set_env_value "WHISPER_PROVIDER" "$wchoice"
            write_ok "WHISPER_PROVIDER=$wchoice"
        else
            write_ok "Whisper-провайдер без изменений: $wchoice"
        fi
        if [ "$wchoice" = "api" ]; then
            local wkey
            wkey="$(read_secret_with_keep "$(get_env_value "WHISPER_API_KEY" "")" "Whisper API key")"
            set_env_value "WHISPER_API_KEY" "$wkey"
        fi
    else
        write_warn "WHISPER disabled in .env — пропуск выбора Whisper-провайдера"
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
    select_providers
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
