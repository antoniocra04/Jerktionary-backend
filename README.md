# Realtime Terms Backend

FastAPI backend для desktop-приложения: audio chunks от Electron по WebSocket, realtime ASR через `faster-whisper`, extraction терминов через Natasha, explanation по hover через локальную LLM Ollama.

## Quick Start

Одна команда для Windows:

```bat
backend.cmd
```

Для macOS / Linux:

```bash
./backend.sh          # или: bash backend.sh
```

`backend.sh` — это bash-порт того же процесса запуска (см. ниже).

Она создаст `.venv`, поставит зависимости, создаст `.env` из `.env.example`, создаст `data/`, установит Ollama при отсутствии и запустит backend на `http://127.0.0.1:8000`.

Перед запуском `run` интерактивно спросит LLM-модель (Ollama) и Whisper-модель из кураторского
списка: достаточно ввести номер или нажать Enter, чтобы оставить текущую/дефолтную. Выбор пишется
в `.env` (`OLLAMA_MODEL`, `WHISPER_MODEL`), после чего модель автоматически скачивается. Чтобы
запустить без диалога (например, в CI/автоматизации), используйте `run -SkipModelSelect`.

Если Python 3.11+ не найден, команда попробует установить `Python.Python.3.11` через `winget`.
Если OneDrive блокирует pip внутри `.venv`, команда автоматически использует venv в `%LOCALAPPDATA%\RealtimeTermsBackend\.venv`.

Команды:

```bat
.\backend.cmd install
.\backend.cmd run
.\backend.cmd test
.\backend.cmd lint
.\backend.cmd format
.\backend.cmd run -Dev
.\backend.cmd run -PullLlm
.\backend.cmd run -SkipOllamaInstall
.\backend.cmd run -SkipModelSelect
.\backend.cmd run -Port 8010
```

Ollama ставится автоматически через `winget`, если `ollama` не найден.
`-SkipOllamaInstall` отключает автоустановку Ollama.
`-PullLlm` скачивает модель `qwen3:8b` через Ollama. Без этого флага backend всё равно стартует, но LLM будет `disabled`, если модель недоступна.
`-SkipModelSelect` пропускает интерактивный выбор LLM- и Whisper-модели — используются значения из `.env`.

## macOS / Linux

Запуск через bash-лаунчер (зеркало `backend.cmd`):

```bash
./backend.sh run
./backend.sh install
./backend.sh test
./backend.sh run -SkipModelSelect
./backend.sh run -SkipOllamaInstall
./backend.sh run -Port 8010
```

Если скрипт не исполняемый: `chmod +x backend.sh scripts/backend.sh`.

**Ollama** не ставится автоматически (на маке/линуксе нет `winget`). Установите вручную:

```bash
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Пока Ollama недоступен, backend стартует, но LLM остаётся `disabled` — как с `-SkipOllamaInstall` на Windows.

**Whisper на macOS:** CUDA на маке нет, поэтому лаунчер при первом запуске записывает в `.env`
`WHISPER_DEVICE=cpu` и `WHISPER_COMPUTE_TYPE=int8` — Whisper работает на CPU через CTranslate2
(нативного Metal у faster-whisper нет). На Linux с NVIDIA GPU можно вручную поставить
`WHISPER_DEVICE=cuda` / `WHISPER_COMPUTE_TYPE=float16` в `.env`.

Если Python 3.11+ не найден, скрипт подскажет `brew install python@3.11` (мак) или установку
через пакетный менеджер дистрибутива (линукс); автоустановки Python, в отличие от Windows, нет.

## Manual Install

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Dev

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Tests

```bash
pytest
```

## Quality

```bash
ruff check .
black .
mypy app
```

## Ollama

```bash
ollama pull qwen3:8b
ollama serve
```

Если Ollama недоступна, backend стартует. `POST /api/terms/explain` вернёт `LLM_UNAVAILABLE`, когда объяснения нет в SQLite cache.

## Без локальных моделей

Обе локальные модели опциональны — фронтенд умеет работать через API-провайдеров:

- `WHISPER_ENABLED=false` в `.env` — backend стартует без загрузки Whisper (экономит
  скачивание модели и VRAM). Распознавание тогда идёт через API-провайдера
  (OpenAI/Groq/совместимый), который выбирается в настройках приложения и передаётся
  первым сообщением по WebSocket.
- `LLM_ENABLED=false` — то же для Ollama. Ответы и объяснения идут через выбранного в
  приложении провайдера: OpenAI, Anthropic (нативный Messages API), Gemini, Groq,
  OpenRouter, DeepSeek или любой OpenAI-совместимый base URL.

## WebSocket

`WS /ws/audio` принимает binary PCM chunks: 16 kHz, mono, int16 little-endian.

Пример события:

```json
{
  "type": "transcript_update",
  "text": "я читал про теорию относительности",
  "is_final": false,
  "terms": [
    {
      "text": "теорию относительности",
      "normalized": "теория относительности",
      "start": 13,
      "end": 36,
      "type": "concept",
      "confidence": 0.9
    }
  ]
}
```

## Explain

```bash
curl -X POST http://127.0.0.1:8000/api/terms/explain ^
  -H "Content-Type: application/json" ^
  -d "{\"term\":\"теория относительности\",\"context\":\"я читал про теорию относительности и скорость света\"}"
```

Ответ:

```json
{
  "title": "Теория относительности",
  "short": "Краткое объяснение термина.",
  "example": "Скорость света одинакова для наблюдателей.",
  "why_important": "Помогает понимать связь пространства, времени и гравитации.",
  "source": "local_llm"
}
```

## Startup

Пример вывода:

```text
Backend startup
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Service         ┃ Status   ┃ Details         ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Config loaded   │ ready    │ env loaded      │
│ SQLite ready    │ ready    │ data/app.sqlite3│
│ Whisper loaded  │ ready    │ small           │
│ Natasha loaded  │ ready    │ pipeline loaded │
│ LLM ready       │ disabled │ Ollama unavailable │
│ API ready       │ ready    │ -               │
│ WebSocket ready │ ready    │ -               │
└─────────────────┴──────────┴─────────────────┘
```
