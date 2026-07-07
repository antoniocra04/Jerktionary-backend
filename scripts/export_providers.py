"""Export the canonical LLM provider catalog as JSON for launcher scripts.

Usage:
    python scripts/export_providers.py   # prints JSON to stdout

The catalog lives in app.core.providers.LLM_PROVIDERS. The "ollama" entry is
excluded because the launchers handle Ollama separately in their local branch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.providers import LLM_PROVIDERS  # noqa: E402

api_providers = [
    {
        "key": p.key,
        "label": p.label,
        "base_url": p.base_url,
        "default_model": p.default_model,
    }
    for p in LLM_PROVIDERS.values()
    if p.key != "ollama"
]

json.dump(api_providers, sys.stdout, indent=2, ensure_ascii=False)
sys.stdout.write("\n")
