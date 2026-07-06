#!/usr/bin/env bash
# Unix entry point — mirrors backend.cmd on Windows.
# Usage:  ./backend.sh run      (or: bash backend.sh run)
set -euo pipefail
exec bash "$(dirname "$0")/scripts/backend.sh" "$@"
