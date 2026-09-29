#!/bin/sh
# No downloads without --install. Explicit PYTHON also rebuilds a venv made with another interpreter.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec "${PYTHON:-python3}" scripts/setup_cli.py "$@"
