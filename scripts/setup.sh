#!/bin/sh
# No downloads without --install. PYTHON selects the interpreter for a new venv.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec "${PYTHON:-python3}" scripts/setup_cli.py "$@"
