#!/usr/bin/env bash
set -euo pipefail
EGOHSI3R_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${EGOHSI3R_PYTHON:-python}" "$EGOHSI3R_ROOT/infer.py" "$@"
