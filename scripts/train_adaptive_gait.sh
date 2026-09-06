#!/usr/bin/env bash
set -euo pipefail
adaptive_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
adaptive_python="${HEXAPOD_PYTHON:-$HOME/.venvs/hexapod-mjx/bin/python}"
if [[ ! -x "$adaptive_python" ]]; then
    echo "Python environment not found: $adaptive_python" >&2
    exit 1
fi
exec "$adaptive_python" -u "$adaptive_repo/mjx/train_adaptive_gait.py" "$@"
