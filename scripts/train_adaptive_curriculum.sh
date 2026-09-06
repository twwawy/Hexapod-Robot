#!/usr/bin/env bash
set -euo pipefail

repo="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    pwd
)"

python_bin="${HEXAPOD_PYTHON:-$HOME/.venvs/hexapod-mjx/bin/python}"

if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    exit 1
fi

exec "$python_bin" -u \
    "$repo/mjx/train_adaptive_curriculum.py" \
    "$@"