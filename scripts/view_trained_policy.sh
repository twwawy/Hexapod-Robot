#!/usr/bin/env bash
set -euo pipefail
policy_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
policy_python="${HEXAPOD_PYTHON:-$HOME/.venvs/hexapod-mjx/bin/python}"
if [[ ! -x "$policy_python" ]]; then
    echo "Python environment not found: $policy_python" >&2
    exit 1
fi
exec env -u LD_LIBRARY_PATH MUJOCO_GL=glfw XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$policy_python" -u "$policy_repo/mjx/view_trained_policy.py" "$@"
