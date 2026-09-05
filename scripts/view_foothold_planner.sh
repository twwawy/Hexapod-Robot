#!/usr/bin/env bash
set -euo pipefail
preview_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
preview_python="${HEXAPOD_PYTHON:-$HOME/.venvs/hexapod-mjx/bin/python}"
if [[ ! -x "$preview_python" ]]; then
    echo "Python environment not found: $preview_python" >&2
    echo "Set HEXAPOD_PYTHON to a Python interpreter with mujoco and numpy installed." >&2
    exit 1
fi
exec env -u LD_LIBRARY_PATH MUJOCO_GL=glfw "$preview_python" -u \
    "$preview_repo/mjx/view_foothold_planner.py" "$@"
