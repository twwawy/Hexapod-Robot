#!/usr/bin/env bash
set -euo pipefail

REPO="${HEXAPOD_ROBOT_REPO:-$HOME/Hexapod-Robot}"
PY="$HOME/.venvs/hexapod-mjx/bin/python"
MODE_ARGS=()
USE_EGL=1

if (($#)) && [[ "$1" == "viewer" ]]; then
  MODE_ARGS+=(--viewer)
  USE_EGL=0
  shift
fi

if [[ "$USE_EGL" == "1" ]]; then
  exec env -u LD_LIBRARY_PATH MUJOCO_GL=egl "$PY" "$REPO/SW/mjx/preview_stand_pose.py" \
    --repo-root "$REPO" \
    "${MODE_ARGS[@]}" \
    "$@"
fi

if [[ -z "${DISPLAY:-}" ]] && xdpyinfo -display :0 >/dev/null 2>&1; then
  exec env -u LD_LIBRARY_PATH DISPLAY=:0 "$PY" "$REPO/SW/mjx/preview_stand_pose.py" \
    --repo-root "$REPO" \
    "${MODE_ARGS[@]}" \
    "$@"
fi

if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a env -u LD_LIBRARY_PATH "$PY" "$REPO/SW/mjx/preview_stand_pose.py" \
    --repo-root "$REPO" \
    "${MODE_ARGS[@]}" \
    "$@"
fi

exec env -u LD_LIBRARY_PATH "$PY" "$REPO/SW/mjx/preview_stand_pose.py" \
  --repo-root "$REPO" \
  "${MODE_ARGS[@]}" \
  "$@"