#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"

"${ISAACLAB_ROOT}/isaaclab.sh" -p -m pip install -e \
  "${REPO_ROOT}/isaaclab_hexapod/source/hexapod_isaaclab"
"${ISAACLAB_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/isaaclab_hexapod/scripts/zero_action.py" \
  --steps "${HEXAPOD_SMOKE_STEPS:-500}" --headless \
  --experience "${ISAACLAB_ROOT}/apps/isaaclab.python.kit"
