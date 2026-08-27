#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"

"${ISAACLAB_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/isaaclab_hexapod/scripts/validate_configs.py" \
  --report "${REPO_ROOT}/isaaclab_hexapod/data/config_validation.json" \
  --headless --experience "${ISAACLAB_ROOT}/apps/isaaclab.python.kit"
