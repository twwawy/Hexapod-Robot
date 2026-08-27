#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"

"${ISAACLAB_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/isaaclab_hexapod/scripts/inspect_asset.py" \
  "${REPO_ROOT}/isaaclab_hexapod/data/usd/hexapod_mjx_parity.usd" \
  --report "${REPO_ROOT}/isaaclab_hexapod/data/usd/asset_inspection.json" \
  --headless --experience "${ISAACLAB_ROOT}/apps/isaaclab.python.kit"
