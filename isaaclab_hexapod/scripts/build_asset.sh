#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"
MJX_PYTHON="${MJX_PYTHON:-/home/huro/.venvs/hexapod-mjx/bin/python}"
INPUT="${REPO_ROOT}/mjx/generated/hexapod_isaac_full_mesh_asset.xml"
OUTPUT="${REPO_ROOT}/isaaclab_hexapod/data/usd/hexapod_full_mesh_mjx_parity.usd"

"${MJX_PYTHON}" "${REPO_ROOT}/isaaclab_hexapod/scripts/sync_latest_mjx_training.py"
"${MJX_PYTHON}" "${REPO_ROOT}/mjx/export_isaac_asset_mjcf.py"
mkdir -p "$(dirname "${OUTPUT}")"
"${ISAACLAB_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/isaaclab_hexapod/scripts/build_asset.py" \
  "${INPUT}" "${OUTPUT}" --headless \
  --experience "${ISAACLAB_ROOT}/apps/isaaclab.python.kit"

test -s "${OUTPUT}"

"${REPO_ROOT}/isaaclab_hexapod/scripts/inspect_asset.sh"

echo "Generated ${OUTPUT}"
