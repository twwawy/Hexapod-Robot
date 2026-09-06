#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"
REPORT="${REPO_ROOT}/isaaclab_hexapod/data/usd/asset_inspection.json"
REPORT_TMP="$(mktemp "${REPORT}.tmp.XXXXXX")"
trap 'rm -f "${REPORT_TMP}"' EXIT

"${ISAACLAB_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/isaaclab_hexapod/scripts/inspect_asset.py" \
  "${REPO_ROOT}/isaaclab_hexapod/data/usd/hexapod_full_mesh_mjx_parity.usd" \
  --report "${REPORT_TMP}" \
  --headless --experience "${ISAACLAB_ROOT}/apps/isaaclab.python.kit"

python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "${REPORT_TMP}"
mv "${REPORT_TMP}" "${REPORT}"
trap - EXIT
