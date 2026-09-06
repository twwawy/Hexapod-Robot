#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"
REPORT="${REPO_ROOT}/isaaclab_hexapod/data/config_validation.json"
REPORT_TMP="$(mktemp "${REPORT}.tmp.XXXXXX")"
trap 'rm -f "${REPORT_TMP}"' EXIT
export PYTHONPATH="${REPO_ROOT}/isaaclab_hexapod/source/hexapod_isaaclab${PYTHONPATH:+:${PYTHONPATH}}"

"${ISAACLAB_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/isaaclab_hexapod/scripts/validate_configs.py" \
  --report "${REPORT_TMP}" \
  --headless --experience "${ISAACLAB_ROOT}/apps/isaaclab.python.kit"

python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "${REPORT_TMP}"
mv "${REPORT_TMP}" "${REPORT}"
trap - EXIT
