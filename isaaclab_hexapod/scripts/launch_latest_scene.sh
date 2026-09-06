#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"
USD="${REPO_ROOT}/isaaclab_hexapod/data/usd/hexapod_full_mesh_mjx_parity.usd"
export PYTHONPATH="${REPO_ROOT}/isaaclab_hexapod/source/hexapod_isaaclab${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${USD}" ]]; then
  echo "Missing full-mesh USD; generating it first."
  "${REPO_ROOT}/isaaclab_hexapod/scripts/build_asset.sh"
fi

exec "${ISAACLAB_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/isaaclab_hexapod/scripts/launch_latest_scene.py" "$@"
