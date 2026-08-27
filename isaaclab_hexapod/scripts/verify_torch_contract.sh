#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/huro/IsaacLab/_isaac_sim/python.sh}"

"${ISAAC_PYTHON}" "${REPO_ROOT}/isaaclab_hexapod/scripts/verify_torch_contract.py" \
  --report "${REPO_ROOT}/isaaclab_hexapod/data/torch_controller_parity.json"
