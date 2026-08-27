#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/huro/isaac-sim-5.1}"

test -x "${ISAACLAB_ROOT}/isaaclab.sh"
test -x "${ISAAC_SIM_ROOT}/python.sh"
if [[ ! -e "${ISAACLAB_ROOT}/_isaac_sim" ]]; then
  ln -s "${ISAAC_SIM_ROOT}" "${ISAACLAB_ROOT}/_isaac_sim"
fi

"${ISAACLAB_ROOT}/_isaac_sim/python.sh" -m pip install \
  gymnasium==1.2.1 rsl-rl-lib==3.1.2 'onnxscript>=0.5'
"${ISAACLAB_ROOT}/_isaac_sim/python.sh" -m pip install -e \
  "${REPO_ROOT}/isaaclab_hexapod/source/hexapod_isaaclab"

echo "HEXAPOD_ISAACLAB_INSTALL_OK"
