#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/huro/IsaacLab}"
NUM_ENVS="${HEXAPOD_NUM_ENVS:-512}"
STAGE_ITERATIONS="${HEXAPOD_STAGE_ITERATIONS:-500}"
DEVICE="${HEXAPOD_DEVICE:-cuda:0}"
WANDB_PROJECT="${HEXAPOD_WANDB_PROJECT:-hexapod-isaac-lidar-depth-curriculum}"
WANDB_ENTITY="${HEXAPOD_WANDB_ENTITY:-hurolilys-inha-university}"
WANDB_SITE_PACKAGES="${HEXAPOD_WANDB_SITE_PACKAGES:-/home/huro/isaacsim-5.1/lib/python3.11/site-packages}"
SOURCE_ROOT="${REPO_ROOT}/isaaclab_hexapod/source"

if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "HEXAPOD_NUM_ENVS must be a positive integer" >&2
  exit 2
fi

export PYTHONPATH="${SOURCE_ROOT}/hexapod_isaaclab${PYTHONPATH:+:${PYTHONPATH}}:${WANDB_SITE_PACKAGES}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_USERNAME="${WANDB_ENTITY}"

echo "Hexapod perceptive curriculum: levels=0..9 envs=${NUM_ENVS} device=${DEVICE}"
echo "Sensors: inverted MID-360 ray caster + provisional 32x24 depth ray caster"
echo "W&B: entity=${WANDB_ENTITY} project=${WANDB_PROJECT} stage_iterations=${STAGE_ITERATIONS}"

exec "${ISAACLAB_ROOT}/isaaclab.sh" -p \
  "${REPO_ROOT}/isaaclab_hexapod/scripts/train_staged_rsl_rl.py" \
  --task Hexapod-Perceptive-Direct-v0 \
  --num_envs "${NUM_ENVS}" \
  --stage_iterations "${STAGE_ITERATIONS}" \
  --device "${DEVICE}" \
  --headless \
  --logger wandb \
  --log_project_name "${WANDB_PROJECT}" \
  "$@"
