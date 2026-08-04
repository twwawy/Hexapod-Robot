#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$HOME/Desktop/Hexapod-MJX-가이드"
MODE_ARGS=()

if (($#)); then
  case "$1" in
    fresh|새로)
      MODE_ARGS+=(--fresh)
      shift
      ;;
    이어서|resume|continue)
      shift
      ;;
    -*)
      ;;
    *)
      echo "Usage: 큰병렬.sh [fresh|이어서] [train/visualize options...]" >&2
      exit 2
      ;;
  esac
fi

exec "$SCRIPT_DIR/residual_rl_run.sh" \
  --run-label largeparallel \
  --duration-sec 5 \
  --num-envs 96 \
  --rollout-steps 96 \
  --num-updates 20 \
  --ppo-epochs 4 \
  --minibatch-size 512 \
  --hidden-size 128 \
  --wandb \
  --wandb-group largeparallel \
  "${MODE_ARGS[@]}" \
  "$@"
