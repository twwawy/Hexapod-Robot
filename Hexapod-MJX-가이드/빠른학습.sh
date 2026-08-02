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
      echo "Usage: 빠른학습.sh [fresh|이어서] [train/visualize options...]" >&2
      exit 2
      ;;
  esac
fi

exec "$SCRIPT_DIR/residual_rl_run.sh" \
  --run-label fasttest \
  --duration-sec 3 \
  --num-envs 8 \
  --rollout-steps 8 \
  --num-updates 2 \
  --ppo-epochs 1 \
  --minibatch-size 16 \
  --hidden-size 32 \
  "${MODE_ARGS[@]}" \
  "$@"
