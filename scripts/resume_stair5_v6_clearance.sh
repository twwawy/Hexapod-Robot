#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint=/home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/stair5-path-v6-warmstart/02_tripod-stair5_try00/checkpoints/000000204800
exec bash "$repo/scripts/train_adaptive_curriculum.sh" \
  --profile hybrid --command-mode terrain --perception teacher \
  --start-index 2 --restore "$checkpoint" --migrate-path-v6 \
  --run-name hybrid-stair5-v6-clearance2cm \
  --stair-clearance-extra 0.02 \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 9 --num-eval-envs 16 --episode-length 8000 \
  --action-profile terrain_mid --discounting .997 \
  --best-video-duration 20 --baseline-comparison-seconds 0 \
  --promote-key eval/episode_terrain_success --promote-threshold .70 \
  --max-retries -1 --on-stage-failure stop \
  --wandb-project hexapod-forward-completion --wandb --wandb-mode online "$@"
