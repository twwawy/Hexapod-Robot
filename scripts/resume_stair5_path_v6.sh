#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="${HEXAPOD_CHECKPOINT:-}"
forward=()
while (($#)); do
  case "$1" in
    --checkpoint)
      if (($# < 2)); then echo "--checkpoint requires a path" >&2; exit 2; fi
      checkpoint="$2"; shift 2 ;;
    --checkpoint=*) checkpoint="${1#*=}"; shift ;;
    --help|-h)
      echo "Usage: bash scripts/resume_stair5_path_v6.sh --checkpoint /path/to/checkpoint [curriculum flags]"
      echo "This is an explicit warm start with a fresh optimizer; source contracts are validated."
      exit 0 ;;
    *) forward+=("$1"); shift ;;
  esac
done
if [[ -z "$checkpoint" || ! -f "$checkpoint/adaptive_contract.json" || ! -f "$checkpoint/ppo_network_config.json" ]]; then
  echo "Provide a complete checkpoint directory with --checkpoint PATH or HEXAPOD_CHECKPOINT." >&2
  exit 2
fi
exec bash "$repo/scripts/train_adaptive_curriculum.sh" \
  --profile hybrid --command-mode terrain --perception teacher \
  --start-index 2 --restore "$checkpoint" --migrate-path-v6 \
  --run-name hybrid-stair5-path-v6-warmstart \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 9 --num-eval-envs 16 --episode-length 8000 \
  --action-profile terrain_mid --discounting .997 \
  --best-video-duration 20 --baseline-comparison-seconds 0 \
  --promote-key eval/episode_terrain_success --promote-threshold .70 \
  --max-retries -1 --on-stage-failure stop \
  --wandb-project hexapod-forward-completion --wandb --wandb-mode online "${forward[@]}"
