#!/usr/bin/env bash
# Local resume preset for best_video_7_0e1017aebbbc13860373.gif.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${HEXAPOD_PYTHON:-$HOME/.venvs/hexapod-mjx/bin/python}"
checkpoint="/home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum/forward-gt-terrain-mid/02_tripod-stair5_try02/checkpoints/000000204800"

# Reject the wrong source BEFORE creating a run, importing JAX or starting W&B.
"$python_bin" - "$repo" "$checkpoint" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
repo, checkpoint = map(Path, sys.argv[1:])
metadata = json.loads((checkpoint/'adaptive_contract.json').read_text())
if not (checkpoint/'ppo_network_config.json').is_file():
    raise SystemExit(f'Incomplete checkpoint: {checkpoint}')
if metadata.get('command_mode') != 'terrain':
    raise SystemExit('This preset requires the recorded terrain task checkpoint.')
mismatches = []
for name, expected in metadata['source_sha256'].items():
    path = (repo/'mjx'/name).resolve()
    if path.parent != (repo/'mjx').resolve() or not path.is_file():
        mismatches.append(name)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        mismatches.append(name)
if mismatches:
    raise SystemExit('Wrong source for the v5 stair checkpoint: '+', '.join(mismatches)+
        '\nUse: bash /home/huro/Hexapod-Robot-stair5-resume/scripts/resume_stair5_best.sh')
print(f'RESUME SOURCE: {repo}\nOBSERVATION: {metadata["observation_contract"]}\nCHECKPOINT: {checkpoint}', flush=True)
PY

exec bash "$repo/scripts/train_adaptive_curriculum.sh" \
  --profile teacher --command-mode terrain --perception teacher \
  --start-index 2 --restore "$checkpoint" \
  --run-name stair5-v5-best-eval-resume \
  --run-root /home/huro/Hexapod-Robot-integration/mjx/runs/adaptive-curriculum \
  --timesteps-per-stage 800000 \
  --num-envs 512 --batch-size 128 --num-minibatches 4 \
  --num-evals 5 --num-eval-envs 16 --episode-length 8000 \
  --action-profile terrain_mid --discounting 0.997 \
  --best-video-duration 40 --baseline-comparison-seconds 20 \
  --promote-key eval/episode_terrain_success --promote-threshold 0.70 \
  --max-retries -1 --on-stage-failure stop \
  --wandb-project hexapod-forward-completion --wandb --wandb-mode online \
  "$@"
