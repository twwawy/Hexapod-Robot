#!/usr/bin/env bash
set -euo pipefail

REPO="${HEXAPOD_ROBOT_REPO:-$HOME/Hexapod-Robot}"
PY="$HOME/.venvs/hexapod-mjx/bin/python"
ARTIFACTS_DIR="SW/mjx/artifacts/residual_rl_runs"
RUN_LABEL="residualrl"
POLICY_PATH=""
LATEST_POLICY_PATH=""
METRICS_PATH=""
METADATA_PATH=""
POSE_IMAGE_PATH=""
POSE_METADATA_PATH=""
OUTPUT_VIDEO=""
DURATION_SEC="12"
RUN_DATE_DIR=""
RUN_OUTPUT_DIR=""
FORWARD_CMD="0.18"
LATERAL_CMD="0.0"
YAW_CMD="0.0"
SKIP_TRAIN=0
FRESH=0
WANDB_ENABLED=0
WANDB_PROJECT="hexapod-residual-rl"
WANDB_ENTITY=""
WANDB_GROUP=""
WANDB_JOB_TYPE="mjx-train"
WANDB_MODE="online"
WANDB_TAGS=""
WANDB_NAME=""
WANDB_RUN_ID=""
TRAIN_ARGS=()
ORIGINAL_ARGS=("$@")
STARTED_AT="$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

NUM_ENVS="24"
ROLLOUT_STEPS="48"
NUM_UPDATES="4"
PPO_EPOCHS="3"
MINIBATCH_SIZE="96"
LEARNING_RATE="3e-4"
HIDDEN_SIZE="128"
SEED="0"
USER_RESUME_PATH=""
AUTO_RESUME_PATH=""
RUN_STEM=""
TRAIN_CMD=()
POSE_EXPORT_CMD=()
VIEW_CMD=()
WANDB_UPLOAD_CMD=()
TRAIN_CMD_STR=""
POSE_EXPORT_CMD_STR=""
VIEW_CMD_STR=""
WANDB_UPLOAD_CMD_STR=""
INVOCATION_CMD=""

print_help() {
  cat <<'EOF'
Usage:
  residual_rl_run.sh [wrapper options] [train_residual_ppo.py options...]

Wrapper options:
  --artifacts-dir PATH      Run artifact root. Default: SW/mjx/artifacts/residual_rl_runs
  --run-label NAME          Short label included in run folder/file names. Default: residualrl
  --policy-path PATH        Best-policy checkpoint path. Default: auto-generated per run
  --latest-policy-path PATH Latest-training checkpoint path. Default: <policy-stem>_latest.pkl
  --metrics-path PATH       Training metrics JSON path. Default: auto-generated per run
  --metadata-path PATH      Run metadata JSON path. Default: auto-generated per run
  --output-video PATH       Output MP4 path. Default: same stem as policy-path with .mp4
  --duration-sec N          Replay length for the saved MP4. Default: 12
  --forward-cmd N           Forward command used during replay. Default: 0.18
  --lateral-cmd N           Lateral command used during replay. Default: 0.0
  --yaw-cmd N               Yaw command used during replay. Default: 0.0
  --wandb                   Enable Weights & Biases logging/uploads.
  --wandb-project NAME      W&B project. Default: hexapod-residual-rl
  --wandb-entity NAME       W&B entity/team.
  --wandb-group NAME        W&B group label.
  --wandb-job-type NAME     W&B job type. Default: mjx-train
  --wandb-mode MODE         W&B mode. Default: online
  --wandb-tags CSV          Comma-separated W&B tags.
  --wandb-name NAME         Override W&B run name. Default: run stem
  --skip-train              Skip training and only render the current best checkpoint.
  --fresh                   Ignore an existing checkpoint instead of resuming from it.
  -h, --help                Show this help.

Everything else is forwarded to SW/mjx/train_residual_ppo.py.
The wrapper writes every new run into `ARTIFACTS_DIR/YYYYMMDD/<run-stem>/`, keeps a
latest checkpoint during training, renders one best-policy MP4 at the end, and when
`--fresh` is used also saves a separate neutral-pose PNG + JSON artifact for that exact
run. In `--skip-train` mode, omitting `--policy-path` automatically picks the newest
saved best-policy checkpoint anywhere under the run artifact root.
EOF
}

join_cmd() {
  local out=""
  local quoted=""
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    out+="${out:+ }${quoted}"
  done
  printf '%s' "$out"
}

resolve_repo_path() {
  if [[ "$1" = /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "$REPO" "$1"
  fi
}
find_latest_best_checkpoint() {
  env REPO="$REPO" ARTIFACTS_DIR="$ARTIFACTS_DIR" "$PY" - <<'PY'
from pathlib import Path
import os
import sys
repo = Path(os.environ["REPO"]).resolve()
artifacts_dir = Path(os.environ["ARTIFACTS_DIR"])
if not artifacts_dir.is_absolute():
    artifacts_dir = (repo / artifacts_dir).resolve()
candidates = [path for path in artifacts_dir.rglob("*.pkl") if not path.name.endswith("_latest.pkl")]
if not candidates:
    sys.exit(1)
latest = max(candidates, key=lambda path: path.stat().st_mtime)
print(latest)
PY
}
find_latest_resume_checkpoint() {
  env REPO="$REPO" ARTIFACTS_DIR="$ARTIFACTS_DIR" RUN_LABEL="$RUN_LABEL" "$PY" - <<'PY'
from pathlib import Path
import os
import sys
repo = Path(os.environ["REPO"]).resolve()
artifacts_dir = Path(os.environ["ARTIFACTS_DIR"])
run_label = os.environ["RUN_LABEL"]
if not artifacts_dir.is_absolute():
    artifacts_dir = (repo / artifacts_dir).resolve()
patterns = [
    (f"*_{run_label}_*_latest.pkl", True),
    (f"*_{run_label}_*.pkl", False),
    ("*_latest.pkl", True),
    ("*.pkl", False),
]
for pattern, latest_only in patterns:
    candidates = []
    for path in artifacts_dir.rglob(pattern):
        if latest_only and not path.name.endswith("_latest.pkl"):
            continue
        if not latest_only and path.name.endswith("_latest.pkl"):
            continue
        candidates.append(path)
    if candidates:
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        print(latest)
        sys.exit(0)
sys.exit(1)
PY
}

parse_train_overrides() {
  local index=0
  while ((index < ${#TRAIN_ARGS[@]})); do
    case "${TRAIN_ARGS[index]}" in
      --num-envs|--num-env)
        NUM_ENVS="${TRAIN_ARGS[index + 1]:?missing value for ${TRAIN_ARGS[index]}}"
        index=$((index + 2))
        ;;
      --num-envs=*|--num-env=*)
        NUM_ENVS="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --rollout-steps)
        ROLLOUT_STEPS="${TRAIN_ARGS[index + 1]:?missing value for --rollout-steps}"
        index=$((index + 2))
        ;;
      --rollout-steps=*)
        ROLLOUT_STEPS="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --num-updates|--num-update)
        NUM_UPDATES="${TRAIN_ARGS[index + 1]:?missing value for ${TRAIN_ARGS[index]}}"
        index=$((index + 2))
        ;;
      --num-updates=*|--num-update=*)
        NUM_UPDATES="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --ppo-epochs)
        PPO_EPOCHS="${TRAIN_ARGS[index + 1]:?missing value for --ppo-epochs}"
        index=$((index + 2))
        ;;
      --ppo-epochs=*)
        PPO_EPOCHS="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --minibatch-size)
        MINIBATCH_SIZE="${TRAIN_ARGS[index + 1]:?missing value for --minibatch-size}"
        index=$((index + 2))
        ;;
      --minibatch-size=*)
        MINIBATCH_SIZE="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --learning-rate)
        LEARNING_RATE="${TRAIN_ARGS[index + 1]:?missing value for --learning-rate}"
        index=$((index + 2))
        ;;
      --learning-rate=*)
        LEARNING_RATE="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --hidden-size)
        HIDDEN_SIZE="${TRAIN_ARGS[index + 1]:?missing value for --hidden-size}"
        index=$((index + 2))
        ;;
      --hidden-size=*)
        HIDDEN_SIZE="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --seed)
        SEED="${TRAIN_ARGS[index + 1]:?missing value for --seed}"
        index=$((index + 2))
        ;;
      --seed=*)
        SEED="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      --resume-path)
        USER_RESUME_PATH="${TRAIN_ARGS[index + 1]:?missing value for --resume-path}"
        index=$((index + 2))
        ;;
      --resume-path=*)
        USER_RESUME_PATH="${TRAIN_ARGS[index]#*=}"
        index=$((index + 1))
        ;;
      *)
        index=$((index + 1))
        ;;
    esac
  done
}

write_run_metadata() {
  local status="$1"
  env \
    METADATA_PATH="$METADATA_PATH" \
    STATUS="$status" \
    STARTED_AT="$STARTED_AT" \
    RUN_STEM="$RUN_STEM" \
    RUN_LABEL="$RUN_LABEL" \
    RUN_DATE_DIR="$RUN_DATE_DIR" \
    RUN_OUTPUT_DIR="$RUN_OUTPUT_DIR" \
    ARTIFACTS_DIR="$ARTIFACTS_DIR" \
    REPO="$REPO" \
    POLICY_PATH="$POLICY_PATH" \
    LATEST_POLICY_PATH="$LATEST_POLICY_PATH" \
    METRICS_PATH="$METRICS_PATH" \
    OUTPUT_VIDEO="$OUTPUT_VIDEO" \
    POSE_IMAGE_PATH="$POSE_IMAGE_PATH" \
    POSE_METADATA_PATH="$POSE_METADATA_PATH" \
    SKIP_TRAIN="$SKIP_TRAIN" \
    FRESH="$FRESH" \
    NUM_ENVS="$NUM_ENVS" \
    ROLLOUT_STEPS="$ROLLOUT_STEPS" \
    NUM_UPDATES="$NUM_UPDATES" \
    PPO_EPOCHS="$PPO_EPOCHS" \
    MINIBATCH_SIZE="$MINIBATCH_SIZE" \
    LEARNING_RATE="$LEARNING_RATE" \
    HIDDEN_SIZE="$HIDDEN_SIZE" \
    SEED="$SEED" \
    INVOCATION_CMD="$INVOCATION_CMD" \
    TRAIN_CMD_STR="$TRAIN_CMD_STR" \
    POSE_EXPORT_CMD_STR="$POSE_EXPORT_CMD_STR" \
    VIEW_CMD_STR="$VIEW_CMD_STR" \
    WANDB_ENABLED="$WANDB_ENABLED" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    WANDB_GROUP="$WANDB_GROUP" \
    WANDB_JOB_TYPE="$WANDB_JOB_TYPE" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_TAGS="$WANDB_TAGS" \
    WANDB_NAME="$WANDB_NAME" \
    WANDB_RUN_ID="$WANDB_RUN_ID" \
    "$PY" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (Path(os.environ["REPO"]) / path)

payload = {
    "status": os.environ["STATUS"],
    "started_at": os.environ["STARTED_AT"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "run_stem": os.environ["RUN_STEM"],
    "run_label": os.environ["RUN_LABEL"],
    "run_date_dir": os.environ.get("RUN_DATE_DIR") or None,
    "run_output_dir": os.environ.get("RUN_OUTPUT_DIR") or None,
    "artifacts_dir": os.environ["ARTIFACTS_DIR"],
    "invocation_command": os.environ["INVOCATION_CMD"],
    "train_command": os.environ.get("TRAIN_CMD_STR", ""),
    "neutral_pose_export_command": os.environ.get("POSE_EXPORT_CMD_STR", ""),
    "visualize_command": os.environ.get("VIEW_CMD_STR", ""),
    "wrapper": {
        "skip_train": os.environ["SKIP_TRAIN"] == "1",
        "fresh": os.environ["FRESH"] == "1",
    },
    "hyperparameters": {
        "num_envs": int(os.environ["NUM_ENVS"]),
        "rollout_steps": int(os.environ["ROLLOUT_STEPS"]),
        "num_updates": int(os.environ["NUM_UPDATES"]),
        "ppo_epochs": int(os.environ["PPO_EPOCHS"]),
        "minibatch_size": int(os.environ["MINIBATCH_SIZE"]),
        "learning_rate": float(os.environ["LEARNING_RATE"]),
        "hidden_size": int(os.environ["HIDDEN_SIZE"]),
        "seed": int(os.environ["SEED"]),
    },
    "wandb": {
        "enabled": os.environ["WANDB_ENABLED"] == "1",
        "project": os.environ["WANDB_PROJECT"] or None,
        "entity": os.environ["WANDB_ENTITY"] or None,
        "group": os.environ["WANDB_GROUP"] or None,
        "job_type": os.environ["WANDB_JOB_TYPE"] or None,
        "mode": os.environ["WANDB_MODE"] or None,
        "tags": [tag for tag in os.environ["WANDB_TAGS"].split(",") if tag],
        "name": os.environ["WANDB_NAME"] or None,
        "run_id": os.environ["WANDB_RUN_ID"] or None,
    },
    "paths": {
        "policy_checkpoint": os.environ["POLICY_PATH"],
        "latest_checkpoint": os.environ["LATEST_POLICY_PATH"],
        "metrics_json": os.environ["METRICS_PATH"],
        "run_metadata_json": os.environ["METADATA_PATH"],
        "neutral_pose_image": os.environ["POSE_IMAGE_PATH"] or None,
        "neutral_pose_metadata": os.environ["POSE_METADATA_PATH"] or None,
        "output_video": os.environ["OUTPUT_VIDEO"],
        "policy_checkpoint_abs": str(resolve(os.environ["POLICY_PATH"])),
        "latest_checkpoint_abs": str(resolve(os.environ["LATEST_POLICY_PATH"])),
        "metrics_json_abs": str(resolve(os.environ["METRICS_PATH"])),
        "run_metadata_json_abs": str(resolve(os.environ["METADATA_PATH"])),
        "neutral_pose_image_abs": str(resolve(os.environ["POSE_IMAGE_PATH"])) if os.environ["POSE_IMAGE_PATH"] else None,
        "neutral_pose_metadata_abs": str(resolve(os.environ["POSE_METADATA_PATH"])) if os.environ["POSE_METADATA_PATH"] else None,
        "output_video_abs": str(resolve(os.environ["OUTPUT_VIDEO"])),
        "run_output_dir_abs": str(resolve(os.environ["RUN_OUTPUT_DIR"])) if os.environ.get("RUN_OUTPUT_DIR") else None,
    },
}
metrics_path = resolve(os.environ["METRICS_PATH"])
if metrics_path.exists():
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics_payload.get("history", [])
    if history:
        last = history[-1]
        payload["metrics_summary"] = {
            "history_length": len(history),
            "final_mean_reward": last.get("mean_reward"),
            "best_mean_reward": last.get("best_mean_reward", last.get("mean_reward")),
            "best_update": last.get("best_update"),
        }
metadata_path = resolve(os.environ["METADATA_PATH"])
pose_metadata_path = os.environ.get("POSE_METADATA_PATH")
if pose_metadata_path:
    pose_metadata = resolve(pose_metadata_path)
    if pose_metadata.exists():
        payload["neutral_pose_summary"] = json.loads(pose_metadata.read_text(encoding="utf-8"))
metadata_path.parent.mkdir(parents=True, exist_ok=True)
metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

on_error() {
  write_run_metadata failed
}

while (($#)); do
  case "$1" in
    --artifacts-dir)
      ARTIFACTS_DIR="${2:?missing value for --artifacts-dir}"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="${2:?missing value for --run-label}"
      shift 2
      ;;
    --policy-path)
      POLICY_PATH="${2:?missing value for --policy-path}"
      shift 2
      ;;
    --latest-policy-path)
      LATEST_POLICY_PATH="${2:?missing value for --latest-policy-path}"
      shift 2
      ;;
    --metrics-path)
      METRICS_PATH="${2:?missing value for --metrics-path}"
      shift 2
      ;;
    --metadata-path)
      METADATA_PATH="${2:?missing value for --metadata-path}"
      shift 2
      ;;
    --output-video)
      OUTPUT_VIDEO="${2:?missing value for --output-video}"
      shift 2
      ;;
    --duration-sec)
      DURATION_SEC="${2:?missing value for --duration-sec}"
      shift 2
      ;;
    --forward-cmd)
      FORWARD_CMD="${2:?missing value for --forward-cmd}"
      shift 2
      ;;
    --lateral-cmd)
      LATERAL_CMD="${2:?missing value for --lateral-cmd}"
      shift 2
      ;;
    --yaw-cmd)
      YAW_CMD="${2:?missing value for --yaw-cmd}"
      shift 2
      ;;
    --wandb)
      WANDB_ENABLED=1
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="${2:?missing value for --wandb-project}"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="${2:?missing value for --wandb-entity}"
      shift 2
      ;;
    --wandb-group)
      WANDB_GROUP="${2:?missing value for --wandb-group}"
      shift 2
      ;;
    --wandb-job-type)
      WANDB_JOB_TYPE="${2:?missing value for --wandb-job-type}"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="${2:?missing value for --wandb-mode}"
      shift 2
      ;;
    --wandb-tags)
      WANDB_TAGS="${2:?missing value for --wandb-tags}"
      shift 2
      ;;
    --wandb-name)
      WANDB_NAME="${2:?missing value for --wandb-name}"
      shift 2
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

parse_train_overrides
RUN_STEM="$(date +%Y%m%d-%H%M%S)_${RUN_LABEL}_env${NUM_ENVS}_roll${ROLLOUT_STEPS}_upd${NUM_UPDATES}_ep${PPO_EPOCHS}_mb${MINIBATCH_SIZE}_hid${HIDDEN_SIZE}_lr${LEARNING_RATE}_seed${SEED}"
RUN_DATE_DIR="$(date +%Y%m%d)"
RUN_OUTPUT_DIR="$ARTIFACTS_DIR/$RUN_DATE_DIR/$RUN_STEM"

if [[ -z "$WANDB_NAME" ]]; then
  WANDB_NAME="$RUN_STEM"
fi
if [[ -z "$WANDB_RUN_ID" ]]; then
  WANDB_RUN_ID="$RUN_STEM"
fi

if [[ "$SKIP_TRAIN" == "1" && -z "$POLICY_PATH" ]]; then
  if ! POLICY_PATH="$(find_latest_best_checkpoint)"; then
    echo "--skip-train could not find any saved best-policy checkpoint. Pass --policy-path or run training first." >&2
    exit 2
  fi
fi

if [[ -z "$POLICY_PATH" ]]; then
  POLICY_PATH="$RUN_OUTPUT_DIR/${RUN_STEM}.pkl"
fi
if [[ -z "$METRICS_PATH" ]]; then
  METRICS_PATH="$RUN_OUTPUT_DIR/${RUN_STEM}_metrics.json"
fi
policy_dir="$(dirname "$POLICY_PATH")"
RUN_OUTPUT_DIR="$policy_dir"
policy_stem="$(basename "$POLICY_PATH")"
policy_stem="${policy_stem%.*}"
if [[ -z "$LATEST_POLICY_PATH" ]]; then
  LATEST_POLICY_PATH="$policy_dir/${policy_stem}_latest.pkl"
fi
if [[ "$SKIP_TRAIN" == "1" && -z "$METADATA_PATH" ]]; then
  METADATA_PATH="$policy_dir/${policy_stem}_view.json"
elif [[ -z "$METADATA_PATH" ]]; then
  METADATA_PATH="$policy_dir/${policy_stem}_run.json"
fi
if [[ -z "$POSE_IMAGE_PATH" ]]; then
  POSE_IMAGE_PATH="$policy_dir/${policy_stem}_neutral_pose.png"
fi
if [[ -z "$POSE_METADATA_PATH" ]]; then
  POSE_METADATA_PATH="$policy_dir/${policy_stem}_neutral_pose.json"
fi
if [[ "$SKIP_TRAIN" == "1" && -z "$OUTPUT_VIDEO" ]]; then
  OUTPUT_VIDEO="$policy_dir/${policy_stem}_view.mp4"
elif [[ -z "$OUTPUT_VIDEO" ]]; then
  OUTPUT_VIDEO="$policy_dir/${policy_stem}.mp4"
fi

INVOCATION_CMD="$(join_cmd "$0" "${ORIGINAL_ARGS[@]}")"
cd "$REPO"
trap on_error ERR

TRAIN_CMD=(env -u LD_LIBRARY_PATH "$PY" SW/mjx/train_residual_ppo.py --output-path "$POLICY_PATH" --latest-output-path "$LATEST_POLICY_PATH" --metrics-path "$METRICS_PATH")
if [[ -n "$USER_RESUME_PATH" ]]; then
  TRAIN_CMD+=(--resume-path "$USER_RESUME_PATH")
elif [[ "$FRESH" != "1" ]]; then
  if AUTO_RESUME_PATH="$(find_latest_resume_checkpoint)"; then
    TRAIN_CMD+=(--resume-path "$AUTO_RESUME_PATH")
  fi
fi
if [[ "$WANDB_ENABLED" == "1" ]]; then
  TRAIN_CMD+=(
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-job-type "$WANDB_JOB_TYPE"
    --wandb-mode "$WANDB_MODE"
    --wandb-name "$WANDB_NAME"
    --wandb-run-id "$WANDB_RUN_ID"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    TRAIN_CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_GROUP" ]]; then
    TRAIN_CMD+=(--wandb-group "$WANDB_GROUP")
  fi
  if [[ -n "$WANDB_TAGS" ]]; then
    TRAIN_CMD+=(--wandb-tags "$WANDB_TAGS")
  fi
fi
TRAIN_CMD+=("${TRAIN_ARGS[@]}")
TRAIN_CMD_STR="$(join_cmd "${TRAIN_CMD[@]}")"
if [[ "$FRESH" == "1" ]]; then
  POSE_EXPORT_CMD=(env -u LD_LIBRARY_PATH MUJOCO_GL=egl "$PY" SW/mjx/export_neutral_pose_artifacts.py --output-image-path "$POSE_IMAGE_PATH" --output-metadata-path "$POSE_METADATA_PATH")
  POSE_EXPORT_CMD_STR="$(join_cmd "${POSE_EXPORT_CMD[@]}")"
fi

VIEW_CMD=(env -u LD_LIBRARY_PATH MUJOCO_GL=egl "$PY" SW/mjx/visualize_residual_policy.py --policy-path "$POLICY_PATH" --output-video "$OUTPUT_VIDEO" --duration-sec "$DURATION_SEC" --forward-cmd "$FORWARD_CMD" --lateral-cmd "$LATERAL_CMD" --yaw-cmd "$YAW_CMD")
VIEW_CMD_STR="$(join_cmd "${VIEW_CMD[@]}")"

if [[ "$WANDB_ENABLED" == "1" ]]; then
  WANDB_UPLOAD_CMD=(env -u LD_LIBRARY_PATH "$PY" SW/mjx/upload_visual_artifacts_wandb.py --checkpoint-path "$POLICY_PATH" --latest-checkpoint-path "$LATEST_POLICY_PATH" --metrics-path "$METRICS_PATH" --run-metadata-path "$METADATA_PATH" --video-path "$OUTPUT_VIDEO")
  if [[ -n "$POSE_IMAGE_PATH" ]]; then
    WANDB_UPLOAD_CMD+=(--neutral-pose-image-path "$POSE_IMAGE_PATH")
  fi
  if [[ -n "$POSE_METADATA_PATH" ]]; then
    WANDB_UPLOAD_CMD+=(--neutral-pose-metadata-path "$POSE_METADATA_PATH")
  fi
  WANDB_UPLOAD_CMD_STR="$(join_cmd "${WANDB_UPLOAD_CMD[@]}")"
fi

write_run_metadata started
printf 'run_stem: %s\nrun_output_dir: %s\npolicy_checkpoint: %s\nlatest_checkpoint: %s\nmetrics_path: %s\nrun_metadata: %s\nneutral_pose_image: %s\nneutral_pose_metadata: %s\noutput_video: %s\n' "$RUN_STEM" "$RUN_OUTPUT_DIR" "$POLICY_PATH" "$LATEST_POLICY_PATH" "$METRICS_PATH" "$METADATA_PATH" "$POSE_IMAGE_PATH" "$POSE_METADATA_PATH" "$OUTPUT_VIDEO"

if [[ "$FRESH" == "1" ]]; then
  printf '[save-neutral-pose] %s\n' "$POSE_EXPORT_CMD_STR"
  "${POSE_EXPORT_CMD[@]}"
  write_run_metadata pose_exported
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  printf '[train-best] %s\n' "$TRAIN_CMD_STR"
  "${TRAIN_CMD[@]}"
fi

printf '[save-mp4] %s\n' "$VIEW_CMD_STR"
"${VIEW_CMD[@]}"

if [[ "$WANDB_ENABLED" == "1" ]]; then
  printf '[wandb-artifacts] %s\n' "$WANDB_UPLOAD_CMD_STR"
  "${WANDB_UPLOAD_CMD[@]}"
fi

write_run_metadata completed
trap - ERR