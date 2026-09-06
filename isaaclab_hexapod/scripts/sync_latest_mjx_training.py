#!/usr/bin/env python3
"""Snapshot the newest MJX training lineage for the Isaac Lab port.

The newest run is not necessarily safe or even checkpointed.  The handoff
therefore records three separate facts: newest attempted/evaluated run, newest
lineage checkpoint, and newest checkpoint that passed the strict safety gate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MJX_ROOT = REPO_ROOT / "mjx"
DEFAULT_OUTPUT = (
    REPO_ROOT / "isaaclab_hexapod/data/training/latest_mjx_training.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _valid_checkpoint(path: str | Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir() and (candidate / "ppo_network_config.json").is_file():
        return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_record(metadata_path: Path) -> dict[str, Any]:
    run_dir = metadata_path.parent
    metadata = _read_json(metadata_path)
    metrics_path = run_dir / "monitor/latest_metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.is_file() else None
    return {
        "run_dir": _relative(run_dir),
        "run_id": metadata.get("run_id", run_dir.name),
        "curriculum_stage": metadata.get("curriculum_stage"),
        "terrain_level": metadata.get("terrain_level"),
        "terrain_name": metadata.get("terrain_name"),
        "terrain_description": metadata.get("terrain_description"),
        "seed": metadata.get("seed"),
        "training_source_commit": metadata.get("git_commit"),
        "action_contract_version": metadata.get("action_contract_version"),
        "observation_contract_version": metadata.get(
            "observation_contract_version"
        ),
        "reward_contract_version": metadata.get("reward_contract_version"),
        "init_checkpoint": _relative(metadata.get("init_checkpoint")),
        "has_local_checkpoint": any(
            (child / "ppo_network_config.json").is_file()
            for child in (run_dir / "checkpoints").glob("*")
            if child.is_dir()
        ),
        "latest_evaluation": None
        if metrics is None
        else {
            "step": metrics.get("step"),
            "updated_at_utc": metrics.get("updated_at_utc"),
            "best_safe": bool(metrics.get("best_safe", False)),
            "best_safe_reasons": metrics.get("best_safe_reasons", []),
            "episode_reward": metrics.get("metrics", {}).get(
                "eval/episode_reward"
            ),
            "terrain_success": metrics.get("metrics", {}).get(
                "eval/episode_terrain_success"
            ),
            "forward_progress_ratio": metrics.get("metrics", {}).get(
                "eval/gait_forward_progress_ratio"
            ),
            "failure_rate": metrics.get("metrics", {}).get(
                "eval/gait_failure_rate"
            ),
            "policy_rejection_rate": metrics.get("metrics", {}).get(
                "eval/gait_policy_rejection_rate"
            ),
            "foot_limited_rate": metrics.get("metrics", {}).get(
                "eval/gait_foot_limited_rate"
            ),
        },
    }


def _newest_safety_checkpoint(
    metadata_paths: list[Path], action_contract: str, observation_contract: str
) -> dict[str, Any] | None:
    for metadata_path in metadata_paths:
        run_dir = metadata_path.parent
        pointer_path = run_dir / "monitor/best_checkpoint.json"
        if not pointer_path.is_file():
            continue
        metadata = _read_json(metadata_path)
        if metadata.get("action_contract_version") != action_contract:
            continue
        if metadata.get("observation_contract_version") != observation_contract:
            continue
        pointer = _read_json(pointer_path)
        checkpoint = _valid_checkpoint(pointer.get("path"))
        if checkpoint is None:
            continue
        return {
            "checkpoint": _relative(checkpoint),
            "run_dir": _relative(run_dir),
            "run_id": metadata.get("run_id", run_dir.name),
            "terrain_level": metadata.get("terrain_level"),
            "score": pointer.get("score"),
            "step": pointer.get("step"),
            "reward_contract": metadata.get("reward_contract_version"),
            "actor_contract_compatible": True,
            "critic_contract_compatible": (
                metadata.get("reward_contract_version")
                == "commanded_progress_motion_gate_v1"
            ),
        }
    return None


def main() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(MJX_ROOT))
    from firmware_mjx_controller import (  # pylint: disable=import-error
        RESIDUAL_SCALE,
        SWING_HEIGHT,
        SWING_HEIGHT_MAX,
        SWING_HEIGHT_MIN,
    )
    from rough_terrain_env import (  # pylint: disable=import-error
        ACTION_CONTRACT_VERSION,
        ACTION_SIZE,
        OBSERVATION_CONTRACT_VERSION,
        OBSERVATION_SIZE,
        REWARD_CONTRACT_VERSION,
        default_config,
    )
    from terrain_curriculum import (  # pylint: disable=import-error
        PLATEAU_DEPTH,
        RAMP_LENGTH,
        STAIR_DEPTH,
        TERRAIN_HALF_WIDTH,
        TERRAIN_LEVELS,
        TERRAIN_START_X,
    )

    metadata_paths = sorted(
        (MJX_ROOT / "runs/terrain").glob("*/run_metadata.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not metadata_paths:
        raise RuntimeError("no MJX terrain run metadata found")

    latest_path = metadata_paths[0]
    latest_metadata = _read_json(latest_path)
    latest_run = _run_record(latest_path)
    latest_action_compatible = (
        latest_metadata.get("action_contract_version")
        == ACTION_CONTRACT_VERSION
    )
    latest_observation_compatible = (
        latest_metadata.get("observation_contract_version")
        == OBSERVATION_CONTRACT_VERSION
    )

    lineage_checkpoint = _valid_checkpoint(latest_metadata.get("init_checkpoint"))
    if lineage_checkpoint is None:
        for metadata_path in metadata_paths:
            pointer = metadata_path.parent / "monitor/level_best_checkpoint.json"
            if pointer.is_file():
                lineage_checkpoint = _valid_checkpoint(_read_json(pointer).get("path"))
                if lineage_checkpoint is not None:
                    break
    if lineage_checkpoint is None:
        raise RuntimeError("no usable checkpoint exists in the latest MJX lineage")
    checkpoint_config = _read_json(lineage_checkpoint / "ppo_network_config.json")

    safety_checkpoint = _newest_safety_checkpoint(
        metadata_paths, ACTION_CONTRACT_VERSION, OBSERVATION_CONTRACT_VERSION
    )
    historical_safety_checkpoint = _newest_safety_checkpoint(
        metadata_paths,
        latest_metadata.get("action_contract_version"),
        latest_metadata.get("observation_contract_version"),
    )
    config = default_config()
    source_paths = (
        MJX_ROOT / "firmware_mjx_controller.py",
        MJX_ROOT / "rough_terrain_env.py",
        MJX_ROOT / "terrain_curriculum.py",
        MJX_ROOT / "train_rough_terrain.py",
    )
    report = {
        "schema_version": 1,
        "purpose": "MJX-to-IsaacLab current training handoff",
        "source_repository_commit": _git_commit(),
        "source_files_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
            for path in source_paths
        },
        "latest_attempt": latest_run,
        "latest_attempt_compatibility": {
            "action": latest_action_compatible,
            "observation": latest_observation_compatible,
            "fully_compatible": (
                latest_action_compatible and latest_observation_compatible
            ),
        },
        "latest_lineage_checkpoint": {
            "checkpoint": _relative(lineage_checkpoint),
            "source": "latest_attempt.init_checkpoint",
            "deployment_eligible": bool(
                latest_action_compatible
                and latest_observation_compatible
                and
                latest_run.get("latest_evaluation", {}).get("best_safe", False)
                if latest_run.get("latest_evaluation")
                else False
            ),
            "action_contract_compatible": latest_action_compatible,
            "observation_contract_compatible": latest_observation_compatible,
            "network_config": checkpoint_config,
        },
        "latest_safety_gated_checkpoint": safety_checkpoint,
        "latest_historical_safety_gated_checkpoint": (
            historical_safety_checkpoint
        ),
        "contracts": {
            "action": {
                "version": ACTION_CONTRACT_VERSION,
                "size": ACTION_SIZE,
                "leg_order": ["RF", "RM", "RB", "LF", "LM", "LB"],
                "per_leg_order": ["x", "y", "z"],
                "normalized_range": [-1.0, 1.0],
                "residual_scale_m": [
                    round(float(value), 6) for value in RESIDUAL_SCALE
                ],
                "swing_height_m": {
                    "action_minus_one": float(SWING_HEIGHT_MIN),
                    "action_zero": float(SWING_HEIGHT),
                    "action_plus_one": float(SWING_HEIGHT_MAX),
                },
            },
            "observation": {
                "version": OBSERVATION_CONTRACT_VERSION,
                "size": OBSERVATION_SIZE,
            },
            "reward": {
                "version": REWARD_CONTRACT_VERSION,
                "weights": {
                    key: float(value) for key, value in config.reward.items()
                },
                "success_bonus": float(config.success_bonus),
                "failure_penalty": float(config.failure_penalty),
            },
            "timing": {
                "physics_dt_s": 0.0025,
                "firmware_dt_s": 0.005,
                "policy_dt_s": 0.02,
                "firmware_ticks_per_policy_step": 4,
                "physics_steps_per_policy_step": 8,
            },
            "network": {
                "actor_hidden_layers": [256, 256, 128],
                "critic_hidden_layers": [256, 256, 128],
                "activation": "silu",
                "distribution": "tanh_normal",
                "normalize_observations": True,
            },
            "terrain_levels": [
                {
                    "level": level.level,
                    "name": level.name,
                    "kind": level.kind,
                    "rough_amplitude_m": level.rough_amplitude,
                    "slope_degrees": level.slope_degrees,
                    "stair_count": level.stair_count,
                    "stair_riser_m": level.stair_riser,
                    "goal_x_m": level.goal_x,
                    "final_height_m": level.final_height,
                }
                for level in TERRAIN_LEVELS
            ],
            "terrain_geometry": {
                "start_x_m": TERRAIN_START_X,
                "half_width_m": TERRAIN_HALF_WIDTH,
                "stair_depth_m": STAIR_DEPTH,
                "ramp_length_m": RAMP_LENGTH,
                "plateau_depth_m": PLATEAU_DEPTH,
            },
        },
        "isaac_transfer_gate": {
            "load_mjx_weights_by_default": False,
            "reason": (
                "The newest run uses the prior residual-v3 action scaling, "
                "while current source uses the 100 mm residual-v4 contract; "
                "retrain before loading any checkpoint."
            ),
        },
    }
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(DEFAULT_OUTPUT.resolve())
    print(
        f"LATEST_RUN={latest_run['run_id']} "
        f"LINEAGE_CHECKPOINT={_relative(lineage_checkpoint)} "
        f"DEPLOYMENT_ELIGIBLE={report['latest_lineage_checkpoint']['deployment_eligible']}"
    )


if __name__ == "__main__":
    main()
