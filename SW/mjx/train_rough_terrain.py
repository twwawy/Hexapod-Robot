#!/usr/bin/env python3
"""Train flat command or mixed-terrain residual PPO on MJX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

# MuJoCo defaults to GLFW, which needs an X11 DISPLAY.  Training is normally
# launched in tmux/SSH without one, but the best-policy GIF still needs an
# offscreen OpenGL context.  Select EGL *before* importing MuJoCo (directly or
# through the environment modules).  An explicit user choice always wins.
if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict

from best_policy_video import make_policy_evaluator, render_policy_video
from command_curriculum_env import HexapodCommandCurriculumEnv
from rough_terrain_env import (
    ACTION_CONTRACT_VERSION,
    ACTION_SIZE,
    ATTITUDE_SOURCE,
    CONTACT_SOURCE,
    OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_SIZE,
    HexapodRoughTerrainEnv,
    default_config,
)
from prepare_rl_scene import MIXED_PATCH_NAMES, MIXED_STAIR_COUNT
from terrain_curriculum import terrain_level


class ScoreMonitor:
    """Persist live evaluation history and the best score outside checkpoints."""

    def __init__(self, directory: Path, task: str, score_key: str) -> None:
        self.directory = directory
        self.task = task
        self.score_key = score_key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_json_path = self.directory / "best_score.json"
        self.best_text_path = self.directory / "best_score.txt"
        self.latest_path = self.directory / "latest_metrics.json"
        self.history_path = self.directory / "metrics_history.jsonl"
        self.best_score = -math.inf
        if self.best_json_path.exists():
            try:
                self.best_score = float(
                    json.loads(self.best_json_path.read_text())["score"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A half-written/old monitor file must not block a new run.
                pass

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    @staticmethod
    def _numeric_metrics(metrics) -> dict[str, float]:
        numeric = {}
        for key, value in metrics.items():
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError):
                continue
        return numeric

    def record(self, step: int, metrics) -> tuple[float, bool]:
        numeric_metrics = self._numeric_metrics(metrics)
        score = numeric_metrics.get(self.score_key, float("nan"))
        payload = {
            "task": self.task,
            "score_key": self.score_key,
            "score": score,
            "step": int(step),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": numeric_metrics,
        }
        self._write_json(self.latest_path, payload)
        with self.history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(payload, sort_keys=True) + "\n")

        is_best = math.isfinite(score) and score > self.best_score
        if is_best:
            self.best_score = score
            self._write_json(self.best_json_path, payload)
            self.best_text_path.write_text(
                "\n".join(
                    (
                        f"task: {self.task}",
                        f"score_key: {self.score_key}",
                        f"best_score: {score:.8f}",
                        f"step: {step}",
                        f"updated_at_utc: {payload['updated_at_utc']}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
        return score, is_best


def progress_video_targets(count: int) -> tuple[float, ...]:
    """Return deterministic 0..1 training fractions for trend snapshots."""
    if count < 1:
        raise ValueError("progress video count must be positive")
    if count == 1:
        return (1.0,)
    return tuple(index / (count - 1) for index in range(count))


def _arguments(default_task: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("command", "terrain"),
        default=default_task,
        help="command = flat walking+yaw; terrain = mixed terrain curriculum.",
    )
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None, help="Optional readable experiment name.")
    parser.add_argument(
        "--competence-stage",
        type=int,
        default=None,
        help="Outer mixed-terrain curriculum stage used for metadata/video labels.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Root for automatic <task>/<run-id>/ experiment directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Checkpoint directory. Defaults to <run-dir>/checkpoints.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="compile reset/step only; no PPO training or GPU required",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=100,
        help="Steps in each zero/random smoke rollout (default: 100 = 2 s).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow PPO on CPU (useful only for very small debugging runs)",
    )
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=0.01)
    parser.add_argument(
        "--discounting",
        type=float,
        default=None,
        help="PPO gamma. Defaults to 0.97 for command and 0.99 for terrain.",
    )
    parser.add_argument(
        "--unroll-length",
        type=int,
        default=None,
        help="Rollout horizon. Defaults to 20 for command and 32 for terrain.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument(
        "--network-layers",
        type=int,
        nargs="+",
        default=(256, 256, 128),
        metavar="WIDTH",
        help="Actor/critic MLP layer widths. Default: 256 256 128.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize policy/normalizer from a compatible Brax checkpoint (e.g. flat -> terrain).",
    )
    parser.add_argument(
        "--init-value-function",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also initialize the critic; disabled by default because terrain rewards differ.",
    )
    parser.add_argument("--phase-time", type=float, default=0.5)
    parser.add_argument("--base-swing-height", type=float, default=0.20)
    parser.add_argument("--base-radial-offset", type=float, default=0.07)
    parser.add_argument("--swing-x", type=float, default=None, help="Active task swing X residual limit [m].")
    parser.add_argument("--swing-y", type=float, default=None, help="Active task swing Y residual limit [m].")
    parser.add_argument("--swing-z-low", type=float, default=None, help="Active task negative swing Z limit [m].")
    parser.add_argument("--swing-z-high", type=float, default=None, help="Active task positive swing Z limit [m].")
    parser.add_argument("--stance-z", type=float, default=None, help="Active task stance-only Z residual limit [m].")
    parser.add_argument(
        "--body-filter-time-constant",
        type=float,
        default=0.15,
        help="Smoothing time constant for the six body residual actions [s].",
    )
    parser.add_argument(
        "--body-translation-limit",
        type=float,
        nargs=3,
        default=None,
        metavar=("FORWARD", "LATERAL", "HEIGHT"),
        help="Override active-task body residual translation limits [m].",
    )
    parser.add_argument(
        "--body-rotation-limit-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Override active-task body residual rotation limits [deg].",
    )
    parser.add_argument("--terrain-speed-min", type=float, default=0.03)
    parser.add_argument("--terrain-speed-max", type=float, default=0.18)
    parser.add_argument(
        "--terrain-yaw-limit",
        type=float,
        default=None,
        help="Override level yaw limit; defaults to the terrain curriculum schedule.",
    )
    parser.add_argument(
        "--terrain-layout",
        choices=("mixed", "stairs"),
        default="mixed",
        help="Terrain scene. mixed batches flat/curb/ramp/blocks/stairs/rough lanes.",
    )
    parser.add_argument(
        "--terrain-level",
        type=int,
        choices=range(5),
        default=2,
        help="0..4 mixed-patch difficulty (also selects stairs height range).",
    )
    parser.add_argument(
        "--terrain-randomize",
        action="store_true",
        help="Randomize terrain; level 4 adds per-env friction/mass/servo/damping.",
    )
    parser.add_argument(
        "--terrain-total-rise",
        type=float,
        default=None,
        help="Override total elevation gain across the complete staircase [m].",
    )
    parser.add_argument(
        "--terrain-step-height",
        type=float,
        default=None,
        help="Advanced override for one stair riser [m]; mutually exclusive with --terrain-total-rise.",
    )
    parser.add_argument("--terrain-step-depth", type=float, default=None, help="Override fixed stair depth [m].")
    parser.add_argument("--terrain-ramp-rise", type=float, default=None, help="Override mixed-lane ramp rise [m].")
    parser.add_argument("--terrain-friction", type=float, default=None, help="Override fixed terrain/foot friction scale.")
    parser.add_argument(
        "--flat-friction",
        type=float,
        default=0.8,
        help="Flat plane/foot sliding friction; 0.8 is dry-asphalt nominal.",
    )
    parser.add_argument("--curriculum-forward-only-steps", type=int, default=250)
    parser.add_argument("--curriculum-limited-yaw-steps", type=int, default=250)
    parser.add_argument(
        "--curriculum-speed-min", type=float, nargs=3, default=(0.03, 0.05, 0.03)
    )
    parser.add_argument(
        "--curriculum-speed-max", type=float, nargs=3, default=(0.10, 0.18, 0.27)
    )
    parser.add_argument(
        "--curriculum-yaw-limit", type=float, nargs=3, default=(0.00, 0.15, 0.35)
    )
    parser.add_argument(
        "--curriculum-lateral-limit", type=float, nargs=3, default=(0.00, 0.05, 0.10)
    )
    parser.add_argument(
        "--monitor-dir",
        type=Path,
        default=None,
        help="Live score files directory. Defaults to <run-dir>/monitor.",
    )
    parser.add_argument(
        "--score-key",
        default="eval/episode_reward",
        help="Evaluation metric used to update best_score.json.",
    )
    parser.add_argument(
        "--best-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render and save the deterministic policy whenever score-key reaches a new best.",
    )
    parser.add_argument(
        "--best-video-path",
        type=Path,
        default=None,
        help=(
            "Full-curriculum/terrain GIF path. Defaults under <run-dir>/videos; "
            "command stage GIFs use the same directory."
        ),
    )
    parser.add_argument("--best-video-duration", type=float, default=20.0)
    parser.add_argument("--best-video-stage0-duration", type=float, default=20.0)
    parser.add_argument("--best-video-stage1-duration", type=float, default=20.0)
    parser.add_argument("--best-video-stage2-duration", type=float, default=20.0)
    parser.add_argument("--best-video-full-duration", type=float, default=20.0)
    parser.add_argument(
        "--progress-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save a deterministic 20-second policy snapshot at evenly spaced "
            "training fractions and upload it to W&B progress/video."
        ),
    )
    parser.add_argument(
        "--progress-video-count",
        type=int,
        default=5,
        help="Snapshots per run, evenly spaced from 0%% through 100%% (default: 5).",
    )
    parser.add_argument(
        "--progress-video-duration",
        type=float,
        default=20.0,
        help="Duration in seconds for every periodic progress snapshot.",
    )
    parser.add_argument(
        "--stage-eval-envs",
        type=int,
        default=8,
        help="Independent reset count for each scripted Stage 0/1/2 evaluation.",
    )
    parser.add_argument("--best-video-fps", type=int, default=20)
    parser.add_argument("--best-video-width", type=int, default=640)
    parser.add_argument("--best-video-height", type=int, default=360)
    parser.add_argument(
        "--best-video-terrain-patch",
        choices=MIXED_PATCH_NAMES,
        default="stairs",
        help="Fixed mixed-terrain lane used for comparable straight best videos.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log Brax PPO progress to the currently logged-in W&B account.")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project used with --wandb (task-specific by default).",
    )
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B run group.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def _make_env(args: argparse.Namespace) -> HexapodRoughTerrainEnv:
    config = default_config()
    config.episode_length = args.episode_length
    config.controller.nominal.phase_time = args.phase_time
    config.controller.nominal.base_swing_height = args.base_swing_height
    config.controller.nominal.base_radial_offset = args.base_radial_offset
    if args.body_filter_time_constant < 0.0:
        raise ValueError("--body-filter-time-constant must be non-negative")
    if args.flat_friction <= 0.0:
        raise ValueError("--flat-friction must be positive")
    config.controller.residual.body_filter_time_constant = (
        args.body_filter_time_constant
    )
    config.terrain.flat_friction = args.flat_friction
    authority = (
        config.controller.residual.command
        if args.task == "command"
        else config.controller.residual.terrain
    )
    for name in (
        "swing_x",
        "swing_y",
        "swing_z_low",
        "swing_z_high",
        "stance_z",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(authority, name, value)
    if args.body_translation_limit is not None:
        if any(value <= 0.0 for value in args.body_translation_limit):
            raise ValueError("--body-translation-limit values must be positive")
        authority.body_translation_limit = tuple(args.body_translation_limit)
    if args.body_rotation_limit_deg is not None:
        if any(not 0.0 < value <= 45.0 for value in args.body_rotation_limit_deg):
            raise ValueError("--body-rotation-limit-deg values must be in (0, 45]")
        authority.body_rotation_limit = tuple(
            np.deg2rad(args.body_rotation_limit_deg)
        )
    config.command_min_speed = args.terrain_speed_min
    config.command_max_speed = args.terrain_speed_max
    config.command_max_yaw_rate = 0.35
    config.command_curriculum.forward_only_steps = args.curriculum_forward_only_steps
    config.command_curriculum.limited_yaw_steps = args.curriculum_limited_yaw_steps
    config.command_curriculum.speed_min = tuple(args.curriculum_speed_min)
    config.command_curriculum.speed_max = tuple(args.curriculum_speed_max)
    config.command_curriculum.lateral_limit = tuple(
        args.curriculum_lateral_limit
    )
    config.command_curriculum.yaw_limit = tuple(args.curriculum_yaw_limit)
    if any(
        minimum < 0.0 or maximum < minimum
        for minimum, maximum in zip(
            args.curriculum_speed_min, args.curriculum_speed_max
        )
    ):
        raise ValueError("each curriculum speed range must satisfy 0 <= min <= max")
    if args.task == "command":
        gait_speed_capacity = (
            config.controller.safety.max_effective_stride
            / config.controller.nominal.phase_time
        )
        if max(args.curriculum_speed_max) > gait_speed_capacity + 1e-6:
            raise ValueError(
                "command curriculum exceeds the controller-owned stride "
                f"capacity ({gait_speed_capacity:.3f} m/s)"
            )
    # Curriculum changes terrain geometry before model-gap difficulty.  Mixed
    # terrain itself is sampled per reset; optional level-4 dynamics are
    # sampled per vectorized environment by domain_randomization.py.
    level_spec = terrain_level(args.terrain_level)
    config.command_max_yaw_rate = (
        args.terrain_yaw_limit
        if args.terrain_yaw_limit is not None
        else level_spec.yaw_limit_rps
    )
    rng = np.random.default_rng(args.seed)
    total_low, total_high = level_spec.stair_total_rise_range_m
    ramp_low, ramp_high = level_spec.ramp_rise_range_m
    randomize_terrain = args.terrain_randomize and args.task == "terrain"
    if args.terrain_total_rise is not None and args.terrain_step_height is not None:
        raise ValueError(
            "--terrain-total-rise and --terrain-step-height are mutually exclusive"
        )
    stair_count = (
        MIXED_STAIR_COUNT
        if args.terrain_layout == "mixed"
        else int(config.terrain.step_count)
    )
    stair_total_rise = (
        args.terrain_total_rise
        if args.terrain_total_rise is not None
        else (
            args.terrain_step_height * stair_count
            if args.terrain_step_height is not None
            else (
                float(rng.uniform(total_low, total_high))
                if randomize_terrain
                else total_high
            )
        )
    )
    if not 0.0 <= stair_total_rise <= 0.20:
        raise ValueError("stair total rise must stay within 0..0.20 m")
    config.terrain.stair_total_rise = stair_total_rise
    config.terrain.step_height = stair_total_rise / stair_count
    config.terrain.ramp_rise = (
        args.terrain_ramp_rise
        if args.terrain_ramp_rise is not None
        else (float(rng.uniform(ramp_low, ramp_high)) if randomize_terrain else ramp_high)
    )
    config.terrain.step_depth = (
        args.terrain_step_depth
        if args.terrain_step_depth is not None
        else (float(rng.uniform(0.180, 0.350)) if randomize_terrain and args.terrain_level >= 3 else 0.250)
    )
    config.terrain.friction = (
        args.terrain_friction
        if args.terrain_friction is not None
        else (float(rng.uniform(0.6, 1.3)) if randomize_terrain and args.terrain_level >= 3 else 1.0)
    )
    if args.task == "terrain":
        config.terrain.patch_probabilities = level_spec.patch_probabilities
    # Physical action semantics stay fixed across levels.  Curriculum changes
    # how expensive intervention is: easy stages strongly prefer the nominal
    # controller, while hard terrain permits the same bounded residual more
    # readily.
    residual_penalties = (
        (-0.040, -0.120, -0.100, -0.060),
        (-0.030, -0.100, -0.080, -0.050),
        (-0.022, -0.080, -0.060, -0.040),
        (-0.015, -0.060, -0.040, -0.030),
        (-0.010, -0.040, -0.025, -0.020),
    )
    if args.task == "terrain":
        (
            config.reward.swing_residual,
            config.reward.stance_residual,
            config.reward.body_translation_residual,
            config.reward.body_rotation_residual,
        ) = residual_penalties[args.terrain_level]
    config.randomization.enabled = randomize_terrain
    # Level 4 dynamics are randomized per vectorized environment by the Brax
    # domain-randomization wrapper.  The base model remains nominal here.
    if args.task == "command":
        return HexapodCommandCurriculumEnv(config=config)
    return HexapodRoughTerrainEnv(config=config, terrain=args.terrain_layout)


def _safe_run_component(value: str) -> str:
    """Keep a human run name portable as a single directory component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not cleaned:
        raise SystemExit("--run-name must contain at least one letter or number")
    return cleaned


def _infer_competence_stage(run_name: str | None) -> int | None:
    """Recover the outer curriculum stage from launcher-generated run names."""
    if not run_name:
        return None
    match = re.search(r"(?:^|-)stage(\d+)(?:-|$)", run_name)
    return int(match.group(1)) if match is not None else None


def _resolve_init_checkpoint(path: Path, network_layers: tuple[int, ...]) -> Path:
    """Resolve and validate a Brax policy-transfer checkpoint."""
    resolved = path.expanduser().resolve()
    if resolved.is_dir() and not (resolved / "ppo_network_config.json").exists():
        candidates = sorted(
            (
                child
                for child in resolved.iterdir()
                if child.is_dir()
                and child.name.isdigit()
                and (child / "ppo_network_config.json").exists()
            ),
            key=lambda child: int(child.name),
        )
        if not candidates:
            raise SystemExit(
                f"--init-checkpoint has no Brax checkpoint directories: {resolved}"
            )
        resolved = candidates[-1]
    config_path = resolved / "ppo_network_config.json"
    if not config_path.exists():
        raise SystemExit(f"missing checkpoint contract: {config_path}")
    try:
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid checkpoint contract: {config_path}") from exc
    observation_shape = checkpoint_config.get("observation_size", {}).get("shape")
    saved_layers = checkpoint_config.get("network_factory_kwargs", {}).get(
        "policy_hidden_layer_sizes"
    )
    if checkpoint_config.get("action_size") != ACTION_SIZE or observation_shape != [OBSERVATION_SIZE]:
        raise SystemExit(
            "--init-checkpoint contract mismatch: "
            f"expected action={ACTION_SIZE} and observation={OBSERVATION_SIZE}"
        )
    if saved_layers is not None and tuple(saved_layers) != tuple(network_layers):
        raise SystemExit(
            f"--init-checkpoint network mismatch: saved={saved_layers}, requested={list(network_layers)}"
        )
    run_metadata_path = resolved.parent.parent / "run_metadata.json"
    if run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        saved_observation_contract = run_metadata.get("observation_contract_version")
        if saved_observation_contract != OBSERVATION_CONTRACT_VERSION:
            raise SystemExit(
                "--init-checkpoint observation semantics mismatch: "
                f"saved={saved_observation_contract!r}, expected={OBSERVATION_CONTRACT_VERSION!r}"
            )
    else:
        raise SystemExit(
            f"--init-checkpoint is missing run metadata required for semantic validation: {run_metadata_path}"
        )
    return resolved


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_run_metadata(args: argparse.Namespace, env: HexapodRoughTerrainEnv) -> None:
    """Persist the exact controller/reward/PPO contract beside each run."""
    try:
        import brax
        brax_version = getattr(brax, "__version__", "unknown")
    except ImportError:
        brax_version = "not-imported"
    config_payload = env._config.to_dict()
    ScoreMonitor._write_json(args.run_dir / "config.json", config_payload)
    ScoreMonitor._write_json(
        args.run_dir / "run_metadata.json",
        {
            "run_id": args.run_id,
            "task": args.task,
            "seed": args.seed,
            "git_commit": _git_commit(),
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
            "contact_source": CONTACT_SOURCE,
            "attitude_source": ATTITUDE_SOURCE,
            "action_size": env.action_size,
            "observation_size": env.observation_size,
            "checkpoint_dir": str(args.output),
            "init_checkpoint": (
                str(args.init_checkpoint) if args.init_checkpoint is not None else None
            ),
            "terrain_layout": args.terrain_layout,
            "terrain_level": args.terrain_level,
            "competence_stage": args.competence_stage,
            "monitor_dir": str(args.monitor_dir),
            "best_video_path": str(args.best_video_path),
            "best_video_paths": args.best_video_paths,
            "progress_video_directory": str(args.progress_video_directory),
            "progress_video_count": args.progress_video_count,
            "progress_video_duration": args.progress_video_duration,
            "jax_version": jax.__version__,
            "mujoco_version": mujoco.__version__,
            "brax_version": brax_version,
            "config": config_payload,
            "ppo": {
                "timesteps": args.timesteps,
                "num_envs": args.num_envs,
                "num_evals": args.num_evals,
                "num_eval_envs": args.num_eval_envs,
                "learning_rate": args.learning_rate,
                "entropy_cost": args.entropy_cost,
                "discounting": args.discounting,
                "unroll_length": args.unroll_length,
                "batch_size": args.batch_size,
                "num_minibatches": args.num_minibatches,
                "num_updates_per_batch": args.num_updates_per_batch,
                "network_layers": list(args.network_layers),
            },
        },
    )


def _smoke_test(env: HexapodRoughTerrainEnv, seed: int, steps: int) -> None:
    if steps < 50:
        raise SystemExit("--smoke-steps must be at least 50 (one nominal gait cycle)")
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    started = time.monotonic()

    def rollout(*, random_actions: bool, rollout_seed: int):
        state = reset(jax.random.PRNGKey(rollout_seed))
        action_key = jax.random.PRNGKey(rollout_seed + 1)
        max_projection = jp.zeros(())
        torso_contacts = jp.zeros(())
        for _ in range(steps):
            action_key, sample_key = jax.random.split(action_key)
            action = (
                jax.random.uniform(
                    sample_key, (env.action_size,), minval=-1.0, maxval=1.0
                )
                if random_actions
                else jp.zeros(env.action_size)
            )
            state = step(state, action)
            max_projection = jp.maximum(
                max_projection, state.metrics["projection_cost"]
            )
            torso_contacts += (
                state.metrics["reward/body_contact"]
                / env._config.reward.body_contact
            )
        state.reward.block_until_ready()
        finite = jp.all(jp.isfinite(state.obs)) & jp.all(jp.isfinite(state.data.qpos))
        if not bool(finite):
            raise RuntimeError("MJX smoke test produced NaN/Inf")
        joint_position = state.data.qpos[env._joint_qpos_ids]
        limit = env._config.controller.safety.joint_limit + 0.05
        if bool(jp.any(jp.abs(joint_position) > limit)):
            raise RuntimeError("MJX smoke test exceeded the guarded joint range")
        if float(torso_contacts) > 0.25 * steps:
            raise RuntimeError("MJX smoke test has persistent torso contact")
        if not random_actions and bool(state.done):
            raise RuntimeError("zero-action nominal gait terminated during smoke test")
        return state, float(max_projection), int(torso_contacts)

    nominal, nominal_projection, nominal_body_contacts = rollout(
        random_actions=False, rollout_seed=seed
    )
    random_state, random_projection, random_body_contacts = rollout(
        random_actions=True, rollout_seed=seed + 1000
    )
    print(
        f"MJX smoke test OK | backend={jax.default_backend()} | "
        f"obs={random_state.obs.shape[-1]} action={env.action_size} "
        f"zero/random={steps}/{steps} steps | "
        f"nominal_projection={nominal_projection:.6f} "
        f"random_projection={random_projection:.6f} "
        f"body_contacts={nominal_body_contacts}/{random_body_contacts} | "
        f"reward={float(random_state.reward):.4f} done={int(random_state.done)} | "
        f"wall={time.monotonic() - started:.2f}s"
    )


def main(default_task: str = "terrain") -> None:
    args = _arguments(default_task)
    if args.competence_stage is None:
        args.competence_stage = _infer_competence_stage(args.run_name)
    if args.competence_stage is not None and args.competence_stage < 0:
        raise SystemExit("--competence-stage must be non-negative")
    if args.discounting is None:
        args.discounting = 0.97 if args.task == "command" else 0.99
    if args.unroll_length is None:
        args.unroll_length = 20 if args.task == "command" else 32
    if args.init_checkpoint is not None:
        args.init_checkpoint = _resolve_init_checkpoint(
            args.init_checkpoint, tuple(args.network_layers)
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    args.run_id = (
        f"{_safe_run_component(args.run_name)}_{timestamp}_seed{args.seed}"
        if args.run_name
        else f"{timestamp}_seed{args.seed}"
    )
    args.run_dir = (args.run_root.expanduser() / args.task / args.run_id).resolve()
    try:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to mix runs in existing directory: {args.run_dir}") from exc
    if args.output is None:
        args.output = args.run_dir / "checkpoints"
    # Orbax checkpointing requires an absolute path.  Resolve user-provided
    # relative paths as well as task defaults before PPO is constructed.
    args.output = args.output.expanduser().resolve()
    if args.wandb_project is None:
        args.wandb_project = (
            "hexapod-command-curriculum"
            if args.task == "command"
            else "hexapod-rough-terrain"
        )
    if args.monitor_dir is None:
        args.monitor_dir = args.run_dir / "monitor"
    args.monitor_dir = args.monitor_dir.expanduser().resolve()
    if args.best_video_path is None:
        args.best_video_path = args.run_dir / "videos" / (
            "best_curriculum_full.gif"
            if args.task == "command"
            else "best_policy.gif"
        )
    args.best_video_path = args.best_video_path.expanduser().resolve()
    if args.best_video_path.suffix.lower() != ".gif":
        raise SystemExit("--best-video-path must end with .gif")
    if args.stage_eval_envs < 1:
        raise SystemExit("--stage-eval-envs must be positive")
    if not 1 <= args.progress_video_count <= 21:
        raise SystemExit("--progress-video-count must be in 1..21")
    if args.progress_video_duration <= 0:
        raise SystemExit("--progress-video-duration must be positive")
    for duration_name in (
        "best_video_duration",
        "best_video_stage0_duration",
        "best_video_stage1_duration",
        "best_video_stage2_duration",
        "best_video_full_duration",
    ):
        if getattr(args, duration_name) <= 0:
            raise SystemExit(f"--{duration_name.replace('_', '-')} must be positive")
    if args.task == "command":
        video_directory = args.best_video_path.parent
        command_video_paths = {
            "stage0": video_directory / "best_stage0_forward.gif",
            "stage1": video_directory / "best_stage1_limited_yaw.gif",
            "stage2": video_directory / "best_stage2_full_command.gif",
            "full": args.best_video_path,
        }
    else:
        command_video_paths = {"policy": args.best_video_path}
    args.best_video_paths = {
        name: str(path.resolve()) for name, path in command_video_paths.items()
    }
    args.progress_video_directory = (
        args.best_video_path.parent / "progress"
    ).resolve()
    for label, directory in (("--output", args.output), ("--monitor-dir", args.monitor_dir)):
        if directory.exists() and (
            not directory.is_dir() or any(directory.iterdir())
        ):
            raise SystemExit(f"refusing to mix a new run into non-empty {label}: {directory}")
    existing_videos = [
        path for path in map(Path, args.best_video_paths.values()) if path.exists()
    ]
    if existing_videos:
        raise SystemExit(f"refusing to overwrite existing best videos: {existing_videos}")
    if args.progress_video_directory.exists() and any(
        args.progress_video_directory.iterdir()
    ):
        raise SystemExit(
            "refusing to mix progress videos into non-empty directory: "
            f"{args.progress_video_directory}"
        )
    print("JAX devices:", jax.devices())
    print(f"run={args.run_dir} contract={ACTION_CONTRACT_VERSION}")
    env = _make_env(args)
    if args.task == "terrain":
        print(
            f"terrain_geometry stage={args.competence_stage} "
            f"level={args.terrain_level} "
            f"stair_total_rise={100.0 * env._config.terrain.stair_total_rise:.2f}cm "
            f"riser={100.0 * env._config.terrain.step_height:.2f}cm "
            f"ramp_rise={100.0 * env._config.terrain.ramp_rise:.2f}cm "
            f"patch_probabilities={tuple(env._config.terrain.patch_probabilities)}"
        )
    _write_run_metadata(args, env)
    if args.smoke:
        _smoke_test(env, args.seed, args.smoke_steps)
        return
    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise SystemExit(
            "GPU JAX backend is required. Install CUDA JAX and verify with: "
            ".venv/bin/python -c 'import jax; print(jax.devices())'\n"
            "Use --allow-cpu only for a tiny debugging run."
        )

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    randomization_fn = None
    if args.task == "terrain" and args.terrain_randomize and args.terrain_level >= 4:
        from domain_randomization import domain_randomize

        randomization_fn = functools.partial(
            domain_randomize,
            friction_range=tuple(env._config.randomization.friction_range),
            mass_range=tuple(env._config.randomization.mass_scale_range),
            actuator_range=tuple(env._config.randomization.actuator_strength_range),
            damping_range=tuple(env._config.randomization.joint_damping_scale_range),
        )

    scripted_envs: dict[str, HexapodCommandCurriculumEnv] = {}
    terrain_video_env: HexapodRoughTerrainEnv | None = None
    if args.task == "command":
        for name, fixed_stage in (
            ("stage0", 0),
            ("stage1", 1),
            ("stage2", 2),
            ("full", None),
        ):
            scripted_envs[name] = HexapodCommandCurriculumEnv(
                config=config_dict.ConfigDict(env._config.to_dict()),
                fixed_curriculum_stage=fixed_stage,
                scripted_commands=True,
            )
    else:
        fixed_patch = (
            MIXED_PATCH_NAMES.index(args.best_video_terrain_patch)
            if args.terrain_layout == "mixed"
            else None
        )
        terrain_video_env = HexapodRoughTerrainEnv(
            config=config_dict.ConfigDict(env._config.to_dict()),
            terrain=args.terrain_layout,
            command_curriculum=True,
            fixed_curriculum_stage=0,
            scripted_commands=True,
            fixed_terrain_patch=fixed_patch,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(args.network_layers),
        value_hidden_layer_sizes=tuple(args.network_layers),
    )
    score_monitor = ScoreMonitor(args.monitor_dir, args.task, args.score_key)

    wandb_run = None
    wandb_module = None
    if args.wandb and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("--wandb requires the wandb package in the active virtual environment") from exc
        wandb_module = wandb
        wandb_config = vars(args).copy()
        # pathlib.Path is convenient for argparse but should be stored as a
        # portable string in the W&B config rather than as a Python object.
        wandb_config["run_root"] = str(args.run_root)
        wandb_config["run_dir"] = str(args.run_dir)
        wandb_config["output"] = str(args.output)
        wandb_config["monitor_dir"] = str(args.monitor_dir)
        wandb_config["best_video_path"] = str(args.best_video_path)
        wandb_config["progress_video_directory"] = str(
            args.progress_video_directory
        )
        wandb_config["init_checkpoint"] = (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or args.run_id,
            group=args.wandb_group,
            tags=(
                [
                    f"competence-stage-{args.competence_stage:02d}",
                    f"terrain-level-{args.terrain_level}",
                ]
                if args.task == "terrain" and args.competence_stage is not None
                else None
            ),
            mode=args.wandb_mode,
            config={
                **wandb_config,
                "action_size": env.action_size,
                "observation_size": int(env.reset(jax.random.PRNGKey(args.seed)).obs.shape[-1]),
                "scene": env.xml_path,
                "task": args.task,
                "controller": "position/heading/posture PI + tripod + 24D foot/body residual",
                "action_contract_version": env.action_contract_version,
                "observation_contract_version": env.observation_contract_version,
                "contact_source": CONTACT_SOURCE,
                "attitude_source": ATTITUDE_SOURCE,
            },
        )
        wandb_run.define_metric("train/global_step")
        wandb_run.define_metric("eval/*", step_metric="train/global_step")
        wandb_run.define_metric("best/*", step_metric="train/global_step")
        wandb_run.define_metric("progress/*", step_metric="train/global_step")
        if args.task == "terrain":
            wandb_run.summary["curriculum/stage"] = args.competence_stage
            wandb_run.summary["curriculum/terrain_level"] = args.terrain_level
            wandb_run.summary["curriculum/stair_total_rise_cm"] = (
                100.0 * env._config.terrain.stair_total_rise
            )
            wandb_run.summary["curriculum/ramp_rise_cm"] = (
                100.0 * env._config.terrain.ramp_rise
            )
            wandb_run.summary["curriculum/yaw_limit_rps"] = (
                env._config.command_max_yaw_rate
            )
            wandb_run.summary["best/video_terrain_patch"] = (
                args.best_video_terrain_patch
            )

    latest_policy: dict[str, object] = {}
    pending_video: tuple[int, float] | None = None
    pending_progress_videos: list[tuple[int, int, float]] = []
    best_video_metadata = args.monitor_dir / "best_video.json"
    progress_video_history = args.monitor_dir / "progress_videos.jsonl"
    stage_metrics_latest = args.monitor_dir / "stage_metrics_latest.json"
    stage_metrics_history = args.monitor_dir / "stage_metrics_history.jsonl"
    latest_stage_metrics: dict[str, object] = {}
    stage_evaluators: dict[str, object] = {}
    progress_steps_seen: set[int] = set()
    stage_video_durations = {
        "stage0": args.best_video_stage0_duration,
        "stage1": args.best_video_stage1_duration,
        "stage2": args.best_video_stage2_duration,
    }
    # Metric evaluation remains compact even though user-facing videos are 20 s.
    stage_eval_durations = {"stage0": 10.0, "stage1": 10.0, "stage2": 12.0}
    progress_targets = progress_video_targets(args.progress_video_count)
    next_progress_target = 0
    rendered_progress_slots: set[int] = set()

    def evaluate_command_stages(step: int, make_policy, params) -> dict[str, float]:
        if args.task != "command":
            return {}
        flattened: dict[str, float] = {}
        for index, name in enumerate(("stage0", "stage1", "stage2")):
            if name not in stage_evaluators:
                stage_evaluators[name] = make_policy_evaluator(
                    env=scripted_envs[name],
                    make_policy=make_policy,
                    duration=stage_eval_durations[name],
                    num_envs=args.stage_eval_envs,
                    seed=args.seed + 10_000 + index * 100,
                )
            result = stage_evaluators[name](params)
            for metric_name, value in result.items():
                flattened[f"eval/{name}/{metric_name}"] = value
        payload = {
            "step": int(step),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": flattened,
        }
        ScoreMonitor._write_json(stage_metrics_latest, payload)
        with stage_metrics_history.open("a", encoding="utf-8") as history:
            history.write(json.dumps(payload, sort_keys=True) + "\n")
        print(
            "stage_eval "
            + " | ".join(
                f"{name}: reward={flattened[f'eval/{name}/reward_mean']:.4f} "
                f"v_err={flattened[f'eval/{name}/velocity_error_mps']:.4f} "
                f"yaw_err={flattened[f'eval/{name}/yaw_error_rps']:.4f} "
                f"torque={flattened[f'eval/{name}/torque_rms_nm']:.3f}Nm "
                f"self_col={flattened[f'eval/{name}/self_collision_rate']:.4f}"
                for name in ("stage0", "stage1", "stage2")
            )
        )
        return flattened

    def save_best_video(step: int, score: float) -> bool:
        if not args.best_video:
            return True
        if latest_policy.get("step") != step:
            return False
        if args.task == "command":
            render_specs = (
                (
                    "stage0",
                    scripted_envs["stage0"],
                    stage_video_durations["stage0"],
                ),
                ("stage1", scripted_envs["stage1"], stage_video_durations["stage1"]),
                ("stage2", scripted_envs["stage2"], stage_video_durations["stage2"]),
                ("full", scripted_envs["full"], args.best_video_full_duration),
            )
        else:
            if terrain_video_env is None:
                raise RuntimeError("terrain video environment was not initialized")
            render_specs = (("policy", terrain_video_env, args.best_video_duration),)

        rendered: dict[str, Path] = {}
        errors: dict[str, str] = {}
        for name, video_env, duration in render_specs:
            try:
                video_path = render_policy_video(
                    env=video_env,
                    make_policy=latest_policy["make_policy"],
                    params=latest_policy["params"],
                    output=Path(args.best_video_paths[name]),
                    seed=args.seed,
                    duration=duration,
                    fps=args.best_video_fps,
                    width=args.best_video_width,
                    height=args.best_video_height,
                    terrain="flat" if args.task == "command" else args.terrain_layout,
                    overlay_title=(
                        f"S{args.competence_stage:02d} L{args.terrain_level} | "
                        f"{args.best_video_terrain_patch.title()} | "
                        f"Stairs {100.0 * env._config.terrain.stair_total_rise:.1f}cm total"
                        if args.task == "terrain" and args.competence_stage is not None
                        else None
                    ),
                )
            except Exception as exc:
                # Rendering must not throw away an otherwise valid, long PPO run.
                errors[name] = f"{type(exc).__name__}: {exc}"
                print(f"best_video_error {name} step={step:,}: {errors[name]}")
                continue
            rendered[name] = video_path
            print(f"best_video {name} step={step:,}: {video_path}")

        ScoreMonitor._write_json(
            best_video_metadata,
            {
                "task": args.task,
                "score_key": args.score_key,
                "score": score,
                "step": step,
                "videos": args.best_video_paths,
                "rendered": sorted(rendered),
                "errors": errors,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if rendered and wandb_run is not None and wandb_module is not None:
            wandb_keys = {
                "stage0": "best/video_stage0_forward",
                "stage1": "best/video_stage1_limited_yaw",
                "stage2": "best/video_stage2_full_command",
                "full": "best/video_curriculum_full",
                "policy": "best/video",
            }
            if args.task == "terrain" and args.competence_stage is not None:
                wandb_keys["policy"] = (
                    f"best/video_stage{args.competence_stage:02d}_"
                    f"level{args.terrain_level}"
                )
            video_payload = {
                wandb_keys[name]: wandb_module.Video(
                    str(path),
                    format="gif",
                    caption=f"{name} | step={step:,} | score={score:.3f}",
                )
                for name, path in rendered.items()
            }
            video_payload.update(
                {
                    "train/global_step": step,
                    "best/video_step": step,
                    "best/video_score": score,
                }
            )
            wandb_run.log(video_payload)
        return True

    def save_progress_video(step: int, slot: int, target_fraction: float) -> bool:
        """Render one fixed progress snapshot once the matching policy is available."""
        if not args.progress_video or slot in rendered_progress_slots:
            return True
        if latest_policy.get("step") != step:
            return False

        target_percent = int(round(100.0 * target_fraction))
        if args.task == "command":
            video_env = scripted_envs["full"]
            stage_token = "flat"
            terrain_name = "flat"
            overlay_title = f"Progress {target_percent}% | Flat baseline"
        else:
            if terrain_video_env is None:
                raise RuntimeError("terrain video environment was not initialized")
            video_env = terrain_video_env
            terrain_name = args.terrain_layout
            if args.competence_stage is None:
                stage_token = f"level{args.terrain_level}"
                stage_label = f"L{args.terrain_level}"
            else:
                stage_token = (
                    f"stage{args.competence_stage:02d}_level{args.terrain_level}"
                )
                stage_label = (
                    f"S{args.competence_stage:02d} L{args.terrain_level}"
                )
            overlay_title = (
                f"Progress {target_percent}% | {stage_label} | "
                f"Stairs {100.0 * env._config.terrain.stair_total_rise:.1f}cm total"
            )

        output_path = args.progress_video_directory / (
            f"{stage_token}_p{target_percent:03d}_step{step:012d}.gif"
        )
        record = {
            "task": args.task,
            "competence_stage": args.competence_stage,
            "terrain_level": args.terrain_level if args.task == "terrain" else None,
            "slot": slot,
            "target_fraction": target_fraction,
            "target_percent": target_percent,
            "step": step,
            "path": str(output_path),
            "wandb_key": "progress/video",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            video_path = render_policy_video(
                env=video_env,
                make_policy=latest_policy["make_policy"],
                params=latest_policy["params"],
                output=output_path,
                seed=args.seed,
                duration=args.progress_video_duration,
                fps=args.best_video_fps,
                width=args.best_video_width,
                height=args.best_video_height,
                terrain=terrain_name,
                overlay_title=overlay_title,
            )
        except Exception as exc:
            # A renderer/encoder failure must not discard a long PPO stage.
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"progress_video_error target={target_percent}% step={step:,}: "
                f"{record['error']}"
            )
            with progress_video_history.open("a", encoding="utf-8") as history:
                history.write(json.dumps(record, sort_keys=True) + "\n")
            return True

        rendered_progress_slots.add(slot)
        record["path"] = str(video_path)
        record["error"] = None
        with progress_video_history.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"progress_video target={target_percent}% step={step:,}: {video_path}"
        )

        if wandb_run is not None and wandb_module is not None:
            stage_caption = (
                "flat baseline"
                if args.task == "command"
                else (
                    f"stage {args.competence_stage}, level {args.terrain_level}"
                    if args.competence_stage is not None
                    else f"terrain level {args.terrain_level}"
                )
            )
            wandb_run.log(
                {
                    "progress/video": wandb_module.Video(
                        str(video_path),
                        format="gif",
                        caption=(
                            f"{stage_caption} | target={target_percent}% | "
                            f"step={step:,}"
                        ),
                    ),
                    "train/global_step": step,
                    "progress/target_fraction": target_fraction,
                    "progress/target_percent": target_percent,
                    "progress/video_step": step,
                    "progress/curriculum_stage": (
                        -1
                        if args.competence_stage is None
                        else args.competence_stage
                    ),
                    "progress/terrain_level": (
                        args.terrain_level if args.task == "terrain" else -1
                    ),
                }
            )
            wandb_run.summary["progress/videos_rendered"] = len(
                rendered_progress_slots
            )
            wandb_run.summary["progress/last_video_step"] = step
        return True

    def schedule_progress_videos(step: int) -> None:
        nonlocal next_progress_target
        if not args.progress_video:
            return
        while next_progress_target < len(progress_targets):
            target_fraction = progress_targets[next_progress_target]
            target_step = int(round(args.timesteps * target_fraction))
            if step < target_step:
                break
            request = (step, next_progress_target, target_fraction)
            if not save_progress_video(*request):
                pending_progress_videos.append(request)
            next_progress_target += 1

    def policy_params(step: int, make_policy, params) -> None:
        nonlocal pending_video, pending_progress_videos
        latest_policy["step"] = step
        latest_policy["make_policy"] = make_policy
        latest_policy["params"] = params
        stage_metrics = evaluate_command_stages(step, make_policy, params)
        latest_stage_metrics["step"] = step
        latest_stage_metrics["metrics"] = stage_metrics
        # Initial Brax evaluation calls progress() before policy_params().
        if stage_metrics and step in progress_steps_seen and wandb_run is not None:
            wandb_run.log({**stage_metrics, "train/global_step": step})
        if pending_video is not None and pending_video[0] == step:
            save_best_video(*pending_video)
            pending_video = None
        still_pending = []
        for requested_step, slot, target_fraction in pending_progress_videos:
            if requested_step <= step:
                save_progress_video(step, slot, target_fraction)
            else:
                still_pending.append((requested_step, slot, target_fraction))
        pending_progress_videos = still_pending

    def progress(step: int, metrics) -> None:
        nonlocal pending_video
        combined_metrics = dict(metrics)
        if latest_stage_metrics.get("step") == step:
            combined_metrics.update(latest_stage_metrics.get("metrics", {}))
        score, is_best = score_monitor.record(step, combined_metrics)
        marker = " NEW_BEST" if is_best else ""
        print(
            f"step={step:,} {args.score_key}={score:.3f}{marker} | "
            f"best={score_monitor.best_score:.3f} | "
            f"monitor={score_monitor.best_text_path}"
        )
        if wandb_run is not None:
            payload = {key: float(value) for key, value in combined_metrics.items()}
            payload["train/global_step"] = step
            wandb_run.log(payload)
            if is_best:
                wandb_run.summary["best/score"] = score
                wandb_run.summary["best/score_key"] = args.score_key
                wandb_run.summary["best/step"] = step
        progress_steps_seen.add(step)
        if is_best:
            if not save_best_video(step, score):
                # At step 0 Brax reports evaluation before exposing policy
                # params.  policy_params() renders it immediately afterward.
                pending_video = (step, score)
        schedule_progress_videos(step)

    train = functools.partial(
        ppo.train,
        num_timesteps=args.timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=1,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        deterministic_eval=True,
        network_factory=network_factory,
        wrap_env_fn=functools.partial(
            wrapper.wrap_for_brax_training, full_reset=True
        ),
        randomization_fn=randomization_fn,
        save_checkpoint_path=str(args.output),
        restore_checkpoint_path=(
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        restore_value_fn=args.init_value_function,
        seed=args.seed,
        progress_fn=progress,
        policy_params_fn=policy_params,
        use_pmap_on_reset=False,
    )
    try:
        train(environment=env)
        # PPO normally evaluates at the exact requested final step.  If its
        # batching rounded differently, still provide every requested trend
        # slot using the final policy instead of silently omitting 100%.
        if args.progress_video and latest_policy:
            final_step = int(latest_policy["step"])
            for _, slot, target_fraction in pending_progress_videos:
                save_progress_video(final_step, slot, target_fraction)
            pending_progress_videos.clear()
            while next_progress_target < len(progress_targets):
                save_progress_video(
                    final_step,
                    next_progress_target,
                    progress_targets[next_progress_target],
                )
                next_progress_target += 1
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
