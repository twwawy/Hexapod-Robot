#!/usr/bin/env python3
"""Train the STM32-firmware residual policy with staged MJX/Brax PPO."""

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
from typing import Any

# Headless training still renders best/progress GIFs.  MuJoCo must select the
# offscreen backend before rough_terrain_env imports MuJoCo.
if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp

from domain_randomization import randomize_batch
from policy_video import render_policy_video
from rough_terrain_env import (
    ACTION_CONTRACT_VERSION,
    ACTION_SIZE,
    OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_SIZE,
    HexapodRoughTerrainEnv,
    default_config,
)
from terrain_curriculum import MAX_TERRAIN_LEVEL, terrain_level as terrain_level_spec
from servo_model import metadata as servo_model_metadata


class ScoreMonitor:
    """Persist evaluation history and the best score independently of W&B."""

    def __init__(self, directory: Path, score_key: str) -> None:
        self.directory = directory
        self.score_key = score_key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_path = directory / "best_score.json"
        self.latest_path = directory / "latest_metrics.json"
        self.history_path = directory / "metrics_history.jsonl"
        self.best_score = -math.inf

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    @staticmethod
    def numeric_metrics(metrics: Any) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    def record(self, step: int, metrics: Any) -> tuple[float, bool]:
        numeric = self.numeric_metrics(metrics)
        score = numeric.get(self.score_key, float("nan"))
        payload = {
            "score_key": self.score_key,
            "score": score,
            "step": int(step),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": numeric,
        }
        self.write_json(self.latest_path, payload)
        with self.history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(payload, sort_keys=True) + "\n")
        # Step 0 is the untrained policy and already has a dedicated 0%
        # progress video.  Best artifacts begin after at least one PPO update,
        # where Brax has also written the matching regular checkpoint.
        is_best = step > 0 and math.isfinite(score) and score > self.best_score
        if is_best:
            self.best_score = score
            self.write_json(self.best_path, payload)
        return score, is_best


def progress_video_targets(count: int) -> tuple[float, ...]:
    """Return evenly spaced fractions including both 0% and 100%."""
    if count < 1:
        raise ValueError("progress video count must be positive")
    if count == 1:
        return (1.0,)
    return tuple(index / (count - 1) for index in range(count))


def _parse_reward_weights(
    values: list[str], valid_keys: tuple[str, ...]
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("reward weights must use key=value")
        key, raw_weight = value.split("=", 1)
        key = key.strip()
        if key not in valid_keys:
            raise ValueError(
                f"unknown reward key '{key}'; valid keys: {', '.join(valid_keys)}"
            )
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"reward weight '{value}' must contain a float value"
            ) from exc
        if not math.isfinite(weight):
            raise ValueError(f"reward weight '{value}' must be finite")
        overrides[key] = weight
    return overrides


def _apply_reward_weights(
    config: Any, overrides: dict[str, float]
) -> None:
    for reward_name, reward_weight in overrides.items():
        config.reward[reward_name] = reward_weight


def _apply_command_config(config: Any, args: argparse.Namespace) -> None:
    config.command.height_min = args.height_cmd_min
    config.command.height_max = args.height_cmd_max
    config.command.pitch_min_deg = args.pitch_cmd_deg_min
    config.command.pitch_max_deg = args.pitch_cmd_deg_max
    config.command.roll_max_deg = args.roll_cmd_deg_max


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Default: 1000 for flat, 2500 for levels 1-8, 5000 for levels 9-12.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--curriculum-stage",
        "--competence-stage",
        dest="curriculum_stage",
        type=int,
        default=None,
        help="Outer curriculum stage; both option names are accepted.",
    )
    parser.add_argument(
        "--terrain-level",
        type=int,
        choices=range(MAX_TERRAIN_LEVEL + 1),
        default=0,
        help="0 flat, 1-2 rough, 3-4 ramps, 5-12 stairs.",
    )
    parser.add_argument("--command-min-speed", type=float, default=0.06)
    parser.add_argument("--command-max-speed", type=float, default=0.12)
    parser.add_argument("--command-delay", type=float, default=1.0)
    parser.add_argument("--height-cmd-min", type=float, default=-0.05)
    parser.add_argument("--height-cmd-max", type=float, default=0.10)
    parser.add_argument("--pitch-cmd-deg-min", type=float, default=-25.0)
    parser.add_argument("--pitch-cmd-deg-max", type=float, default=25.0)
    parser.add_argument("--roll-cmd-deg-max", type=float, default=15.0)
    parser.add_argument(
        "--dr-bank-size",
        type=int,
        choices=(1, 16),
        default=16,
        help="Plant bank size: 1 disables DR; 16 enables model/state DR.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("lower_leg", "terrain", "feet", "full"),
        default="lower_leg",
        help=(
            "lower_leg keeps foot/tibia/torso terrain contacts without costly "
            "robot self-collision; full restores every collider."
        ),
    )
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=32)
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run policy evaluations; use --no-eval only for throughput checks.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=0.01)
    parser.add_argument(
        "--reward-weight",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a reward coefficient; repeat for multiple keys.",
    )
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--unroll-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument(
        "--network-layers", type=int, nargs="+", default=(256, 256, 128)
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-value-function",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--score-key", default="eval/episode_reward")
    parser.add_argument(
        "--best-video", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--best-video-path", type=Path, default=None)
    parser.add_argument("--best-video-duration", type=float, default=20.0)
    parser.add_argument(
        "--stage-video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stage-video-duration", type=float, default=20.0)
    parser.add_argument(
        "--progress-video", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--progress-video-count", type=int, default=5)
    parser.add_argument("--progress-video-duration", type=float, default=20.0)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="hexapod-firmware-terrain")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=10)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--jax-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "hexapod-mjx" / "jax",
        help="Persistent XLA cache reused by restarts with identical shapes.",
    )
    args = parser.parse_args(argv)
    try:
        args.reward_weights = _parse_reward_weights(
            args.reward_weight, tuple(sorted(default_config().reward.keys()))
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not cleaned:
        raise SystemExit("--run-name must contain at least one letter or number")
    return cleaned


def _infer_stage(run_name: str | None) -> int | None:
    if not run_name:
        return None
    match = re.search(r"(?:^|-)stage(\d+)(?:-|$)", run_name)
    return int(match.group(1)) if match else None


def _resolve_checkpoint(path: Path, network_layers: tuple[int, ...]) -> Path:
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
            raise SystemExit(f"no Brax checkpoint found under {resolved}")
        resolved = candidates[-1]
    contract_path = resolved / "ppo_network_config.json"
    if not contract_path.exists():
        raise SystemExit(f"missing checkpoint contract: {contract_path}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid checkpoint contract: {contract_path}") from exc
    observation_shape = contract.get("observation_size", {}).get("shape")
    saved_layers = contract.get("network_factory_kwargs", {}).get(
        "policy_hidden_layer_sizes"
    )
    if contract.get("action_size") != ACTION_SIZE:
        raise SystemExit(
            f"checkpoint action tensor mismatch: expected action={ACTION_SIZE}"
        )
    if saved_layers is not None and tuple(saved_layers) != network_layers:
        raise SystemExit(
            f"checkpoint network mismatch: saved={saved_layers}, "
            f"requested={list(network_layers)}"
        )
    metadata_path = resolved.parent.parent / "run_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"checkpoint is missing semantic metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # v3 changes every Z output from a Cartesian endpoint offset into a
    # phase-gated swing-height command.  Tensor shape alone is therefore not
    # enough for safe restoration from v1/v2 policies.
    compatible_action_contracts = {ACTION_CONTRACT_VERSION}
    if metadata.get("action_contract_version") not in compatible_action_contracts:
        raise SystemExit("checkpoint action semantics do not match this firmware policy")
    saved_observation_contract = metadata.get("observation_contract_version")
    if (
        saved_observation_contract != OBSERVATION_CONTRACT_VERSION
        or observation_shape != [OBSERVATION_SIZE]
    ):
        raise ValueError(
            f"checkpoint observation contract '{saved_observation_contract}' != required "
            f"'{OBSERVATION_CONTRACT_VERSION}' (legacy checkpoints incompatible — "
            "start fresh)"
        )
    return resolved


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _configure_jax_cache(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(resolved))
    return resolved


def _training_schedule(args: argparse.Namespace) -> dict[str, int]:
    samples_per_training_step = (
        args.batch_size * args.num_minibatches * args.unroll_length
    )
    epochs = max(args.num_evals - 1, 1)
    training_steps_per_epoch = math.ceil(
        args.timesteps / (epochs * samples_per_training_step)
    )
    return {
        "samples_per_training_step": samples_per_training_step,
        "training_steps_per_epoch": training_steps_per_epoch,
        "samples_per_epoch": training_steps_per_epoch * samples_per_training_step,
        "actual_timesteps": (
            epochs * training_steps_per_epoch * samples_per_training_step
        ),
        "evaluation_timesteps": (
            args.num_evals * args.num_eval_envs * args.episode_length
            if getattr(args, "eval", True)
            else 0
        ),
        "rollout_batches_per_training_step": (
            args.batch_size * args.num_minibatches // args.num_envs
        ),
    }


def _prepare_run(args: argparse.Namespace) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = _safe_component(args.run_name) if args.run_name else "firmware-terrain"
    args.run_id = f"{prefix}_{timestamp}_seed{args.seed}"
    args.run_dir = (args.run_root.expanduser() / "terrain" / args.run_id).resolve()
    try:
        args.run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to mix runs in {args.run_dir}") from exc
    args.output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.run_dir / "checkpoints"
    )
    args.monitor_dir = args.run_dir / "monitor"
    videos = args.run_dir / "videos"
    stage_token = (
        f"stage{args.curriculum_stage:02d}_level{args.terrain_level}"
        if args.curriculum_stage is not None
        else f"level{args.terrain_level}"
    )
    args.stage_token = stage_token
    args.best_video_path = (
        args.best_video_path.expanduser().resolve()
        if args.best_video_path is not None
        else videos / f"best_{stage_token}.gif"
    )
    args.stage_video_path = videos / f"stage_final_{stage_token}.gif"
    args.progress_video_dir = videos / "progress"
    if args.best_video_path.suffix.lower() != ".gif":
        raise SystemExit("--best-video-path must end with .gif")
    for label, directory in (("--output", args.output),):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise SystemExit(f"refusing to mix a new run into non-empty {label}: {directory}")


def _write_metadata(args: argparse.Namespace, env: HexapodRoughTerrainEnv) -> None:
    ScoreMonitor.write_json(args.run_dir / "config.json", env._config.to_dict())
    ScoreMonitor.write_json(
        args.run_dir / "run_metadata.json",
        {
            "run_id": args.run_id,
            "seed": args.seed,
            "git_commit": _git_commit(),
            "controller": "STM32 firmware base + safety-gated 18D Cartesian foot residual",
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
            "action_size": env.action_size,
            "observation_size": env.observation_size,
            "curriculum_stage": args.curriculum_stage,
            "terrain_level": env.curriculum_level,
            "terrain_name": env.terrain_name,
            "terrain_kind": env.terrain_kind,
            "terrain_description": env.terrain_description,
            "terrain_goal_x_m": env.terrain_goal_x,
            "terrain_stair_count": env.terrain_stair_count,
            "mjx_contact_slots": env.contact_slots,
            "mjx_constraint_rows": env.constraint_rows,
            "collision_mode": env.collision_mode,
            "dr_bank_size": args.dr_bank_size,
            "servo_model": servo_model_metadata(),
            "terrain_step_height_m": env.terrain_step_height,
            "terrain_total_rise_m": env.terrain_total_rise,
            "checkpoint_dir": str(args.output),
            "best_checkpoint_pointer": str(args.monitor_dir / "best_checkpoint.json"),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "best_video_path": str(args.best_video_path),
            "stage_video_path": str(args.stage_video_path),
            "progress_video_dir": str(args.progress_video_dir),
            "progress_video_targets": list(
                progress_video_targets(args.progress_video_count)
            ),
            "ppo": {
                "timesteps": args.timesteps,
                "num_envs": args.num_envs,
                "episode_length": args.episode_length,
                "num_evals": args.num_evals,
                "num_eval_envs": args.num_eval_envs,
                "eval": args.eval,
                "learning_rate": args.learning_rate,
                "entropy_cost": args.entropy_cost,
                "discounting": args.discounting,
                "unroll_length": args.unroll_length,
                "batch_size": args.batch_size,
                "num_minibatches": args.num_minibatches,
                "num_updates_per_batch": args.num_updates_per_batch,
                "network_layers": list(args.network_layers),
                **_training_schedule(args),
            },
        },
    )


def _smoke_test(env: HexapodRoughTerrainEnv, seed: int, steps: int) -> None:
    if steps < 1:
        raise SystemExit("--smoke-steps must be positive")
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(seed))
    action = jp.zeros(env.action_size)
    started = time.monotonic()
    for _ in range(steps):
        state = step(state, action)
    state.reward.block_until_ready()
    if not bool(jp.all(jp.isfinite(state.obs))):
        raise RuntimeError("MJX smoke test produced NaN/Inf")
    print(
        f"MJX smoke OK | backend={jax.default_backend()} | "
        f"level={env.curriculum_level} terrain={env.terrain_description} | "
        f"obs={state.obs.shape[-1]} action={env.action_size} | "
        f"reward={float(state.reward):.4f} done={int(state.done)} | "
        f"swing_mean={float(state.metrics['swing_height_mean_m']):.3f}m "
        f"swing_max={float(state.metrics['swing_height_max_m']):.3f}m "
        f"boost={float(state.metrics['swing_height_boost_fraction']):.3f} "
        f"scuff={float(state.metrics['early_swing_contact_fraction']):.3f} | "
        f"wall={time.monotonic() - started:.2f}s"
    )


def _wandb_config(args: argparse.Namespace, env: HexapodRoughTerrainEnv) -> dict[str, Any]:
    excluded = {"run_dir", "monitor_dir"}
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in excluded:
            continue
        config[key] = str(value) if isinstance(value, Path) else value
    config.update(
        {
            "run_dir": str(args.run_dir),
            "action_size": env.action_size,
            "observation_size": env.observation_size,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
            "terrain_name": env.terrain_name,
            "terrain_kind": env.terrain_kind,
            "terrain_description": env.terrain_description,
            "terrain_step_height_m": env.terrain_step_height,
            "terrain_total_rise_m": env.terrain_total_rise,
            "mjx_contact_slots": env.contact_slots,
            "mjx_constraint_rows": env.constraint_rows,
            "collision_mode": env.collision_mode,
        }
    )
    return config


def main() -> None:
    args = _arguments()
    args.jax_cache_dir = _configure_jax_cache(args.jax_cache_dir)
    spec = terrain_level_spec(args.terrain_level)
    if args.episode_length is None:
        args.episode_length = (
            1000 if spec.level == 0 else (2500 if spec.level <= 8 else 5000)
        )
    if args.curriculum_stage is None:
        args.curriculum_stage = _infer_stage(args.run_name)
    if args.curriculum_stage is not None and args.curriculum_stage < 0:
        raise SystemExit("--curriculum-stage must be non-negative")
    if args.timesteps < 1 or args.num_envs < 1 or args.episode_length < 1:
        raise SystemExit("timesteps, num-envs and episode-length must be positive")
    if args.command_min_speed < 0 or args.command_max_speed < args.command_min_speed:
        raise SystemExit("command speeds must satisfy 0 <= min <= max")
    if args.height_cmd_min > args.height_cmd_max:
        raise SystemExit("height commands must satisfy min <= max")
    if args.pitch_cmd_deg_min > args.pitch_cmd_deg_max:
        raise SystemExit("pitch commands must satisfy min <= max")
    if args.roll_cmd_deg_max < 0:
        raise SystemExit("roll command maximum must be non-negative")
    if args.num_evals < 1 or args.num_eval_envs < 1:
        raise SystemExit("num-evals and num-eval-envs must be positive")
    rollout_envs = args.batch_size * args.num_minibatches
    if rollout_envs % args.num_envs:
        raise SystemExit(
            "batch-size * num-minibatches must be divisible by num-envs: "
            f"{args.batch_size} * {args.num_minibatches} % {args.num_envs} != 0"
        )
    if not 1 <= args.progress_video_count <= 21:
        raise SystemExit("--progress-video-count must be in 1..21")
    if min(
        args.best_video_duration,
        args.stage_video_duration,
        args.progress_video_duration,
        args.video_fps,
        args.video_width,
        args.video_height,
    ) <= 0:
        raise SystemExit("video duration, fps and dimensions must be positive")

    network_layers = tuple(args.network_layers)
    if args.init_checkpoint is not None:
        args.init_checkpoint = _resolve_checkpoint(args.init_checkpoint, network_layers)
    config = default_config()
    config.episode_length = args.episode_length
    config.command_min_speed = args.command_min_speed
    config.command_max_speed = args.command_max_speed
    config.command_delay = args.command_delay
    _apply_command_config(config, args)
    config.collision_mode = args.collision_mode
    config.dr_enabled = args.dr_bank_size == 16
    _apply_reward_weights(config, args.reward_weights)
    env = HexapodRoughTerrainEnv(config=config, terrain_level=args.terrain_level)
    print("JAX devices:", jax.devices())
    print(
        f"contract action={ACTION_CONTRACT_VERSION} observation={OBSERVATION_CONTRACT_VERSION} | "
        f"stage={args.curriculum_stage} level={env.curriculum_level} "
        f"terrain={env.terrain_description} episode={args.episode_length} "
        f"collision={env.collision_mode} contact_slots={env.contact_slots} "
        f"constraint_rows={env.constraint_rows}"
    )
    schedule = _training_schedule(args)
    print(
        "schedule "
        f"requested={args.timesteps:,} actual={schedule['actual_timesteps']:,} "
        f"per_eval={schedule['samples_per_epoch']:,} "
        f"rollout_batches={schedule['rollout_batches_per_training_step']} "
        f"eval_sim_steps={schedule['evaluation_timesteps']:,} "
        f"jax_cache={args.jax_cache_dir}"
    )
    if args.smoke:
        _smoke_test(env, args.seed, args.smoke_steps)
        return
    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise SystemExit(
            "GPU JAX backend is required. Verify with `python -c 'import jax; "
            "print(jax.devices())'`; use --allow-cpu only for tiny debug runs."
        )

    _prepare_run(args)
    _write_metadata(args, env)
    print(f"RUN_DIR={args.run_dir}")

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    args.output.mkdir(parents=True, exist_ok=True)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=network_layers,
        value_hidden_layer_sizes=network_layers,
    )
    monitor = ScoreMonitor(args.monitor_dir, args.score_key)
    randomization_fn = (
        None
        if args.dr_bank_size == 1
        else functools.partial(
            randomize_batch,
            bank_size=args.dr_bank_size,
            seed=args.seed,
        )
    )

    wandb_run = None
    wandb_module = None
    if args.wandb and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("--wandb requires `pip install wandb`") from exc
        wandb_module = wandb
        tags = [f"terrain-level-{args.terrain_level}", ACTION_CONTRACT_VERSION]
        if args.curriculum_stage is not None:
            tags.append(f"curriculum-stage-{args.curriculum_stage:02d}")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or args.run_id,
            group=args.wandb_group,
            dir=str(args.run_dir),
            mode=args.wandb_mode,
            tags=tags,
            config=_wandb_config(args, env),
        )
        wandb_run.define_metric("train/global_step")
        for namespace in ("eval/*", "best/*", "stage/*", "progress/*"):
            wandb_run.define_metric(namespace, step_metric="train/global_step")
        wandb_run.summary["curriculum/stage"] = args.curriculum_stage
        wandb_run.summary["curriculum/terrain_level"] = args.terrain_level
        wandb_run.summary["curriculum/terrain_name"] = env.terrain_name
        wandb_run.summary["curriculum/terrain_kind"] = env.terrain_kind
        wandb_run.summary["curriculum/stair_riser_cm"] = 100.0 * env.terrain_step_height
        wandb_run.summary["curriculum/stair_total_rise_cm"] = 100.0 * env.terrain_total_rise

    latest_policy: dict[str, Any] = {}
    pending_best: tuple[int, float] | None = None
    pending_progress: list[tuple[int, int, float]] = []
    progress_targets = progress_video_targets(args.progress_video_count)
    next_progress_target = 0
    rendered_progress: dict[int, Path] = {}
    progress_history = args.monitor_dir / "progress_videos.jsonl"
    artifact_history = args.monitor_dir / "artifacts.jsonl"

    def append_artifact(record: dict[str, Any]) -> None:
        record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        artifact_history.parent.mkdir(parents=True, exist_ok=True)
        with artifact_history.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, sort_keys=True) + "\n")

    def save_best_artifacts(step: int, score: float) -> bool:
        if latest_policy.get("step") != step:
            return False
        errors: dict[str, str] = {}
        checkpoint_path = args.output / f"{step:012d}"
        if checkpoint_path.exists():
            print(f"best_checkpoint step={step:,}: {checkpoint_path}")
            ScoreMonitor.write_json(
                args.monitor_dir / "best_checkpoint.json",
                {"step": step, "score": score, "path": str(checkpoint_path)},
            )
        else:
            errors["checkpoint"] = f"matching Brax checkpoint not found: {checkpoint_path}"
            print(f"best_checkpoint_error step={step:,}: {errors['checkpoint']}")

        rendered = False
        if args.best_video:
            try:
                render_policy_video(
                    env=env,
                    make_policy=latest_policy["make_policy"],
                    params=latest_policy["params"],
                    output=args.best_video_path,
                    seed=args.seed + 20_000,
                    duration=args.best_video_duration,
                    fps=args.video_fps,
                    width=args.video_width,
                    height=args.video_height,
                )
                rendered = True
                print(f"best_video step={step:,}: {args.best_video_path}")
            except Exception as exc:
                errors["video"] = f"{type(exc).__name__}: {exc}"
                print(f"best_video_error step={step:,}: {errors['video']}")
        append_artifact(
            {
                "kind": "best",
                "step": step,
                "score": score,
                "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
                "video": str(args.best_video_path) if rendered else None,
                "errors": errors,
            }
        )
        if wandb_run is not None:
            payload: dict[str, Any] = {
                "train/global_step": step,
                "best/score": score,
                "best/checkpoint_step": step,
            }
            if rendered and wandb_module is not None:
                caption = f"best | step={step:,} | score={score:.3f}"
                payload["best/video"] = wandb_module.Video(
                    str(args.best_video_path),
                    format="gif",
                    caption=caption,
                )
                payload[f"best/video_{args.stage_token}"] = wandb_module.Video(
                    str(args.best_video_path), format="gif", caption=caption
                )
            wandb_run.log(payload)
            wandb_run.summary["best/score"] = score
            wandb_run.summary["best/step"] = step
            if checkpoint_path.exists():
                wandb_run.summary["best/checkpoint"] = str(checkpoint_path)
        return True

    def save_progress_video(step: int, slot: int, fraction: float) -> bool:
        if not args.progress_video or slot in rendered_progress:
            return True
        if latest_policy.get("step") != step:
            return False
        percent = int(round(100.0 * fraction))
        output = args.progress_video_dir / (
            f"{args.stage_token}_p{percent:03d}_step{step:012d}.gif"
        )
        record: dict[str, Any] = {
            "kind": "progress",
            "step": step,
            "slot": slot,
            "target_fraction": fraction,
            "target_percent": percent,
            "path": str(output),
        }
        try:
            render_policy_video(
                env=env,
                make_policy=latest_policy["make_policy"],
                params=latest_policy["params"],
                output=output,
                seed=args.seed + 30_000,
                duration=args.progress_video_duration,
                fps=args.video_fps,
                width=args.video_width,
                height=args.video_height,
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"progress_video_error target={percent}% step={step:,}: {record['error']}")
        else:
            rendered_progress[slot] = output
            record["error"] = None
            print(f"progress_video target={percent}% step={step:,}: {output}")
            if wandb_run is not None and wandb_module is not None:
                caption = f"{args.stage_token} | {percent}% | step={step:,}"
                generic_video = wandb_module.Video(
                    str(output),
                    format="gif",
                    caption=caption,
                )
                wandb_run.log(
                    {
                        "progress/video": generic_video,
                        f"progress/video_{args.stage_token}_p{percent:03d}": wandb_module.Video(
                            str(output), format="gif", caption=caption
                        ),
                        "progress/target_percent": percent,
                        "progress/video_step": step,
                        "train/global_step": step,
                    }
                )
        record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        progress_history.parent.mkdir(parents=True, exist_ok=True)
        with progress_history.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, sort_keys=True) + "\n")
        return True

    def schedule_progress(step: int) -> None:
        nonlocal next_progress_target
        if not args.progress_video:
            return
        while next_progress_target < len(progress_targets):
            fraction = progress_targets[next_progress_target]
            if step < int(round(args.timesteps * fraction)):
                break
            request = (step, next_progress_target, fraction)
            if not save_progress_video(*request):
                pending_progress.append(request)
            next_progress_target += 1

    def policy_params(step: int, make_policy: Any, params: Any) -> None:
        nonlocal pending_best, pending_progress
        latest_policy.update(
            {"step": int(step), "make_policy": make_policy, "params": params}
        )
        if pending_best is not None and pending_best[0] <= step:
            save_best_artifacts(step, pending_best[1])
            pending_best = None
        remaining: list[tuple[int, int, float]] = []
        for requested_step, slot, fraction in pending_progress:
            if requested_step <= step:
                save_progress_video(step, slot, fraction)
            else:
                remaining.append((requested_step, slot, fraction))
        pending_progress = remaining

    def progress(step: int, metrics: Any) -> None:
        nonlocal pending_best
        numeric = ScoreMonitor.numeric_metrics(metrics)
        score, is_best = monitor.record(step, numeric)
        success = numeric.get("eval/episode_terrain_success", 0.0)
        failure = sum(
            numeric.get(f"eval/episode_termination/{name}", 0.0)
            for name in (
                "controller_invalid",
                "joint_limit",
                "dynamics",
                "tilt",
                "clearance",
                "body_contact",
                "nonfinite",
            )
        )
        marker = " NEW_BEST" if is_best else ""
        print(
            f"step={step:,} {args.score_key}={score:.3f}{marker} | "
            f"success={success:.3f} failure={failure:.3f} | best={monitor.best_score:.3f}"
        )
        if wandb_run is not None:
            wandb_run.log({**numeric, "train/global_step": step})
        if is_best and not save_best_artifacts(step, score):
            pending_best = (step, score)
        schedule_progress(step)

    def save_stage_final(step: int) -> None:
        if not args.stage_video or not latest_policy:
            return
        errors: dict[str, str] = {}
        try:
            args.stage_video_path.parent.mkdir(parents=True, exist_ok=True)
            render_policy_video(
                env=env,
                make_policy=latest_policy["make_policy"],
                params=latest_policy["params"],
                output=args.stage_video_path,
                seed=args.seed + 40_000,
                duration=args.stage_video_duration,
                fps=args.video_fps,
                width=args.video_width,
                height=args.video_height,
            )
        except Exception as exc:
            errors["video"] = f"{type(exc).__name__}: {exc}"
            print(f"stage_video_error step={step:,}: {errors['video']}")
            rendered = False
        else:
            rendered = True
            print(f"stage_video step={step:,}: {args.stage_video_path}")
        append_artifact(
            {
                "kind": "stage_final",
                "step": step,
                "video": str(args.stage_video_path) if rendered else None,
                "errors": errors,
            }
        )
        if rendered and wandb_run is not None and wandb_module is not None:
            caption = f"stage final | {args.stage_token} | step={step:,}"
            generic_video = wandb_module.Video(
                str(args.stage_video_path),
                format="gif",
                caption=caption,
            )
            wandb_run.log(
                {
                    "stage/final_video": generic_video,
                    f"stage/final_video_{args.stage_token}": wandb_module.Video(
                        str(args.stage_video_path), format="gif", caption=caption
                    ),
                    "stage/final_step": step,
                    "train/global_step": step,
                }
            )
            wandb_run.summary["stage/final_video"] = str(args.stage_video_path)
            wandb_run.summary["stage/final_step"] = step

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
        run_evals=args.eval,
        network_factory=network_factory,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
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
        final_step = int(latest_policy.get("step", 0))
        if args.progress_video and latest_policy:
            for _, slot, fraction in list(pending_progress):
                save_progress_video(final_step, slot, fraction)
            pending_progress.clear()
            while next_progress_target < len(progress_targets):
                save_progress_video(
                    final_step,
                    next_progress_target,
                    progress_targets[next_progress_target],
                )
                next_progress_target += 1
        save_stage_final(final_step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
