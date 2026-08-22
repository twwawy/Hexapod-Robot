#!/usr/bin/env python3
"""Train flat command curriculum or stair-terrain residual PPO on MJX."""

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

from best_policy_video import render_policy_video
from command_curriculum_env import HexapodCommandCurriculumEnv
from rough_terrain_env import (
    ACTION_CONTRACT_VERSION,
    HexapodRoughTerrainEnv,
    default_config,
)


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


def _arguments(default_task: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("command", "terrain"),
        default=default_task,
        help="command = flat walking+yaw curriculum; terrain = staircase task.",
    )
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None, help="Optional readable experiment name.")
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
        "--allow-cpu",
        action="store_true",
        help="allow PPO on CPU (useful only for very small debugging runs)",
    )
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=0.01)
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--unroll-length", type=int, default=20)
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
    parser.add_argument("--phase-time", type=float, default=0.5)
    parser.add_argument("--base-swing-height", type=float, default=0.07)
    parser.add_argument("--base-radial-offset", type=float, default=0.01)
    parser.add_argument("--swing-x", type=float, default=None, help="Active task swing X residual limit [m].")
    parser.add_argument("--swing-y", type=float, default=None, help="Active task swing Y residual limit [m].")
    parser.add_argument("--swing-z-low", type=float, default=None, help="Active task negative swing Z limit [m].")
    parser.add_argument("--swing-z-high", type=float, default=None, help="Active task positive swing Z limit [m].")
    parser.add_argument("--stance-z", type=float, default=None, help="Active task stance-only Z residual limit [m].")
    parser.add_argument("--stride-half-range", type=float, default=None, help="Stride scale authority around one; 0.2 = 0.8..1.2.")
    parser.add_argument("--frequency-half-range", type=float, default=None, help="Frequency authority around one; 0.15 = 0.85..1.15.")
    parser.add_argument("--swing-height-min", type=float, default=None, help="Active task global swing height minimum [m].")
    parser.add_argument("--swing-height-max", type=float, default=None, help="Active task global swing height maximum [m].")
    parser.add_argument("--radial-min", type=float, default=None, help="Active task radial offset minimum [m].")
    parser.add_argument("--radial-max", type=float, default=None, help="Active task radial offset maximum [m].")
    parser.add_argument("--terrain-speed-min", type=float, default=0.03)
    parser.add_argument("--terrain-speed-max", type=float, default=0.18)
    parser.add_argument("--terrain-yaw-limit", type=float, default=0.35)
    parser.add_argument(
        "--terrain-level",
        type=int,
        choices=range(5),
        default=2,
        help="0: <=20 mm, 1: 20..35 mm, 2: 35..50 mm, 3/4: 20..60 mm terrain.",
    )
    parser.add_argument(
        "--terrain-randomize",
        action="store_true",
        help="Sample one terrain/dynamics model per run; level 4 also samples mass/actuator/damping.",
    )
    parser.add_argument("--terrain-step-height", type=float, default=None, help="Override fixed stair height [m].")
    parser.add_argument("--terrain-step-depth", type=float, default=None, help="Override fixed stair depth [m].")
    parser.add_argument("--terrain-friction", type=float, default=None, help="Override fixed terrain/foot friction scale.")
    parser.add_argument("--curriculum-forward-only-steps", type=int, default=250)
    parser.add_argument("--curriculum-limited-yaw-steps", type=int, default=250)
    parser.add_argument(
        "--curriculum-speed-min", type=float, nargs=3, default=(0.03, 0.05, 0.03)
    )
    parser.add_argument(
        "--curriculum-speed-max", type=float, nargs=3, default=(0.08, 0.12, 0.18)
    )
    parser.add_argument(
        "--curriculum-yaw-limit", type=float, nargs=3, default=(0.00, 0.15, 0.35)
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
        help="GIF path for the current best policy. Defaults to <monitor-dir>/best_policy.gif.",
    )
    parser.add_argument("--best-video-duration", type=float, default=10.0)
    parser.add_argument("--best-video-fps", type=int, default=20)
    parser.add_argument("--best-video-width", type=int, default=640)
    parser.add_argument("--best-video-height", type=int, default=360)
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
        "stride_half_range",
        "frequency_half_range",
        "swing_height_min",
        "swing_height_max",
        "radial_min",
        "radial_max",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(authority, name, value)
    config.command_min_speed = args.terrain_speed_min
    config.command_max_speed = args.terrain_speed_max
    config.command_max_yaw_rate = args.terrain_yaw_limit
    config.command_curriculum.forward_only_steps = args.curriculum_forward_only_steps
    config.command_curriculum.limited_yaw_steps = args.curriculum_limited_yaw_steps
    config.command_curriculum.speed_min = tuple(args.curriculum_speed_min)
    config.command_curriculum.speed_max = tuple(args.curriculum_speed_max)
    config.command_curriculum.yaw_limit = tuple(args.curriculum_yaw_limit)
    # Curriculum changes terrain difficulty before model-gap difficulty.  The
    # first version is deliberately run-level rather than per-reset so every
    # result has a reproducible physical scene recorded in run_metadata.json.
    level_ranges = (
        (0.001, 0.020),
        (0.020, 0.035),
        (0.035, 0.050),
        (0.020, 0.060),
        (0.020, 0.060),
    )
    rng = np.random.default_rng(args.seed)
    height_low, height_high = level_ranges[args.terrain_level]
    randomize_terrain = args.terrain_randomize and args.task == "terrain"
    config.terrain.step_height = (
        args.terrain_step_height
        if args.terrain_step_height is not None
        else (float(rng.uniform(height_low, height_high)) if randomize_terrain else height_high)
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
    config.randomization.enabled = randomize_terrain
    if randomize_terrain and args.terrain_level >= 4:
        config.randomization.mass_scale = float(rng.uniform(0.90, 1.10))
        config.randomization.actuator_strength_scale = float(rng.uniform(0.85, 1.15))
        config.randomization.joint_damping_scale = float(rng.uniform(0.80, 1.20))
    if args.task == "command":
        return HexapodCommandCurriculumEnv(config=config)
    return HexapodRoughTerrainEnv(config=config)


def _safe_run_component(value: str) -> str:
    """Keep a human run name portable as a single directory component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not cleaned:
        raise SystemExit("--run-name must contain at least one letter or number")
    return cleaned


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
            "action_size": env.action_size,
            "observation_size": env.observation_size,
            "checkpoint_dir": str(args.output),
            "monitor_dir": str(args.monitor_dir),
            "best_video_path": str(args.best_video_path),
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


def _smoke_test(env: HexapodRoughTerrainEnv, seed: int) -> None:
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(seed))
    action_key = jax.random.PRNGKey(seed + 1)
    started = time.monotonic()
    for _ in range(10):
        action_key, sample_key = jax.random.split(action_key)
        action = jax.random.uniform(
            sample_key, (env.action_size,), minval=-1.0, maxval=1.0
        )
        state = step(state, action)
    state.reward.block_until_ready()
    if bool(jp.any(jp.isnan(state.obs))) or bool(jp.any(jp.isnan(state.data.qpos))):
        raise RuntimeError("MJX smoke test produced NaN under bounded random actions")
    print(
        f"MJX smoke test OK | backend={jax.default_backend()} | "
        f"obs={state.obs.shape[-1]} action={env.action_size} random-actions=10 | "
        f"reward={float(state.reward):.4f} done={int(state.done)} | "
        f"wall={time.monotonic() - started:.2f}s"
    )


def main(default_task: str = "terrain") -> None:
    args = _arguments(default_task)
    generated_run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    generated_run_id = f"{generated_run_id}_seed{args.seed}"
    args.run_id = _safe_run_component(args.run_name) if args.run_name else generated_run_id
    args.run_dir = (args.run_root.expanduser() / args.task / args.run_id).resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
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
        args.best_video_path = args.run_dir / "best_policy.gif"
    args.best_video_path = args.best_video_path.expanduser().resolve()
    if args.best_video_path.suffix.lower() != ".gif":
        raise SystemExit("--best-video-path must end with .gif")
    print("JAX devices:", jax.devices())
    print(f"run={args.run_dir} contract={ACTION_CONTRACT_VERSION}")
    env = _make_env(args)
    _write_run_metadata(args, env)
    if args.smoke:
        _smoke_test(env, args.seed)
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
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config={
                **wandb_config,
                "action_size": env.action_size,
                "observation_size": int(env.reset(jax.random.PRNGKey(args.seed)).obs.shape[-1]),
                "scene": env.xml_path,
                "task": args.task,
                "controller": "classical tripod + contact/workspace-safe 22D residual",
                "action_contract_version": env.action_contract_version,
            },
        )

    latest_policy: dict[str, object] = {}
    pending_video: tuple[int, float] | None = None
    best_video_metadata = args.monitor_dir / "best_video.json"

    def save_best_video(step: int, score: float) -> bool:
        if not args.best_video:
            return True
        if latest_policy.get("step") != step:
            return False
        try:
            video_path = render_policy_video(
                env=env,
                make_policy=latest_policy["make_policy"],
                params=latest_policy["params"],
                output=args.best_video_path,
                seed=args.seed,
                duration=args.best_video_duration,
                fps=args.best_video_fps,
                width=args.best_video_width,
                height=args.best_video_height,
                terrain="flat" if args.task == "command" else "stairs",
            )
        except Exception as exc:
            # Rendering must not throw away an otherwise valid, long PPO run.
            print(f"best_video_error step={step:,}: {type(exc).__name__}: {exc}")
            return True

        ScoreMonitor._write_json(
            best_video_metadata,
            {
                "task": args.task,
                "score_key": args.score_key,
                "score": score,
                "step": step,
                "video": str(video_path),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"best_video step={step:,}: {video_path}")
        if wandb_run is not None and wandb_module is not None:
            wandb_run.log(
                {
                    "best/video": wandb_module.Video(
                        str(video_path),
                        format="gif",
                        caption=f"{args.task} | step={step:,} | score={score:.3f}",
                    ),
                    "best/video_step": step,
                    "best/video_score": score,
                },
                step=step,
            )
        return True

    def policy_params(step: int, make_policy, params) -> None:
        nonlocal pending_video
        latest_policy["step"] = step
        latest_policy["make_policy"] = make_policy
        latest_policy["params"] = params
        if pending_video is not None and pending_video[0] == step:
            save_best_video(*pending_video)
            pending_video = None

    def progress(step: int, metrics) -> None:
        nonlocal pending_video
        score, is_best = score_monitor.record(step, metrics)
        marker = " NEW_BEST" if is_best else ""
        print(
            f"step={step:,} {args.score_key}={score:.3f}{marker} | "
            f"best={score_monitor.best_score:.3f} | "
            f"monitor={score_monitor.best_text_path}"
        )
        if wandb_run is not None:
            payload = {key: float(value) for key, value in metrics.items()}
            payload["train/global_step"] = step
            wandb_run.log(payload, step=step)
            if is_best:
                wandb_run.summary["best/score"] = score
                wandb_run.summary["best/score_key"] = args.score_key
                wandb_run.summary["best/step"] = step
        if is_best:
            if not save_best_video(step, score):
                # At step 0 Brax reports evaluation before exposing policy
                # params.  policy_params() renders it immediately afterward.
                pending_video = (step, score)

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
        wrap_env_fn=wrapper.wrap_for_brax_training,
        save_checkpoint_path=str(args.output),
        seed=args.seed,
        progress_fn=progress,
        policy_params_fn=policy_params,
        use_pmap_on_reset=False,
    )
    try:
        train(environment=env)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
