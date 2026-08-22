#!/usr/bin/env python3
"""Train flat command curriculum or stair-terrain residual PPO on MJX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jp

from best_policy_video import render_policy_video
from command_curriculum_env import HexapodCommandCurriculumEnv
from rough_terrain_env import HexapodRoughTerrainEnv, default_config


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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Checkpoint directory. Defaults to a task-specific directory.",
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
    parser.add_argument(
        "--residual-scale",
        type=float,
        nargs=3,
        default=(0.04, 0.03, 0.09),
        metavar=("X_M", "Y_M", "Z_M"),
        help="Per-leg local Cartesian residual limits in metres.",
    )
    parser.add_argument("--terrain-speed-min", type=float, default=0.03)
    parser.add_argument("--terrain-speed-max", type=float, default=0.18)
    parser.add_argument("--terrain-yaw-limit", type=float, default=0.35)
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
        help="Live score files directory. Defaults to SW/mjx/artifacts/<task>/monitor.",
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
    config.phase_time = args.phase_time
    config.base_swing_height = args.base_swing_height
    config.base_radial_offset = args.base_radial_offset
    config.residual_scale = list(args.residual_scale)
    config.command_min_speed = args.terrain_speed_min
    config.command_max_speed = args.terrain_speed_max
    config.command_max_yaw_rate = args.terrain_yaw_limit
    config.command_curriculum.forward_only_steps = args.curriculum_forward_only_steps
    config.command_curriculum.limited_yaw_steps = args.curriculum_limited_yaw_steps
    config.command_curriculum.speed_min = tuple(args.curriculum_speed_min)
    config.command_curriculum.speed_max = tuple(args.curriculum_speed_max)
    config.command_curriculum.yaw_limit = tuple(args.curriculum_yaw_limit)
    if args.task == "command":
        return HexapodCommandCurriculumEnv(config=config)
    return HexapodRoughTerrainEnv(config=config)


def _smoke_test(env: HexapodRoughTerrainEnv, seed: int) -> None:
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(seed))
    action = jp.zeros(env.action_size)
    started = time.monotonic()
    for _ in range(10):
        state = step(state, action)
    state.reward.block_until_ready()
    print(
        f"MJX smoke test OK | backend={jax.default_backend()} | "
        f"obs={state.obs.shape[-1]} action={env.action_size} | "
        f"reward={float(state.reward):.4f} done={int(state.done)} | "
        f"wall={time.monotonic() - started:.2f}s"
    )


def main(default_task: str = "terrain") -> None:
    args = _arguments(default_task)
    if args.output is None:
        output_name = (
            "command_curriculum" if args.task == "command" else "rough_terrain"
        )
        args.output = Path(__file__).resolve().parent / "checkpoints" / output_name
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
        args.monitor_dir = (
            Path(__file__).resolve().parent / "artifacts" / args.task / "monitor"
        )
    args.monitor_dir = args.monitor_dir.expanduser().resolve()
    if args.best_video_path is None:
        args.best_video_path = args.monitor_dir / "best_policy.gif"
    args.best_video_path = args.best_video_path.expanduser().resolve()
    if args.best_video_path.suffix.lower() != ".gif":
        raise SystemExit("--best-video-path must end with .gif")
    print("JAX devices:", jax.devices())
    env = _make_env(args)
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
                "controller": "tripod + 22D Cartesian/gait residual",
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
