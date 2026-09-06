#!/usr/bin/env python3
"""
Train one stage of the adaptive 24-D v4 residual locomotion policy.

Features
--------
- Stage 1: Tripod only
- Stage 2: Wave only
- Stage 3: deterministic Hybrid supervisor
- teacher / lidar / blind perception
- PPO checkpoint save / restore
- 24-D adaptive contract persistence
- W&B metrics
- best-score checkpoint selection
- deterministic best-policy GIF at stage end
- curriculum-manager friendly monitor files

Stage 0 geometry/planner validation must be completed before PPO training.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import functools
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Headless training still needs offscreen MuJoCo rendering for best.gif.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    """Atomically write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def numeric_metrics(metrics: Any) -> dict[str, float]:
    """Convert Brax/JAX metric values into JSON-safe floats."""
    result: dict[str, float] = {}

    for name, value in metrics.items():
        try:
            result[name] = float(value)
        except (TypeError, ValueError):
            continue

    return result


def block_tree(tree: Any) -> None:
    """Synchronize every JAX leaf in a pytree/dict observation."""
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def render_policy_video(
    *,
    env: Any,
    make_policy: Any,
    params: Any,
    output: Path,
    seed: int,
    duration: float,
    fps: int,
    width: int,
    height: int,
) -> Path:
    """
    Render deterministic PPO rollout as GIF.

    Implemented here instead of depending on legacy renderer assumptions,
    because adaptive v4 uses dict/pytree observations.
    """

    if duration <= 0:
        raise ValueError("video duration must be positive")
    if fps <= 0:
        raise ValueError("video fps must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")

    if output.suffix.lower() != ".gif":
        raise ValueError("best video output must end in .gif")

    import jax
    import numpy as np
    from PIL import Image
    import mujoco
    from mujoco import mjx

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_name(
        f".{output.stem}.tmp.gif"
    )
    temporary.unlink(missing_ok=True)

    policy = jax.jit(
        make_policy(
            params,
            deterministic=True,
        )
    )

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(
        jax.random.PRNGKey(seed)
    )
    block_tree(state.obs)

    key = jax.random.PRNGKey(seed + 1)

    control_steps = max(
        1,
        int(math.ceil(duration / float(env.dt))),
    )

    requested_frames = max(
        1,
        int(math.ceil(duration * fps)),
    )

    frame_steps = np.floor(
        np.arange(requested_frames, dtype=float)
        / (fps * float(env.dt))
    ).astype(int)

    frame_steps = np.clip(
        frame_steps,
        0,
        control_steps - 1,
    )

    renderer = mujoco.Renderer(
        env.mj_model,
        height=height,
        width=width,
    )

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    frames: list[Image.Image] = []
    frame_index = 0

    try:
        for control_step in range(control_steps):
            key, action_key = jax.random.split(key)

            action, _ = policy(
                state.obs,
                action_key,
            )

            state = step(
                state,
                action,
            )
            block_tree(state.obs)

            done = bool(
                np.asarray(state.done)
            )

            should_render = (
                frame_index < requested_frames
                and control_step >= frame_steps[frame_index]
            )

            if should_render or done:
                host_data = mjx.get_data(
                    env.mj_model,
                    state.data,
                )

                base = np.asarray(
                    host_data.qpos[:3],
                    dtype=float,
                )

                camera.lookat[:] = (
                    base[0] + 0.55,
                    base[1],
                    max(base[2] - 0.15, 0.12),
                )

                camera.distance = 1.75
                camera.azimuth = 135
                camera.elevation = -24

                renderer.update_scene(
                    host_data,
                    camera=camera,
                )

                frame = Image.fromarray(
                    renderer.render()
                )

                frames.append(
                    frame.convert(
                        "P",
                        palette=Image.ADAPTIVE,
                    )
                )

                frame_index += 1

            if done:
                break

    finally:
        renderer.close()

    if not frames:
        raise RuntimeError(
            "no frames were produced while rendering best policy"
        )

    try:
        frames[0].save(
            temporary,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / fps),
            loop=0,
            optimize=False,
        )

        temporary.replace(output)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    # A short GIF can mean success, watchdog termination, or physical failure.
    # Preserve the cause beside the video instead of making users infer it.
    report = {
        "elapsed_s": (control_step + 1) * float(env.dt),
        "requested_duration_s": duration,
        "done": done,
        "termination": {
            key: float(np.asarray(value))
            for key, value in state.metrics.items()
            if key.startswith("termination/") and float(np.asarray(value)) != 0.
        },
        "terrain_success": float(np.asarray(state.metrics.get("terrain_success", 0.))),
    }
    output.with_suffix(".termination.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Policy video termination: {report}", flush=True)
    return output


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('--migrate-flat-boxes', action='store_true',
                        help='Explicit reviewed flat 786ff09 checkpoint transfer; requires --restore.')

    # -------------------------------------------------------
    # Adaptive environment
    # -------------------------------------------------------

    parser.add_argument(
        "--perception",
        choices=("lidar", "teacher", "blind"),
        default="lidar",
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help=(
            "1: Tripod only, "
            "2: Wave only, "
            "3: deterministic hybrid"
        ),
    )

    parser.add_argument(
        "--terrain-level",
        type=int,
        default=0,
    )

    # -------------------------------------------------------
    # PPO
    # -------------------------------------------------------

    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000_000,
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--episode-length",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--num-evals",
        type=int,
        default=10,
    )
    parser.add_argument('--num-eval-envs', type=int, default=16)

    parser.add_argument(
        "--seed",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--entropy-cost",
        type=float,
        default=0.005,
    )

    # -------------------------------------------------------
    # LiDAR
    # -------------------------------------------------------

    parser.add_argument(
        "--azimuths",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--elevations",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--range-noise",
        type=float,
        default=0.005,
    )

    # -------------------------------------------------------
    # Run / checkpoint
    # -------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Stage run directory. "
            "checkpoints/, monitor/, videos/ are created inside."
        ),
    )

    parser.add_argument(
        "--restore",
        type=Path,
        default=None,
        help=(
            "Restore compatible 24-D checkpoint. "
            "Weights + observation normalizer are restored."
        ),
    )

    parser.add_argument(
        "--init-teacher",
        type=Path,
        default=None,
        help=(
            "Initialize LiDAR PPO from a trained 24-D teacher. "
            "Only valid with --perception lidar."
        ),
    )

    # -------------------------------------------------------
    # Curriculum metadata
    # -------------------------------------------------------

    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--score-key",
        default="eval/episode_reward",
        help="Metric used to choose the best checkpoint.",
    )

    # -------------------------------------------------------
    # W&B
    # -------------------------------------------------------

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload evaluation, cycle best score, best video and artifact to W&B.",
    )

    parser.add_argument(
        "--wandb-project",
        default="hexapod-adaptive-gait",
    )

    parser.add_argument(
        "--wandb-entity",
        default=None,
    )

    parser.add_argument(
        "--wandb-group",
        default=None,
    )

    parser.add_argument(
        "--wandb-name",
        default=None,
    )

    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )

    # -------------------------------------------------------
    # Best video
    # -------------------------------------------------------

    parser.add_argument(
        "--best-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--best-video-duration",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--video-width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--video-height",
        type=int,
        default=360,
    )
    parser.add_argument(
        "--action-profile",
        choices=(
            "flat_safe",
            "terrain_mid",
            "terrain_high",
            "full",
    ),
    default="full",
)

    args = parser.parse_args()
    if args.migrate_flat_boxes and not args.restore:
        parser.error('--migrate-flat-boxes requires --restore')

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    positive_values = (
        args.timesteps,
        args.num_envs,
        args.batch_size,
        args.num_minibatches,
        args.episode_length,
        args.num_evals,
        args.num_eval_envs,
    )

    if min(positive_values) < 1:
        parser.error(
            "training counts must all be positive"
        )

    if (
        args.batch_size
        * args.num_minibatches
    ) % args.num_envs:
        parser.error(
            "batch-size * num-minibatches "
            "must be divisible by num-envs"
        )

    if args.learning_rate <= 0:
        parser.error(
            "--learning-rate must be positive"
        )

    if args.entropy_cost < 0:
        parser.error(
            "--entropy-cost cannot be negative"
        )

    if args.restore and args.init_teacher:
        parser.error(
            "choose either --restore or --init-teacher"
        )

    if (
        args.init_teacher
        and args.perception != "lidar"
    ):
        parser.error(
            "--init-teacher requires --perception lidar"
        )

    if args.curriculum_stage is not None:
        if args.curriculum_stage < 0:
            parser.error(
                "--curriculum-stage cannot be negative"
            )

    if args.best_video_duration <= 0:
        parser.error(
            "--best-video-duration must be positive"
        )

    if args.video_fps <= 0:
        parser.error(
            "--video-fps must be positive"
        )

    if (
        args.video_width <= 0
        or args.video_height <= 0
    ):
        parser.error(
            "video dimensions must be positive"
        )

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Heavy imports only after MUJOCO_GL is configured.
    from brax.training.agents.ppo import (
        checkpoint,
        train as ppo,
    )

    from mujoco_playground._src import wrapper

    from adaptive_gait_env import (
        AdaptiveGaitEnv,
        default_config,
    )

    from adaptive_gait_policy import (
        contract,
        network_factory,
        read_contract,
    )

    # -----------------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------------

    cfg = default_config()
    cfg.episode_length = args.episode_length
    # Flat bootstrap curriculum:
    # Give the policy enough time to recover from poor early residuals while
    # strongly rewarding commanded forward motion.
    if args.action_profile == "flat_safe":
        # Fixed easy command first: remove command-speed randomness.
        cfg.command_min_speed = 0.08
        cfg.command_max_speed = 0.08

        # 3 s is too aggressive for the first PPO stage.
        cfg.no_progress_timeout = 8.0
        cfg.no_progress_min_delta = 0.01

        # Make standing still clearly unattractive.
        cfg.reward.velocity = 3.0
        cfg.reward.progress = 3.0
        cfg.reward.under_speed = -3.0

        # Keep learned residuals close to the known-good classical gait initially.
        cfg.reward.residual = -0.03
        cfg.reward.action_rate = -0.03

    gait_mode = {
        1: "tripod",
        2: "wave",
        3: "hybrid",
    }[args.stage]

    env = AdaptiveGaitEnv(
        terrain_level=args.terrain_level,
        perception=args.perception,
        config=cfg,
        azimuths=args.azimuths,
        elevations=args.elevations,
        dropout=args.dropout,
        noise=args.range_noise,
        gait_mode=gait_mode,
        action_profile=args.action_profile,
    )

    # -----------------------------------------------------------------------
    # Restore / teacher initialization
    # -----------------------------------------------------------------------

    restore: Path | None = None

    if args.restore or args.init_teacher:
        restore, old = read_contract(
            args.restore or args.init_teacher, migrate_flat_boxes=args.migrate_flat_boxes
        )

        expected_source = (
            "teacher"
            if args.init_teacher
            else args.perception
        )

        if old["actor_source"] != expected_source:
            raise SystemExit(
                "checkpoint actor source mismatch: "
                f"expected={expected_source}, "
                f"got={old['actor_source']}"
            )

    # -----------------------------------------------------------------------
    # Run directory
    # -----------------------------------------------------------------------

    if args.output is None:
        stage_token = (
            f"stage{args.stage}"
            f"-level{args.terrain_level}"
        )

        output = (
            Path(__file__).resolve().parent
            / "runs"
            / (
                f"adaptive-{args.perception}-"
                f"{stage_token}-"
                f"{datetime.now():%Y%m%d-%H%M%S}"
            )
        ).resolve()

    else:
        output = (
            args.output
            .expanduser()
            .resolve()
        )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    (output / ".gitignore").write_text(
        "*\n",
        encoding="utf-8",
    )

    checkpoint_dir = output / "checkpoints"
    monitor_dir = output / "monitor"
    video_dir = output / "videos"

    checkpoint_dir.mkdir()
    monitor_dir.mkdir()
    video_dir.mkdir()

    # -----------------------------------------------------------------------
    # Contract
    # -----------------------------------------------------------------------

    metadata = contract(env)
    if args.migrate_flat_boxes:
        if old.get('action_profile') != args.action_profile:
            raise ValueError('Migration must preserve the source action_profile')
        metadata['explicit_migration'] = old['explicit_migration']
        print('Explicit checkpoint migration: ' + json.dumps(metadata['explicit_migration']), flush=True)

    metadata["action_profile"] = args.action_profile

    metadata["initial_checkpoint"] = (
        str(restore)
        if restore is not None
        else None
    )

    metadata["initialization"] = (
        "teacher_weight_transfer_then_asymmetric_ppo"
        if args.init_teacher
        else (
            "restore"
            if args.restore
            else "ppo"
        )
    )

    metadata["gait_stage"] = args.stage
    metadata["curriculum_stage"] = (
        args.curriculum_stage
    )
    metadata["score_key"] = args.score_key

    metadata["training"] = {
        "timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "batch_size": args.batch_size,
        "num_minibatches": args.num_minibatches,
        "episode_length": args.episode_length,
        "num_evals": args.num_evals,
        "num_eval_envs": args.num_eval_envs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
    }

    write_json(
        output / "adaptive_contract.json",
        metadata,
    )

    print(
        json.dumps(
            {
                key: metadata[key]
                for key in (
                    "action_contract",
                    "observation_contract",
                    "observation_size",
                    "limits",
                )
            },
            indent=2,
        ),
        flush=True,
    )

    print(
        "Timing baselines: "
        "Tripod=1.0 s, Wave=1.0 s.",
        flush=True,
    )

    print(
        f"Output: {output}",
        flush=True,
    )

    print(
        "Actor: "
        f"{args.perception} | "
        f"gait={env.gait_mode} | "
        f"terrain={args.terrain_level} | "
        f"action={env.action_size} | "
        f"obs={env.observation_size}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # PPO network
    # -----------------------------------------------------------------------

    factory = network_factory()

    network_config = checkpoint.network_config(
        env.observation_size,
        env.action_size,
        True,
        factory,
    )

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------

    wandb_run = None
    wandb_module = None

    if (
        args.wandb
        and args.wandb_mode != "disabled"
    ):
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "--wandb requires `pip install wandb`"
            ) from exc

        wandb_module = wandb

        tags = [
            "adaptive-v4",
            f"gait-stage-{args.stage}",
            f"terrain-{args.terrain_level}",
            args.perception,
        ]

        if args.curriculum_stage is not None:
            tags.append(
                f"curriculum-{args.curriculum_stage:02d}"
            )

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=(
                args.wandb_name
                or output.name
            ),
            dir=str(output),
            mode=args.wandb_mode,
            tags=tags,
            config=metadata,
        )

        wandb_run.define_metric(
            "train/global_step"
        )

        wandb_run.define_metric(
            "eval/*",
            step_metric="train/global_step",
        )

        wandb_run.define_metric(
            "training/*",
            step_metric="train/global_step",
        )

        wandb_run.define_metric(
            "stage/*",
            step_metric="train/global_step",
        )

        # A curriculum stage is one training cycle. Keep these keys separate
        # from legacy stage/* dashboards so cycle best values update at every
        # evaluation as soon as a new best is observed.
        wandb_run.define_metric(
            "cycle/*",
            step_metric="train/global_step",
        )

        wandb_run.summary["stage/gait"] = (
            args.stage
        )

        wandb_run.summary[
            "stage/terrain_level"
        ] = args.terrain_level

        wandb_run.summary[
            "stage/perception"
        ] = args.perception

        wandb_run.summary[
            "stage/curriculum_index"
        ] = args.curriculum_stage

    # -----------------------------------------------------------------------
    # Best-policy state
    # -----------------------------------------------------------------------

    best_score = -math.inf
    best_step: int | None = None
    best_metrics: dict[str, float] | None = None

    latest_policy: dict[str, Any] = {}

    # Metrics whose matching checkpoint may be written in the adjacent callback.
    pending_best: tuple[int, float, dict[str, float]] | None = None

    # -----------------------------------------------------------------------
    # Checkpoint helper
    # -----------------------------------------------------------------------

    def checkpoint_for_step(
        step: int,
    ) -> Path | None:
        candidates = [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and int(path.name) == int(step)
            and (
                path / "ppo_network_config.json"
            ).exists()
        ]

        if len(candidates) == 1:
            return candidates[0]

        return None

    def persist_best_pointer() -> bool:
        """
        Save best checkpoint metadata once the checkpoint for best_step exists.
        """
        if best_step is None:
            return False

        best_checkpoint = checkpoint_for_step(
            best_step
        )

        if best_checkpoint is None:
            return False

        pointer = {
            "step": best_step,
            "score": best_score,
            "score_key": args.score_key,
            "path": str(best_checkpoint),
            "metrics": best_metrics,
        }

        write_json(
            monitor_dir / "best_checkpoint.json",
            pointer,
        )

        if wandb_run is not None:
            wandb_run.summary[
                "best/score"
            ] = best_score

            wandb_run.summary[
                "best/step"
            ] = best_step

            wandb_run.summary[
                "best/checkpoint"
            ] = str(best_checkpoint)

        return True

    # -----------------------------------------------------------------------
    # Brax progress callback
    # -----------------------------------------------------------------------

    def progress(
        step: int,
        metrics: Any,
    ) -> None:
        nonlocal best_score, best_step, best_metrics, pending_best

        step = int(step)
        numeric = numeric_metrics(metrics)

        payload = {
            "step": step,
            **numeric,
        }

        with (
            output / "metrics.jsonl"
        ).open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                json.dumps(payload)
                + "\n"
            )

        write_json(
            monitor_dir
            / "latest_metrics.json",
            {
                "step": step,
                "metrics": numeric,
            },
        )

        score = numeric.get(
            args.score_key
        )

        new_best = (
            step > 0
            and score is not None
            and math.isfinite(score)
            and score > best_score
        )

        if new_best:
            best_score = float(score)
            best_step = step
            best_metrics = dict(numeric)

            write_json(
                monitor_dir
                / "best_metrics.json",
                {
                    "step": best_step,
                    "score_key": args.score_key,
                    "score": best_score,
                    "metrics": best_metrics,
                },
            )

            pending_best = (
                best_step,
                best_score,
                best_metrics,
            )

            print(
                "NEW BEST | "
                f"step={best_step:,} "
                f"{args.score_key}="
                f"{best_score:.5f}",
                flush=True,
            )

        else:
            print(
                "EVAL | "
                f"step={step:,} "
                f"{args.score_key}="
                f"{score if score is not None else float('nan'):.5f} "
                f"best={best_score:.5f}",
                flush=True,
            )

        if wandb_run is not None:
            cycle_metrics = {}
            if best_step is not None:
                cycle_metrics = {'cycle/best_score': best_score, 'cycle/best_step': best_step}
                wandb_run.summary.update(cycle_metrics)
                wandb_run.summary['cycle/score_key'] = args.score_key
            wandb_run.log(
                {
                    **numeric,
                    **cycle_metrics,
                    "train/global_step": step,
                }
            )

            if new_best:
                wandb_run.log(
                    {
                        "stage/best_score": best_score,
                        "stage/best_step": best_step,
                        "cycle/best_score": best_score,
                        "cycle/best_step": best_step,
                        "train/global_step": step,
                    }
                )

    # -----------------------------------------------------------------------
    # Brax policy/checkpoint callback
    # -----------------------------------------------------------------------

    def save_policy(
        step: int,
        make_policy: Any,
        params: Any,
    ) -> None:
        nonlocal pending_best

        step = int(step)

        latest_policy.clear()
        latest_policy.update(
            {
                "step": step,
                "make_policy": make_policy,
                "params": params,
            }
        )

        # Step 0 is the untrained policy.
        if step == 0:
            return

        checkpoint.save(
            str(checkpoint_dir),
            step,
            params,
            network_config,
        )

        completed = checkpoint_for_step(
            step
        )

        if completed is None:
            raise RuntimeError(
                "checkpoint was not produced "
                f"for step {step}"
            )

        # Every checkpoint carries semantic contract metadata.
        write_json(
            completed
            / "adaptive_contract.json",
            metadata,
        )

        write_json(
            output / "latest_checkpoint.json",
            {
                "step": step,
                "path": str(completed),
            },
        )

        # If progress() selected this same evaluation as best,
        # the checkpoint now exists and can safely be pointed to.
        if (
            pending_best is not None
            and pending_best[0] == step
        ):
            persist_best_pointer()
            pending_best = None

        # Also handles callback ordering where best was already known.
        if best_step == step:
            persist_best_pointer()

    # -----------------------------------------------------------------------
    # Stage finalization
    # -----------------------------------------------------------------------

    def finalize_best() -> None:
        if best_step is None:
            print(
                "No trained evaluation produced a "
                "finite best score.",
                flush=True,
            )
            return

        if not persist_best_pointer():
            raise RuntimeError(
                "best evaluation exists but the matching "
                f"checkpoint is missing: step={best_step}"
            )

        pointer_path = (
            monitor_dir
            / "best_checkpoint.json"
        )

        pointer = json.loads(
            pointer_path.read_text(
                encoding="utf-8"
            )
        )

        best_checkpoint = Path(
            pointer["path"]
        ).resolve()

        print(
            "\n"
            "============================================",
            flush=True,
        )

        print(
            f"BEST STEP       : {best_step:,}",
            flush=True,
        )

        print(
            f"BEST SCORE      : {best_score:.6f}",
            flush=True,
        )

        print(
            f"BEST CHECKPOINT : {best_checkpoint}",
            flush=True,
        )

        # ---------------------------------------------------------------
        # Render deterministic best rollout
        # ---------------------------------------------------------------

        if args.best_video:
            if not latest_policy:
                raise RuntimeError(
                    "policy inference factory unavailable; "
                    "cannot render best video"
                )

            best_params = checkpoint.load(
                best_checkpoint
            )

            video_path = (
                video_dir
                / "best.gif"
            )

            try:
                render_policy_video(
                    env=env,
                    make_policy=latest_policy[
                        "make_policy"
                    ],
                    params=best_params,
                    output=video_path,
                    seed=args.seed + 20_000,
                    duration=args.best_video_duration,
                    fps=args.video_fps,
                    width=args.video_width,
                    height=args.video_height,
                )

            except Exception as exc:
                write_json(
                    monitor_dir
                    / "best_video_error.json",
                    {
                        "step": best_step,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    },
                )

                print(
                    "BEST VIDEO ERROR: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            else:
                print(
                    f"BEST VIDEO      : {video_path}",
                    flush=True,
                )

                pointer["video"] = str(
                    video_path
                )

                write_json(
                    pointer_path,
                    pointer,
                )

                # -------------------------------------------------------
                # W&B video + artifact
                # -------------------------------------------------------

                if (
                    wandb_run is not None
                    and wandb_module is not None
                ):
                    caption = (
                        f"Adaptive v4 | "
                        f"gait stage {args.stage} | "
                        f"terrain {args.terrain_level} | "
                        f"step {best_step:,} | "
                        f"score {best_score:.3f}"
                    )

                    wandb_run.log(
                        {
                            "stage/best_video":
                                wandb_module.Video(
                                    str(video_path),
                                    format="gif",
                                    caption=caption,
                                ),
                            "cycle/best_video":
                                wandb_module.Video(
                                    str(video_path),
                                    format="gif",
                                    caption=caption,
                                ),
                            "stage/best_score":
                                best_score,
                            "stage/best_step":
                                best_step,
                            "cycle/best_score":
                                best_score,
                            "cycle/best_step":
                                best_step,
                            "train/global_step":
                                latest_policy['step'],
                        }
                    )

                    wandb_run.summary[
                        "stage/best_video"
                    ] = str(video_path)
                    wandb_run.summary["cycle/best_video_path"] = str(video_path)
                    wandb_run.summary["cycle/best_score"] = best_score
                    wandb_run.summary["cycle/best_step"] = best_step

                    artifact = (
                        wandb_module.Artifact(
                            name=(
                                f"{output.name}"
                                "-best-policy"
                            ),
                            type="policy",
                            metadata={
                                "step":
                                    best_step,
                                "score":
                                    best_score,
                                "score_key":
                                    args.score_key,
                                "gait_stage":
                                    args.stage,
                                "terrain_level":
                                    args.terrain_level,
                                "perception":
                                    args.perception,
                            },
                        )
                    )

                    artifact.add_file(
                        str(video_path),
                        name="best.gif",
                    )
                    artifact.add_dir(str(best_checkpoint), name='checkpoint')
                    artifact.add_file(str(output / 'metrics.jsonl'), name='metrics.jsonl')
                    termination_report = video_path.with_suffix('.termination.json')
                    if termination_report.is_file():
                        artifact.add_file(str(termination_report), name='best.termination.json')

                    artifact.add_file(
                        str(
                            pointer_path
                        ),
                        name=(
                            "best_checkpoint.json"
                        ),
                    )

                    wandb_run.log_artifact(
                        artifact,
                        aliases=[
                            "best",
                            f"step-{best_step}",
                        ],
                    )

        print(
            "============================================\n",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # PPO
    # -----------------------------------------------------------------------

    print(
        "Starting PPO training...",
        flush=True,
    )

    try:
        ppo.train(
            environment=env,

            num_timesteps=args.timesteps,
            num_envs=args.num_envs,
            episode_length=args.episode_length,

            action_repeat=1,

            learning_rate=args.learning_rate,
            entropy_cost=args.entropy_cost,
            discounting=0.99,

            unroll_length=20,

            batch_size=args.batch_size,
            num_minibatches=args.num_minibatches,
            num_updates_per_batch=4,

            normalize_observations=True,

            clipping_epsilon=0.2,
            gae_lambda=0.95,
            max_grad_norm=1.0,

            num_evals=args.num_evals,

            num_eval_envs=min(
                args.num_envs,
                args.num_eval_envs,
            ),

            deterministic_eval=True,

            network_factory=factory,

            wrap_env_fn=functools.partial(
                wrapper.wrap_for_brax_training,
                full_reset=True,
            ),

            restore_checkpoint_path=(
                str(restore)
                if restore is not None
                else None
            ),

            seed=args.seed,

            progress_fn=progress,
            policy_params_fn=save_policy,

            use_pmap_on_reset=False,
        )

        finalize_best()

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
