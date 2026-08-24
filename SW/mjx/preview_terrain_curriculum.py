#!/usr/bin/env python3
"""Render short straight-line ramp/stair previews for every terrain level."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

from best_policy_video import render_policy_video
from prepare_rl_scene import MIXED_PATCH_NAMES, MIXED_STAIR_COUNT
from rough_terrain_env import ACTION_SIZE, HexapodRoughTerrainEnv, default_config
from terrain_curriculum import TERRAIN_LEVELS, terrain_level


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="terrain-curriculum-preview")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "terrain_previews",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        choices=range(len(TERRAIN_LEVELS)),
        default=list(range(len(TERRAIN_LEVELS))),
    )
    parser.add_argument(
        "--patches",
        nargs="+",
        choices=("ramp", "stairs"),
        default=("ramp", "stairs"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional Brax PPO checkpoint or checkpoints directory; zero residual otherwise.",
    )
    parser.add_argument("--terrain-randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="hexapod-rough-terrain")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    if args.duration <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("video duration, fps, width, and height must be positive")
    return args


def _resolve_checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (resolved / "ppo_network_config.json").exists():
        return resolved
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
        raise SystemExit(f"no PPO checkpoint found under {resolved}")
    return candidates[-1]


def _zero_make_policy(_params, deterministic: bool = True):
    del deterministic

    def policy(observation, _key):
        return jp.zeros(observation.shape[:-1] + (ACTION_SIZE,)), {}

    return policy


def _checkpoint_policy(checkpoint_path: Path):
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    inference = ppo_checkpoint.load_policy(checkpoint_path, deterministic=True)

    def make_policy(_params, deterministic: bool = True):
        del deterministic
        return inference

    return make_policy


def main() -> None:
    args = _arguments()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_dir / f"{args.run_name}_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = (
        _resolve_checkpoint(args.checkpoint) if args.checkpoint is not None else None
    )
    make_policy = _checkpoint_policy(checkpoint) if checkpoint else _zero_make_policy

    wandb_run = None
    wandb_module = None
    if args.wandb:
        import wandb

        wandb_module = wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}_{timestamp}",
            group=args.run_name,
            mode=args.wandb_mode,
            job_type="terrain-curriculum-preview",
            config={
                "levels": args.levels,
                "patches": args.patches,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "terrain_randomize": args.terrain_randomize,
                "duration": args.duration,
                "fps": args.fps,
            },
        )

    records: list[dict] = []
    try:
        for level in args.levels:
            spec = terrain_level(level)
            rng = np.random.default_rng(args.seed + level)
            stair_total_rise = (
                float(rng.uniform(*spec.stair_total_rise_range_m))
                if args.terrain_randomize
                else spec.stair_total_rise_range_m[1]
            )
            ramp_rise = (
                float(rng.uniform(*spec.ramp_rise_range_m))
                if args.terrain_randomize
                else spec.ramp_rise_range_m[1]
            )
            config = default_config()
            config.terrain.stair_total_rise = stair_total_rise
            config.terrain.step_height = stair_total_rise / MIXED_STAIR_COUNT
            config.terrain.ramp_rise = ramp_rise
            config.terrain.patch_probabilities = spec.patch_probabilities
            config.command_max_yaw_rate = spec.yaw_limit_rps

            for patch in args.patches:
                env = HexapodRoughTerrainEnv(
                    config=config_dict.ConfigDict(config.to_dict()),
                    terrain="mixed",
                    command_curriculum=True,
                    fixed_curriculum_stage=0,
                    scripted_commands=True,
                    fixed_terrain_patch=MIXED_PATCH_NAMES.index(patch),
                )
                output = run_dir / f"stage{level:02d}_level{level}_{patch}.gif"
                render_policy_video(
                    env=env,
                    make_policy=make_policy,
                    params=None,
                    output=output,
                    seed=args.seed + level,
                    duration=args.duration,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    terrain="mixed",
                    overlay_title=(
                        f"S{level:02d} L{level} {patch.title()} | "
                        f"Stairs {100.0 * stair_total_rise:.1f}cm total | "
                        f"Ramp {100.0 * ramp_rise:.1f}cm"
                    ),
                )
                key = f"preview/video_stage{level:02d}_level{level}_{patch}"
                record = {
                    "stage": level,
                    "level": level,
                    "patch": patch,
                    "stair_total_rise_m": stair_total_rise,
                    "ramp_rise_m": ramp_rise,
                    "video": str(output),
                    "wandb_key": key,
                }
                records.append(record)
                if wandb_run is not None and wandb_module is not None:
                    wandb_run.log(
                        {
                            key: wandb_module.Video(
                                str(output),
                                format="gif",
                                caption=(
                                    f"Stage {level:02d} | Level {level} | {patch} | "
                                    f"stair_total={100.0 * stair_total_rise:.1f}cm | "
                                    f"ramp={100.0 * ramp_rise:.1f}cm"
                                ),
                            ),
                            "preview/stage": level,
                        }
                    )
                print(f"preview_video stage={level} patch={patch}: {output}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    (run_dir / "preview_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"preview_complete: {run_dir}")


if __name__ == "__main__":
    main()
