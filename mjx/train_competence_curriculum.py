#!/usr/bin/env python3
"""Launch flat-to-stairs firmware-residual PPO stages with checkpoint transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="firmware-stairs")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--stage-timesteps", type=int, default=5_000_000)
    parser.add_argument("--stage-0-timesteps", type=int, default=None)
    parser.add_argument(
        "--flat-baseline-timesteps",
        type=int,
        default=1_000_000,
        help="Level-0 pretraining budget when no initial checkpoint is supplied.",
    )
    parser.add_argument("--start-level", type=int, choices=range(5), default=1)
    parser.add_argument("--max-level", type=int, choices=range(5), default=4)
    parser.add_argument(
        "--level-progression",
        choices=("competence", "sequential"),
        default="competence",
    )
    parser.add_argument("--promote-threshold", type=float, default=0.80)
    parser.add_argument("--demote-threshold", type=float, default=0.50)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-value-function",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore the initial critic too; later stages always restore it.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="hexapod-firmware-stairs")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Extra train_rough_terrain.py options after '--'.",
    )
    args = parser.parse_args()
    extras = args.trainer_args
    if extras and extras[0] == "--":
        extras = extras[1:]
    if args.stages < 1 or args.stage_timesteps < 1:
        parser.error("--stages and --stage-timesteps must be positive")
    if args.flat_baseline_timesteps < 0:
        parser.error("--flat-baseline-timesteps cannot be negative")
    if args.stage_0_timesteps is not None and args.stage_0_timesteps < 1:
        parser.error("--stage-0-timesteps must be positive")
    if args.start_level > args.max_level:
        parser.error("--start-level cannot exceed --max-level")
    if not 0.0 <= args.demote_threshold <= args.promote_threshold <= 1.0:
        parser.error("thresholds must satisfy 0 <= demote <= promote <= 1")
    managed = {
        "--timesteps",
        "--seed",
        "--run-name",
        "--run-root",
        "--output",
        "--terrain-level",
        "--curriculum-stage",
        "--competence-stage",
        "--init-checkpoint",
        "--wandb-group",
        "--wandb-project",
        "--wandb-entity",
        "--wandb-mode",
    }
    conflicts = [
        token
        for token in extras
        if token.split("=", 1)[0] in managed
    ]
    if conflicts:
        parser.error(f"launcher-managed trainer arguments cannot follow '--': {conflicts}")
    return args, extras


def _newest_checkpoint(directory: Path) -> Path:
    candidates = [
        child
        for child in directory.iterdir()
        if child.is_dir()
        and child.name.isdigit()
        and (child / "ppo_network_config.json").exists()
    ]
    if not candidates:
        raise RuntimeError(f"no checkpoint produced in {directory}")
    return max(candidates, key=lambda child: int(child.name))


def _new_run(terrain_root: Path, before: set[Path], prefix: str) -> Path:
    created = [path for path in set(terrain_root.iterdir()) - before if path.is_dir()]
    matching = [path for path in created if path.name.startswith(prefix + "_")]
    if len(matching) != 1:
        raise RuntimeError(f"could not uniquely resolve run {prefix!r}: {matching}")
    return matching[0]


def _write_history(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _video_paths(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    return {
        "best_video": metadata["best_video_path"],
        "stage_video": metadata["stage_video_path"],
        "progress_videos": metadata["progress_video_dir"],
        "progress_targets": metadata["progress_video_targets"],
    }


def _run_trainer(
    *,
    trainer: Path,
    args: argparse.Namespace,
    extras: list[str],
    prefix: str,
    timesteps: int,
    seed: int,
    level: int,
    stage: int | None,
    init_checkpoint: Path | None,
    restore_value: bool,
    terrain_root: Path,
) -> Path:
    before = set(terrain_root.iterdir())
    command = [
        sys.executable,
        str(trainer),
        "--timesteps",
        str(timesteps),
        "--seed",
        str(seed),
        "--run-name",
        prefix,
        "--run-root",
        str(args.run_root),
        "--terrain-level",
        str(level),
        "--wandb-group",
        args.run_name,
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
    ]
    if stage is not None:
        command.extend(("--curriculum-stage", str(stage)))
    if args.wandb_entity:
        command.extend(("--wandb-entity", args.wandb_entity))
    if args.wandb:
        command.append("--wandb")
    if init_checkpoint is not None:
        command.extend(("--init-checkpoint", str(init_checkpoint)))
        if restore_value:
            command.append("--init-value-function")
    command.extend(extras)
    print(
        f"launch stage={stage} level={level} timesteps={timesteps:,} "
        f"init={init_checkpoint}"
    )
    subprocess.run(command, check=True)
    return _new_run(terrain_root, before, prefix)


def main() -> None:
    args, extras = _arguments()
    args.run_root = args.run_root.expanduser().resolve()
    trainer = Path(__file__).resolve().parent / "train_rough_terrain.py"
    terrain_root = args.run_root / "terrain"
    terrain_root.mkdir(parents=True, exist_ok=True)
    state_path = args.run_root / "curricula" / args.run_name / "curriculum_history.json"
    history: list[dict[str, Any]] = []
    init_checkpoint = (
        args.init_checkpoint.expanduser().resolve()
        if args.init_checkpoint is not None
        else None
    )

    if init_checkpoint is None and args.flat_baseline_timesteps > 0:
        prefix = f"{args.run_name}-flat-baseline"
        run_dir = _run_trainer(
            trainer=trainer,
            args=args,
            extras=extras,
            prefix=prefix,
            timesteps=args.flat_baseline_timesteps,
            seed=args.seed,
            level=0,
            stage=None,
            init_checkpoint=None,
            restore_value=False,
            terrain_root=terrain_root,
        )
        init_checkpoint = _newest_checkpoint(run_dir / "checkpoints")
        history.append(
            {
                "phase": "flat_baseline",
                "terrain_level": 0,
                "timesteps": args.flat_baseline_timesteps,
                "run_dir": str(run_dir),
                "checkpoint": str(init_checkpoint),
                **_video_paths(run_dir),
            }
        )
        _write_history(state_path, history)

    level = args.start_level
    for stage in range(args.stages):
        timesteps = (
            args.stage_0_timesteps
            if stage == 0 and args.stage_0_timesteps is not None
            else args.stage_timesteps
        )
        prefix = f"{args.run_name}-stage{stage:02d}-level{level}"
        run_dir = _run_trainer(
            trainer=trainer,
            args=args,
            extras=extras,
            prefix=prefix,
            timesteps=timesteps,
            seed=args.seed + stage + 1,
            level=level,
            stage=stage,
            init_checkpoint=init_checkpoint,
            restore_value=(stage > 0 or args.init_value_function),
            terrain_root=terrain_root,
        )
        latest = json.loads(
            (run_dir / "monitor" / "latest_metrics.json").read_text(encoding="utf-8")
        )
        success = float(latest.get("metrics", {}).get("eval/episode_terrain_success", 0.0))
        checkpoint = _newest_checkpoint(run_dir / "checkpoints")
        if args.level_progression == "sequential":
            next_level = min(args.max_level, level + 1)
        elif success >= args.promote_threshold:
            next_level = min(args.max_level, level + 1)
        elif success < args.demote_threshold:
            next_level = max(0, level - 1)
        else:
            next_level = level
        history.append(
            {
                "phase": "stairs",
                "stage": stage,
                "terrain_level": level,
                "timesteps": timesteps,
                "success_rate": success,
                "next_level": next_level,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                **_video_paths(run_dir),
            }
        )
        _write_history(state_path, history)
        print(
            f"complete stage={stage} success={success:.3f} "
            f"level={level}->{next_level} checkpoint={checkpoint}"
        )
        init_checkpoint = checkpoint
        level = next_level

    print(f"CURRICULUM_HISTORY={state_path}")


if __name__ == "__main__":
    main()
