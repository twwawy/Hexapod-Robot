#!/usr/bin/env python3
"""Launch competence-gated flat-to-final-stairs firmware residual PPO."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from terrain_curriculum import MAX_TERRAIN_LEVEL, terrain_level


def _arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="firmware-terrain-final-stairs")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stages", type=int, default=36, help="Maximum competence attempts."
    )
    parser.add_argument("--stage-timesteps", type=int, default=5_000_000)
    parser.add_argument("--stage-0-timesteps", type=int, default=None)
    parser.add_argument(
        "--flat-baseline-timesteps",
        type=int,
        default=262_144,
        help="Minimal fixed level-0 budget; default is four PPO updates.",
    )
    parser.add_argument(
        "--start-level", type=int, choices=range(MAX_TERRAIN_LEVEL + 1), default=1
    )
    parser.add_argument(
        "--max-level",
        type=int,
        choices=range(MAX_TERRAIN_LEVEL + 1),
        default=MAX_TERRAIN_LEVEL,
    )
    parser.add_argument(
        "--level-progression",
        choices=("competence", "sequential"),
        default="competence",
    )
    parser.add_argument("--promote-threshold", type=float, default=0.80)
    parser.add_argument(
        "--max-stages-per-level",
        type=int,
        default=None,
        help=(
            "In competence mode, promote after this many attempts even when "
            "the success threshold is not met."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("latest", "best"),
        default="latest",
        help="Checkpoint passed to the next stage.",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-value-function",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore the initial critic too; later stages always restore it.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="hexapod-firmware-terrain")
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
    if not 0.0 <= args.promote_threshold <= 1.0:
        parser.error("--promote-threshold must be in [0, 1]")
    if args.max_stages_per_level is not None and args.max_stages_per_level < 1:
        parser.error("--max-stages-per-level must be positive")
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


def _selected_checkpoint(run_dir: Path, selection: str) -> Path:
    if selection == "latest":
        return _newest_checkpoint(run_dir / "checkpoints")
    if selection != "best":
        raise ValueError(f"unsupported checkpoint selection: {selection}")

    pointer = run_dir / "monitor" / "best_checkpoint.json"
    if not pointer.exists():
        raise RuntimeError(f"best checkpoint pointer was not produced: {pointer}")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    checkpoint = Path(payload.get("path", "")).expanduser().resolve()
    if not checkpoint.is_dir() or not (checkpoint / "ppo_network_config.json").exists():
        raise RuntimeError(f"invalid best checkpoint in {pointer}: {checkpoint}")
    return checkpoint


def _level_transition(
    *,
    level: int,
    max_level: int,
    progression: str,
    success: float,
    threshold: float,
    level_attempt: int,
    max_stages_per_level: int | None,
) -> tuple[int, bool, str]:
    competence_met = success >= threshold
    attempt_limit_reached = (
        max_stages_per_level is not None
        and level_attempt >= max_stages_per_level
    )
    if progression == "sequential":
        should_advance = True
        reason = "sequential"
    elif competence_met:
        should_advance = True
        reason = "competence"
    elif attempt_limit_reached:
        should_advance = True
        reason = "attempt_limit"
    else:
        should_advance = False
        reason = "retry"

    next_level = min(max_level, level + 1) if should_advance else level
    completed = level == max_level and should_advance
    return next_level, completed, reason


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


def _trainer_int_arg(extras: list[str], option: str, default: int) -> int:
    value = default
    for index, token in enumerate(extras):
        if token == option and index + 1 < len(extras):
            value = int(extras[index + 1])
        elif token.startswith(option + "="):
            value = int(token.split("=", 1)[1])
    return value


def _replace_trainer_arg(extras: list[str], option: str, value: int) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(extras):
        token = extras[index]
        if token == option:
            index += 2
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    result.extend((option, str(value)))
    return result


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
        baseline_extras = extras
        steps_per_update = (
            _trainer_int_arg(extras, "--batch-size", 256)
            * _trainer_int_arg(extras, "--num-minibatches", 8)
            * _trainer_int_arg(extras, "--unroll-length", 32)
        )
        requested_evals = _trainer_int_arg(extras, "--num-evals", 10)
        max_useful_evals = math.ceil(
            args.flat_baseline_timesteps / steps_per_update
        ) + 1
        if requested_evals > max_useful_evals:
            baseline_extras = _replace_trainer_arg(
                extras, "--num-evals", max_useful_evals
            )
            print(
                f"flat_num_evals={requested_evals}->{max_useful_evals} "
                f"for minimal {args.flat_baseline_timesteps:,}-step baseline"
            )
        run_dir = _run_trainer(
            trainer=trainer,
            args=args,
            extras=baseline_extras,
            prefix=prefix,
            timesteps=args.flat_baseline_timesteps,
            seed=args.seed,
            level=0,
            stage=None,
            init_checkpoint=None,
            restore_value=False,
            terrain_root=terrain_root,
        )
        init_checkpoint = _selected_checkpoint(run_dir, args.checkpoint_selection)
        history.append(
            {
                "phase": "flat_baseline",
                "terrain_level": 0,
                "terrain_name": terrain_level(0).name,
                "timesteps": args.flat_baseline_timesteps,
                "run_dir": str(run_dir),
                "checkpoint": str(init_checkpoint),
                **_video_paths(run_dir),
            }
        )
        _write_history(state_path, history)

    level = args.start_level
    level_attempt = 0
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
        checkpoint = _selected_checkpoint(run_dir, args.checkpoint_selection)
        level_attempt += 1
        next_level, completed, promotion_reason = _level_transition(
            level=level,
            max_level=args.max_level,
            progression=args.level_progression,
            success=success,
            threshold=args.promote_threshold,
            level_attempt=level_attempt,
            max_stages_per_level=args.max_stages_per_level,
        )
        spec = terrain_level(level)
        history.append(
            {
                "phase": "terrain",
                "stage": stage,
                "terrain_level": level,
                "terrain_name": spec.name,
                "terrain_description": spec.description,
                "timesteps": timesteps,
                "success_rate": success,
                "promote_threshold": args.promote_threshold,
                "level_attempt": level_attempt,
                "max_stages_per_level": args.max_stages_per_level,
                "promotion_reason": promotion_reason,
                "next_level": next_level,
                "curriculum_complete": completed,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_selection": args.checkpoint_selection,
                **_video_paths(run_dir),
            }
        )
        _write_history(state_path, history)
        print(
            f"complete stage={stage} success={success:.3f} "
            f"level={level}->{next_level} attempt={level_attempt} "
            f"reason={promotion_reason} checkpoint={checkpoint}"
        )
        init_checkpoint = checkpoint
        if next_level != level:
            level_attempt = 0
        level = next_level
        if completed:
            print(f"CURRICULUM_COMPLETE level={level} success={success:.3f}")
            break

    print(f"CURRICULUM_HISTORY={state_path}")


if __name__ == "__main__":
    main()
