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


EASY_LEVEL_LAST = 3
EASY_MAX_STAGES_PER_LEVEL = 2
HARD_MAX_STAGES_PER_LEVEL = 4


def _max_stages_for_level(level: int, override: int | None = None) -> int:
    """Return the competence-attempt cap for one terrain level."""
    if override is not None:
        return override
    if level <= EASY_LEVEL_LAST:
        return EASY_MAX_STAGES_PER_LEVEL
    return HARD_MAX_STAGES_PER_LEVEL


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
        "--start-stage",
        type=int,
        default=0,
        help="Stage label offset used when continuing from an earlier run.",
    )
    parser.add_argument(
        "--stages", type=int, default=44, help="Maximum competence attempts."
    )
    parser.add_argument("--stage-timesteps", type=int, default=5_000_000)
    parser.add_argument("--stage-0-timesteps", type=int, default=None)
    parser.add_argument(
        "--flat-baseline-timesteps",
        type=int,
        default=262_144,
        help=(
            "Initial level-0 budget; default is four PPO updates. "
            "Competence mode may run one additional flat stage."
        ),
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
            "Override the default per-level cap (levels 0-3: 2 attempts, "
            "levels 4-12: 4 attempts)."
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
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=None,
        help="Level-indexed frozen-teacher paths and distillation weights.",
    )
    parser.add_argument("--teacher-huber-delta", type=float, default=0.10)
    parser.add_argument("--teacher-max-policy-rejection-rate", type=float, default=0.01)
    parser.add_argument("--teacher-max-foot-limited-rate", type=float, default=0.01)
    parser.add_argument("--teacher-max-failure-rate", type=float, default=0.05)
    parser.add_argument(
        "--init-student-from-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize a fresh curriculum actor from its level-0 v3 teacher.",
    )
    parser.add_argument(
        "--teacher-stop-on-attempt-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop instead of force-promoting a teacher student that missed competence.",
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
    if args.start_stage < 0:
        parser.error("--start-stage cannot be negative")
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
        "--teacher-v3-checkpoint",
        "--teacher-v2-checkpoint",
        "--distill-v3-weight",
        "--distill-v2-xy-weight",
        "--distill-huber-delta",
        "--init-student-from-teacher",
        "--no-init-student-from-teacher",
        "--best-safe-max-policy-rejection-rate",
        "--best-safe-max-foot-limited-rate",
        "--best-safe-max-failure-rate",
        "--teacher-video",
        "--no-teacher-video",
    }
    conflicts = [
        token
        for token in extras
        if token.split("=", 1)[0] in managed
    ]
    if conflicts:
        parser.error(f"launcher-managed trainer arguments cannot follow '--': {conflicts}")
    if args.teacher_huber_delta <= 0:
        parser.error("--teacher-huber-delta must be positive")
    if min(
        args.teacher_max_policy_rejection_rate,
        args.teacher_max_foot_limited_rate,
        args.teacher_max_failure_rate,
    ) < 0:
        parser.error("teacher best-safe rate limits cannot be negative")
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


def _guard_teacher_attempt_limit(
    args: argparse.Namespace,
    *,
    level: int,
    success: float,
    next_level: int,
    completed: bool,
    reason: str,
) -> tuple[int, bool, str]:
    if (
        args.teacher_manifest is not None
        and args.teacher_stop_on_attempt_limit
        and reason == "attempt_limit"
        and success < args.promote_threshold
    ):
        return level, False, "attempt_limit_stop"
    return next_level, completed, reason


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
        "teacher_distillation": metadata.get("teacher_distillation"),
    }


def _load_teacher_manifest(args: argparse.Namespace) -> None:
    args.teacher_levels = {}
    if args.teacher_manifest is None:
        return
    manifest_path = args.teacher_manifest.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid teacher manifest {manifest_path}: {exc}") from exc
    levels = payload.get("levels")
    if not isinstance(levels, dict):
        raise SystemExit(f"teacher manifest must contain an object named 'levels': {manifest_path}")
    parsed: dict[int, dict[str, Any]] = {}
    for raw_level, entry in levels.items():
        try:
            level = int(raw_level)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid teacher level {raw_level!r}") from exc
        if level not in range(MAX_TERRAIN_LEVEL + 1) or not isinstance(entry, dict):
            raise SystemExit(f"invalid teacher entry for level {raw_level!r}")
        parsed[level] = dict(entry)
    args.teacher_manifest = manifest_path
    args.teacher_levels = parsed


def _teacher_path(args: argparse.Namespace, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    run_relative = (args.run_root / path).resolve()
    if run_relative.exists():
        return run_relative
    assert args.teacher_manifest is not None
    return (args.teacher_manifest.parent / path).resolve()


def _teacher_trainer_args(
    args: argparse.Namespace,
    *,
    level: int,
    initialize_student: bool,
) -> list[str]:
    entry = args.teacher_levels.get(level)
    if entry is None:
        return []
    command: list[str] = []
    v3_path = entry.get("v3_checkpoint")
    v2_path = entry.get("v2_xy_checkpoint")
    v3_weight = float(entry.get("v3_weight", 0.0))
    v2_weight = float(entry.get("v2_xy_weight", 0.0))
    if min(v3_weight, v2_weight) < 0:
        raise SystemExit(f"teacher weights cannot be negative at level {level}")
    if v3_path is not None:
        command.extend(("--teacher-v3-checkpoint", str(_teacher_path(args, v3_path))))
        command.append("--teacher-video")
    elif v3_weight > 0:
        raise SystemExit(f"level {level} v3_weight requires v3_checkpoint")
    if v2_path is not None:
        command.extend(("--teacher-v2-checkpoint", str(_teacher_path(args, v2_path))))
    elif v2_weight > 0:
        raise SystemExit(f"level {level} v2_xy_weight requires v2_xy_checkpoint")
    command.extend(
        (
            "--distill-v3-weight",
            str(v3_weight),
            "--distill-v2-xy-weight",
            str(v2_weight),
            "--distill-huber-delta",
            str(args.teacher_huber_delta),
            "--best-safe-max-policy-rejection-rate",
            str(args.teacher_max_policy_rejection_rate),
            "--best-safe-max-foot-limited-rate",
            str(args.teacher_max_foot_limited_rate),
            "--best-safe-max-failure-rate",
            str(args.teacher_max_failure_rate),
        )
    )
    if initialize_student and args.init_student_from_teacher:
        if v3_path is None:
            raise SystemExit(
                f"fresh student initialization requires a v3 teacher at level {level}"
            )
        command.append("--init-student-from-teacher")
    return command


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
    teacher_args: list[str],
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
    command.extend(teacher_args)
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
    _load_teacher_manifest(args)
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
    level = args.start_level
    level_attempt = 0

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
            teacher_args=_teacher_trainer_args(
                args, level=0, initialize_student=True
            ),
        )
        init_checkpoint = _selected_checkpoint(run_dir, args.checkpoint_selection)
        latest = json.loads(
            (run_dir / "monitor" / "latest_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        success = float(
            latest.get("metrics", {}).get("eval/episode_terrain_success", 0.0)
        )
        level_attempt = 1
        stage_limit = _max_stages_for_level(0, args.max_stages_per_level)
        next_level, completed, promotion_reason = _level_transition(
            level=0,
            max_level=args.max_level,
            progression=args.level_progression,
            success=success,
            threshold=args.promote_threshold,
            level_attempt=level_attempt,
            max_stages_per_level=stage_limit,
        )
        next_level, completed, promotion_reason = _guard_teacher_attempt_limit(
            args,
            level=0,
            success=success,
            next_level=next_level,
            completed=completed,
            reason=promotion_reason,
        )
        if next_level != 0:
            next_level = max(next_level, args.start_level)
        history.append(
            {
                "phase": "flat_baseline",
                "terrain_level": 0,
                "terrain_name": terrain_level(0).name,
                "timesteps": args.flat_baseline_timesteps,
                "success_rate": success,
                "promote_threshold": args.promote_threshold,
                "level_attempt": level_attempt,
                "max_stages_per_level": stage_limit,
                "promotion_reason": promotion_reason,
                "next_level": next_level,
                "curriculum_complete": completed,
                "run_dir": str(run_dir),
                "checkpoint": str(init_checkpoint),
                **_video_paths(run_dir),
            }
        )
        _write_history(state_path, history)
        print(
            f"complete stage=flat_baseline success={success:.3f} "
            f"level=0->{next_level} attempt={level_attempt}/{stage_limit} "
            f"reason={promotion_reason} checkpoint={init_checkpoint}"
        )
        if completed:
            print(f"CURRICULUM_COMPLETE level=0 success={success:.3f}")
            print(f"CURRICULUM_HISTORY={state_path}")
            return
        if promotion_reason == "attempt_limit_stop":
            print(
                f"CURRICULUM_STOPPED level=0 success={success:.3f} "
                "reason=teacher_attempt_limit"
            )
            print(f"CURRICULUM_HISTORY={state_path}")
            return
        if next_level != 0:
            level_attempt = 0
        level = next_level

    for attempt_index in range(args.stages):
        stage = args.start_stage + attempt_index
        timesteps = (
            args.stage_0_timesteps
            if attempt_index == 0 and args.stage_0_timesteps is not None
            else args.stage_timesteps
        )
        prefix = f"{args.run_name}-stage{stage:02d}-level{level}"
        initialize_student = init_checkpoint is None
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
            restore_value=(
                attempt_index > 0
                or level_attempt > 0
                or args.init_value_function
            ),
            terrain_root=terrain_root,
            teacher_args=_teacher_trainer_args(
                args,
                level=level,
                initialize_student=initialize_student,
            ),
        )
        latest = json.loads(
            (run_dir / "monitor" / "latest_metrics.json").read_text(encoding="utf-8")
        )
        success = float(latest.get("metrics", {}).get("eval/episode_terrain_success", 0.0))
        checkpoint = _selected_checkpoint(run_dir, args.checkpoint_selection)
        level_attempt += 1
        stage_limit = _max_stages_for_level(level, args.max_stages_per_level)
        next_level, completed, promotion_reason = _level_transition(
            level=level,
            max_level=args.max_level,
            progression=args.level_progression,
            success=success,
            threshold=args.promote_threshold,
            level_attempt=level_attempt,
            max_stages_per_level=stage_limit,
        )
        next_level, completed, promotion_reason = _guard_teacher_attempt_limit(
            args,
            level=level,
            success=success,
            next_level=next_level,
            completed=completed,
            reason=promotion_reason,
        )
        if level == 0 and next_level != 0:
            next_level = max(next_level, args.start_level)
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
                "max_stages_per_level": stage_limit,
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
            f"level={level}->{next_level} attempt={level_attempt}/{stage_limit} "
            f"reason={promotion_reason} checkpoint={checkpoint}"
        )
        init_checkpoint = checkpoint
        if next_level != level:
            level_attempt = 0
        level = next_level
        if completed:
            print(f"CURRICULUM_COMPLETE level={level} success={success:.3f}")
            break
        if promotion_reason == "attempt_limit_stop":
            print(
                f"CURRICULUM_STOPPED level={level} success={success:.3f} "
                "reason=teacher_attempt_limit"
            )
            break

    print(f"CURRICULUM_HISTORY={state_path}")


if __name__ == "__main__":
    main()
