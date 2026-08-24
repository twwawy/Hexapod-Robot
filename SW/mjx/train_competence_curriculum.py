#!/usr/bin/env python3
"""Competence-based mixed-terrain curriculum launcher.

When no checkpoint is supplied, a short flat command stage first identifies
the controller-following baseline and small residual.  Terrain stages then
move from rough patches into stairs whose complete rise reaches 20 cm.  Every
stage keeps the same 24-D action and 113-D observation contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="terrain-competence")
    parser.add_argument(
        "--run-root", type=Path, default=Path(__file__).resolve().parent / "runs"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--stage-timesteps", type=int, default=5_000_000)
    parser.add_argument(
        "--flat-baseline-timesteps",
        type=int,
        default=1_000_000,
        help=(
            "Short flat pretraining budget used only when --init-checkpoint "
            "is absent; set 0 to start terrain from scratch."
        ),
    )
    parser.add_argument(
        "--stage-0-timesteps",
        type=int,
        default=None,
        help="Override timesteps for stage 0 only; later stages use --stage-timesteps.",
    )
    parser.add_argument("--start-level", type=int, choices=range(5), default=0)
    parser.add_argument("--max-level", type=int, choices=range(5), default=4)
    parser.add_argument(
        "--level-progression",
        choices=("competence", "sequential"),
        default="competence",
        help=(
            "competence promotes/demotes from evaluation success; sequential "
            "advances one level after every stage."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Compatible flat/terrain checkpoint used to initialize stage 0.",
    )
    parser.add_argument(
        "--init-value-function",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore the critic at stage 0; later terrain stages always restore it.",
    )
    parser.add_argument("--promote-threshold", type=float, default=0.80)
    parser.add_argument("--demote-threshold", type=float, default=0.50)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default="hexapod-rough-terrain",
        help="Shared W&B project for the flat baseline and every terrain stage.",
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Additional train_rough_terrain.py arguments after '--'.",
    )
    args = parser.parse_args()
    extras = args.trainer_args
    if extras and extras[0] == "--":
        extras = extras[1:]
    if (
        args.stages < 1
        or args.stage_timesteps < 1
        or args.flat_baseline_timesteps < 0
        or (args.stage_0_timesteps is not None and args.stage_0_timesteps < 1)
    ):
        parser.error("--stages and stage timestep values must be positive")
    if not 0.0 <= args.demote_threshold <= args.promote_threshold <= 1.0:
        parser.error("thresholds must satisfy 0 <= demote <= promote <= 1")
    return args, extras


def newest_checkpoint(directory: Path) -> Path:
    candidates = [child for child in directory.iterdir() if child.is_dir() and child.name.isdigit()]
    if not candidates:
        raise RuntimeError(f"no checkpoint produced in {directory}")
    return max(candidates, key=lambda child: int(child.name))


def trainer_int_arg(extras: list[str], option: str, default: int) -> int:
    """Read a scalar integer trainer option, honoring its last occurrence."""
    value = default
    for index, token in enumerate(extras):
        if token == option and index + 1 < len(extras):
            value = int(extras[index + 1])
        elif token.startswith(option + "="):
            value = int(token.split("=", 1)[1])
    return value


def replace_trainer_arg(extras: list[str], option: str, value: int) -> list[str]:
    """Return trainer args with one scalar option replaced."""
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


def main() -> None:
    args, extras = parse_args()
    trainer = Path(__file__).resolve().parent / "train_rough_terrain.py"
    terrain_root = args.run_root.expanduser().resolve() / "terrain"
    terrain_root.mkdir(parents=True, exist_ok=True)
    curriculum_dir = terrain_root / f"{args.run_name}_curriculum_state"
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    state_path = curriculum_dir / "curriculum_history.json"
    history: list[dict] = []
    level = args.start_level
    init_checkpoint = (
        args.init_checkpoint.expanduser().resolve()
        if args.init_checkpoint is not None
        else None
    )

    if init_checkpoint is None and args.flat_baseline_timesteps > 0:
        command_root = args.run_root.expanduser().resolve() / "command"
        command_root.mkdir(parents=True, exist_ok=True)
        baseline_prefix = f"{args.run_name}-flat-baseline"
        baseline_extras = extras
        batch_size = trainer_int_arg(extras, "--batch-size", 256)
        num_minibatches = trainer_int_arg(extras, "--num-minibatches", 8)
        unroll_length = trainer_int_arg(extras, "--unroll-length", 20)
        requested_evals = trainer_int_arg(extras, "--num-evals", 10)
        steps_per_update = batch_size * num_minibatches * unroll_length
        max_useful_evals = math.ceil(
            args.flat_baseline_timesteps / steps_per_update
        ) + 1
        if requested_evals > max_useful_evals:
            baseline_extras = replace_trainer_arg(
                extras, "--num-evals", max_useful_evals
            )
        before = set(command_root.iterdir())
        baseline_command = [
            sys.executable,
            str(trainer),
            "--task",
            "command",
            "--timesteps",
            str(args.flat_baseline_timesteps),
            "--seed",
            str(args.seed),
            "--run-name",
            baseline_prefix,
            "--run-root",
            str(args.run_root.expanduser().resolve()),
            "--wandb-group",
            args.run_name,
            "--wandb-project",
            args.wandb_project,
        ]
        if args.wandb:
            baseline_command.append("--wandb")
        baseline_command.extend(baseline_extras)
        print(
            f"flat_baseline timesteps={args.flat_baseline_timesteps} "
            f"init={init_checkpoint}"
        )
        subprocess.run(baseline_command, check=True)
        created = [
            path
            for path in set(command_root.iterdir()) - before
            if path.is_dir()
        ]
        matching = [
            path for path in created if path.name.startswith(baseline_prefix + "_")
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"could not uniquely resolve flat baseline {baseline_prefix!r}: {matching}"
            )
        baseline_run = matching[0]
        init_checkpoint = newest_checkpoint(baseline_run / "checkpoints")
        history.append(
            {
                "phase": "flat_baseline",
                "timesteps": args.flat_baseline_timesteps,
                "run_dir": str(baseline_run),
                "checkpoint": str(init_checkpoint),
                "best_video": str(
                    baseline_run / "videos" / "best_curriculum_full.gif"
                ),
                "progress_videos": str(baseline_run / "videos" / "progress"),
                "wandb_progress_video_key": "progress/video",
            }
        )
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    for stage in range(args.stages):
        prefix = f"{args.run_name}-stage{stage:02d}-level{level}"
        stage_timesteps = (
            args.stage_0_timesteps
            if stage == 0 and args.stage_0_timesteps is not None
            else args.stage_timesteps
        )
        stage_extras = extras
        if stage == 0 and args.stage_0_timesteps is not None:
            batch_size = trainer_int_arg(extras, "--batch-size", 256)
            num_minibatches = trainer_int_arg(extras, "--num-minibatches", 8)
            unroll_length = trainer_int_arg(extras, "--unroll-length", 32)
            requested_evals = trainer_int_arg(extras, "--num-evals", 10)
            steps_per_update = batch_size * num_minibatches * unroll_length
            max_useful_evals = math.ceil(stage_timesteps / steps_per_update) + 1
            if requested_evals > max_useful_evals:
                stage_extras = replace_trainer_arg(
                    extras, "--num-evals", max_useful_evals
                )
                print(
                    f"stage0_num_evals={requested_evals}->{max_useful_evals} "
                    f"to keep requested timesteps near {stage_timesteps}"
                )
        before = set(terrain_root.iterdir())
        command = [
            sys.executable,
            str(trainer),
            "--task",
            "terrain",
            "--terrain-layout",
            "mixed",
            "--terrain-level",
            str(level),
            "--competence-stage",
            str(stage),
            "--timesteps",
            str(stage_timesteps),
            "--seed",
            str(args.seed + stage),
            "--run-name",
            prefix,
            "--run-root",
            str(args.run_root.expanduser().resolve()),
            "--wandb-group",
            args.run_name,
            "--wandb-project",
            args.wandb_project,
        ]
        if args.wandb:
            command.append("--wandb")
        if init_checkpoint is not None:
            command.extend(("--init-checkpoint", str(init_checkpoint)))
            if stage > 0 or args.init_value_function:
                command.append("--init-value-function")
        command.extend(stage_extras)
        print(
            f"curriculum_stage={stage} level={level} "
            f"timesteps={stage_timesteps} init={init_checkpoint}"
        )
        subprocess.run(command, check=True)

        created = [path for path in set(terrain_root.iterdir()) - before if path.is_dir()]
        matching = [path for path in created if path.name.startswith(prefix + "_")]
        if len(matching) != 1:
            raise RuntimeError(f"could not uniquely resolve stage run for prefix {prefix!r}: {matching}")
        run_dir = matching[0]
        metrics_path = run_dir / "monitor" / "latest_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        success = float(metrics.get("eval/episode_terrain_success", 0.0))
        init_checkpoint = newest_checkpoint(run_dir / "checkpoints")
        if args.level_progression == "sequential":
            next_level = min(args.max_level, level + 1)
        else:
            next_level = level
            if success > args.promote_threshold:
                next_level = min(args.max_level, level + 1)
            elif success < args.demote_threshold:
                next_level = max(0, level - 1)
        record = {
            "phase": "terrain",
            "stage": stage,
            "level": level,
            "level_progression": args.level_progression,
            "success_rate": success,
            "next_level": next_level,
            "run_dir": str(run_dir),
            "checkpoint": str(init_checkpoint),
            "best_video": str(run_dir / "videos" / "best_policy.gif"),
            "wandb_video_key": f"best/video_stage{stage:02d}_level{level}",
            "progress_videos": str(run_dir / "videos" / "progress"),
            "wandb_progress_video_key": "progress/video",
        }
        history.append(record)
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)
        print(
            f"competence success={success:.3f} level={level}->{next_level} "
            f"state={state_path}"
        )
        level = next_level


if __name__ == "__main__":
    main()
