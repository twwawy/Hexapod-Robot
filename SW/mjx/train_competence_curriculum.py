#!/usr/bin/env python3
"""Competence-based mixed-terrain curriculum launcher.

Each stage trains on one fixed, reproducible patch distribution.  Evaluation
success promotes/demotes the next distribution, and the compatible 22-D/110-D
policy checkpoint initializes the next stage.  This avoids changing action
semantics or rebuilding XML inside a compiled PPO stage.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--start-level", type=int, choices=range(5), default=0)
    parser.add_argument("--max-level", type=int, choices=range(5), default=4)
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
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Additional train_rough_terrain.py arguments after '--'.",
    )
    args = parser.parse_args()
    extras = args.trainer_args
    if extras and extras[0] == "--":
        extras = extras[1:]
    if args.stages < 1 or args.stage_timesteps < 1:
        parser.error("--stages and --stage-timesteps must be positive")
    if not 0.0 <= args.demote_threshold <= args.promote_threshold <= 1.0:
        parser.error("thresholds must satisfy 0 <= demote <= promote <= 1")
    return args, extras


def newest_checkpoint(directory: Path) -> Path:
    candidates = [child for child in directory.iterdir() if child.is_dir() and child.name.isdigit()]
    if not candidates:
        raise RuntimeError(f"no checkpoint produced in {directory}")
    return max(candidates, key=lambda child: int(child.name))


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

    for stage in range(args.stages):
        prefix = f"{args.run_name}-stage{stage:02d}-level{level}"
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
            "--timesteps",
            str(args.stage_timesteps),
            "--seed",
            str(args.seed + stage),
            "--run-name",
            prefix,
            "--run-root",
            str(args.run_root.expanduser().resolve()),
            "--wandb-group",
            args.run_name,
        ]
        if args.wandb:
            command.append("--wandb")
        if init_checkpoint is not None:
            command.extend(("--init-checkpoint", str(init_checkpoint)))
            if stage > 0 or args.init_value_function:
                command.append("--init-value-function")
        command.extend(extras)
        print(f"curriculum_stage={stage} level={level} init={init_checkpoint}")
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
        next_level = level
        if success > args.promote_threshold:
            next_level = min(args.max_level, level + 1)
        elif success < args.demote_threshold:
            next_level = max(0, level - 1)
        record = {
            "stage": stage,
            "level": level,
            "success_rate": success,
            "next_level": next_level,
            "run_dir": str(run_dir),
            "checkpoint": str(init_checkpoint),
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
