#!/usr/bin/env python3
"""Automatic curriculum manager for adaptive 24-D v4 PPO."""

from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import dataclass
import json
import itertools
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    gait_stage: int
    terrain_level: int
    name: str


# 빠르게 실전적으로 올라가는 curriculum
FAST_CURRICULUM = (
    # Tripod
    StageSpec(1, 0, "tripod-flat"),
    StageSpec(1, 1, "tripod-rough25"),
    StageSpec(1, 2, "tripod-rough50"),
    StageSpec(1, 3, "tripod-ramp8"),
    StageSpec(1, 4, "tripod-ramp15"),
    StageSpec(1, 5, "tripod-stair5"),
    StageSpec(1, 7, "tripod-stair8"),
    StageSpec(1, 8, "tripod-stair10"),

    # Wave
    StageSpec(2, 5, "wave-stair5"),
    StageSpec(2, 7, "wave-stair8"),
    StageSpec(2, 8, "wave-stair10"),

    # Hybrid
    StageSpec(3, 2, "hybrid-rough50"),
    StageSpec(3, 4, "hybrid-ramp15"),
    StageSpec(3, 5, "hybrid-stair5"),
    StageSpec(3, 7, "hybrid-stair8"),
    StageSpec(3, 8, "hybrid-stair10"),
)


FULL_CURRICULUM = FAST_CURRICULUM + (
    StageSpec(3, 9, "hybrid-stair15"),
    StageSpec(3, 10, "hybrid-stair20"),
)

# Learn smooth terrain adaptation before sparse/rubble footholds; source is GT.
TEACHER_CURRICULUM = tuple(StageSpec(1, level, name) for level, name in (
    (0, 'tripod-flat'), (3, 'tripod-ramp8'), (5, 'tripod-stair5'),
    (7, 'tripod-stair8'), (1, 'tripod-rough25'), (4, 'tripod-ramp15'),
    (8, 'tripod-stair10'), (2, 'tripod-rough50'))) + tuple(
        stage for stage in FULL_CURRICULUM if stage.gait_stage != 1)

# Free steering must first be learned without a finite one-way stair course.
RC_CURRICULUM = (StageSpec(1, 0, 'rc-tripod-flat'), StageSpec(2, 0, 'rc-wave-flat'),
                 StageSpec(3, 0, 'rc-hybrid-flat'))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrate-flat-boxes', action='store_true',
                        help='Reviewed flat checkpoint transfer on the first cycle only; requires --restore.')
    parser.add_argument('--migrate-completion-reward', action='store_true',
                        help='Transfer reviewed grid v5 checkpoint to completion reward on first cycle only.')
    parser.add_argument('--stair-clearance-extra', type=float, default=0.)
    parser.add_argument('--migrate-path-v6', action='store_true', help='Reviewed v5->v6 migration on the first cycle only.')
    parser.add_argument('--migrate-recontact', action='store_true',
                        help='Transfer reviewed completion-reward weights to boundary recontact controller.')

    parser.add_argument(
        "--profile",
        choices=("observe", "fast", "full", "teacher", "rc"),
        default="teacher",
    )

    parser.add_argument(
        "--perception",
        choices=("teacher", "lidar"),
        default="teacher",
    )
    parser.add_argument('--cycles', type=int, default=5,
                        help='Number of flat Tripod cycles for --profile observe; no promotion gate.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--num-eval-envs', type=int, default=4)
    parser.add_argument('--discounting', type=float, default=.997)
    parser.add_argument('--command-mode', choices=('terrain', 'rc'), default='terrain')
    parser.add_argument('--baseline-comparison-seconds', type=float, default=None,
                        help='Cycle-end paired check; default 20s for teacher profile, otherwise disabled.')

    parser.add_argument(
        "--run-name",
        default="adaptive-v4-curriculum",
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "adaptive-curriculum",
    )

    parser.add_argument(
        "--timesteps-per-stage",
        type=int,
        default=10_000_000,
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-evals",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--episode-length",
        type=int,
        default=2000,
        help="Policy steps per cycle episode; 800 is a compact 16-second observation window.",
    )

    parser.add_argument(
        "--action-profile",
        choices=("flat_safe", "terrain_mid", "terrain_high", "full"),
        default="full",
    )

    parser.add_argument(
        "--best-video-duration",
        type=float,
        default=40.0,
        help="Seconds rendered for each completed cycle's best-score video.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--promote-key",
        default="eval/episode_terrain_success",
    )

    parser.add_argument(
        "--promote-threshold",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=-1,
        help='Retries per terrain; -1 retries until the promotion criterion passes.',
    )
    parser.add_argument('--on-stage-failure', choices=('stop', 'advance'), default='stop',
                        help='After retries: stop, or advance with the final attempt best without claiming success.')

    parser.add_argument(
        "--restore",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--init-teacher",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload every cycle's evaluation, best score, video and artifact to W&B.",
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
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )

    args = parser.parse_args()
    if args.migrate_flat_boxes and not args.restore:
        parser.error('--migrate-flat-boxes requires --restore')

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    if args.timesteps_per_stage < 1:
        parser.error("--timesteps-per-stage must be positive")

    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if min(args.cycles, args.batch_size, args.num_minibatches, args.num_eval_envs) < 1:
        parser.error('cycle/batch/evaluation counts must be positive')
    if (args.batch_size * args.num_minibatches) % args.num_envs:
        parser.error('batch-size * num-minibatches must be divisible by num-envs')

    if args.num_evals < 1:
        parser.error("--num-evals must be positive")

    if args.episode_length < 1:
        parser.error("--episode-length must be positive")

    if args.best_video_duration <= 0:
        parser.error("--best-video-duration must be positive")

    if not 0.0 <= args.promote_threshold <= 1.0:
        parser.error("--promote-threshold must be in [0, 1]")

    if args.max_retries < -1:
        parser.error('--max-retries must be -1 (unlimited) or nonnegative')
    if args.max_retries == -1 and args.on_stage_failure == 'advance':
        parser.error('Unlimited retries requires --on-stage-failure stop; it advances only on success')
    if args.migrate_path_v6 and (not args.restore or args.init_teacher or args.profile == 'rc' or args.command_mode != 'terrain' or args.migrate_recontact or args.migrate_completion_reward or args.migrate_flat_boxes):
        parser.error('--migrate-path-v6 requires terrain --restore and excludes other migrations')
    if args.migrate_recontact and (not args.restore or args.migrate_completion_reward or args.migrate_flat_boxes):
        parser.error('--migrate-recontact requires --restore and excludes other migrations')

    if args.migrate_completion_reward and (not args.restore or args.migrate_flat_boxes):
        parser.error('--migrate-completion-reward requires --restore and excludes --migrate-flat-boxes')
    if not 0. < args.discounting < 1.:
        parser.error('--discounting must be between 0 and 1')
    if args.restore and args.init_teacher:
        parser.error("choose either --restore or --init-teacher")
    if args.profile == 'rc':
        args.command_mode = 'rc'
    if args.command_mode == 'rc' and args.profile != 'rc':
        parser.error('Use --profile rc for free steering; terrain completion is a different task')
    if args.command_mode == 'rc' and args.promote_key == 'eval/episode_terrain_success':
        args.promote_key = 'eval/episode_command_success'

    if args.init_teacher and args.perception != "lidar":
        parser.error("--init-teacher requires --perception lidar")

    curriculum = (
        FAST_CURRICULUM
        if args.profile == "fast"
        else FULL_CURRICULUM
    )
    if args.profile == 'observe':
        curriculum = tuple(StageSpec(1, 0, 'tripod-flat') for _ in range(args.cycles))
    if args.profile == 'rc':
        curriculum = RC_CURRICULUM
    if args.profile == 'teacher':
        if args.perception != 'teacher':
            parser.error('--profile teacher requires --perception teacher')
        curriculum = TEACHER_CURRICULUM
    comparison_seconds = args.baseline_comparison_seconds
    if comparison_seconds is None:
        comparison_seconds = 20. if args.profile == 'teacher' else 0.
    import math
    if not math.isfinite(comparison_seconds) or comparison_seconds < 0:
        parser.error('--baseline-comparison-seconds must be finite and nonnegative')
    # Validate all future stages before launching even the first training job.
    from terrain_curriculum import terrain_level
    for stage in curriculum:
        spec = terrain_level(stage.terrain_level)
        if spec.kind not in {'flat', 'rough', 'ramp', 'stairs'} or stage.gait_stage not in (1, 2, 3):
            parser.error(f'Unsupported adaptive curriculum stage: {stage}')

    if not 0 <= args.start_index < len(curriculum):
        parser.error(
            f"--start-index must be 0..{len(curriculum) - 1}"
        )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    root = (
        args.run_root.expanduser().resolve()
        / args.run_name
    )

    # Repeating the same command starts a new experiment without overwriting or
    # accidentally mixing scores/checkpoints from a previous interrupted run.
    if root.exists():
        root = root.with_name(root.name + datetime.now().strftime('-%Y%m%d-%H%M%S-%f'))
    root.mkdir(parents=True, exist_ok=False)
    args.run_name = root.name
    print(f'Experiment: {root}', flush=True)

    trainer = (
        Path(__file__).resolve().parent
        / "train_adaptive_gait.py"
    )

    if not trainer.is_file():
        raise SystemExit(
            f"trainer not found: {trainer}"
        )

    history_path = root / "curriculum_history.json"
    state_path = root / "curriculum_state.json"
    final_path = root / "final_checkpoint.json"

    history: list[dict[str, Any]] = []

    if history_path.exists():
        try:
            history = read_json(history_path)
        except Exception:
            history = []

    restore = (
        args.restore.expanduser().resolve()
        if args.restore is not None
        else None
    )

    teacher_init = (
        args.init_teacher.expanduser().resolve()
        if args.init_teacher is not None
        else None
    )

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    for stage_index in range(
        args.start_index,
        len(curriculum),
    ):
        spec = curriculum[stage_index]

        print()
        print("=" * 80)
        print(
            f"CURRICULUM STAGE {stage_index}/{len(curriculum) - 1}"
        )
        print(
            f"name={spec.name} | "
            f"gait_stage={spec.gait_stage} | "
            f"terrain_level={spec.terrain_level}"
        )
        print("=" * 80)

        stage_passed = False
        selected_checkpoint: Path | None = None
        final_promote_value = 0.0

        retries = itertools.count() if args.max_retries == -1 else range(args.max_retries+1)
        for retry in retries:
            run_dir = (
                root
                / (
                    f"{stage_index:02d}_"
                    f"{spec.name}_"
                    f"try{retry:02d}"
                )
            )

            if run_dir.exists():
                raise RuntimeError(
                    f"stage output already exists: {run_dir}\n"
                    "Use a different --run-name or remove the old smoke run."
                )

            stage_seed = args.seed if args.profile == 'observe' else (
                args.seed
                + stage_index * 100
                + retry
            )

            command = [
                sys.executable,
                "-u",
                str(trainer),

                "--stage",
                str(spec.gait_stage),

                "--terrain-level",
                str(spec.terrain_level),

                "--perception",
                args.perception,

                "--timesteps",
                str(args.timesteps_per_stage),

                "--num-envs",
                str(args.num_envs),
                '--batch-size', str(args.batch_size),
                '--num-minibatches', str(args.num_minibatches),
                '--num-eval-envs', str(args.num_eval_envs),

                "--num-evals",
                str(args.num_evals),

                "--episode-length",
                str(args.episode_length),

                "--action-profile",
                args.action_profile,

                "--best-video-duration",
                str(args.best_video_duration),

                "--seed",
                str(stage_seed),

                "--curriculum-stage",
                str(stage_index),

                "--output",
                str(run_dir),

                "--wandb-project",
                args.wandb_project,

                "--wandb-group",
                args.run_name,

                "--wandb-name",
                (
                    f"{stage_index:02d}-"
                    f"{spec.name}-"
                    f"try{retry:02d}"
                ),

                "--wandb-mode",
                args.wandb_mode,

                "--best-video",
            ]

            command.append('--wandb' if args.wandb else '--no-wandb')
            command.extend(['--baseline-comparison-seconds', str(comparison_seconds)])
            command.extend(['--discounting', str(args.discounting)])
            command.extend(['--command-mode', args.command_mode])
            command.extend(['--stair-clearance-extra', str(args.stair_clearance_extra)])
            if args.migrate_flat_boxes and stage_index == args.start_index and retry == 0:
                command.append('--migrate-flat-boxes')
            if args.migrate_completion_reward and stage_index == args.start_index and retry == 0:
                command.append('--migrate-completion-reward')
            if args.migrate_path_v6 and stage_index == args.start_index and retry == 0:
                command.append('--migrate-path-v6')
            if args.migrate_recontact and stage_index == args.start_index and retry == 0:
                command.append('--migrate-recontact')

            if args.wandb_entity:
                command.extend(
                    [
                        "--wandb-entity",
                        args.wandb_entity,
                    ]
                )

            # 첫 stage에서 LiDAR student를 teacher로 초기화하는 경우
            if (
                stage_index == args.start_index
                and retry == 0
                and teacher_init is not None
            ):
                command.extend(
                    [
                        "--init-teacher",
                        str(teacher_init),
                    ]
                )

            # 그 외에는 직전 best checkpoint를 이어받음
            elif restore is not None:
                command.extend(
                    [
                        "--restore",
                        str(restore),
                    ]
                )

            print()
            print("-" * 80)
            print(
                f"LAUNCH | stage={stage_index} "
                f"retry={retry} "
                f"seed={stage_seed}"
            )
            print(
                f"restore={restore}"
            )
            print("-" * 80)
            print(
                " ".join(command),
                flush=True,
            )

            write_json(root / 'active_cycle.json', dict(stage=stage_index, retry=retry,
                        command=command, output=str(run_dir)))
            # Preserve the exact error even when the terminal scrollback is lost.
            with (root / f'cycle-{stage_index:02d}-try{retry:02d}.log').open('w') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    stopped = threading.Event()
                    def heartbeat():
                        while not stopped.wait(30.):
                            if process.poll() is not None:
                                return
                            print(f'WORKER WAIT | pid={process.pid} still running; waiting for next trainer output (not a progress guarantee)', flush=True)
                    threading.Thread(target=heartbeat, daemon=True).start()
                    try:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                            log.flush()
                        returncode = process.wait()
                    except KeyboardInterrupt:
                        process.terminate()
                        try:
                            process.wait(timeout=10.)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                    finally:
                        stopped.set()

                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)

            # ----------------------------------------------------------
            # Trainer output
            # ----------------------------------------------------------

            pointer_path = (
                run_dir
                / "monitor"
                / "best_checkpoint.json"
            )

            metrics_path = (
                run_dir
                / "monitor"
                / "best_metrics.json"
            )

            video_path = (
                run_dir
                / "videos"
                / "best.gif"
            )

            if not pointer_path.is_file():
                raise RuntimeError(
                    "train_adaptive_gait.py did not produce:\n"
                    f"  {pointer_path}"
                )

            if not metrics_path.is_file():
                raise RuntimeError(
                    "train_adaptive_gait.py did not produce:\n"
                    f"  {metrics_path}"
                )

            if not video_path.is_file():
                raise RuntimeError(
                    "train_adaptive_gait.py did not produce the required "
                    f"cycle best-score video:\n  {video_path}"
                )

            pointer = read_json(pointer_path)
            metrics = read_json(metrics_path)

            checkpoint_path = Path(
                pointer["path"]
            ).expanduser().resolve()

            if not checkpoint_path.is_dir():
                raise RuntimeError(
                    f"best checkpoint does not exist: {checkpoint_path}"
                )

            score = float(
                pointer["score"]
            )

            metric_table = metrics.get(
                "metrics",
                {},
            )

            promote_value = float(
                metric_table.get(
                    args.promote_key,
                    0.0,
                )
            )

            final_promote_value = promote_value
            selected_checkpoint = checkpoint_path

            record = {
                "curriculum_index": stage_index,
                "retry": retry,
                "name": spec.name,
                "gait_stage": spec.gait_stage,
                "terrain_level": spec.terrain_level,
                "seed": stage_seed,
                "run_dir": str(run_dir),
                "best_checkpoint": str(
                    checkpoint_path
                ),
                "best_video": str(video_path),
                "best_score": score,
                "score_key": pointer.get(
                    "score_key"
                ),
                "promote_key": args.promote_key,
                "promote_value": promote_value,
                "promote_threshold": (
                    args.promote_threshold
                ),
            }

            history.append(record)

            write_json(
                history_path,
                history,
            )

            # 중요:
            # retry 시에도 이번 attempt의 best를 이어서 학습한다.
            restore = checkpoint_path

            print()
            print(
                f"BEST SCORE      = {score:.6f}"
            )
            print(
                f"{args.promote_key} = "
                f"{promote_value:.6f}"
            )
            print(
                f"BEST CHECKPOINT = {checkpoint_path}"
            )
            print(
                f"BEST VIDEO      = "
                f"{video_path}"
            )

            if (
                args.profile == 'observe' or promote_value
                >= args.promote_threshold
            ):
                stage_passed = True

                if args.profile == 'observe':
                    print('CYCLE COMPLETE: next flat cycle restores this cycle best checkpoint.')
                else:
                    print(f'PROMOTE ({promote_value:.4f} >= {args.promote_threshold:.4f})')

                break

            print(
                "RETRY REQUIRED "
                f"({promote_value:.4f} < "
                f"{args.promote_threshold:.4f})"
            )

        # --------------------------------------------------------------
        # Stage result
        # --------------------------------------------------------------

        if not stage_passed:
            write_json(root / 'failed_stage.json', {
                'stage_index': stage_index, 'name': spec.name,
                'attempts': retry + 1,
                'promote_key': args.promote_key, 'promote_value': final_promote_value,
                'threshold': args.promote_threshold,
                'checkpoint': str(selected_checkpoint),
                'next_stage_index': stage_index + 1,
                'on_stage_failure': args.on_stage_failure,
            })
            message = (
                f"Curriculum stage failed: {spec.name}\n"
                f"best {args.promote_key}="
                f"{final_promote_value:.4f} "
                f"< threshold={args.promote_threshold:.4f}\n"
                f"attempts={retry + 1}"
            )
            if args.on_stage_failure == 'stop':
                raise RuntimeError(message + '\nCheckpoint and resume indices saved in failed_stage.json')
            print(message + '\nADVANCE requested: promotion criterion NOT met; using final attempt best.', flush=True)

        assert selected_checkpoint is not None

        write_json(
            state_path,
            {
                "completed_stage_index": stage_index,
                "completed_name": spec.name,
                'promotion_passed': stage_passed,
                'advanced_without_passing': not stage_passed,
                "gait_stage": spec.gait_stage,
                "terrain_level": spec.terrain_level,
                "checkpoint": str(
                    selected_checkpoint
                ),
                "next_stage_index": (
                    stage_index + 1
                ),
                "promote_key": args.promote_key,
                "promote_value": (
                    final_promote_value
                ),
                "history": str(
                    history_path
                ),
            },
        )

    # ------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------

    assert restore is not None

    write_json(
        final_path,
        {
            "profile": args.profile,
            "perception": args.perception,
            "checkpoint": str(restore),
            "history": str(history_path),
            "state": str(state_path),
            "stages": len(curriculum),
        },
    )

    print()
    print("=" * 80)
    print("CURRICULUM COMPLETE")
    print(f"FINAL CHECKPOINT: {restore}")
    print(f"HISTORY         : {history_path}")
    print(f"FINAL MANIFEST  : {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
