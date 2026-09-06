#!/usr/bin/env python3
"""Automatic curriculum launcher for adaptive 24-D v4 PPO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    gait_stage: int
    terrain_level: int
    name: str
    action_profile: str


FAST_CURRICULUM = (
    # --------------------------------------------------
    # Tripod: baseline을 절대 먼저 깨뜨리지 않는다.
    # --------------------------------------------------
    StageSpec(1, 0, "tripod-flat", "flat_safe"),

    StageSpec(1, 3, "tripod-ramp8", "terrain_mid"),
    StageSpec(1, 4, "tripod-ramp15", "terrain_mid"),

    StageSpec(1, 5, "tripod-stair5", "terrain_mid"),
    StageSpec(1, 6, "tripod-stair6p5", "terrain_high"),
    StageSpec(1, 7, "tripod-stair8", "terrain_high"),
    StageSpec(1, 8, "tripod-stair10", "full"),

    # --------------------------------------------------
    # Wave
    # --------------------------------------------------
    StageSpec(2, 5, "wave-stair5", "terrain_mid"),
    StageSpec(2, 6, "wave-stair6p5", "terrain_high"),
    StageSpec(2, 7, "wave-stair8", "terrain_high"),
    StageSpec(2, 8, "wave-stair10", "full"),

    # --------------------------------------------------
    # Hybrid
    # --------------------------------------------------
    StageSpec(3, 3, "hybrid-ramp8", "terrain_mid"),
    StageSpec(3, 4, "hybrid-ramp15", "terrain_high"),
    StageSpec(3, 5, "hybrid-stair5", "terrain_high"),
    StageSpec(3, 6, "hybrid-stair6p5", "terrain_high"),
    StageSpec(3, 7, "hybrid-stair8", "full"),
    StageSpec(3, 8, "hybrid-stair10", "full"),
)


FULL_CURRICULUM = FAST_CURRICULUM + (
    StageSpec(3, 9, 'hybrid-stair15'),
    StageSpec(3, 10, 'hybrid-stair20'),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(
        json.dumps(payload, indent=2) + '\n'
    )
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--profile',
        choices=('fast', 'full'),
        default='fast',
    )

    parser.add_argument(
        '--perception',
        choices=('teacher', 'lidar'),
        default='teacher',
    )

    parser.add_argument(
        '--run-name',
        default='adaptive-v4-curriculum',
    )

    parser.add_argument(
        '--run-root',
        type=Path,
        default=Path(__file__).resolve().parent
        / 'runs'
        / 'adaptive-curriculum',
    )

    parser.add_argument(
        '--timesteps-per-stage',
        type=int,
        default=10_000_000,
    )

    parser.add_argument(
        '--num-envs',
        type=int,
        default=64,
    )

    parser.add_argument(
        '--num-evals',
        type=int,
        default=10,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=40,
    )

    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--promote-key',
        default='eval/episode_terrain_success',
    )

    parser.add_argument(
        '--promote-threshold',
        type=float,
        default=.70,
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
    )

    parser.add_argument(
        '--restore',
        type=Path,
        default=None,
    )

    parser.add_argument(
        '--init-teacher',
        type=Path,
        default=None,
    )

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument(
        '--wandb-project',
        default='hexapod-adaptive-gait',
    )
    parser.add_argument(
        '--wandb-entity',
        default=None,
    )
    parser.add_argument(
        '--wandb-mode',
        choices=('online', 'offline', 'disabled'),
        default='online',
    )

    args = parser.parse_args()

    if not 0.0 <= args.promote_threshold <= 1.0:
        parser.error('--promote-threshold must be in [0, 1]')

    if args.max_retries < 0:
        parser.error('--max-retries cannot be negative')

    if args.restore and args.init_teacher:
        parser.error(
            'choose either --restore or --init-teacher'
        )

    if args.init_teacher and args.perception != 'lidar':
        parser.error(
            '--init-teacher requires --perception lidar'
        )

    curriculum = (
        FAST_CURRICULUM
        if args.profile == 'fast'
        else FULL_CURRICULUM
    )

    if not 0 <= args.start_index < len(curriculum):
        parser.error(
            f'--start-index must be 0..{len(curriculum)-1}'
        )

    root = (
        args.run_root.expanduser().resolve()
        / args.run_name
    )

    root.mkdir(parents=True, exist_ok=True)

    trainer = (
        Path(__file__).resolve().parent
        / 'train_adaptive_gait.py'
    )

    history_path = root / 'curriculum_history.json'
    final_path = root / 'final_checkpoint.json'

    history = []

    restore = (
        args.restore.expanduser().resolve()
        if args.restore
        else None
    )

    teacher_init = (
        args.init_teacher.expanduser().resolve()
        if args.init_teacher
        else None
    )

    for stage_index in range(
        args.start_index,
        len(curriculum),
    ):
        spec = curriculum[stage_index]

        success = 0.0
        selected_checkpoint = None

        for retry in range(args.max_retries + 1):
            run_dir = (
                root
                / (
                    f'{stage_index:02d}_'
                    f'{spec.name}_'
                    f'try{retry}'
                )
            )

            command = [
                sys.executable,
                str(trainer),

                '--stage',
                str(spec.gait_stage),

                '--terrain-level',
                str(spec.terrain_level),

                '--perception',
                args.perception,

                '--timesteps',
                str(args.timesteps_per_stage),

                '--num-envs',
                str(args.num_envs),

                '--num-evals',
                str(args.num_evals),

                '--seed',
                str(args.seed + stage_index * 100 + retry),

                '--curriculum-stage',
                str(stage_index),

                '--output',
                str(run_dir),

                '--wandb-project',
                args.wandb_project,

                '--wandb-group',
                args.run_name,

                '--wandb-mode',
                args.wandb_mode,

                '--wandb-name',
                f'{stage_index:02d}-{spec.name}-try{retry}',

                '--best-video',
                
                "--action-profile",
                spec.action_profile,
            ]

            if args.wandb:
                command.append('--wandb')

            if args.wandb_entity:
                command.extend(
                    [
                        '--wandb-entity',
                        args.wandb_entity,
                    ]
                )

            if (
                stage_index == args.start_index
                and retry == 0
                and teacher_init is not None
            ):
                command.extend(
                    [
                        '--init-teacher',
                        str(teacher_init),
                    ]
                )

            elif restore is not None:
                command.extend(
                    [
                        '--restore',
                        str(restore),
                    ]
                )

            print()
            print('=' * 72)
            print(
                f'CURRICULUM {stage_index}/{len(curriculum)-1}'
            )
            print(
                f'gait={spec.gait_stage} '
                f'terrain={spec.terrain_level} '
                f'name={spec.name} '
                f'retry={retry}'
            )
            print(f'restore={restore}')
            print('=' * 72)

            subprocess.run(
                command,
                check=True,
            )

            pointer_path = (
                run_dir
                / 'monitor'
                / 'best_checkpoint.json'
            )

            metrics_path = (
                run_dir
                / 'monitor'
                / 'best_metrics.json'
            )

            if not pointer_path.exists():
                raise RuntimeError(
                    f'best checkpoint missing: {pointer_path}'
                )

            if not metrics_path.exists():
                raise RuntimeError(
                    f'best metrics missing: {metrics_path}'
                )

            pointer = json.loads(
                pointer_path.read_text()
            )

            metrics = json.loads(
                metrics_path.read_text()
            )

            selected_checkpoint = Path(
                pointer['path']
            ).resolve()

            promote_value = float(
                metrics['metrics'].get(
                    args.promote_key,
                    0.0,
                )
            )

            success = promote_value

            video_path = (
                run_dir
                / 'videos'
                / 'best.gif'
            )

            record = {
                'curriculum_index': stage_index,
                'retry': retry,
                'gait_stage': spec.gait_stage,
                'terrain_level': spec.terrain_level,
                'name': spec.name,
                'run_dir': str(run_dir),
                'best_checkpoint': str(
                    selected_checkpoint
                ),
                'best_video': str(video_path),
                'best_score': pointer['score'],
                'promote_key': args.promote_key,
                'promote_value': promote_value,
                'threshold': args.promote_threshold,
            }

            history.append(record)

            write_json(
                history_path,
                history,
            )

            print(
                f'BEST SCORE = {pointer["score"]:.4f}'
            )
            print(
                f'{args.promote_key} = '
                f'{promote_value:.4f}'
            )
            print(
                f'BEST VIDEO = {video_path}'
            )

            restore = selected_checkpoint

            if promote_value >= args.promote_threshold:
                print(
                    f'PROMOTE -> next curriculum stage '
                    f'({promote_value:.3f} >= '
                    f'{args.promote_threshold:.3f})'
                )
                break

            print(
                f'NOT READY '
                f'({promote_value:.3f} < '
                f'{args.promote_threshold:.3f})'
            )

        else:
            raise RuntimeError(
                f'curriculum stage {stage_index} failed'
            )

        if success < args.promote_threshold:
            raise RuntimeError(
                f'{spec.name} failed after '
                f'{args.max_retries + 1} attempts'
            )

        write_json(
            final_path,
            {
                'curriculum_index': stage_index,
                'gait_stage': spec.gait_stage,
                'terrain_level': spec.terrain_level,
                'name': spec.name,
                'checkpoint': str(
                    selected_checkpoint
                ),
                'history': str(history_path),
            },
        )

    print()
    print('=' * 72)
    print('CURRICULUM COMPLETE')
    print(f'FINAL CHECKPOINT: {restore}')
    print(f'HISTORY: {history_path}')
    print('=' * 72)


if __name__ == '__main__':
    main()