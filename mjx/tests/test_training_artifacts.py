"""Tests for local best/progress artifact bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


MJX_DIR = Path(__file__).resolve().parents[1]
if str(MJX_DIR) not in sys.path:
    sys.path.insert(0, str(MJX_DIR))

from prepare_rl_scene import STAIR_TOTAL_RISE, STEP_COUNT, STEP_HEIGHT  # noqa: E402
from terrain_curriculum import (  # noqa: E402
    MAX_TERRAIN_LEVEL,
    ROUGH_COLUMNS,
    ROUGH_HFIELD_NCOL,
    ROUGH_HFIELD_NROW,
    ROUGH_ROWS,
    TERRAIN_LEVELS,
    rough_heightfield_grid,
)
from train_competence_curriculum import (  # noqa: E402
    _guard_teacher_attempt_limit,
    _level_transition,
    _max_stages_for_level,
    _replace_trainer_arg,
    _selected_checkpoint,
    _trainer_int_arg,
)
from train_rough_terrain import (  # noqa: E402
    ScoreMonitor,
    _training_schedule,
    essential_wandb_metrics,
    level_best_video_keys,
    progress_video_targets,
)


class TrainingArtifactTest(unittest.TestCase):
    def test_wandb_keeps_only_essential_metrics(self) -> None:
        metrics = {
            "eval/episode_reward": 12.0,
            "eval/gait_failure_rate": 0.01,
            "training/distill_v3_action_rmse": 0.2,
            "eval/episode_root_angular_speed_std": 99.0,
        }
        self.assertEqual(
            essential_wandb_metrics(metrics),
            {
                "eval/episode_reward": 12.0,
                "eval/gait_failure_rate": 0.01,
                "training/distill_v3_action_rmse": 0.2,
            },
        )

    def test_stage_caps_switch_from_two_to_four_at_steep_ramp(self) -> None:
        self.assertEqual(
            tuple(_max_stages_for_level(level) for level in range(4)),
            (2, 2, 2, 2),
        )
        self.assertEqual(
            tuple(_max_stages_for_level(level) for level in range(4, 13)),
            (4,) * 9,
        )
        self.assertEqual(_max_stages_for_level(0, override=3), 3)
        self.assertEqual(_max_stages_for_level(12, override=3), 3)

    def test_default_progress_targets_are_exact_quarters(self) -> None:
        self.assertEqual(progress_video_targets(5), (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_level_best_video_has_stable_and_level_specific_wandb_keys(self) -> None:
        self.assertEqual(
            level_best_video_keys(5),
            ("level/best_video", "level/best_video_level5"),
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            level_best_video_keys(-1)

    def test_untrained_step_zero_is_not_selected_as_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = ScoreMonitor(Path(directory), "eval/episode_reward")
            _, initial_best = monitor.record(0, {"eval/episode_reward": 10.0})
            _, trained_best = monitor.record(100, {"eval/episode_reward": 1.0})
            self.assertFalse(initial_best)
            self.assertTrue(trained_best)
            payload = json.loads(monitor.best_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["step"], 100)

    def test_best_safe_gate_rejects_high_policy_rejection_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = ScoreMonitor(
                Path(directory),
                "eval/episode_reward",
                max_policy_rejection_rate=0.01,
                max_foot_limited_rate=0.01,
                max_failure_rate=0.05,
            )
            _, unsafe = monitor.record(
                100,
                {
                    "eval/episode_reward": 100.0,
                    "eval/avg_episode_length": 1000.0,
                    "eval/episode_policy_rejection_fraction": 20.0,
                },
            )
            _, safe = monitor.record(
                200,
                {
                    "eval/episode_reward": 90.0,
                    "eval/avg_episode_length": 1000.0,
                    "eval/episode_policy_rejection_fraction": 1.0,
                },
            )
            self.assertFalse(unsafe)
            self.assertTrue(safe)
            payload = json.loads(monitor.best_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["step"], 200)
            self.assertTrue(payload["best_safe"])
            level_payload = json.loads(
                monitor.level_best_path.read_text(encoding="utf-8")
            )
            self.assertEqual(level_payload["step"], 100)
            self.assertFalse(level_payload["best_safe"])

    def test_curriculum_order_and_final_ten_by_twenty_centimeter_stairs(self) -> None:
        self.assertEqual(MAX_TERRAIN_LEVEL, 12)
        self.assertEqual(tuple(spec.level for spec in TERRAIN_LEVELS), tuple(range(13)))
        self.assertEqual(
            tuple(spec.kind for spec in TERRAIN_LEVELS[:5]),
            ("flat", "rough", "rough", "ramp", "ramp"),
        )
        self.assertAlmostEqual(TERRAIN_LEVELS[1].rough_amplitude, 0.025)
        self.assertAlmostEqual(TERRAIN_LEVELS[2].rough_amplitude, 0.050)
        self.assertAlmostEqual(TERRAIN_LEVELS[3].slope_degrees, 8.0)
        self.assertAlmostEqual(TERRAIN_LEVELS[4].slope_degrees, 15.0)
        self.assertTrue(all(spec.kind == "stairs" for spec in TERRAIN_LEVELS[5:]))
        seven_step = TERRAIN_LEVELS[5:9]
        self.assertEqual(tuple(spec.stair_count for spec in seven_step), (7,) * 4)
        self.assertEqual(
            tuple(spec.stair_riser for spec in seven_step),
            (0.05, 0.10, 0.15, 0.20),
        )
        for spec, expected in zip(seven_step, (0.35, 0.70, 1.05, 1.40)):
            self.assertAlmostEqual(spec.final_height, expected)
        ten_step = TERRAIN_LEVELS[9:13]
        self.assertEqual(tuple(spec.stair_count for spec in ten_step), (10,) * 4)
        self.assertEqual(
            tuple(spec.stair_riser for spec in ten_step),
            (0.05, 0.10, 0.15, 0.20),
        )
        final = TERRAIN_LEVELS[-1]
        self.assertEqual(final.stair_count, 10)
        self.assertAlmostEqual(final.stair_riser, 0.20)
        self.assertAlmostEqual(final.final_height, 2.0)
        self.assertEqual(STEP_COUNT, 10)
        self.assertAlmostEqual(STEP_HEIGHT, 0.20)
        self.assertAlmostEqual(STAIR_TOTAL_RISE, 2.0)

    def test_flat_eval_count_is_replaced_without_touching_other_args(self) -> None:
        extras = ["--num-envs", "2048", "--num-evals", "20", "--best-video"]
        replaced = _replace_trainer_arg(extras, "--num-evals", 5)
        self.assertEqual(_trainer_int_arg(replaced, "--num-evals", 10), 5)
        self.assertIn("--num-envs", replaced)
        self.assertIn("--best-video", replaced)

    def test_competence_retries_then_promotes_at_attempt_limit(self) -> None:
        retry = _level_transition(
            level=1,
            max_level=12,
            progression="competence",
            success=0.40,
            threshold=0.80,
            level_attempt=2,
            max_stages_per_level=3,
        )
        promote = _level_transition(
            level=1,
            max_level=12,
            progression="competence",
            success=0.40,
            threshold=0.80,
            level_attempt=3,
            max_stages_per_level=3,
        )
        self.assertEqual(retry, (1, False, "retry"))
        self.assertEqual(promote, (2, False, "attempt_limit"))

    def test_competence_can_promote_before_attempt_limit(self) -> None:
        transition = _level_transition(
            level=4,
            max_level=12,
            progression="competence",
            success=0.8125,
            threshold=0.80,
            level_attempt=1,
            max_stages_per_level=3,
        )
        self.assertEqual(transition, (5, False, "competence"))

    def test_attempt_limit_completes_final_level(self) -> None:
        transition = _level_transition(
            level=12,
            max_level=12,
            progression="competence",
            success=0.25,
            threshold=0.80,
            level_attempt=3,
            max_stages_per_level=3,
        )
        self.assertEqual(transition, (12, True, "attempt_limit"))

    def test_teacher_curriculum_stops_instead_of_force_promoting(self) -> None:
        args = SimpleNamespace(
            teacher_manifest=Path("teachers.json"),
            teacher_stop_on_attempt_limit=True,
            promote_threshold=0.80,
        )
        guarded = _guard_teacher_attempt_limit(
            args,
            level=4,
            success=0.50,
            next_level=5,
            completed=False,
            reason="attempt_limit",
        )
        self.assertEqual(guarded, (4, False, "attempt_limit_stop"))

    def test_best_checkpoint_selection_uses_monitor_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "checkpoints" / "000000000123"
            checkpoint.mkdir(parents=True)
            (checkpoint / "ppo_network_config.json").write_text("{}")
            monitor = run_dir / "monitor"
            monitor.mkdir()
            (monitor / "best_checkpoint.json").write_text(
                json.dumps({"path": str(checkpoint), "score": 1.0, "step": 123})
            )
            self.assertEqual(_selected_checkpoint(run_dir, "best"), checkpoint)

    def test_rough_heightfield_replaces_box_grid_without_losing_pattern(self) -> None:
        grid = rough_heightfield_grid(0.025)
        self.assertEqual(len(grid), ROUGH_HFIELD_NROW)
        self.assertTrue(all(len(row) == ROUGH_HFIELD_NCOL for row in grid))
        self.assertEqual(ROUGH_HFIELD_NROW, ROUGH_ROWS + 2)
        self.assertEqual(ROUGH_HFIELD_NCOL, ROUGH_COLUMNS + 2)
        self.assertEqual(grid[0], grid[1])
        self.assertEqual(grid[-1], grid[-2])
        self.assertTrue(all(row[0] == row[1] for row in grid))
        self.assertTrue(all(row[-1] == row[-2] for row in grid))
        self.assertAlmostEqual(max(max(row) for row in grid), 0.025)

    def test_num_envs_changes_rollout_parallelism_not_sample_budget(self) -> None:
        common = {
            "batch_size": 256,
            "num_minibatches": 8,
            "unroll_length": 32,
            "num_evals": 20,
            "timesteps": 4_672_320,
            "num_eval_envs": 32,
            "episode_length": 2500,
        }
        large = _training_schedule(SimpleNamespace(**common, num_envs=2048))
        small = _training_schedule(SimpleNamespace(**common, num_envs=1024))
        self.assertEqual(large["actual_timesteps"], small["actual_timesteps"])
        self.assertEqual(large["rollout_batches_per_training_step"], 1)
        self.assertEqual(small["rollout_batches_per_training_step"], 2)


if __name__ == "__main__":
    unittest.main()
