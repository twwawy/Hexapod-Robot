"""Tests for local best/progress artifact bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


MJX_DIR = Path(__file__).resolve().parents[1]
if str(MJX_DIR) not in sys.path:
    sys.path.insert(0, str(MJX_DIR))

from prepare_rl_scene import STAIR_TOTAL_RISE, STEP_COUNT  # noqa: E402
from rough_terrain_env import (  # noqa: E402
    CURRICULUM_STEP_HEIGHTS,
    CURRICULUM_TOTAL_RISES,
)
from train_rough_terrain import ScoreMonitor, progress_video_targets  # noqa: E402


class TrainingArtifactTest(unittest.TestCase):
    def test_default_progress_targets_are_exact_quarters(self) -> None:
        self.assertEqual(progress_video_targets(5), (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_untrained_step_zero_is_not_selected_as_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = ScoreMonitor(Path(directory), "eval/episode_reward")
            _, initial_best = monitor.record(0, {"eval/episode_reward": 10.0})
            _, trained_best = monitor.record(100, {"eval/episode_reward": 1.0})
            self.assertFalse(initial_best)
            self.assertTrue(trained_best)
            payload = json.loads(monitor.best_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["step"], 100)

    def test_curriculum_preserves_fixed_tensor_contract_across_heights(self) -> None:
        self.assertEqual(CURRICULUM_TOTAL_RISES, (0.0, 0.05, 0.10, 0.15, 0.20))
        self.assertEqual(CURRICULUM_TOTAL_RISES[-1], STAIR_TOTAL_RISE)
        for step_height, total_rise in zip(
            CURRICULUM_STEP_HEIGHTS, CURRICULUM_TOTAL_RISES
        ):
            self.assertAlmostEqual(step_height * STEP_COUNT, total_rise)


if __name__ == "__main__":
    unittest.main()
