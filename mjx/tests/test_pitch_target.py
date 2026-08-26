from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import firmware_mjx_controller as firmware
from rough_terrain_env import PITCH_TARGET_MAX_RAD, _pitch_target


class PitchTargetTest(unittest.TestCase):
    def _converged(self, height: float, support: float = 0.0) -> float:
        target = jp.zeros(())
        heights = jp.full((9,), height)
        for _ in range(500):
            target = _pitch_target(heights, jp.asarray(support), target, 0.02)
        return float(target)

    def test_specification_vectors_use_relative_rise(self) -> None:
        self.assertAlmostEqual(self._converged(0.0), 0.0, places=6)
        self.assertAlmostEqual(self._converged(0.10), 0.15264, delta=2.0e-4)
        self.assertAlmostEqual(self._converged(0.30), 0.43256, delta=2.0e-4)
        self.assertAlmostEqual(self._converged(0.50), PITCH_TARGET_MAX_RAD, places=5)
        self.assertAlmostEqual(
            self._converged(1.10, support=1.0), 0.15264, delta=2.0e-4
        )

    def test_nan_is_finite_and_deadband_is_zero(self) -> None:
        nan_target = _pitch_target(jp.full((9,), jp.nan), 0.0, jp.nan, 0.02)
        self.assertTrue(math.isfinite(float(nan_target)))
        self.assertEqual(float(nan_target), 0.0)
        self.assertEqual(self._converged(0.0049), 0.0)

    def test_filter_converges_monotonically(self) -> None:
        target = jp.zeros(())
        values = []
        for _ in range(20):
            target = _pitch_target(jp.full((9,), 0.30), 0.0, target, 0.02)
            values.append(float(target))
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        self.assertLess(values[-1], 0.43256)

    def test_requires_nine_forward_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly nine"):
            _pitch_target(jp.zeros(8), 0.0, 0.0, 0.02)

    def test_controller_pitch_error_has_required_sign(self) -> None:
        state = firmware.initial_state()._replace(first_step=jp.asarray(False))
        next_state, _ = firmware.step(
            state,
            target_velocity=jp.zeros(2),
            body_position_world=jp.asarray((0.0, 0.0, 0.316)),
            attitude_rpy=jp.deg2rad(jp.asarray((0.0, 10.0, 0.0))),
            contacts=jp.ones(6, dtype=jp.bool_),
            policy_action=jp.zeros(18),
            pitch_target=jp.deg2rad(-20.0),
        )
        expected_step = -jp.deg2rad(15.0) * firmware.FIRMWARE_CONTROL_DT
        self.assertAlmostEqual(
            float(next_state.posture_command[1]), float(expected_step), places=7
        )


if __name__ == "__main__":
    unittest.main()
