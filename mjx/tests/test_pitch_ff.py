from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import jax.numpy as jp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import firmware_mjx_controller as firmware
from rough_terrain_env import (
    PITCH_FF_MAX_RAD,
    STAIR_SWING_CLEARANCE_M,
    _coupled_swing_height_floor,
    _pitch_ff,
)


class PitchFeedforwardTest(unittest.TestCase):
    def _converged(self, height: float, support: float = 0.0) -> float:
        feedforward = jp.zeros(())
        heights = jp.full((9,), height)
        for _ in range(500):
            feedforward = _pitch_ff(
                heights, jp.asarray(support), feedforward, 0.02
            )
        return float(feedforward)

    def test_approved_vectors_use_relative_rise_and_uphill_negative_sign(self) -> None:
        self.assertAlmostEqual(self._converged(0.0), 0.0, places=6)
        self.assertAlmostEqual(self._converged(0.10), -0.15264, delta=2.0e-4)
        self.assertAlmostEqual(self._converged(0.30), -0.43256, delta=2.0e-4)
        self.assertAlmostEqual(self._converged(0.50), -PITCH_FF_MAX_RAD, places=5)
        self.assertAlmostEqual(
            self._converged(1.10, support=1.0), -0.15264, delta=2.0e-4
        )

    def test_nan_is_finite_and_deadband_is_zero(self) -> None:
        nan_ff = _pitch_ff(jp.full((9,), jp.nan), 0.0, jp.nan, 0.02)
        self.assertTrue(math.isfinite(float(nan_ff)))
        self.assertEqual(float(nan_ff), 0.0)
        self.assertEqual(self._converged(0.0049), 0.0)

    def test_filter_converges_monotonically(self) -> None:
        feedforward = jp.zeros(())
        values = []
        for _ in range(20):
            feedforward = _pitch_ff(
                jp.full((9,), 0.30), 0.0, feedforward, 0.02
            )
            values.append(float(feedforward))
        self.assertTrue(
            all(left > right for left, right in zip(values, values[1:]))
        )
        self.assertGreater(values[-1], -0.43256)

    def test_requires_nine_forward_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly nine"):
            _pitch_ff(jp.zeros(8), 0.0, 0.0, 0.02)

    def test_controller_pitch_error_has_required_sign(self) -> None:
        state = firmware.initial_state()._replace(first_step=jp.asarray(False))
        next_state, _ = firmware.step(
            state,
            target_velocity=jp.zeros(2),
            body_position_world=jp.asarray((0.0, 0.0, 0.316)),
            attitude_rpy=jp.deg2rad(jp.asarray((0.0, 10.0, 0.0))),
            contacts=jp.ones(6, dtype=jp.bool_),
            policy_action=jp.zeros(18),
            pitch_ff=jp.deg2rad(-20.0),
        )
        expected_step = -jp.deg2rad(15.0) * firmware.FIRMWARE_CONTROL_DT
        self.assertAlmostEqual(
            float(next_state.posture_command[1]), float(expected_step), places=7
        )

    def test_level6_pitch_and_swing_floor_rise_together(self) -> None:
        target = -jp.arctan2(jp.asarray(0.10), jp.asarray(0.25))
        half_pitch = 0.5 * target
        half_floor = _coupled_swing_height_floor(half_pitch, target, 0.10)
        full_floor = _coupled_swing_height_floor(target, target, 0.10)

        expected_full = 0.10 + STAIR_SWING_CLEARANCE_M
        self.assertAlmostEqual(float(full_floor), expected_full, places=7)
        self.assertAlmostEqual(
            float(half_floor),
            firmware.SWING_HEIGHT_MIN
            + 0.5 * (expected_full - firmware.SWING_HEIGHT_MIN),
            places=7,
        )

    def test_disabled_stair_assist_keeps_nominal_swing_height(self) -> None:
        floor = _coupled_swing_height_floor(
            jp.asarray(-0.2), jp.asarray(0.0), 0.10
        )
        self.assertAlmostEqual(float(floor), firmware.SWING_HEIGHT_MIN, places=7)

    def test_uphill_target_strengthens_but_does_not_exceed_pitch_cap(self) -> None:
        target = -jp.arctan2(jp.asarray(0.10), jp.asarray(0.25))
        feedforward = jp.zeros(())
        for _ in range(500):
            feedforward = _pitch_ff(
                jp.full((9,), 0.10),
                jp.asarray(0.0),
                feedforward,
                0.02,
                target,
            )
        self.assertLess(float(feedforward), -0.20)
        self.assertGreaterEqual(float(feedforward), -PITCH_FF_MAX_RAD)


if __name__ == "__main__":
    unittest.main()
