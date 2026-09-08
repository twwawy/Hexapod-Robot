from __future__ import annotations

from pathlib import Path
import sys
import unittest

import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import firmware_mjx_controller as firmware
from rough_terrain_env import SWING_BOOST_MAX_M, _swing_boost


class SwingBoostTest(unittest.TestCase):
    def test_terrain_boost_vectors(self) -> None:
        self.assertEqual(float(_swing_boost(jp.zeros(9), 0.0)), 0.0)
        heights = jp.asarray((0.0, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(
            float(_swing_boost(heights, 0.0)), SWING_BOOST_MAX_M, places=7
        )
        self.assertAlmostEqual(
            float(_swing_boost(heights + 1.0, 1.0)), SWING_BOOST_MAX_M, places=7
        )

    def test_requires_nine_forward_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly nine"):
            _swing_boost(jp.zeros(8), 0.0)

    def test_swing_boost_is_phase_gated_and_hard_clipped(self) -> None:
        action = jp.zeros(18)
        swing = jp.full(6, firmware.LEG_SWING)
        endpoint, _ = firmware._phase_gated_policy_residual(
            action,
            swing,
            jp.asarray((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)),
            SWING_BOOST_MAX_M,
        )
        np.testing.assert_allclose(np.asarray(endpoint[:, 2]), 0.0, atol=1.0e-7)

        midpoint, _ = firmware._phase_gated_policy_residual(
            action, swing, jp.full(6, 0.5), SWING_BOOST_MAX_M
        )
        np.testing.assert_allclose(
            np.asarray(midpoint[:, 2]), SWING_BOOST_MAX_M, atol=1.0e-7
        )

        maximum_action = jp.tile(jp.asarray((0.0, 0.0, 1.0)), 6)
        clipped, _ = firmware._phase_gated_policy_residual(
            maximum_action, swing, jp.full(6, 0.5), SWING_BOOST_MAX_M
        )
        np.testing.assert_allclose(
            np.asarray(clipped[:, 2]),
            firmware.SWING_HEIGHT_MAX - firmware.SWING_HEIGHT,
            atol=1.0e-7,
        )

    def test_stance_and_late_landing_are_unchanged(self) -> None:
        action = jp.tile(jp.asarray((1.0, 1.0, 1.0)), 6)
        gait_state = jp.asarray(
            (
                firmware.LEG_STANCE,
                firmware.LEG_LATE_LANDING,
                firmware.LEG_STANCE,
                firmware.LEG_LATE_LANDING,
                firmware.LEG_STANCE,
                firmware.LEG_LATE_LANDING,
            )
        )
        legacy, _ = firmware._phase_gated_policy_residual(
            action, gait_state, jp.full(6, 0.5), 0.0
        )
        boosted, _ = firmware._phase_gated_policy_residual(
            action, gait_state, jp.full(6, 0.5), SWING_BOOST_MAX_M
        )
        np.testing.assert_allclose(np.asarray(boosted), np.asarray(legacy), atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
