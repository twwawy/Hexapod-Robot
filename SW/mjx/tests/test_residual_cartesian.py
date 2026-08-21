"""Contract tests for the first Cartesian residual-RL curriculum."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import jax.numpy as jnp


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexapod_mjx.residual_controller import (  # noqa: E402
    ACTION_DIM,
    ResidualControllerConfig,
    _apply_contact_adaptation,
    residual_action_metres,
)


class CartesianResidualContractTest(unittest.TestCase):
    def test_action_is_six_bounded_vertical_residuals(self) -> None:
        config = ResidualControllerConfig()
        residual = residual_action_metres(jnp.full((1, ACTION_DIM), 100.0), config)
        self.assertEqual(ACTION_DIM, 6)
        self.assertEqual(residual.shape, (1, 6))
        self.assertLessEqual(float(jnp.max(jnp.abs(residual))), config.residual_swing_z)
        self.assertAlmostEqual(float(jnp.max(jnp.abs(residual))), config.residual_swing_z, places=7)

    def test_stance_mask_blocks_residual_and_early_contact_holds_foot(self) -> None:
        nominal = jnp.zeros((1, 6, 3), dtype=jnp.float32)
        current = jnp.asarray(
            [[[0.1, 0.2, -0.3], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2]]],
            dtype=jnp.float32,
        )
        swing_mask = jnp.asarray([[True, False, True, False, True, False]])
        residual_z = jnp.asarray([[0.01, 0.02, 0.03, -0.01, -0.02, -0.03]], dtype=jnp.float32)
        contacts = jnp.asarray([[True, False, False, False, False, False]])

        corrected, applied = _apply_contact_adaptation(
            nominal,
            residual_z,
            swing_mask,
            contacts,
            current,
        )

        # Swing leg 0 landed early: safety wins over the RL command.
        self.assertTrue(jnp.allclose(corrected[:, 0, :], current[:, 0, :]))
        self.assertEqual(float(applied[0, 0]), 0.0)
        # Stance leg 1 ignores even a large residual.
        self.assertEqual(float(corrected[0, 1, 2]), 0.0)
        self.assertEqual(float(applied[0, 1]), 0.0)
        # A free swing leg keeps its Cartesian vertical correction.
        self.assertAlmostEqual(float(corrected[0, 2, 2]), 0.03, places=6)
        self.assertAlmostEqual(float(applied[0, 2]), 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
