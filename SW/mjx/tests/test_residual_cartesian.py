"""Contract tests for the first Cartesian residual-RL curriculum."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from types import SimpleNamespace

import jax.numpy as jnp


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexapod_mjx.residual_controller import (  # noqa: E402
    ACTION_DIM,
    ResidualControllerConfig,
    _apply_contact_adaptation,
    _apply_nominal_posture_overlay,
    _classical_body_twist,
    _documented_inverse_kinematics,
    residual_action_metres,
    reset_controller_state,
)
from hexapod_mjx.residual_rl import PPOConfig, _bounded_policy_log_std  # noqa: E402


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

    def test_classical_position_heading_pi_contributes_nominal_body_twist(self) -> None:
        config = ResidualControllerConfig(
            body_twist_limits=(1.0, 1.0, 1.0),
            translation_pi_kp=(0.50, 0.50),
            translation_pi_ki=(0.05, 0.05),
            heading_pi_kp=1.00,
            heading_pi_ki=0.05,
        )
        state = reset_controller_state(1)
        filtered_command = jnp.asarray([[0.10, 0.0, 0.20]], dtype=jnp.float32)
        body_position = jnp.asarray([[0.0, 0.0, 0.20]], dtype=jnp.float32)
        body_quat = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
        twist, desired_pos, desired_heading, translation_integral, heading_integral = _classical_body_twist(
            config,
            state,
            filtered_command,
            body_position,
            body_quat,
            0.02,
        )

        self.assertGreater(float(twist[0, 0]), 0.10)
        self.assertGreater(float(twist[0, 2]), 0.20)
        self.assertLess(float(desired_pos[0, 1]), 0.0)
        self.assertGreater(float(desired_heading[0]), 0.0)
        self.assertGreater(float(translation_integral[0, 0]), 0.0)
        self.assertGreater(float(heading_integral[0]), 0.0)

    def test_posture_pi_lowers_targets_when_body_is_below_reference_height(self) -> None:
        config = ResidualControllerConfig(
            posture_pi_kp=(0.50, 0.50, 0.80),
            posture_pi_ki=(0.03, 0.03, 0.05),
        )
        feet = jnp.zeros((1, ACTION_DIM, 3), dtype=jnp.float32)
        quat = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
        bundle = SimpleNamespace(reset_root_height=jnp.asarray(0.20, dtype=jnp.float32))
        corrected, next_integral = _apply_nominal_posture_overlay(
            feet,
            quat,
            jnp.asarray([0.18], dtype=jnp.float32),
            bundle,
            config,
            jnp.zeros((1, 3), dtype=jnp.float32),
            0.02,
        )

        self.assertLess(float(corrected[0, 0, 2]), 0.0)
        self.assertGreater(float(next_integral[0, 2]), 0.0)

    def test_imported_analytical_ik_reproduces_documented_home_pose(self) -> None:
        # This is the exact 30°/50° home configuration in
        # Downloads/mjx/prepare_scene.py, expressed in a simple leg frame.
        bundle = SimpleNamespace(
            hip_body_pos=jnp.zeros((6, 3), dtype=jnp.float32),
            leg_outward_body=jnp.tile(jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32), (6, 1)),
            leg_tangent_body=jnp.tile(jnp.asarray([[0.0, 1.0, 0.0]], dtype=jnp.float32), (6, 1)),
            right_leg_mask=jnp.asarray([False, False, False, True, True, True]),
        )
        foot = jnp.tile(jnp.asarray([[[0.218728, 0.0, -0.287006]]], dtype=jnp.float32), (1, 6, 1))
        target = _documented_inverse_kinematics(foot, bundle)
        left_expected = jnp.asarray([0.0, jnp.pi / 6.0, -5.0 * jnp.pi / 18.0])
        right_expected = jnp.asarray([0.0, -jnp.pi / 6.0, 5.0 * jnp.pi / 18.0])
        self.assertTrue(jnp.allclose(target[0, :3], left_expected, atol=3e-5))
        self.assertTrue(jnp.allclose(target[0, 3:], right_expected, atol=3e-5))

    def test_residual_policy_exploration_std_is_bounded(self) -> None:
        config = PPOConfig()
        params = {"policy_log_std": jnp.full((ACTION_DIM,), 9.0, dtype=jnp.float32)}
        bounded = _bounded_policy_log_std(params, config)
        self.assertTrue(jnp.all(bounded <= config.max_policy_log_std))
        self.assertTrue(jnp.all(bounded >= config.min_policy_log_std))


if __name__ == "__main__":
    unittest.main()
