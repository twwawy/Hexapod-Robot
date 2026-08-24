"""Contract tests for the first Cartesian residual-RL curriculum."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexapod_mjx.residual_controller import (  # noqa: E402
    ACTION_DIM,
    ResidualControllerConfig,
    _apply_contact_adaptation,
    _apply_nominal_posture_overlay,
    _classical_body_twist,
    _documented_inverse_kinematics,
    build_residual_controller,
    residual_action_metres,
    reset_controller_state,
)
from hexapod_mjx.model import load_hexapod_model, repo_root_from  # noqa: E402
from hexapod_mjx.residual_rl import PPOConfig, _bounded_policy_log_std  # noqa: E402
from urdf_kinematics import (  # noqa: E402
    NOMINAL_FOOT_RADIAL,
    NOMINAL_FOOT_VERTICAL,
    SHOULDER_LATERAL_OFFSET,
    TIBIA_LENGTH,
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

    def test_urdf_analytical_ik_reproduces_home_pose(self) -> None:
        lateral = jnp.asarray(
            [-SHOULDER_LATERAL_OFFSET] * 3
            + [SHOULDER_LATERAL_OFFSET] * 3,
            dtype=jnp.float32,
        )
        bundle = SimpleNamespace(
            hip_body_pos=jnp.zeros((6, 3), dtype=jnp.float32),
            leg_outward_body=jnp.tile(jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32), (6, 1)),
            leg_tangent_body=jnp.tile(jnp.asarray([[0.0, 1.0, 0.0]], dtype=jnp.float32), (6, 1)),
            right_leg_mask=jnp.asarray([False, False, False, True, True, True]),
            shoulder_lateral=lateral,
            distal_length=jnp.full((6,), TIBIA_LENGTH, dtype=jnp.float32),
            distal_angle_offset=jnp.zeros((6,), dtype=jnp.float32),
        )
        foot = jnp.stack(
            (
                jnp.full((6,), NOMINAL_FOOT_RADIAL),
                lateral,
                jnp.full((6,), NOMINAL_FOOT_VERTICAL),
            ),
            axis=-1,
        )[None, :, :]
        target = _documented_inverse_kinematics(foot, bundle)
        left_expected = jnp.asarray([0.0, jnp.pi / 6.0, -5.0 * jnp.pi / 18.0])
        right_expected = jnp.asarray([0.0, -jnp.pi / 6.0, 5.0 * jnp.pi / 18.0])
        self.assertTrue(jnp.allclose(target[0, :3], left_expected, atol=3e-5))
        self.assertTrue(jnp.allclose(target[0, 3:], right_expected, atol=3e-5))

    def test_cad_support_point_ik_matches_model_neutral_pose(self) -> None:
        model_bundle = load_hexapod_model(repo_root_from(Path(__file__)))
        config = ResidualControllerConfig()
        controller_bundle = build_residual_controller(model_bundle, config)
        recovered = _documented_inverse_kinematics(
            controller_bundle.reference_foot_body_pos[None, :, :],
            controller_bundle,
        )[0]
        expected = controller_bundle.neutral_joint_pose[
            controller_bundle.leg_joint_indices
        ]
        self.assertTrue(jnp.allclose(recovered, expected, atol=3e-5))

        # Also verify a non-home pose so the test covers the full fixed-offset
        # chain rather than only one calibrated point.
        data = mujoco.MjData(model_bundle.model)
        data.qpos[0:3] = np.array([0.0, 0.0, 0.0])
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        desired = np.asarray(model_bundle.default_joint_pose).copy()
        raw_pose = np.asarray(
            [[0.12, 0.40, -0.70]] * 3 + [[0.12, -0.40, 0.70]] * 3,
            dtype=np.float64,
        )
        desired[np.asarray(controller_bundle.leg_joint_indices)] = raw_pose
        data.qpos[model_bundle.joint_qpos_adr] = desired
        mujoco.mj_forward(model_bundle.model, data)

        root_id = model_bundle.model.body("hexapod_root").id
        root_pos = data.xpos[root_id]
        root_rot = data.xmat[root_id].reshape(3, 3)
        support_ids = np.asarray(controller_bundle.foot_support_geom_ids)
        foot_world = np.mean(data.geom_xpos[support_ids], axis=1)
        foot_body = (foot_world - root_pos) @ root_rot
        recovered = _documented_inverse_kinematics(
            jnp.asarray(foot_body)[None, :, :], controller_bundle
        )[0]
        self.assertTrue(jnp.allclose(recovered, raw_pose, atol=3e-5))

    def test_residual_policy_exploration_std_is_bounded(self) -> None:
        config = PPOConfig()
        params = {"policy_log_std": jnp.full((ACTION_DIM,), 9.0, dtype=jnp.float32)}
        bounded = _bounded_policy_log_std(params, config)
        self.assertTrue(jnp.all(bounded <= config.max_policy_log_std))
        self.assertTrue(jnp.all(bounded >= config.min_policy_log_std))


if __name__ == "__main__":
    unittest.main()
