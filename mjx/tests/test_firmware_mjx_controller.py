from __future__ import annotations

import unittest
from pathlib import Path
import sys

import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firmware_controller import FirmwareController
import firmware_mjx_controller as firmware_mjx


class FirmwareMjxControllerTest(unittest.TestCase):
    def test_policy_y_authority_and_workspace_margin_are_conservative(self) -> None:
        np.testing.assert_allclose(
            np.asarray(firmware_mjx.RESIDUAL_SCALE),
            np.asarray((0.10, 0.10, 0.10)),
            atol=1.0e-8,
        )
        self.assertAlmostEqual(firmware_mjx.WORKSPACE_MARGIN, 0.001)
        self.assertAlmostEqual(firmware_mjx.SWING_HEIGHT, 0.06)
        self.assertAlmostEqual(firmware_mjx.SWING_HEIGHT_MIN, 0.04)
        self.assertAlmostEqual(firmware_mjx.SWING_HEIGHT_MAX, 0.25)
        peak = firmware_mjx._swing(
            jp.asarray(0.5), firmware_mjx.BASE_FEET, firmware_mjx.BASE_FEET
        )
        np.testing.assert_allclose(
            np.asarray(peak[:, 2] - firmware_mjx.BASE_FEET[:, 2]),
            np.full(6, 0.06),
            atol=1.0e-7,
        )

        outer_local = jp.tile(
            jp.asarray(
                (
                    firmware_mjx.LINK_1
                    + firmware_mjx.LINK_2
                    + firmware_mjx.LINK_3,
                    0.0,
                    0.0,
                )
            ),
            (6, 1),
        )
        limited_body, was_limited = firmware_mjx._limit_foot_reach(
            firmware_mjx._leg_to_body(outer_local)
        )
        limited_local = firmware_mjx._body_to_leg(limited_body)
        planar_reach = jp.linalg.norm(limited_local[0, :2]) - firmware_mjx.LINK_1
        self.assertTrue(bool(was_limited[0]))
        self.assertAlmostEqual(
            float(planar_reach),
            firmware_mjx.LINK_2
            + firmware_mjx.LINK_3
            - firmware_mjx.WORKSPACE_MARGIN,
            places=6,
        )

    def test_adaptive_swing_height_is_phase_gated_and_state_specific(self) -> None:
        action = np.zeros((6, 3), dtype=np.float32)
        action[:, 0] = 1.0
        action[:, 1] = 1.0
        action[:, 2] = np.asarray((1.0, -1.0, 0.5, -0.5, 1.0, 1.0))
        gait_state = jp.asarray(
            (
                firmware_mjx.LEG_SWING,
                firmware_mjx.LEG_SWING,
                firmware_mjx.LEG_STANCE,
                firmware_mjx.LEG_STANCE,
                firmware_mjx.LEG_LATE_LANDING,
                firmware_mjx.LEG_SWING,
            )
        )
        progress = jp.asarray((0.5, 0.5, 0.5, 0.5, 0.5, 0.0))
        residual, heights = firmware_mjx._phase_gated_policy_residual(
            jp.asarray(action.reshape(-1)), gait_state, progress
        )
        residual = np.asarray(residual)
        heights = np.asarray(heights)

        self.assertAlmostEqual(heights[0], 0.25, places=6)
        self.assertAlmostEqual(heights[1], 0.04, places=6)
        self.assertAlmostEqual(heights[2], 0.155, places=6)
        self.assertAlmostEqual(residual[0, 2], 0.19, places=6)
        self.assertAlmostEqual(residual[1, 2], -0.02, places=6)
        self.assertAlmostEqual(residual[2, 2], 0.05, places=6)
        self.assertAlmostEqual(residual[3, 2], -0.05, places=6)
        np.testing.assert_allclose(residual[4], np.zeros(3), atol=1.0e-8)
        self.assertAlmostEqual(residual[5, 2], 0.0, places=7)

        endpoint_residual, _ = firmware_mjx._phase_gated_policy_residual(
            jp.asarray(action.reshape(-1)),
            jp.full(6, firmware_mjx.LEG_SWING),
            jp.asarray((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)),
        )
        np.testing.assert_allclose(
            np.asarray(endpoint_residual)[:, 2], np.zeros(6), atol=1.0e-7
        )

    def test_terrain_swing_floor_overrides_a_low_policy_command(self) -> None:
        low_action = jp.tile(jp.asarray((0.0, 0.0, -1.0)), 6)
        residual, commanded_height = firmware_mjx._phase_gated_policy_residual(
            low_action,
            jp.full(6, firmware_mjx.LEG_SWING),
            jp.full(6, 0.5),
            swing_height_floor=jp.asarray(0.13),
        )
        np.testing.assert_allclose(np.asarray(commanded_height), 0.13, atol=1.0e-7)
        np.testing.assert_allclose(
            np.asarray(residual[:, 2]),
            0.13 - firmware_mjx.SWING_HEIGHT,
            atol=1.0e-7,
        )

    def test_initial_model_targets_match_home_pose(self) -> None:
        output = firmware_mjx.initial_output()
        expected_servo = np.tile(np.deg2rad((0.0, 30.0, 50.0)), (6, 1))
        expected_servo[0:3, 1] *= -1.0
        expected_servo[3:6, 2] *= -1.0
        np.testing.assert_allclose(
            np.asarray(output.servo_joint_targets), expected_servo, atol=1.0e-5
        )
        np.testing.assert_allclose(
            np.asarray(output.model_joint_targets), expected_servo, atol=1.0e-5
        )
        self.assertTrue(np.all(np.asarray(output.ik_valid)))

    def test_zero_residual_matches_compiled_stm32_controller(self) -> None:
        native = FirmwareController()
        jax_state = firmware_mjx.initial_state()
        contacts = np.ones(6, dtype=bool)
        try:
            for _ in range(120):
                native_output = native.step(
                    target_vx=0.08,
                    target_wz=0.0,
                    body_position=np.asarray((0.0, 0.0, 0.316)),
                    attitude=np.zeros(3),
                    contacts=contacts,
                )
                jax_state, jax_output = firmware_mjx.step(
                    jax_state,
                    target_velocity=jp.asarray((0.08, 0.0)),
                    body_position_world=jp.asarray((0.0, 0.0, 0.316)),
                    attitude_rpy=jp.zeros(3),
                    contacts=jp.asarray(contacts),
                    policy_action=jp.zeros(18),
                )

            np.testing.assert_allclose(
                np.asarray(jax_output.servo_joint_targets).reshape(18),
                (
                    native_output.joint_angles.reshape(6, 3)
                    * np.asarray(firmware_mjx.MODEL_SIGNS)
                ).reshape(18),
                atol=2.0e-5,
            )
            np.testing.assert_allclose(
                np.asarray(jax_output.applied_twist),
                native_output.applied_twist,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                np.asarray(jax_output.gait_progress),
                native_output.gait_progress,
                atol=1.0e-6,
            )
            np.testing.assert_array_equal(
                np.asarray(jax_output.gait_state), native_output.gait_state
            )
        finally:
            native.close()

    def test_policy_never_bypasses_joint_limits(self) -> None:
        state = firmware_mjx.initial_state()
        contacts = jp.ones(6, dtype=jp.bool_)
        for _ in range(160):
            state, output = firmware_mjx.step(
                state,
                target_velocity=jp.asarray((0.12, 0.0)),
                body_position_world=jp.asarray((0.0, 0.0, 0.316)),
                attitude_rpy=jp.zeros(3),
                contacts=contacts,
                policy_action=jp.ones(18),
            )
        self.assertLessEqual(
            float(jp.max(jp.abs(output.servo_joint_targets))),
            float(firmware_mjx.JOINT_LIMIT) + 1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
