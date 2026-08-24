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
    def test_initial_model_targets_match_home_pose(self) -> None:
        output = firmware_mjx.initial_output()
        expected_servo = np.tile(np.deg2rad((0.0, 30.0, 50.0)), (6, 1))
        np.testing.assert_allclose(
            np.asarray(output.servo_joint_targets), expected_servo, atol=1.0e-5
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
                native_output.joint_angles,
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
