from __future__ import annotations

from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import firmware_mjx_controller as firmware


class CommandInputTest(unittest.TestCase):
    def _step(self, **commands):
        state = firmware.initial_state()._replace(first_step=jp.asarray(False))
        return firmware.step(
            state,
            target_velocity=jp.zeros(2),
            body_position_world=jp.asarray((0.0, 0.0, 0.316)),
            attitude_rpy=commands.pop("attitude_rpy", jp.zeros(3)),
            contacts=jp.ones(6, dtype=jp.bool_),
            policy_action=jp.zeros(18),
            **commands,
        )

    def test_explicit_zero_commands_are_bit_exact_with_t2_defaults(self) -> None:
        default_state, default_output = self._step()
        zero_state, zero_output = self._step(
            pitch_ff=jp.asarray(0.0),
            roll_cmd=jp.asarray(0.0),
            pitch_cmd=jp.asarray(0.0),
            height_offset=jp.asarray(0.0),
        )
        for left, right in zip(default_state, zero_state):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(default_output, zero_output):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_roll_command_error_has_positive_sign(self) -> None:
        state, _ = self._step(roll_cmd=jp.deg2rad(10.0))
        expected = jp.deg2rad(15.0) * firmware.FIRMWARE_CONTROL_DT
        self.assertAlmostEqual(float(state.posture_command[0]), float(expected), places=7)

    def test_effective_pitch_is_clipped_to_32_degrees(self) -> None:
        combined_state, _ = self._step(
            pitch_cmd=jp.deg2rad(-25.0), pitch_ff=jp.deg2rad(-28.0)
        )
        clipped_state, _ = self._step(pitch_cmd=-firmware.EFFECTIVE_PITCH_MAX_RAD)
        np.testing.assert_allclose(
            np.asarray(combined_state.posture_command),
            np.asarray(clipped_state.posture_command),
            atol=0.0,
            rtol=0.0,
        )

    def test_positive_height_offset_composes_all_feet_down_five_centimeters(self) -> None:
        nominal = jp.asarray(
            ((0.1, 0.2, -0.3),) * 6,
        )
        shifted = firmware._apply_height_offset(nominal, jp.asarray(0.05))
        np.testing.assert_array_equal(
            np.asarray(shifted[:, :2]), np.asarray(nominal[:, :2])
        )
        np.testing.assert_allclose(
            np.asarray(shifted[:, 2]),
            np.asarray(nominal[:, 2]) - 0.05,
            atol=1.0e-7,
        )

    def test_reachable_positive_height_offset_is_applied_to_output(self) -> None:
        _, baseline = self._step()
        _, raised = self._step(height_offset=jp.asarray(0.03))
        self.assertTrue(bool(raised.posture_accepted))
        np.testing.assert_allclose(
            np.asarray(raised.foot_targets_body[:, 2]),
            np.asarray(baseline.foot_targets_body[:, 2]) - 0.03,
            atol=1.0e-7,
        )

    def test_unreachable_height_and_posture_are_rejected_to_safe_nominal(self) -> None:
        _, baseline = self._step()
        _, rejected = self._step(
            height_offset=jp.asarray(0.10),
            roll_cmd=jp.deg2rad(45.0),
            pitch_cmd=jp.deg2rad(-32.0),
        )
        self.assertFalse(bool(rejected.posture_accepted))
        np.testing.assert_allclose(
            np.asarray(rejected.foot_targets_body),
            np.asarray(baseline.foot_targets_body),
            atol=1.0e-7,
        )

    def test_seeded_zero_command_rollout_is_deterministic(self) -> None:
        def rollout(explicit: bool) -> np.ndarray:
            state = firmware.initial_state()
            outputs = []
            key = jax.random.PRNGKey(7)
            for _ in range(24):
                key, action_key = jax.random.split(key)
                kwargs = {}
                if explicit:
                    kwargs = dict(
                        pitch_ff=jp.asarray(0.0),
                        roll_cmd=jp.asarray(0.0),
                        pitch_cmd=jp.asarray(0.0),
                        height_offset=jp.asarray(0.0),
                    )
                state, output = firmware.step(
                    state,
                    target_velocity=jp.asarray((0.08, 0.0)),
                    body_position_world=jp.asarray((0.0, 0.0, 0.316)),
                    attitude_rpy=jp.zeros(3),
                    contacts=jp.ones(6, dtype=jp.bool_),
                    policy_action=jax.random.uniform(
                        action_key, (18,), minval=-1.0, maxval=1.0
                    ),
                    **kwargs,
                )
                outputs.append(np.asarray(output.servo_joint_targets))
            return np.asarray(outputs)

        np.testing.assert_allclose(rollout(False), rollout(True), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
