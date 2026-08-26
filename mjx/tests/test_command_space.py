from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rough_terrain_env import (
    EFFECTIVE_PITCH_MAX_RAD,
    OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_SIZE,
    HexapodRoughTerrainEnv,
    _absolute_tilt_failure,
    _base_reward_terms,
    _effective_posture_target,
    _posture_success,
    _terrain_posture_command,
    _upright_reward,
    default_config,
)
from servo_model import SERVO_STALL_TORQUE_NM
from train_rough_terrain import _apply_command_config, _arguments


class CommandSpaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = HexapodRoughTerrainEnv(terrain_level=0)

    def _sample(self, terrain_kind: str, slope: float, seed: int) -> np.ndarray:
        key_a, key_b = jax.random.split(jax.random.PRNGKey(seed))
        return np.asarray(
            _terrain_posture_command(
                height_key=key_a,
                pitch_key=key_b,
                terrain_kind=terrain_kind,
                slope_degrees=slope,
                command_config=default_config().command,
            )
        )

    def test_observation_shape_contract_and_flat_command(self) -> None:
        state = self.env.reset(jax.random.PRNGKey(0))
        self.assertEqual(state.obs.shape, (146,))
        self.assertEqual(OBSERVATION_SIZE, 146)
        self.assertEqual(
            OBSERVATION_CONTRACT_VERSION,
            "firmware_state_collision_terrain_command5_pitch_v3",
        )
        self.assertEqual(state.info["command"].shape, (5,))
        np.testing.assert_array_equal(np.asarray(state.info["command"])[2:], 0.0)

    def test_flat_and_rough_posture_commands_are_zero_for_100_seeds(self) -> None:
        for terrain_kind in ("flat", "rough"):
            for seed in range(100):
                np.testing.assert_array_equal(
                    self._sample(terrain_kind, 0.0, seed), np.zeros(3)
                )

    def test_ramp_and_stair_curriculum_bounds(self) -> None:
        for seed in range(100):
            ramp = self._sample("ramp", 15.0, seed)
            self.assertGreaterEqual(ramp[0], -0.05)
            self.assertLessEqual(ramp[0], 0.0)
            self.assertAlmostEqual(ramp[1], math.radians(-15.0), places=6)
            self.assertEqual(ramp[2], 0.0)
            stairs = self._sample("stairs", 0.0, seed)
            self.assertGreaterEqual(stairs[0], -0.05)
            self.assertLessEqual(stairs[0], 0.0)
            self.assertGreaterEqual(stairs[1], math.radians(-25.0))
            self.assertLessEqual(stairs[1], math.radians(-5.0))
            self.assertEqual(stairs[2], 0.0)

    def test_effective_target_reward_height_and_success_kernels(self) -> None:
        command = jp.asarray((0.2, 0.0, 0.05, -0.30, 0.10))
        target = _effective_posture_target(command, jp.asarray(-0.40))
        self.assertAlmostEqual(float(target[0]), 0.10, places=7)
        self.assertAlmostEqual(float(target[1]), -EFFECTIVE_PITCH_MAX_RAD, places=7)
        self.assertAlmostEqual(float(_upright_reward(target, target)), 1.0, places=7)
        self.assertTrue(bool(_posture_success(target, target)))
        self.assertFalse(bool(_posture_success(target + jp.asarray((0.22, 0.0)), target)))

        terms = _base_reward_terms(
            forward_velocity=jp.asarray(0.2),
            command=command,
            yaw_velocity=jp.asarray(0.0),
            attitude=target,
            posture_target=target,
            height_command=command[2],
            clearance=jp.asarray(0.366),
            target_clearance=jp.asarray(0.316),
            root_angular_speed=jp.asarray(0.0),
            joint_proximity=jp.zeros(18),
            action=jp.zeros(18),
            last_action=jp.zeros(18),
            swing_height_cost=jp.asarray(0.0),
            early_swing_contact=jp.asarray(0.0),
            vertical_velocity=jp.asarray(0.0),
            lateral_velocity=jp.asarray(0.0),
            joint_velocity=jp.zeros(18),
            actuator_force=jp.zeros(18),
            torque_limit=jp.asarray(SERVO_STALL_TORQUE_NM),
            torque_saturation=jp.asarray(0.0),
            gait_accepted=jp.asarray(True),
            posture_accepted=jp.asarray(True),
            policy_valid=jp.ones(6, dtype=jp.bool_),
            foot_limited=jp.zeros(6, dtype=jp.bool_),
            body_contact=jp.asarray(False),
            self_collision=jp.asarray(False),
        )
        self.assertAlmostEqual(float(terms["upright"]), 1.0, places=7)
        self.assertAlmostEqual(float(terms["height"]), 1.0, places=7)

    def test_absolute_fifty_degree_tilt_still_fails(self) -> None:
        self.assertTrue(
            bool(
                _absolute_tilt_failure(
                    jp.deg2rad(jp.asarray((0.0, 50.0, 0.0))),
                    default_config().safety.max_tilt,
                )
            )
        )

    def test_cli_flags_apply_to_config(self) -> None:
        args = _arguments(
            [
                "--height-cmd-min", "-0.04",
                "--height-cmd-max", "0.08",
                "--pitch-cmd-deg-min", "-20",
                "--pitch-cmd-deg-max", "18",
                "--roll-cmd-deg-max", "12",
            ]
        )
        config = default_config()
        _apply_command_config(config, args)
        self.assertEqual(config.command.height_min, -0.04)
        self.assertEqual(config.command.height_max, 0.08)
        self.assertEqual(config.command.pitch_min_deg, -20.0)
        self.assertEqual(config.command.pitch_max_deg, 18.0)
        self.assertEqual(config.command.roll_max_deg, 12.0)


if __name__ == "__main__":
    unittest.main()
