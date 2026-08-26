from __future__ import annotations

from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rough_terrain_env import (
    OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_SIZE,
    HexapodRoughTerrainEnv,
    _absolute_tilt_failure,
    _posture_success,
    _upright_reward,
    default_config,
)


class PostureContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = HexapodRoughTerrainEnv(terrain_level=0)

    def test_observation_shape_and_contract(self) -> None:
        state = self.env.reset(jax.random.PRNGKey(0))
        self.assertEqual(state.obs.shape, (143,))
        self.assertEqual(OBSERVATION_SIZE, 143)
        self.assertEqual(
            OBSERVATION_CONTRACT_VERSION,
            "firmware_state_collision_terrain_pitch_v3",
        )

    def test_upright_reward_tracks_target_relative_pitch(self) -> None:
        reward = _upright_reward(jp.asarray((0.0, 0.2, 0.0)), jp.asarray(0.0))
        self.assertAlmostEqual(float(reward), 0.71653, delta=1.0e-4)
        aligned = _upright_reward(jp.asarray((0.0, 0.2, 0.0)), jp.asarray(0.2))
        self.assertAlmostEqual(float(aligned), 1.0, places=7)

    def test_success_uses_target_relative_twelve_degree_gate(self) -> None:
        self.assertTrue(
            bool(
                _posture_success(
                    jp.deg2rad(jp.asarray((0.0, 25.0, 0.0))),
                    jp.deg2rad(20.0),
                )
            )
        )
        self.assertFalse(
            bool(
                _posture_success(
                    jp.deg2rad(jp.asarray((25.0, 0.0, 0.0))),
                    jp.asarray(0.0),
                )
            )
        )

    def test_absolute_fifty_degree_tilt_still_fails(self) -> None:
        max_tilt = default_config().safety.max_tilt
        self.assertTrue(
            bool(
                _absolute_tilt_failure(
                    jp.deg2rad(jp.asarray((0.0, 50.0, 0.0))), max_tilt
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
