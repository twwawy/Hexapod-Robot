from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rough_terrain_env import (
    _base_reward_terms,
    _edge_margin_cost,
    _foot_clearance_terrain_cost,
    _scale_reward_terms,
    _touchdown_impact_cost,
    _update_progress_watchdog,
    default_config,
)
from servo_model import SERVO_STALL_TORQUE_NM


class StairPenaltyTest(unittest.TestCase):
    def _base_kwargs(self) -> dict[str, jp.ndarray]:
        return {
            "forward_velocity": jp.asarray(0.2),
            "command": jp.asarray((0.2, 0.0, 0.0, 0.0, 0.0)),
            "yaw_velocity": jp.asarray(0.0),
            "attitude": jp.zeros(3),
            "posture_target": jp.zeros(2),
            "height_command": jp.asarray(0.0),
            "clearance": jp.asarray(0.316),
            "target_clearance": jp.asarray(0.316),
            "root_angular_speed": jp.asarray(0.0),
            "joint_proximity": jp.zeros(18),
            "action": jp.zeros(18),
            "last_action": jp.zeros(18),
            "swing_height_cost": jp.asarray(0.0),
            "early_swing_contact": jp.asarray(0.0),
            "vertical_velocity": jp.asarray(0.0),
            "lateral_velocity": jp.asarray(0.0),
            "joint_velocity": jp.zeros(18),
            "actuator_force": jp.zeros(18),
            "torque_limit": jp.asarray(SERVO_STALL_TORQUE_NM),
            "torque_saturation": jp.asarray(0.0),
            "gait_accepted": jp.asarray(True),
            "posture_accepted": jp.asarray(True),
            "policy_valid": jp.ones(6, dtype=jp.bool_),
            "foot_limited": jp.zeros(6, dtype=jp.bool_),
            "body_contact": jp.asarray(False),
            "self_collision": jp.asarray(False),
        }

    def _assert_terms(
        self, terms: dict[str, jp.ndarray], expected: dict[str, float]
    ) -> None:
        self.assertEqual(set(terms), set(expected))
        np.testing.assert_allclose(
            np.asarray([float(terms[name]) for name in sorted(terms)]),
            np.asarray([expected[name] for name in sorted(expected)]),
            atol=1.0e-6,
            rtol=0.0,
        )

    def _hover_expected(self) -> dict[str, float]:
        return {
            "velocity": 1.0,
            "yaw": 1.0,
            "upright": 1.0,
            "height": 1.0,
            "progress": 1.0,
            "under_speed": 0.0,
            "stability": 1.0,
            "joint_margin": 1.0,
            "action_rate": 0.0,
            "residual": 0.0,
            "swing_height": 0.0,
            "early_swing_contact": 0.0,
            "vertical_velocity": 0.0,
            "lateral_velocity": 0.0,
            "joint_velocity": 0.0,
            "torque": 0.0,
            "torque_saturation": 0.0,
            "gait_rejected": 0.0,
            "posture_rejected": 0.0,
            "policy_rejected": 0.0,
            "foot_limited": 0.0,
            "body_contact": 0.0,
            "self_collision": 0.0,
        }

    def test_legacy_golden_level_hover(self) -> None:
        terms = _base_reward_terms(**self._base_kwargs())
        self._assert_terms(
            terms,
            self._hover_expected(),
        )

    def test_legacy_golden_ten_degree_pitch(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["attitude"] = jp.deg2rad(jp.asarray((0.0, 10.0, 0.0)))
        terms = _base_reward_terms(**kwargs)
        expected = self._hover_expected()
        expected["upright"] = math.exp(-(math.radians(10.0) ** 2) / 0.12)
        self._assert_terms(terms, expected)

    def test_legacy_golden_contact_mock(self) -> None:
        kwargs = self._base_kwargs()
        kwargs.update(
            {
                "forward_velocity": jp.asarray(0.0),
                "yaw_velocity": jp.asarray(0.3),
                "clearance": jp.asarray(0.216),
                "root_angular_speed": jp.asarray(2.0),
                "joint_proximity": jp.full(18, 0.5),
                "action": jp.ones(18),
                "swing_height_cost": jp.asarray(0.25),
                "early_swing_contact": jp.asarray(0.5),
                "vertical_velocity": jp.asarray(0.1),
                "lateral_velocity": jp.asarray(-0.2),
                "joint_velocity": jp.full(18, 2.0),
                "actuator_force": jp.full(18, SERVO_STALL_TORQUE_NM / 2.0),
                "torque_saturation": jp.asarray(0.25),
                "gait_accepted": jp.asarray(False),
                "posture_accepted": jp.asarray(False),
                "policy_valid": jp.asarray((False, True, True, True, True, True)),
                "foot_limited": jp.asarray((True, True, False, False, False, False)),
                "body_contact": jp.asarray(True),
                "self_collision": jp.asarray(True),
            }
        )
        self._assert_terms(
            _base_reward_terms(**kwargs),
            {
                "velocity": math.exp(-25.0),
                "yaw": 0.0,
                "upright": 0.0,
                "height": 0.0,
                "progress": 0.0,
                "under_speed": 1.0,
                "stability": 0.0,
                "joint_margin": 0.0,
                "action_rate": 1.0,
                "residual": 1.0,
                "swing_height": 0.25,
                "early_swing_contact": 0.5,
                "vertical_velocity": 0.01,
                "lateral_velocity": 0.04,
                "joint_velocity": 0.04,
                "torque": 0.25,
                "torque_saturation": 0.25,
                "gait_rejected": 1.0,
                "posture_rejected": 1.0,
                "policy_rejected": 1.0 / 6.0,
                "foot_limited": 2.0 / 6.0,
                "body_contact": 1.0,
                "self_collision": 1.0,
            },
        )

    def test_command_aware_golden_nonzero_height_pitch_and_roll(self) -> None:
        kwargs = self._base_kwargs()
        command = jp.asarray(
            (0.2, 0.0, 0.05, math.radians(-15.0), math.radians(5.0))
        )
        target = jp.asarray((math.radians(5.0), math.radians(-15.0)))
        kwargs.update(
            {
                "command": command,
                "attitude": target,
                "posture_target": target,
                "height_command": command[2],
                "clearance": jp.asarray(0.366),
            }
        )
        self._assert_terms(_base_reward_terms(**kwargs), self._hover_expected())

    def test_new_penalty_vectors(self) -> None:
        swing = jp.ones(6, dtype=jp.bool_)
        terrain = jp.full(6, 0.30)
        self.assertEqual(
            float(_foot_clearance_terrain_cost(terrain + 0.05, terrain, swing)),
            0.0,
        )
        self.assertAlmostEqual(
            float(_foot_clearance_terrain_cost(terrain + 0.01, terrain, swing)),
            0.01,
            places=6,
        )
        edge_cost = _edge_margin_cost(
            jp.asarray((0.55, 0.56, 0.58, 0.59, 0.65, 0.80)),
            jp.asarray((0.55,)),
        )
        self.assertAlmostEqual(float(edge_cost), (0.03 + 0.02) / 6.0, places=7)
        impact = _touchdown_impact_cost(
            jp.zeros(6, dtype=jp.bool_),
            jp.asarray((True, False, False, False, False, False)),
            jp.asarray((-2.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        self.assertAlmostEqual(float(impact), 4.0 / 6.0, places=7)

    def test_reward_weight_is_wired_into_scaling(self) -> None:
        config = default_config()
        terms = {"foot_clearance_terrain": jp.asarray(0.01)}
        self.assertAlmostEqual(
            float(_scale_reward_terms(terms, config.reward)["foot_clearance_terrain"]),
            -0.02,
            places=7,
        )
        config.reward.foot_clearance_terrain = -4.0
        self.assertAlmostEqual(
            float(_scale_reward_terms(terms, config.reward)["foot_clearance_terrain"]),
            -0.04,
            places=7,
        )

    def test_stationary_return_is_negative_and_target_speed_is_best(self) -> None:
        config = default_config()
        stationary = self._base_kwargs()
        stationary["command"] = jp.asarray((0.08, 0.0, 0.0, 0.0, 0.0))
        stationary["forward_velocity"] = jp.asarray(0.0)
        half_speed = dict(stationary, forward_velocity=jp.asarray(0.04))
        target_speed = dict(stationary, forward_velocity=jp.asarray(0.08))

        returns = []
        for kwargs in (stationary, half_speed, target_speed):
            scaled = _scale_reward_terms(_base_reward_terms(**kwargs), config.reward)
            returns.append(sum(float(value) for value in scaled.values()))

        self.assertLess(returns[0], 0.0)
        self.assertLess(returns[0], returns[1])
        self.assertLess(returns[1], returns[2])

    def test_progress_watchdog_credits_forward_or_upward_motion(self) -> None:
        anchor = jp.asarray(0.0)
        steps = jp.asarray(149, dtype=jp.int32)
        anchor, steps, timed_out = _update_progress_watchdog(
            potential=jp.asarray(0.019),
            anchor=anchor,
            stagnant_steps=steps,
            command_active=jp.asarray(True),
            success=jp.asarray(False),
            dt=0.02,
            min_delta=0.02,
            timeout=3.0,
        )
        self.assertTrue(bool(timed_out))

        anchor, steps, timed_out = _update_progress_watchdog(
            potential=jp.asarray(0.02),
            anchor=jp.asarray(0.0),
            stagnant_steps=jp.asarray(149, dtype=jp.int32),
            command_active=jp.asarray(True),
            success=jp.asarray(False),
            dt=0.02,
            min_delta=0.02,
            timeout=3.0,
        )
        self.assertAlmostEqual(float(anchor), 0.02, places=7)
        self.assertEqual(int(steps), 0)
        self.assertFalse(bool(timed_out))


if __name__ == "__main__":
    unittest.main()
