"""Contract tests for the classical-first 22-D terrain residual v2."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import jax
import jax.numpy as jp


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rough_terrain_env import (  # noqa: E402
    ACTION_CONTRACT_VERSION,
    ACTION_SIZE,
    OBSERVATION_SIZE,
    _quat_rotate,
    _quat_rotate_inverse,
)
from tripod_core import (  # noqa: E402
    LINK1,
    heading_aligned_points,
    phase_masked_residual,
    project_workspace,
    scale_asymmetric,
)


class RoughTerrainContractTest(unittest.TestCase):
    def test_fixed_action_and_observation_contract(self) -> None:
        self.assertEqual(ACTION_SIZE, 22)
        self.assertEqual(OBSERVATION_SIZE, 110)
        self.assertEqual(ACTION_CONTRACT_VERSION, "cartesian_gait_residual_v2")

    def test_stance_xy_is_exactly_zero_and_swing_ranges_are_asymmetric(self) -> None:
        raw = jp.array(
            [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]] + [[0.0, 0.0, 0.0]] * 4
        )
        residual = phase_masked_residual(
            raw,
            jp.array([True, False, True, False, True, False]),
            swing_x=0.025,
            swing_y=0.015,
            swing_z_low=-0.010,
            swing_z_high=0.050,
            stance_z=0.008,
        )
        self.assertTrue(jp.allclose(residual[0], jp.array([0.025, -0.015, 0.050])))
        self.assertTrue(jp.allclose(residual[1], jp.array([0.0, 0.0, -0.008])))
        self.assertTrue(jp.allclose(residual[3:, :2], 0.0))
        self.assertEqual(float(scale_asymmetric(jp.array(0.0), -0.010, 0.050)), 0.0)

    def test_contact_priority_holds_early_swing_and_searches_only_downward(self) -> None:
        from tripod_core import contact_adapt_targets

        requested = jp.zeros((6, 3))
        current = jp.tile(jp.array([[0.2, 0.0, -0.25]]), (6, 1))
        adapted, early, lost = contact_adapt_targets(
            requested,
            current,
            jp.array([True, False, True, False, True, False]),
            jp.array([True, False, False, False, False, False]),
            lost_contact_search=0.010,
        )
        self.assertTrue(bool(early[0]))
        self.assertTrue(jp.allclose(adapted[0], current[0]))
        self.assertTrue(bool(lost[1]))
        self.assertTrue(jp.allclose(adapted[1], jp.array([0.0, 0.0, -0.010])))

    def test_workspace_projection_makes_extreme_targets_feasible(self) -> None:
        requested = jp.array(
            [[1.0, 0.0, -1.0], [0.001, 0.0, 0.0], [0.22, 0.02, -0.29]]
        )
        projected, cost = project_workspace(requested)
        radial = jp.linalg.norm(projected[:, :2], axis=-1)
        distance = jp.sqrt(jp.square(radial - LINK1) + jp.square(projected[:, 2]))
        self.assertTrue(jp.all(distance >= 0.112 - 1e-6))
        self.assertTrue(jp.all(distance <= 0.345 + 1e-6))
        self.assertGreater(float(cost), 0.0)

    def test_body_frame_vector_is_yaw_invariant(self) -> None:
        body_vector = jp.array([0.31, -0.18, -0.29])
        yaw_90 = jp.array([jp.sqrt(0.5), 0.0, 0.0, jp.sqrt(0.5)])
        world_vector = _quat_rotate(yaw_90, body_vector)
        recovered = _quat_rotate_inverse(yaw_90, world_vector)
        self.assertTrue(jp.allclose(recovered, body_vector, atol=1e-6))

    def test_heading_grid_rotates_with_robot_heading(self) -> None:
        offsets = jp.array([[0.25, 0.0], [0.25, 0.22]])
        at_x = heading_aligned_points(jp.array([0.0, 0.0]), jp.array([1.0, 0.0]), offsets)
        at_y = heading_aligned_points(jp.array([0.0, 0.0]), jp.array([0.0, 1.0]), offsets)
        self.assertTrue(jp.allclose(at_x[0], jp.array([0.25, 0.0])))
        self.assertTrue(jp.allclose(at_y[0], jp.array([0.0, 0.25])))
        self.assertTrue(jp.allclose(at_y[1], jp.array([-0.22, 0.25])))

    def test_core_is_jittable(self) -> None:
        compiled = jax.jit(lambda feet: project_workspace(feet)[0])
        output = compiled(jp.ones((6, 3)) * jp.array([0.22, 0.0, -0.28]))
        self.assertEqual(output.shape, (6, 3))


if __name__ == "__main__":
    unittest.main()
