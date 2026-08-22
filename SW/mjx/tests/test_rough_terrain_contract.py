"""Contract tests for the classical-first 22-D terrain residual v2."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rough_terrain_env import (  # noqa: E402
    ACTION_CONTRACT_VERSION,
    ACTION_SIZE,
    OBSERVATION_SIZE,
    OBSERVATION_CONTRACT_VERSION,
    HexapodRoughTerrainEnv,
    default_config,
    _quat_rotate,
    _quat_rotate_inverse,
)
from tripod_core import (  # noqa: E402
    LINK1,
    analytical_ik,
    heading_aligned_points,
    hysteretic_clearance_contact,
    median_support_height,
    nominal_foot_targets,
    phase_masked_residual,
    project_workspace,
    scale_asymmetric,
    update_airborne_state,
)
from prepare_rl_scene import prepare_flat_rl_scene  # noqa: E402
from tripod_controller import (  # noqa: E402
    GaitConfig,
    LEG_PREFIXES,
    RIGHT_LEGS,
    TRIPOD_A,
    TripodGaitController,
)
from domain_randomization import domain_randomize  # noqa: E402


class RoughTerrainContractTest(unittest.TestCase):
    def test_fixed_action_and_observation_contract(self) -> None:
        self.assertEqual(ACTION_SIZE, 22)
        self.assertEqual(OBSERVATION_SIZE, 110)
        self.assertEqual(ACTION_CONTRACT_VERSION, "cartesian_gait_residual_v2")
        self.assertEqual(
            OBSERVATION_CONTRACT_VERSION, "body_state_coarse9_touchdown6_v1"
        )
        config = default_config()
        self.assertEqual(
            config.controller.residual.command.to_dict(),
            config.controller.residual.terrain.to_dict(),
        )

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
            jp.zeros(6, dtype=jp.bool_),
            lost_contact_search=0.010,
        )
        # Initial liftoff contact is not an early landing.
        self.assertFalse(bool(early[0]))
        self.assertTrue(jp.allclose(adapted[0], requested[0]))
        self.assertTrue(bool(lost[1]))
        self.assertTrue(jp.allclose(adapted[1], jp.array([0.0, 0.0, -0.010])))

        airborne = update_airborne_state(
            jp.zeros(6, dtype=jp.bool_),
            jp.array([True, False, True, False, True, False]),
            jp.array([False, False, False, False, False, False]),
        )
        adapted, early, _ = contact_adapt_targets(
            requested,
            current,
            jp.array([True, False, True, False, True, False]),
            jp.array([True, False, False, False, False, False]),
            airborne,
            lost_contact_search=0.010,
        )
        self.assertTrue(bool(early[0]))
        self.assertTrue(jp.allclose(adapted[0], current[0]))

    def test_support_height_uses_contact_stance_median(self) -> None:
        height = median_support_height(
            jp.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25]),
            jp.array([True, True, False, True, False, False]),
            jp.array(0.9),
        )
        self.assertAlmostEqual(float(height), 0.05, places=7)
        fallback = median_support_height(
            jp.zeros(6), jp.zeros(6, dtype=jp.bool_), jp.array(0.13)
        )
        self.assertAlmostEqual(float(fallback), 0.13, places=7)

    def test_clearance_contact_has_enter_release_hysteresis(self) -> None:
        clearance = jp.array([0.034, 0.040, 0.046])
        new_contact = hysteretic_clearance_contact(
            clearance,
            jp.array([False, False, False]),
            enter_clearance=0.035,
            release_clearance=0.045,
        )
        latched = hysteretic_clearance_contact(
            clearance,
            jp.array([True, True, True]),
            enter_clearance=0.035,
            release_clearance=0.045,
        )
        self.assertTrue(jp.array_equal(new_contact, jp.array([True, False, False])))
        self.assertTrue(jp.array_equal(latched, jp.array([True, True, False])))

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

    def test_zero_action_numpy_classical_matches_jax_nominal(self) -> None:
        prepare_flat_rl_scene()
        env = HexapodRoughTerrainEnv(terrain="flat", command_curriculum=True)
        model = env.mj_model
        config = GaitConfig(
            control_dt=0.02,
            phase_time=0.5,
            swing_height=0.07,
            radial_offset=0.01,
        )
        classical = TripodGaitController(model, config)
        origins = jp.array([classical._origins[prefix] for prefix in LEG_PREFIXES])
        outward = jp.array([classical._outward[prefix] for prefix in LEG_PREFIXES])
        tripod_a = jp.array([prefix in TRIPOD_A for prefix in LEG_PREFIXES])
        signs = jp.array(
            [(1.0, -1.0, 1.0) if prefix in RIGHT_LEGS else (1.0, 1.0, -1.0) for prefix in LEG_PREFIXES]
        )
        actuator_ids = np.array(
            [
                [model.actuator(f"{prefix}_{joint}_position").id for joint in (1, 2, 3)]
                for prefix in LEG_PREFIXES
            ]
        )
        for phase in (0.0, 0.13, 0.49, 0.50, 0.77, 0.99):
            for command in ((0.0, 0.0), (0.08, 0.0), (0.12, 0.25)):
                feet, _ = nominal_foot_targets(
                    origins=origins,
                    outward=outward,
                    tripod_a=tripod_a,
                    phase=jp.array(phase),
                    command=jp.array(command),
                    phase_time=0.5,
                    step_scale=jp.array(1.0),
                    swing_height=jp.array(0.07),
                    radial_offset=jp.array(0.01),
                )
                jax_targets = np.zeros(model.nu)
                jax_targets[actuator_ids] = np.asarray(analytical_ik(feet) * signs)
                numpy_targets = classical.nominal_targets(phase, command)
                self.assertLess(
                    float(np.max(np.abs(jax_targets - numpy_targets))), 1e-4
                )
                zero_action_targets, _ = env._controller_targets(
                    jp.zeros(ACTION_SIZE),
                    jp.array(phase),
                    jp.array(command),
                    jp.array(1.0),
                    jp.ones(6, dtype=jp.bool_),
                    jp.zeros(6, dtype=jp.bool_),
                    jp.zeros((6, 3)),
                )
                self.assertLess(
                    float(
                        np.max(
                            np.abs(np.asarray(zero_action_targets) - numpy_targets)
                        )
                    ),
                    1e-4,
                )

    def test_actual_env_reset_jit_and_full_gait_cycle(self) -> None:
        env = HexapodRoughTerrainEnv(terrain="flat", command_curriculum=True)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        state = reset(jax.random.PRNGKey(7))
        self.assertEqual(state.obs.shape, (OBSERVATION_SIZE,))
        self.assertTrue(bool(jp.all(jp.isfinite(state.obs))))
        for _ in range(50):
            state = step(state, jp.zeros(ACTION_SIZE))
        state.obs.block_until_ready()
        self.assertEqual(state.obs.shape, (OBSERVATION_SIZE,))
        self.assertTrue(bool(jp.all(jp.isfinite(state.obs))))
        phase = float(state.info["phase"])
        self.assertLess(min(phase, 1.0 - phase), 1e-4)

    def test_mixed_terrain_height_families_and_per_env_reset(self) -> None:
        env = HexapodRoughTerrainEnv(terrain="mixed")
        points = jp.array(
            [
                [0.70, 0.0],
                [0.70, 3.0],
                [1.05, 6.0],
                [0.82, 9.20],
                [0.50, 12.0],
                [0.78, 15.0],
            ]
        )
        heights = env._terrain_height(points)
        self.assertAlmostEqual(float(heights[0]), 0.0, places=6)
        self.assertAlmostEqual(float(heights[1]), 0.04, places=6)
        self.assertGreater(float(heights[2]), 0.0)
        self.assertAlmostEqual(float(heights[3]), 0.06, places=6)
        self.assertAlmostEqual(float(heights[4]), 0.035, places=6)
        self.assertAlmostEqual(float(heights[5]), 0.02, places=6)
        patches = {
            int(env.reset(jax.random.PRNGKey(seed)).info["terrain_patch"])
            for seed in range(12)
        }
        self.assertGreaterEqual(len(patches), 3)

    def test_dynamics_randomization_keeps_force_safety_fixed(self) -> None:
        env = HexapodRoughTerrainEnv(terrain="mixed")
        randomized, in_axes = domain_randomize(
            env.mjx_model, jax.random.split(jax.random.PRNGKey(3), 4)
        )
        self.assertIsNone(in_axes.actuator_forcerange)
        self.assertTrue(
            jp.array_equal(
                randomized.actuator_forcerange, env.mjx_model.actuator_forcerange
            )
        )
        self.assertTrue(jp.all(jp.abs(randomized.actuator_gainprm[:, :, 0]) <= 120.0))


if __name__ == "__main__":
    unittest.main()
