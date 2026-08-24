"""Contracts for classical whole-body control plus the bounded 24-D residual."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rough_terrain_env import (  # noqa: E402
    ACTION_CONTRACT_VERSION,
    ACTION_SIZE,
    BODY_RESIDUAL_SIZE,
    FOOT_RESIDUAL_SIZE,
    OBSERVATION_SIZE,
    OBSERVATION_CONTRACT_VERSION,
    HexapodRoughTerrainEnv,
    default_config,
    mujoco_terrain_foot_contacts,
    _quat_rotate,
    _quat_rotate_inverse,
)
from tripod_core import (  # noqa: E402
    LINK1,
    advance_contact_gated_phase,
    apply_body_pose_overlay,
    analytical_ik,
    body_to_leg_local,
    classical_body_twist,
    feasible_yaw_limit,
    heading_aligned_points,
    leg_local_to_body,
    limit_effective_stride,
    median_support_height,
    nominal_foot_targets,
    phase_masked_residual,
    project_workspace,
    posture_pi_candidate,
    scale_asymmetric,
    self_collision_detected,
    smooth_gait_action,
    torque_saturation_cost,
    update_airborne_state,
    workspace_valid,
)
from prepare_rl_scene import (  # noqa: E402
    DRY_ASPHALT_FRICTION,
    FLAT_RL_SCENE_OUTPUT,
    MIXED_PATCH_NAMES,
    prepare_flat_rl_scene,
)
from tripod_controller import (  # noqa: E402
    GaitConfig,
    LEG_PREFIXES,
    RIGHT_LEGS,
    TRIPOD_A,
    TripodGaitController,
)
from domain_randomization import domain_randomize  # noqa: E402
from best_policy_video import (  # noqa: E402
    FOOT_LABELS,
    _ground_truth_controller_rpy,
    make_policy_evaluator,
)
from train_rough_terrain import progress_video_targets  # noqa: E402
from terrain_curriculum import TERRAIN_LEVELS  # noqa: E402
from urdf_kinematics import (  # noqa: E402
    FOOT_COLLISION_RADIUS,
    NOMINAL_FOOT_RADIAL,
    NOMINAL_FOOT_VERTICAL,
    SHOULDER_LATERAL_OFFSET,
    SHOULDER_VERTICAL_OFFSET,
    STAND_ROOT_HEIGHT,
)


SHOULDER_LATERAL = jp.array(
    [SHOULDER_LATERAL_OFFSET] * 3 + [-SHOULDER_LATERAL_OFFSET] * 3
)


class RoughTerrainContractTest(unittest.TestCase):
    def test_progress_video_targets_cover_quarters(self) -> None:
        self.assertEqual(progress_video_targets(5), (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(progress_video_targets(1), (1.0,))
        with self.assertRaises(ValueError):
            progress_video_targets(0)

    def test_fixed_action_and_observation_contract(self) -> None:
        self.assertEqual(FOOT_RESIDUAL_SIZE, 18)
        self.assertEqual(BODY_RESIDUAL_SIZE, 6)
        self.assertEqual(ACTION_SIZE, 24)
        self.assertEqual(OBSERVATION_SIZE, 113)
        self.assertEqual(
            ACTION_CONTRACT_VERSION,
            "classical_wbc_cartesian_body6d_residual_v1",
        )
        self.assertEqual(
            OBSERVATION_CONTRACT_VERSION,
            "gt_attitude_collision_contact6_coarse9_touchdown6_v3",
        )
        config = default_config()
        self.assertEqual(
            config.controller.residual.command.to_dict(),
            config.controller.residual.terrain.to_dict(),
        )
        self.assertEqual(tuple(config.command_curriculum.speed_max), (0.10, 0.18, 0.27))
        self.assertAlmostEqual(config.controller.safety.max_effective_stride, 0.140)
        self.assertAlmostEqual(
            config.controller.residual.body_filter_time_constant, 0.15
        )
        self.assertAlmostEqual(config.controller.nominal.base_swing_height, 0.20)
        self.assertAlmostEqual(config.controller.nominal.base_radial_offset, 0.07)
        self.assertEqual(tuple(config.controller.command.position_kp), (1.0, 1.0))
        self.assertEqual(tuple(config.controller.command.position_ki), (0.0, 0.0))
        self.assertAlmostEqual(config.controller.command.heading_kp, 2.0)
        self.assertAlmostEqual(config.controller.command.heading_ki, 0.0)
        self.assertTrue(
            np.allclose(
                tuple(config.controller.posture.angular_rate_limit),
                (np.pi / 12.0,) * 3,
            )
        )
        self.assertAlmostEqual(config.terrain.flat_friction, DRY_ASPHALT_FRICTION)
        self.assertAlmostEqual(config.reward.torque, -0.020)
        self.assertAlmostEqual(config.reward.torque_saturation, -0.050)
        self.assertAlmostEqual(config.reward.slip, -0.080)

    def test_classical_position_heading_pi_owns_the_gait_twist(self) -> None:
        result = classical_body_twist(
            command_target=jp.array((0.10, 0.04, 0.20)),
            filtered_command=jp.zeros(3),
            desired_position_xy=jp.zeros(2),
            desired_heading=jp.zeros(()),
            position_integral=jp.zeros(2),
            heading_integral=jp.zeros(()),
            applied_twist=jp.zeros(3),
            body_position_xy=jp.array((-0.02, -0.01)),
            body_heading=jp.array(-0.05),
            dt=0.02,
            command_deadzone=0.005,
            command_rate_limit=jp.array((0.9, 0.9, 1.2)),
            position_kp=jp.array((0.5, 0.5)),
            position_ki=jp.array((0.05, 0.05)),
            position_integral_limit=jp.array((0.25, 0.25)),
            position_feedback_limit=jp.array((0.08, 0.08)),
            heading_kp=1.0,
            heading_ki=0.05,
            heading_integral_limit=0.5,
            heading_feedback_limit=0.35,
            twist_limit=jp.array((0.28, 0.28, jp.pi / 4.0)),
            twist_rate_limit=jp.array((0.9, 0.9, 1.2)),
        )
        twist, filtered, desired_position, desired_heading = result[:4]
        self.assertTrue(jp.all(filtered > 0.0))
        self.assertTrue(jp.all(twist > 0.0))
        self.assertGreater(float(jp.linalg.norm(desired_position)), 0.0)
        self.assertGreater(float(desired_heading), 0.0)

        invalid_result = classical_body_twist(
            command_target=jp.array((0.10, 0.04, 0.0)),
            filtered_command=jp.zeros(3),
            desired_position_xy=jp.ones(2),
            desired_heading=jp.zeros(()),
            position_integral=jp.ones(2),
            heading_integral=jp.zeros(()),
            applied_twist=jp.zeros(3),
            body_position_xy=jp.array((0.2, -0.1)),
            body_heading=jp.zeros(()),
            dt=0.02,
            command_deadzone=0.005,
            command_rate_limit=jp.array((0.5, 0.5, 1.57)),
            position_kp=jp.ones(2),
            position_ki=jp.zeros(2),
            position_integral_limit=jp.array((0.2, 0.2)),
            position_feedback_limit=jp.array((0.05, 0.05)),
            heading_kp=2.0,
            heading_ki=0.0,
            heading_integral_limit=0.5,
            heading_feedback_limit=jp.pi / 12.0,
            twist_limit=jp.array((0.28, 0.28, jp.pi / 4.0)),
            twist_rate_limit=jp.array((0.5, 0.5, 1.57)),
            position_valid=jp.array(False),
        )
        self.assertTrue(
            jp.allclose(invalid_result[2], jp.array((0.2, -0.1)))
        )
        self.assertTrue(jp.allclose(invalid_result[4], 0.0))
        self.assertTrue(jp.allclose(invalid_result[6], 0.0))

    def test_full_body_residual_uses_all_six_dofs_before_workspace_gate(self) -> None:
        env = HexapodRoughTerrainEnv(terrain="flat", command_curriculum=True)
        feet, _ = nominal_foot_targets(
            origins=env._origins,
            outward=env._outward,
            shoulder_lateral=env._shoulder_lateral,
            tripod_a=env._tripod_a,
            phase=jp.zeros(()),
            command=jp.zeros(3),
            phase_time=0.5,
            step_scale=jp.ones(()),
            swing_height=jp.array(0.20),
            radial_offset=jp.array(0.07),
        )
        feet_body = leg_local_to_body(feet, env._origins, env._outward)
        pose_translation = jp.array((0.003, -0.003, 0.003))
        pose_rpy = jp.array((0.01, -0.01, 0.01))
        overlaid = apply_body_pose_overlay(
            feet_body, pose_translation, pose_rpy
        )
        overlaid_local = body_to_leg_local(
            overlaid, env._origins, env._outward
        )
        self.assertFalse(bool(jp.allclose(overlaid_local, feet)))
        self.assertTrue(
            bool(
                workspace_valid(
                    overlaid_local,
                    env._shoulder_lateral,
                    min_distance=env._config.controller.safety.workspace_min_distance,
                    max_distance=env._config.controller.safety.workspace_max_distance,
                    joint_limit=env._config.controller.safety.joint_limit,
                )
            )
        )
        posture, integral, error = posture_pi_candidate(
            target_rpy=pose_rpy,
            measured_rpy=jp.zeros(3),
            desired_heading=jp.zeros(()),
            measured_heading=jp.zeros(()),
            posture_integral=jp.zeros(3),
            posture_command=jp.zeros(3),
            dt=0.02,
            kp=jp.array((2.0, 2.0, 1.2)),
            ki=jp.array((0.08, 0.08, 0.04)),
            integral_limit=jp.array((0.5, 0.5, 0.35)),
            angular_rate_limit=jp.array((1.57, 1.57, 1.04)),
            posture_limit=jp.array((jp.pi / 4, jp.pi / 4, jp.pi / 7.2)),
        )
        self.assertTrue(jp.all(jp.abs(posture) > 0.0))
        self.assertTrue(jp.all(jp.abs(integral) > 0.0))
        self.assertTrue(jp.allclose(error, pose_rpy))

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

        adapted, early, searched = contact_adapt_targets(
            requested,
            current,
            jp.array([True, False, True, False, True, False]),
            jp.array([True, True, False, True, False, True]),
            airborne,
            lost_contact_search=0.010,
            lost_contact_inward=0.008,
            early_landing_allowed=jp.array(False),
            late_landing=jp.array([False, False, True, False, False, False]),
        )
        self.assertFalse(bool(early[0]))
        self.assertTrue(bool(searched[2]))
        self.assertTrue(
            jp.allclose(adapted[2], current[2] + jp.array((-0.008, 0.0, -0.010)))
        )

    def test_tripod_phase_waits_for_all_airborne_swing_feet_to_land(self) -> None:
        swing = jp.array([True, False, True, False, True, False])
        airborne = swing
        held_phase, late, completed = advance_contact_gated_phase(
            phase=jp.array(0.49),
            gait_enabled=jp.array(True),
            swing=swing,
            contacts=jp.array([True, True, False, True, True, True]),
            airborne=airborne,
            dt=0.02,
            phase_time=0.5,
        )
        self.assertTrue(bool(late))
        self.assertFalse(bool(completed))
        self.assertLess(float(held_phase), 0.5)
        self.assertGreater(float(held_phase), 0.499)

        next_phase, late, completed = advance_contact_gated_phase(
            phase=held_phase,
            gait_enabled=jp.array(True),
            swing=swing,
            contacts=jp.ones(6, dtype=jp.bool_),
            airborne=airborne,
            dt=0.02,
            phase_time=0.5,
        )
        self.assertFalse(bool(late))
        self.assertTrue(bool(completed))
        self.assertAlmostEqual(float(next_phase), 0.5, places=6)

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

    def test_contact_uses_only_active_foot_to_world_collisions(self) -> None:
        geom_body_ids = jp.ones(20, dtype=jp.int32)
        geom_body_ids = geom_body_ids.at[2].set(0).at[3].set(0)
        contacts = mujoco_terrain_foot_contacts(
            contact_geom=jp.array(
                ((10, 2), (11, 12), (3, 12), (13, 2), (-1, -1))
            ),
            contact_distance=jp.array((-0.001, -0.001, 0.0, 0.001, -1.0)),
            foot_geom_ids=jp.array((10, 11, 12, 13, 14, 15)),
            geom_body_ids=geom_body_ids,
        )
        self.assertTrue(
            jp.array_equal(
                contacts, jp.array((True, False, True, False, False, False))
            )
        )

    def test_video_attitude_is_ground_truth_and_scene_has_no_imu_sensor(self) -> None:
        env = HexapodRoughTerrainEnv(terrain="flat")
        self.assertEqual(FOOT_LABELS, LEG_PREFIXES)
        self.assertEqual(env.mj_model.nsensor, 0)
        self.assertEqual(
            mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu"), -1
        )
        home_quat = np.asarray(env._home_qpos[3:7])
        self.assertTrue(
            np.allclose(_ground_truth_controller_rpy(home_quat), np.zeros(3), atol=1e-6)
        )

    def test_workspace_projection_makes_extreme_targets_feasible(self) -> None:
        requested = jp.array(
            [[1.0, 0.0, -1.0], [0.001, 0.0, 0.0], [0.22, 0.02, -0.29]]
        )
        shoulder_lateral = jp.array(
            [SHOULDER_LATERAL_OFFSET, -SHOULDER_LATERAL_OFFSET, 0.0]
        )
        projected, cost = project_workspace(requested, shoulder_lateral)
        radial = jp.linalg.norm(projected[:, :2], axis=-1)
        planar_radius = jp.sqrt(
            jp.maximum(radial**2 - shoulder_lateral**2, 0.0)
        )
        distance = jp.sqrt(
            jp.square(planar_radius - LINK1)
            + jp.square(projected[:, 2] - SHOULDER_VERTICAL_OFFSET)
        )
        self.assertTrue(jp.all(distance >= 0.112 - 1e-6))
        self.assertTrue(jp.all(distance <= 0.345 + 1e-6))
        self.assertGreater(float(cost), 0.0)

    def test_effective_stride_caps_forward_and_yaw_stroke(self) -> None:
        env = HexapodRoughTerrainEnv(terrain="flat", command_curriculum=True)
        for command in (
            jp.array((0.27, 0.0, 0.0)),
            jp.array((0.27, 0.0, 0.35)),
        ):
            applied_scale, effective_stride = limit_effective_stride(
                requested_scale=jp.array(1.2),
                command=command,
                origins=env._origins,
                outward=env._outward,
                shoulder_lateral=env._shoulder_lateral,
                phase_time=0.5,
                max_stride=0.140,
            )
            self.assertLessEqual(float(effective_stride), 0.140001)
            self.assertLess(float(applied_scale), 1.2)
        straight_scale, straight_stride = limit_effective_stride(
            requested_scale=jp.array(1.2),
            command=jp.array((0.27, 0.0, 0.0)),
            origins=env._origins,
            outward=env._outward,
            shoulder_lateral=env._shoulder_lateral,
            phase_time=0.5,
            max_stride=0.140,
        )
        self.assertAlmostEqual(float(straight_stride), 0.140, places=6)
        self.assertAlmostEqual(float(straight_scale), 0.140 / 0.135, places=5)

    def test_speed_curriculum_limits_yaw_to_reachable_gait_envelope(self) -> None:
        env = HexapodRoughTerrainEnv(terrain="flat", command_curriculum=True)
        capacity = 0.140 / 0.5
        high_speed_limit = feasible_yaw_limit(
            speed=jp.array(0.27),
            requested_yaw_limit=jp.array(0.35),
            origins=env._origins,
            outward=env._outward,
            shoulder_lateral=env._shoulder_lateral,
            phase_time=0.5,
            max_stride=0.140,
            max_frequency_scale=1.0,
        )
        self.assertGreater(float(high_speed_limit), 0.0)
        self.assertLess(float(high_speed_limit), 0.35)
        for seed in range(20):
            command = env._sample_command(
                jax.random.PRNGKey(seed), jp.array(2, dtype=jp.int32)
            )
            tangent = jp.stack(
                (-env._outward[:, 1], env._outward[:, 0], jp.zeros(6)), axis=-1
            )
            nominal_body = (
                env._origins
                + env._outward * NOMINAL_FOOT_RADIAL
                + tangent * env._shoulder_lateral[:, None]
            )
            nominal_body = nominal_body.at[:, 2].set(
                env._origins[:, 2] + NOMINAL_FOOT_VERTICAL
            )
            yaw_velocity = jp.cross(
                jp.tile(jp.array((0.0, 0.0, command[2])), (6, 1)), nominal_body
            )
            foot_velocity = jp.array((command[1], -command[0], 0.0)) + yaw_velocity
            peak = jp.max(jp.linalg.norm(foot_velocity[:, :2], axis=-1))
            self.assertLessEqual(float(command[0]), 0.270001)
            self.assertLessEqual(float(peak), capacity + 1e-5)

    def test_self_collision_and_torque_saturation_costs(self) -> None:
        geom_body_ids = jp.array((0, 1, 2, 3))
        self.assertTrue(
            bool(
                self_collision_detected(
                    jp.array(((1, 2), (1, 0))),
                    jp.array((-0.01, -0.01)),
                    geom_body_ids,
                )
            )
        )
        self.assertFalse(
            bool(
                self_collision_detected(
                    jp.array(((1, 0), (2, 0))),
                    jp.array((-0.01, -0.01)),
                    geom_body_ids,
                )
            )
        )
        saturation = torque_saturation_cost(
            jp.array((0.0, 6.8, 8.0)), force_limit=8.0
        )
        self.assertAlmostEqual(float(saturation), 1.0 / 3.0, places=6)

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
        compiled = jax.jit(
            lambda feet: project_workspace(feet, SHOULDER_LATERAL)[0]
        )
        output = compiled(jp.ones((6, 3)) * jp.array([0.22, 0.0, -0.28]))
        self.assertEqual(output.shape, (6, 3))

    def test_global_gait_action_is_smoothed_but_converges(self) -> None:
        previous = jp.zeros(4)
        requested = jp.ones(4)
        first = smooth_gait_action(
            previous, requested, control_dt=0.02, time_constant=0.15
        )
        self.assertTrue(jp.all(first > 0.0))
        self.assertTrue(jp.all(first < 0.2))
        applied = previous
        for _ in range(75):
            applied = smooth_gait_action(
                applied, requested, control_dt=0.02, time_constant=0.15
            )
        self.assertTrue(jp.allclose(applied, requested, atol=1e-4))
        immediate = smooth_gait_action(
            previous, requested, control_dt=0.02, time_constant=0.0
        )
        self.assertTrue(jp.array_equal(immediate, requested))

    def test_flat_scene_uses_dry_asphalt_contact_friction(self) -> None:
        prepare_flat_rl_scene()
        env = HexapodRoughTerrainEnv(terrain="flat", command_curriculum=True)
        model = env.mj_model
        self.assertAlmostEqual(
            float(model.geom("floor").friction[0]), DRY_ASPHALT_FRICTION
        )
        for prefix in LEG_PREFIXES:
            self.assertAlmostEqual(
                float(model.geom(f"{prefix}_foot_collision").friction[0]),
                DRY_ASPHALT_FRICTION,
            )

    def test_zero_action_numpy_classical_matches_jax_nominal(self) -> None:
        prepare_flat_rl_scene()
        env_config = default_config()
        env = HexapodRoughTerrainEnv(
            config=env_config, terrain="flat", command_curriculum=True
        )
        model = env.mj_model
        config = GaitConfig(
            control_dt=0.02,
            phase_time=0.5,
            swing_height=0.20,
            radial_offset=0.07,
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
            for command in ((0.08, 0.0), (0.12, 0.25)):
                feet, _ = nominal_foot_targets(
                    origins=origins,
                    outward=outward,
                    shoulder_lateral=SHOULDER_LATERAL,
                    tripod_a=tripod_a,
                    phase=jp.array(phase),
                    command=jp.array(command),
                    phase_time=0.5,
                    step_scale=jp.array(1.0),
                    swing_height=jp.array(0.20),
                    radial_offset=jp.array(0.07),
                )
                jax_targets = np.zeros(model.nu)
                jax_targets[actuator_ids] = np.asarray(
                    analytical_ik(feet, SHOULDER_LATERAL) * signs
                )
                numpy_targets = classical.nominal_targets(phase, command)
                self.assertLess(
                    float(np.max(np.abs(jax_targets - numpy_targets))), 1e-4
                )
                zero_action_targets, _ = env._controller_targets(
                    jp.zeros(ACTION_SIZE),
                    jp.array(phase),
                    jp.array((command[0], 0.0, command[1])),
                    jp.array(1.0),
                    jp.ones(6, dtype=jp.bool_),
                    jp.zeros(6, dtype=jp.bool_),
                    jp.zeros((6, 3)),
                    jp.zeros(BODY_RESIDUAL_SIZE),
                    jp.zeros(3),
                    jp.zeros(BODY_RESIDUAL_SIZE),
                )
                self.assertLess(
                    float(
                        np.max(
                            np.abs(np.asarray(zero_action_targets) - numpy_targets)
                        )
                    ),
                    1e-4,
                )

    def test_ik_matches_the_converted_urdf_fk_and_home_ground_clearance(self) -> None:
        """Guard against replacing the CAD chain with an idealized 3-link leg."""
        prepare_flat_rl_scene()
        model = mujoco.MjModel.from_xml_path(str(FLAT_RL_SCENE_OUTPUT))
        data = mujoco.MjData(model)
        root_id = model.body("hexapod").id
        lateral = np.asarray(SHOULDER_LATERAL)
        signs = np.asarray(
            [
                (1.0, -1.0, 1.0) if prefix in RIGHT_LEGS else (1.0, 1.0, -1.0)
                for prefix in LEG_PREFIXES
            ]
        )

        for servo_pose in (
            np.array((0.0, np.pi / 6.0, 5.0 * np.pi / 18.0)),
            np.array((0.12, 0.40, 0.70)),
        ):
            data.qpos[:] = model.key("home").qpos
            data.qpos[2] = STAND_ROOT_HEIGHT
            for leg_index, prefix in enumerate(LEG_PREFIXES):
                raw = servo_pose * signs[leg_index]
                for joint_number, value in enumerate(raw, start=1):
                    joint_id = model.joint(f"{prefix}_{joint_number}").id
                    data.qpos[model.jnt_qposadr[joint_id]] = value
            mujoco.mj_forward(model, data)

            root_pos = data.xpos[root_id]
            root_rot = data.xmat[root_id].reshape(3, 3)
            feet_local = []
            for prefix in LEG_PREFIXES:
                hip_id = model.body(f"{prefix}_motor_horn_1_1").id
                site_id = model.site(f"{prefix}_foot_site").id
                hip = root_rot.T @ (data.xpos[hip_id] - root_pos)
                foot = root_rot.T @ (data.site_xpos[site_id] - root_pos)
                outward = hip.copy()
                outward[2] = 0.0
                outward /= np.linalg.norm(outward)
                tangent = np.array((-outward[1], outward[0], 0.0))
                relative = foot - hip
                feet_local.append(
                    (relative @ outward, relative @ tangent, relative[2])
                )

            recovered = np.asarray(
                analytical_ik(jp.asarray(feet_local), jp.asarray(lateral))
            )
            self.assertTrue(np.allclose(recovered, servo_pose[None, :], atol=2e-5))

        # The home keyframe places the 32 mm proxy sphere on, not through, z=0.
        data.qpos[:] = model.key("home").qpos
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(float(data.qpos[2]), STAND_ROOT_HEIGHT, places=7)
        for prefix in LEG_PREFIXES:
            geom_id = model.geom(f"{prefix}_foot_collision").id
            bottom = data.geom_xpos[geom_id, 2] - FOOT_COLLISION_RADIUS
            self.assertAlmostEqual(float(bottom), 0.0, places=5)

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
        # Tripod_Enable waits for the rate-limited body twist and each half
        # phase waits for all airborne swing feet to land.  A small contact-
        # dependent delay relative to the ideal one-second clock is expected.
        self.assertLess(min(phase, 1.0 - phase), 0.06)

    def test_fixed_stage_script_is_deterministic_and_independent(self) -> None:
        stage1 = HexapodRoughTerrainEnv(
            terrain="flat",
            command_curriculum=True,
            fixed_curriculum_stage=1,
            scripted_commands=True,
        )
        self.assertEqual(int(stage1._curriculum_stage(jp.array(0))), 1)
        self.assertEqual(int(stage1._curriculum_stage(jp.array(999))), 1)
        self.assertTrue(
            jp.allclose(
                stage1._scripted_command(jp.array(0), jp.array(1)),
                jp.array((0.12, 0.00, 0.12)),
            )
        )
        self.assertTrue(
            jp.allclose(
                stage1._scripted_command(jp.array(160), jp.array(1)),
                jp.array((0.12, 0.04, -0.12)),
            )
        )

        stage2 = HexapodRoughTerrainEnv(
            terrain="flat",
            command_curriculum=True,
            fixed_curriculum_stage=2,
            scripted_commands=True,
        )
        expected = (
            (0, (0.10, 0.00, 0.00)),
            (150, (0.14, 0.08, 0.30)),
            (300, (0.14, -0.08, -0.30)),
            (450, (0.27, 0.00, 0.00)),
        )
        for steps, command in expected:
            self.assertTrue(
                jp.allclose(
                    stage2._scripted_command(jp.array(steps), jp.array(2)),
                    jp.array(command),
                )
            )

    def test_scripted_stage_policy_evaluator_reports_tracking_metrics(self) -> None:
        env = HexapodRoughTerrainEnv(
            terrain="flat",
            command_curriculum=True,
            fixed_curriculum_stage=0,
            scripted_commands=True,
        )

        def make_policy(params, deterministic=False):
            del params, deterministic

            def policy(obs, key):
                del key
                return jp.zeros(obs.shape[:-1] + (ACTION_SIZE,)), {}

            return policy

        evaluator = make_policy_evaluator(
            env=env, make_policy=make_policy, duration=0.04, num_envs=2, seed=9
        )
        metrics = evaluator(None)
        self.assertEqual(
            set(metrics),
            {
                "reward_mean",
                "velocity_error_mps",
                "yaw_error_rps",
                "survival_fraction",
                "torque_rms_nm",
                "torque_saturation_mean",
                "self_collision_rate",
                "effective_stride_mean_m",
                "effective_stride_max_m",
                "gait_step_scale_mean",
                "gait_frequency_scale_mean",
                "gait_swing_height_mean_m",
                "gait_radial_offset_mean_m",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreater(metrics["survival_fraction"], 0.0)

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
        self.assertAlmostEqual(float(heights[4]), 0.05, places=6)
        self.assertAlmostEqual(float(heights[5]), 0.02, places=6)
        patches = {
            int(env.reset(jax.random.PRNGKey(seed)).info["terrain_patch"])
            for seed in range(12)
        }
        self.assertGreaterEqual(len(patches), 3)

    def test_mixed_curriculum_reaches_twenty_centimeter_total_stair_rise(self) -> None:
        self.assertEqual(
            [spec.stair_total_rise_range_m[1] for spec in TERRAIN_LEVELS],
            [0.04, 0.08, 0.12, 0.16, 0.20],
        )
        self.assertTrue(
            all(
                abs(sum(spec.patch_probabilities) - 1.0) < 1e-9
                for spec in TERRAIN_LEVELS
            )
        )
        config = default_config()
        config.terrain.stair_total_rise = 0.20
        config.terrain.step_height = 0.20 / 6.0
        config.terrain.step_depth = 0.28
        config.terrain.ramp_rise = 0.24
        env = HexapodRoughTerrainEnv(config=config, terrain="mixed")
        heights = env._terrain_height(
            jp.array([[0.50, 12.0], [1.90, 12.0], [1.65, 6.0]])
        )
        self.assertAlmostEqual(float(heights[0]), 0.20 / 6.0, places=6)
        self.assertAlmostEqual(float(heights[1]), 0.20, places=6)
        self.assertAlmostEqual(float(heights[2]), 0.24, places=6)
        first_stair = env.mj_model.geom("mixed_stair_1")
        top_stair = env.mj_model.geom("mixed_stair_6")
        self.assertAlmostEqual(float(first_stair.pos[2]), 0.20 / 12.0, places=6)
        self.assertAlmostEqual(float(first_stair.size[2]), 0.20 / 12.0, places=6)
        self.assertAlmostEqual(float(top_stair.pos[2]), 0.10, places=6)
        self.assertAlmostEqual(float(top_stair.size[2]), 0.10, places=6)

    def test_fixed_stair_video_reset_is_straight_and_reproducible(self) -> None:
        env = HexapodRoughTerrainEnv(
            terrain="mixed",
            command_curriculum=True,
            fixed_curriculum_stage=0,
            scripted_commands=True,
            fixed_terrain_patch=MIXED_PATCH_NAMES.index("stairs"),
        )
        state = env.reset(jax.random.PRNGKey(123))
        self.assertEqual(
            int(state.info["terrain_patch"]), MIXED_PATCH_NAMES.index("stairs")
        )
        self.assertTrue(
            jp.allclose(state.info["command"], jp.array((0.08, 0.0, 0.0)))
        )

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
