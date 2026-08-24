"""MJX terrain environments with full classical control plus 6-DOF residuals."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

from prepare_rl_scene import (
    DRY_ASPHALT_FRICTION,
    FLAT_RL_SCENE_OUTPUT,
    MIXED_BLOCKS,
    MIXED_CURB,
    MIXED_LANE_HALF_WIDTH,
    MIXED_PATCH_NAMES,
    MIXED_PATCH_Y,
    MIXED_RAMP,
    MIXED_RL_SCENE_OUTPUT,
    MIXED_ROUGH,
    MIXED_STAIR_COUNT,
    MIXED_STAIR_START_X,
    RL_SCENE_OUTPUT,
    STEP_COUNT,
    STEP_DEPTH,
    STEP_HEIGHT,
    STEP_START_X,
    prepare_flat_rl_scene,
    prepare_rl_scene,
)
from tripod_controller import LEG_PREFIXES, RIGHT_LEGS, TRIPOD_A
from tripod_core import (
    MODEL_LATERAL,
    MODEL_FORWARD,
    advance_contact_gated_phase,
    apply_body_pose_overlay,
    analytical_ik,
    body_to_leg_local,
    classical_body_twist,
    contact_adapt_targets,
    feasible_yaw_limit,
    heading_aligned_points,
    hysteretic_clearance_contact,
    leg_local_to_body,
    limit_effective_stride,
    median_support_height,
    nominal_foot_targets,
    nominal_touchdown_body_targets,
    phase_masks,
    phase_masked_residual,
    posture_pi_candidate,
    project_workspace,
    self_collision_detected,
    smooth_gait_action,
    torque_saturation_cost,
    update_airborne_state,
    workspace_valid,
    wrap_angle,
)
from urdf_kinematics import NOMINAL_FOOT_RADIAL, shoulder_lateral_offset


FOOT_RESIDUAL_SIZE = 18
BODY_RESIDUAL_SIZE = 6
ACTION_SIZE = FOOT_RESIDUAL_SIZE + BODY_RESIDUAL_SIZE
OBSERVATION_SIZE = 113
ACTION_CONTRACT_VERSION = "classical_wbc_cartesian_body6d_residual_v1"
OBSERVATION_CONTRACT_VERSION = "body_state_command3_coarse9_touchdown6_v2"


def default_config() -> config_dict.ConfigDict:
    """Return the v2 contract, grouped by ownership instead of implementation."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.0025,
        episode_length=1000,
        impl="jax",
        nconmax=256,
        njmax=128,
        command_min_speed=0.03,
        command_max_speed=0.18,
        command_max_lateral_speed=0.10,
        command_max_yaw_rate=0.35,
        command_curriculum=config_dict.create(
            forward_only_steps=250,
            limited_yaw_steps=250,
            speed_min=(0.03, 0.05, 0.03),
            speed_max=(0.10, 0.18, 0.27),
            lateral_limit=(0.00, 0.05, 0.10),
            yaw_limit=(0.00, 0.15, 0.35),
            resample_seconds=(1.5, 4.0),
        ),
        controller=config_dict.create(
            nominal=config_dict.create(
                phase_time=0.5,
                # Values from the updated source controller.  The policy no
                # longer owns these gait parameters.
                base_swing_height=0.20,
                base_radial_offset=0.07,
                gait_enable_speed=0.005,
                gait_enable_yaw_rate=0.0174532925,
            ),
            command=config_dict.create(
                deadzone=0.005,
                rate_limit=(0.5, 0.5, 1.5707963268),
                twist_limit=(0.28, 0.28, 0.7853981633974483),
                twist_rate_limit=(0.5, 0.5, 1.5707963268),
                position_kp=(1.0, 1.0),
                position_ki=(0.0, 0.0),
                position_integral_limit=(0.20, 0.20),
                position_feedback_limit=(0.05, 0.05),
                heading_kp=2.0,
                heading_ki=0.0,
                heading_integral_limit=0.50,
                heading_feedback_limit=0.2617993878,
            ),
            residual=config_dict.create(
                body_filter_time_constant=0.15,
                command=config_dict.create(
                    swing_x=0.025,
                    swing_y=0.015,
                    swing_z_low=-0.010,
                    swing_z_high=0.050,
                    stance_z=0.008,
                    body_translation_limit=(0.050, 0.050, 0.100),
                    body_rotation_limit=(0.7853981634, 0.7853981634, 0.4363323130),
                ),
                terrain=config_dict.create(
                    swing_x=0.025,
                    swing_y=0.015,
                    swing_z_low=-0.010,
                    swing_z_high=0.050,
                    stance_z=0.008,
                    body_translation_limit=(0.050, 0.050, 0.100),
                    body_rotation_limit=(0.7853981634, 0.7853981634, 0.4363323130),
                ),
            ),
            posture=config_dict.create(
                kp=(2.0, 2.0, 2.0),
                ki=(0.0, 0.0, 0.0),
                integral_limit=(0.50, 0.50, 0.50),
                angular_rate_limit=(0.2617993878, 0.2617993878, 0.2617993878),
                angle_limit=(0.7853981634, 0.7853981634, 0.4363323130),
            ),
            estimator=config_dict.create(
                slip_distance=0.05,
                slip_confirm_steps=5,
            ),
            contact=config_dict.create(
                contact_enter_clearance=0.035,
                contact_release_clearance=0.045,
                search_down_speed=0.20,
                search_inward_ratio=0.8,
            ),
            safety=config_dict.create(
                workspace_min_distance=0.112,
                workspace_max_distance=0.345,
                projection_reference=0.050,
                joint_limit=2.356194,
                max_joint_speed=5.511455,
                max_effective_stride=0.140,
                actuator_force_limit=8.0,
                rollover_angle=1.3962634016,
                max_joint_jump=0.7853981634,
            ),
        ),
        observation=config_dict.create(
            # Nine coarse heading-aligned samples plus six nominal touchdown
            # heights keep the terrain observation at 15 dimensions.
            forward_offsets=(0.05, 0.35, 0.65),
            lateral_offsets=(-0.22, 0.0, 0.22),
        ),
        terrain=config_dict.create(
            step_start_x=STEP_START_X,
            step_depth=STEP_DEPTH,
            step_height=STEP_HEIGHT,
            step_count=STEP_COUNT,
            stair_total_rise=min(0.20, STEP_HEIGHT * STEP_COUNT),
            ramp_rise=MIXED_RAMP[2],
            friction=1.0,
            flat_friction=DRY_ASPHALT_FRICTION,
            patch_probabilities=(0.10, 0.15, 0.20, 0.20, 0.20, 0.15),
        ),
        # Ranges are consumed by the optional per-env randomization wrapper.
        randomization=config_dict.create(
            enabled=False,
            stair_total_rise_range=(0.020, 0.200),
            step_depth_range=(0.180, 0.350),
            friction_range=(0.6, 1.3),
            mass_scale_range=(0.90, 1.10),
            actuator_strength_range=(0.85, 1.00),
            joint_damping_scale_range=(0.80, 1.20),
            observation_noise=False,
            latency_steps=(0, 2),
            mass_scale=1.0,
            actuator_strength_scale=1.0,
            joint_damping_scale=1.0,
        ),
        reward=config_dict.create(
            velocity=2.0,
            yaw=0.5,
            upright=0.5,
            height=0.3,
            progress=0.2,
            # Flat command training strongly prefers the nominal controller.
            # Terrain levels relax these weights without changing action scale.
            swing_residual=-0.040,
            stance_residual=-0.120,
            body_translation_residual=-0.080,
            body_rotation_residual=-0.040,
            foot_action_rate=-0.005,
            body_action_rate=-0.020,
            vertical_velocity=-0.10,
            lateral_velocity=-0.10,
            joint_velocity=-0.002,
            torque=-0.020,
            torque_saturation=-0.050,
            slip=-0.080,
            projection=-0.50,
            body_contact=-1.0,
            self_collision=-1.0,
            termination=-2.0,
        ),
    )


def _quat_rotate(quat: jax.Array, vector: jax.Array) -> jax.Array:
    scalar = quat[0]
    xyz = quat[1:]
    temp = 2.0 * jp.cross(xyz, vector)
    return vector + scalar * temp + jp.cross(xyz, temp)


def _quat_rotate_inverse(quat: jax.Array, vector: jax.Array) -> jax.Array:
    return _quat_rotate(quat * jp.array((1.0, -1.0, -1.0, -1.0)), vector)


def _quat_to_matrix(quat: jax.Array) -> jax.Array:
    """MuJoCo wxyz quaternion to a body-to-world rotation matrix."""
    w, x, y, z = quat
    return jp.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def _body_heading(quat: jax.Array) -> jax.Array:
    forward_world = _quat_rotate(quat, MODEL_FORWARD)
    return jp.arctan2(forward_world[1], forward_world[0])


def _controller_roll_pitch(quat: jax.Array) -> jax.Array:
    """Measured roll/pitch in the controller frame, not the rotated CAD frame."""
    basis = jp.stack(
        (MODEL_FORWARD, MODEL_LATERAL, jp.array((0.0, 0.0, 1.0))), axis=-1
    )
    controller_rotation = basis.T @ _quat_to_matrix(quat) @ basis
    pitch = jp.arcsin(jp.clip(-controller_rotation[2, 0], -1.0, 1.0))
    roll = jp.arctan2(controller_rotation[2, 1], controller_rotation[2, 2])
    return jp.array((roll, pitch))


class HexapodRoughTerrainEnv(mjx_env.MjxEnv):
    """Classical controller owns locomotion; RL supplies bounded corrections.

    ``action[0:18]`` is swing XYZ / stance Z-only foot residual.  The final six
    terms request body forward/lateral/height plus roll/pitch/yaw.  They pass
    through a single posture PI and a whole-body workspace acceptance gate;
    policy output never changes gait phase, stride, frequency, swing height,
    contact handling, IK, or joint protection.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[dict[str, Any]] = None,
        *,
        terrain: str = "mixed",
        command_curriculum: bool = False,
        fixed_curriculum_stage: Optional[int] = None,
        scripted_commands: bool = False,
        fixed_terrain_patch: Optional[int] = None,
    ) -> None:
        if terrain not in {"flat", "stairs", "mixed"}:
            raise ValueError(
                f"terrain must be 'flat', 'stairs', or 'mixed', got {terrain!r}"
            )
        self._terrain = terrain
        self._command_curriculum = command_curriculum
        if fixed_curriculum_stage is not None and fixed_curriculum_stage not in (0, 1, 2):
            raise ValueError("fixed_curriculum_stage must be 0, 1, 2, or None")
        if (fixed_curriculum_stage is not None or scripted_commands) and not command_curriculum:
            raise ValueError(
                "fixed/scripted curriculum evaluation requires command_curriculum=True"
            )
        if fixed_terrain_patch is not None and (
            terrain != "mixed" or not 0 <= fixed_terrain_patch < len(MIXED_PATCH_NAMES)
        ):
            raise ValueError(
                "fixed_terrain_patch requires mixed terrain and a valid patch index"
            )
        self._fixed_curriculum_stage = fixed_curriculum_stage
        self._scripted_commands = scripted_commands
        self._fixed_terrain_patch = fixed_terrain_patch
        super().__init__(config, config_overrides)
        if terrain == "stairs":
            prepare_rl_scene(
                RL_SCENE_OUTPUT,
                terrain="stairs",
                step_start_x=self._config.terrain.step_start_x,
                step_depth=self._config.terrain.step_depth,
                step_height=self._config.terrain.step_height,
                step_count=self._config.terrain.step_count,
                friction=self._config.terrain.friction,
            )
            scene_path = RL_SCENE_OUTPUT
        elif terrain == "mixed":
            prepare_rl_scene(
                MIXED_RL_SCENE_OUTPUT,
                terrain="mixed",
                step_depth=self._config.terrain.step_depth,
                step_height=self._config.terrain.step_height,
                ramp_rise=self._config.terrain.ramp_rise,
                friction=self._config.terrain.friction,
            )
            scene_path = MIXED_RL_SCENE_OUTPUT
        else:
            prepare_flat_rl_scene(
                FLAT_RL_SCENE_OUTPUT, friction=self._config.terrain.flat_friction
            )
            scene_path = FLAT_RL_SCENE_OUTPUT
        self._xml_path = str(scene_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        # Optional base-model scales remain reproducible in metadata.  Level-4
        # training adds per-env model randomization in the Brax wrapper.
        randomization = self._config.randomization
        self._mj_model.body_mass[1:] *= randomization.mass_scale
        self._mj_model.body_inertia[1:] *= randomization.mass_scale
        # Motor uncertainty changes the realized position-servo response, not
        # the hardware safety envelope.  Keep every actuator hard-clamped at
        # ±8 Nm while scaling its kp/kv realization together.
        force_limit = float(self._config.controller.safety.actuator_force_limit)
        self._mj_model.actuator_forcerange[:, 0] = -force_limit
        self._mj_model.actuator_forcerange[:, 1] = force_limit
        self._mj_model.actuator_gainprm[:, 0] *= randomization.actuator_strength_scale
        self._mj_model.actuator_biasprm[:, 1:3] *= randomization.actuator_strength_scale
        self._mj_model.dof_damping *= randomization.joint_damping_scale
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        home_id = self._mj_model.key("home").id
        self._home_qpos = jp.array(self._mj_model.key_qpos[home_id])
        self._home_ctrl = jp.array(self._mj_model.key_ctrl[home_id])
        self._joint_qpos_ids = jp.array(
            [
                self._mj_model.jnt_qposadr[self._mj_model.joint(f"{prefix}_{joint}").id]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._joint_qvel_ids = jp.array(
            [
                self._mj_model.jnt_dofadr[self._mj_model.joint(f"{prefix}_{joint}").id]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._actuator_ids = jp.array(
            [
                [self._mj_model.actuator(f"{prefix}_{joint}_position").id for joint in (1, 2, 3)]
                for prefix in LEG_PREFIXES
            ]
        )
        self._foot_site_ids = jp.array(
            [self._mj_model.site(f"{prefix}_foot_site").id for prefix in LEG_PREFIXES]
        )
        self._foot_geom_ids = jp.array(
            [self._mj_model.geom(f"{prefix}_foot_collision").id for prefix in LEG_PREFIXES]
        )
        self._torso_geom_id = self._mj_model.geom("torso_collision").id
        self._geom_body_ids = jp.array(self._mj_model.geom_bodyid)

        origins, outward, raw_signs, tripod_a, shoulder_lateral = [], [], [], [], []
        for prefix in LEG_PREFIXES:
            body_id = self._mj_model.body(f"{prefix}_motor_horn_1_1").id
            origin = np.asarray(self._mj_model.body_pos[body_id], dtype=float)
            direction = origin.copy()
            direction[2] = 0.0
            direction /= np.linalg.norm(direction)
            origins.append(origin)
            outward.append(direction)
            raw_signs.append((1.0, -1.0, 1.0) if prefix in RIGHT_LEGS else (1.0, 1.0, -1.0))
            tripod_a.append(prefix in TRIPOD_A)
            shoulder_lateral.append(
                shoulder_lateral_offset(right=prefix in RIGHT_LEGS)
            )
        self._origins = jp.array(origins)
        self._outward = jp.array(outward)
        self._raw_signs = jp.array(raw_signs)
        self._tripod_a = jp.array(tripod_a)
        self._shoulder_lateral = jp.array(shoulder_lateral)

        terrain_config = self._config.terrain
        self._step_centers = jp.array(
            [terrain_config.step_start_x + terrain_config.step_depth * i for i in range(terrain_config.step_count)]
        )
        self._step_heights = jp.array(
            [terrain_config.step_height * (i + 1) for i in range(terrain_config.step_count)]
        )
        self._mixed_patch_y = jp.array(MIXED_PATCH_Y)
        self._mixed_blocks = jp.array(MIXED_BLOCKS)
        self._mixed_rough = jp.array(MIXED_ROUGH)
        forward, lateral = np.meshgrid(
            np.asarray(self._config.observation.forward_offsets),
            np.asarray(self._config.observation.lateral_offsets),
        )
        self._terrain_offsets = jp.array(np.stack((forward.ravel(), lateral.ravel()), axis=-1))

    def _authority(self) -> config_dict.ConfigDict:
        return self._config.controller.residual.command if self._terrain == "flat" else self._config.controller.residual.terrain

    def _terrain_height(self, xy: jax.Array) -> jax.Array:
        if self._terrain == "flat":
            return jp.zeros(xy.shape[:-1])
        if self._terrain == "mixed":
            x = xy[..., 0]
            y = xy[..., 1]
            height = jp.zeros(x.shape)

            curb_start, curb_length, curb_height = MIXED_CURB
            curb_inside = (
                (x >= curb_start)
                & (x <= curb_start + curb_length)
                & (jp.abs(y - MIXED_PATCH_Y[1]) <= MIXED_LANE_HALF_WIDTH)
            )
            height = jp.maximum(height, jp.where(curb_inside, curb_height, 0.0))

            ramp_start, ramp_length, _ = MIXED_RAMP
            ramp_rise = self._config.terrain.ramp_rise
            ramp_lane = jp.abs(y - MIXED_PATCH_Y[2]) <= MIXED_LANE_HALF_WIDTH
            ramp_surface = jp.where(
                x < ramp_start,
                0.0,
                jp.where(
                    x <= ramp_start + ramp_length,
                    (x - ramp_start) * ramp_rise / ramp_length,
                    jp.where(x <= ramp_start + ramp_length + 0.9, ramp_rise, 0.0),
                ),
            )
            height = jp.maximum(height, jp.where(ramp_lane, ramp_surface, 0.0))

            def box_family(boxes: jax.Array, lane_y: float) -> jax.Array:
                bx = boxes[:, 0]
                by = boxes[:, 1] + lane_y
                length = boxes[:, 2]
                width = boxes[:, 3]
                box_height = boxes[:, 4]
                inside = (
                    (jp.abs(x[..., None] - bx) <= length / 2.0)
                    & (jp.abs(y[..., None] - by) <= width / 2.0)
                )
                return jp.max(jp.where(inside, box_height, 0.0), axis=-1)

            height = jp.maximum(
                height, box_family(self._mixed_blocks, MIXED_PATCH_Y[3])
            )
            stair_x = x[..., None]
            stair_centers = jp.array(
                [
                    MIXED_STAIR_START_X + self._config.terrain.step_depth * i
                    for i in range(MIXED_STAIR_COUNT)
                ]
            )
            stair_inside = (
                (
                    jp.abs(stair_x - stair_centers)
                    <= self._config.terrain.step_depth / 2.0
                )
                & (jp.abs(y[..., None] - MIXED_PATCH_Y[4]) <= MIXED_LANE_HALF_WIDTH)
            )
            stair_height = jp.max(
                jp.where(
                    stair_inside,
                    jp.array(
                        [
                            self._config.terrain.step_height * (i + 1)
                            for i in range(MIXED_STAIR_COUNT)
                        ]
                    ),
                    0.0,
                ),
                axis=-1,
            )
            height = jp.maximum(height, stair_height)
            height = jp.maximum(
                height, box_family(self._mixed_rough, MIXED_PATCH_Y[5])
            )
            return height
        x = xy[..., 0, None]
        y = xy[..., 1, None]
        inside = (
            (jp.abs(x - self._step_centers) <= self._config.terrain.step_depth / 2.0)
            & (jp.abs(y) <= 1.0)
        )
        return jp.max(jp.where(inside, self._step_heights, 0.0), axis=-1)

    def _curriculum_stage(self, steps: jax.Array) -> jax.Array:
        if self._fixed_curriculum_stage is not None:
            return jp.asarray(self._fixed_curriculum_stage, dtype=jp.int32)
        if not self._command_curriculum:
            return jp.zeros((), dtype=jp.int32)
        first_end = int(self._config.command_curriculum.forward_only_steps)
        second_end = first_end + int(self._config.command_curriculum.limited_yaw_steps)
        return jp.where(steps < first_end, 0, jp.where(steps < second_end, 1, 2)).astype(jp.int32)

    def _scripted_command(self, steps: jax.Array, stage: jax.Array) -> jax.Array:
        """Deterministic command used only for comparable evaluation/video.

        Training continues to call :meth:`_sample_command`.  A fixed-stage
        evaluation uses time from episode reset, while the full curriculum
        script resets its local clock at the 0->1 and 1->2 boundaries.
        """
        first_end = int(self._config.command_curriculum.forward_only_steps)
        second_end = first_end + int(
            self._config.command_curriculum.limited_yaw_steps
        )
        if self._fixed_curriculum_stage is None:
            local_steps = jp.where(
                stage == 0,
                steps,
                jp.where(stage == 1, steps - first_end, steps - second_end),
            )
        else:
            local_steps = steps
        local_seconds = local_steps.astype(jp.float32) * self.dt

        stage0 = jp.array((0.08, 0.0, 0.0))
        stage1_index = jp.clip((local_seconds / 3.0).astype(jp.int32), 0, 2)
        stage1 = jp.array(
            ((0.12, 0.00, 0.12), (0.12, 0.04, -0.12), (0.16, -0.04, 0.08))
        )[stage1_index]
        stage2_index = jp.clip((local_seconds / 3.0).astype(jp.int32), 0, 3)
        stage2 = jp.array(
            (
                (0.10, 0.00, 0.00),
                (0.14, 0.08, 0.30),
                (0.14, -0.08, -0.30),
                (0.27, 0.00, 0.00),
            )
        )[stage2_index]
        return jp.where(stage == 0, stage0, jp.where(stage == 1, stage1, stage2))

    def _sample_command(self, rng: jax.Array, stage: jax.Array) -> jax.Array:
        speed_key, lateral_key, yaw_key = jax.random.split(rng, 3)
        if self._command_curriculum:
            speed_min = jp.array(self._config.command_curriculum.speed_min)[stage]
            speed_max = jp.array(self._config.command_curriculum.speed_max)[stage]
            lateral_limit = jp.array(
                self._config.command_curriculum.lateral_limit
            )[stage]
            yaw_limit = jp.array(self._config.command_curriculum.yaw_limit)[stage]
        else:
            speed_min = self._config.command_min_speed
            speed_max = self._config.command_max_speed
            lateral_limit = self._config.command_max_lateral_speed
            yaw_limit = self._config.command_max_yaw_rate
        speed = jax.random.uniform(speed_key, (), minval=speed_min, maxval=speed_max)
        linear_capacity = (
            self._config.controller.safety.max_effective_stride
            / self._config.controller.nominal.phase_time
        )
        reachable_lateral = jp.sqrt(
            jp.maximum(linear_capacity * linear_capacity - speed * speed, 0.0)
        )
        applied_lateral_limit = jp.minimum(lateral_limit, reachable_lateral)
        lateral = jax.random.uniform(
            lateral_key,
            (),
            minval=-applied_lateral_limit,
            maxval=applied_lateral_limit,
        )
        linear_speed = jp.hypot(speed, lateral)
        safe_yaw_limit = feasible_yaw_limit(
            speed=linear_speed,
            requested_yaw_limit=yaw_limit,
            origins=self._origins,
            outward=self._outward,
            shoulder_lateral=self._shoulder_lateral,
            phase_time=self._config.controller.nominal.phase_time,
            max_stride=self._config.controller.safety.max_effective_stride,
            max_frequency_scale=1.0,
        )
        tangent = jp.stack(
            (-self._outward[:, 1], self._outward[:, 0], jp.zeros(6)), axis=-1
        )
        nominal_body = (
            self._origins
            + self._outward * NOMINAL_FOOT_RADIAL
            + tangent * self._shoulder_lateral[:, None]
        )
        max_foot_radius = jp.max(jp.linalg.norm(nominal_body[:, :2], axis=-1))
        conservative_yaw_limit = jp.maximum(
            linear_capacity - linear_speed, 0.0
        ) / jp.maximum(max_foot_radius, 1e-6)
        safe_yaw_limit = jp.minimum(safe_yaw_limit, conservative_yaw_limit)
        yaw_rate = jax.random.uniform(
            yaw_key, (), minval=-safe_yaw_limit, maxval=safe_yaw_limit
        )
        return jp.array((speed, lateral, yaw_rate))

    def _sample_command_interval(self, rng: jax.Array) -> jax.Array:
        low_seconds, high_seconds = self._config.command_curriculum.resample_seconds
        low_steps = max(1, int(np.ceil(float(low_seconds) / self.dt)))
        high_steps = max(low_steps, int(np.floor(float(high_seconds) / self.dt)))
        return jax.random.randint(
            rng, (), minval=low_steps, maxval=high_steps + 1, dtype=jp.int32
        )

    def _feet_body(self, data: mjx.Data) -> jax.Array:
        """Foot vectors in root body frame; their meaning survives world yaw."""
        relative_world = data.site_xpos[self._foot_site_ids] - data.qpos[None, :3]
        return jax.vmap(_quat_rotate_inverse, in_axes=(None, 0))(data.qpos[3:7], relative_world)

    def _feet_leg_local(self, data: mjx.Data) -> jax.Array:
        feet_body = self._feet_body(data)
        relative = feet_body - self._origins
        tangent = jp.stack((-self._outward[:, 1], self._outward[:, 0], jp.zeros(6)), axis=-1)
        return jp.stack(
            (jp.sum(relative * self._outward, axis=-1), jp.sum(relative * tangent, axis=-1), relative[:, 2]),
            axis=-1,
        )

    def _foot_contacts(
        self, data: mjx.Data, previous: Optional[jax.Array] = None
    ) -> jax.Array:
        """Collision contact fused with a hysteretic clearance estimate."""
        contact = data._impl.contact
        active = contact.dist < 0.0
        foot_match = jp.any(contact.geom[:, :, None] == self._foot_geom_ids[None, None, :], axis=1)
        physical = jp.any(active[:, None] & foot_match, axis=0)
        feet = data.site_xpos[self._foot_site_ids]
        clearance = feet[:, 2] - self._terrain_height(feet[:, :2])
        if previous is None:
            previous = jp.zeros(6, dtype=jp.bool_)
        geometric = hysteretic_clearance_contact(
            clearance,
            previous,
            enter_clearance=self._config.controller.contact.contact_enter_clearance,
            release_clearance=self._config.controller.contact.contact_release_clearance,
        )
        return physical | geometric

    def _support_height(
        self, data: mjx.Data, contacts: jax.Array, swing: jax.Array
    ) -> jax.Array:
        feet = data.site_xpos[self._foot_site_ids]
        terrain_heights = self._terrain_height(feet[:, :2])
        support_mask = contacts & (~swing)
        fallback = self._terrain_height(data.qpos[:2])
        return median_support_height(terrain_heights, support_mask, fallback)

    def _update_body_position_estimate(
        self,
        data: mjx.Data,
        contacts: jax.Array,
        swing: jax.Array,
        info: dict[str, Any],
    ) -> tuple[jax.Array, ...]:
        """FK/contact stance-anchor estimator with source slip rejection."""
        stance_contact = contacts & (~swing)
        previous_valid = info["stance_anchor_valid"]
        new_anchor = stance_contact & (~previous_valid)
        feet_body = self._feet_body(data)
        rotated_feet = jax.vmap(_quat_rotate, in_axes=(None, 0))(
            data.qpos[3:7], feet_body
        )
        # Match the deployable estimator: a new anchor comes from the previous
        # body estimate plus FK, never from MuJoCo's absolute site position.
        estimated_anchor = info["body_position_estimate"] + rotated_feet
        anchors = jp.where(
            new_anchor[:, None], estimated_anchor, info["stance_anchors_world"]
        )
        anchor_valid = stance_contact
        candidates = anchors - rotated_feet
        pair_distance = jp.linalg.norm(
            candidates[:, None, :] - candidates[None, :, :], axis=-1
        )
        other = ~jp.eye(6, dtype=jp.bool_)
        neighbor = jp.any(
            (pair_distance <= self._config.controller.estimator.slip_distance)
            & other
            & anchor_valid[:, None]
            & anchor_valid[None, :],
            axis=1,
        )
        valid_count = jp.sum(anchor_valid.astype(jp.int32))
        # The source accepts a lone stance candidate.  Pairwise agreement and
        # slip confirmation begin only when two or more candidates exist.
        isolated = anchor_valid & (~neighbor) & (valid_count >= 2)
        slip_counts = jp.where(
            isolated,
            info["slip_counts"] + 1,
            jp.zeros(6, dtype=jp.int32),
        )
        slip_mask = anchor_valid & (
            info["slip_mask"]
            | (slip_counts >= self._config.controller.estimator.slip_confirm_steps)
        )
        usable = anchor_valid & (~isolated) & (~slip_mask)
        usable_count = jp.sum(usable.astype(jp.float32))
        estimate = jp.sum(
            jp.where(usable[:, None], candidates, jp.zeros_like(candidates)), axis=0
        ) / jp.maximum(usable_count, 1.0)
        estimate = jp.where(
            usable_count > 0.0, estimate, info["body_position_estimate"]
        )
        return estimate, anchors, anchor_valid, slip_counts, slip_mask

    def _torso_contact(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        return jp.any((contact.dist < 0.0) & jp.any(contact.geom == self._torso_geom_id, axis=-1))

    def _self_collision(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        return self_collision_detected(
            contact.geom, contact.dist, self._geom_body_ids
        )

    def _terrain_features(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Nine coarse heights plus six nominal-touchdown terrain heights."""
        forward = _quat_rotate(data.qpos[3:7], MODEL_FORWARD)[:2]
        sample_world = heading_aligned_points(data.qpos[:2], forward, self._terrain_offsets)
        coarse = self._terrain_height(sample_world) - info["support_height"]
        touchdown_body = nominal_touchdown_body_targets(
            origins=self._origins,
            outward=self._outward,
            shoulder_lateral=self._shoulder_lateral,
            command=info["applied_twist"],
            phase_time=self._config.controller.nominal.phase_time,
        )
        touchdown_world = data.qpos[None, :3] + jax.vmap(
            _quat_rotate, in_axes=(None, 0)
        )(data.qpos[3:7], touchdown_body)
        touchdown = (
            self._terrain_height(touchdown_world[:, :2]) - info["support_height"]
        )
        return jp.concatenate((coarse, touchdown))

    def _gait_terms(
        self, command: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return controller-owned stride scaling and realized stroke."""
        nominal = self._config.controller.nominal
        step_scale, effective_stride = limit_effective_stride(
            requested_scale=jp.ones(()),
            command=command,
            origins=self._origins,
            outward=self._outward,
            shoulder_lateral=self._shoulder_lateral,
            phase_time=nominal.phase_time,
            max_stride=self._config.controller.safety.max_effective_stride,
        )
        return step_scale, effective_stride

    def _physical_body_residual(self, raw: jax.Array) -> jax.Array:
        """Map six normalized residual terms to metres and radians."""
        authority = self._authority()
        translation = jp.clip(raw[:3], -1.0, 1.0) * jp.asarray(
            authority.body_translation_limit
        )
        rotation = jp.clip(raw[3:], -1.0, 1.0) * jp.asarray(
            authority.body_rotation_limit
        )
        return jp.concatenate((translation, rotation))

    def _controller_targets(
        self,
        action: jax.Array,
        phase: jax.Array,
        command: jax.Array,
        blend: jax.Array,
        contacts: jax.Array,
        airborne: jax.Array,
        current_feet_local: jax.Array,
        body_pose_target: jax.Array,
        posture_candidate: jax.Array,
        previous_body_pose: jax.Array,
        gait_enable: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Classical gait -> foot residual -> contact -> body 6-DOF -> IK."""
        action = jp.clip(action, -1.0, 1.0)
        authority = self._authority()
        step_scale, effective_stride = self._gait_terms(command)
        nominal_config = self._config.controller.nominal
        swing_height = jp.clip(
            nominal_config.base_swing_height + body_pose_target[2],
            0.15,
            0.25,
        )
        nominal, swing = nominal_foot_targets(
            origins=self._origins,
            outward=self._outward,
            shoulder_lateral=self._shoulder_lateral,
            tripod_a=self._tripod_a,
            phase=phase,
            command=command,
            phase_time=nominal_config.phase_time,
            step_scale=step_scale,
            swing_height=swing_height,
            radial_offset=nominal_config.base_radial_offset,
        )
        motion_request = (
            jp.linalg.norm(command[:2]) >= nominal_config.gait_enable_speed
        ) | (jp.abs(command[2]) >= nominal_config.gait_enable_yaw_rate)
        gait_enabled = motion_request if gait_enable is None else gait_enable
        swing = swing & gait_enabled
        home_feet, _ = nominal_foot_targets(
            origins=self._origins,
            outward=self._outward,
            shoulder_lateral=self._shoulder_lateral,
            tripod_a=self._tripod_a,
            phase=jp.zeros_like(phase),
            command=jp.zeros_like(command),
            phase_time=nominal_config.phase_time,
            step_scale=jp.ones(()),
            swing_height=nominal_config.base_swing_height,
            radial_offset=nominal_config.base_radial_offset,
        )
        nominal = jp.where(gait_enabled, nominal, home_feet)
        residual = phase_masked_residual(
            action[:FOOT_RESIDUAL_SIZE].reshape(6, 3), swing,
            swing_x=authority.swing_x,
            swing_y=authority.swing_y,
            swing_z_low=authority.swing_z_low,
            swing_z_high=authority.swing_z_high,
            stance_z=authority.stance_z,
        )
        requested = nominal + blend * residual
        _, phase_progress = phase_masks(self._tripod_a, phase)
        late_landing = swing & (phase_progress >= 0.999) & (~contacts)
        adapted, early_landing, lost_contact = contact_adapt_targets(
            requested, current_feet_local, swing, contacts, airborne,
            lost_contact_search=(
                self._config.controller.contact.search_down_speed * self.dt
            ),
            lost_contact_inward=(
                self._config.controller.contact.search_down_speed
                * self._config.controller.contact.search_inward_ratio
                * self.dt
            ),
            early_landing_allowed=phase_progress >= 0.5,
            late_landing=late_landing,
        )
        effective_swing = swing & (~early_landing)

        base_body = leg_local_to_body(adapted, self._origins, self._outward)
        candidate_pose = jp.concatenate(
            (body_pose_target[:3], posture_candidate)
        )
        candidate_body = apply_body_pose_overlay(
            base_body, candidate_pose[:3], candidate_pose[3:]
        )
        candidate_feet = body_to_leg_local(
            candidate_body, self._origins, self._outward
        )
        posture_accepted = workspace_valid(
            candidate_feet,
            self._shoulder_lateral,
            min_distance=self._config.controller.safety.workspace_min_distance,
            max_distance=self._config.controller.safety.workspace_max_distance,
            joint_limit=self._config.controller.safety.joint_limit,
        )
        applied_body_pose = jp.where(
            posture_accepted, candidate_pose, previous_body_pose
        )
        overlaid_body = apply_body_pose_overlay(
            base_body, applied_body_pose[:3], applied_body_pose[3:]
        )
        overlaid_feet = body_to_leg_local(
            overlaid_body, self._origins, self._outward
        )
        safe_feet, projection_cost = project_workspace(
            overlaid_feet,
            self._shoulder_lateral,
            min_distance=self._config.controller.safety.workspace_min_distance,
            max_distance=self._config.controller.safety.workspace_max_distance,
        )
        raw = analytical_ik(safe_feet, self._shoulder_lateral) * self._raw_signs
        targets = self._home_ctrl.at[self._actuator_ids].set(raw)
        targets = jp.clip(targets, -self._config.controller.safety.joint_limit, self._config.controller.safety.joint_limit)
        targets = self._home_ctrl + blend * (targets - self._home_ctrl)
        return targets, {
            "frequency_scale": jp.ones(()),
            "step_scale": step_scale,
            "swing_height": swing_height,
            "radial_offset": jp.asarray(nominal_config.base_radial_offset),
            "effective_stride": effective_stride,
            "swing": effective_swing,
            "phase_swing": swing,
            "residual": residual,
            "early_landing": early_landing,
            "lost_contact": lost_contact,
            "projection_cost": projection_cost,
            "requested_feet": requested,
            "safe_feet": safe_feet,
            "gait_enabled": gait_enabled,
            "motion_request": motion_request,
            "body_pose": applied_body_pose,
            "posture_accepted": posture_accepted,
        }

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        local_gravity = _quat_rotate_inverse(quat, jp.array((0.0, 0.0, -1.0)))
        obs = jp.concatenate(
            (
                info["command"], local_velocity, data.qvel[3:6], local_gravity,
                data.qpos[self._joint_qpos_ids] - self._home_qpos[self._joint_qpos_ids],
                0.1 * data.qvel[self._joint_qvel_ids],
                jp.ravel(self._feet_body(data)), info["contact_state"].astype(jp.float32),
                self._terrain_features(data, info),
                jp.array((jp.sin(2 * jp.pi * info["phase"]), jp.cos(2 * jp.pi * info["phase"]))),
                info["last_action"],
            )
        )
        return obs

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, q_key, vel_key, command_key, interval_key, patch_key = jax.random.split(rng, 6)
        qpos = self._home_qpos.at[self._joint_qpos_ids].add(
            jax.random.uniform(q_key, (18,), minval=-0.015, maxval=0.015)
        )
        if self._terrain == "mixed":
            if self._fixed_terrain_patch is None:
                probabilities = jp.array(self._config.terrain.patch_probabilities)
                probabilities = probabilities / jp.sum(probabilities)
                terrain_patch = jax.random.choice(
                    patch_key, len(MIXED_PATCH_NAMES), (), p=probabilities
                )
            else:
                terrain_patch = jp.asarray(
                    self._fixed_terrain_patch, dtype=jp.int32
                )
            qpos = qpos.at[1].set(self._mixed_patch_y[terrain_patch])
        else:
            terrain_patch = jp.zeros((), dtype=jp.int32)
        qvel = jp.zeros(self.mjx_model.nv).at[:6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.02, maxval=0.02)
        )
        curriculum_stage = self._curriculum_stage(jp.zeros((), dtype=jp.int32))
        data = mjx.make_data(self.mj_model, impl=self.mjx_model.impl.value, naconmax=self._config.nconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data.replace(qpos=qpos, qvel=qvel, ctrl=self._home_ctrl))
        contact_state = self._foot_contacts(data)
        initial_swing, _ = phase_masks(self._tripod_a, jp.zeros(()))
        support_height = self._support_height(data, contact_state, initial_swing)
        command = (
            self._scripted_command(jp.zeros((), dtype=jp.int32), curriculum_stage)
            if self._scripted_commands
            else self._sample_command(command_key, curriculum_stage)
        )
        initial_heading = _body_heading(data.qpos[3:7])
        info = {
            "rng": rng,
            "command": command,
            "filtered_command": jp.zeros(3),
            "applied_twist": jp.zeros(3),
            "desired_position_xy": data.qpos[:2],
            "desired_heading": initial_heading,
            "position_integral": jp.zeros(2),
            "heading_integral": jp.zeros(()),
            "body_position_estimate": data.qpos[:3],
            "stance_anchors_world": data.site_xpos[self._foot_site_ids],
            "stance_anchor_valid": contact_state,
            "slip_counts": jp.zeros(6, dtype=jp.int32),
            "slip_mask": jp.zeros(6, dtype=jp.bool_),
            "body_pose_filter": jp.zeros(BODY_RESIDUAL_SIZE),
            "body_pose_command": jp.zeros(BODY_RESIDUAL_SIZE),
            "posture_integral": jp.zeros(3),
            "gait_running": jp.zeros((), dtype=jp.bool_),
            "stop_pending": jp.zeros((), dtype=jp.bool_),
            "command_steps_remaining": self._sample_command_interval(interval_key),
            "phase": jp.zeros(()),
            "steps": jp.zeros((), dtype=jp.int32),
            "curriculum_stage": curriculum_stage,
            "last_action": jp.zeros(ACTION_SIZE),
            "last_policy_action": jp.zeros(ACTION_SIZE),
            "previous_foot_positions": data.site_xpos[self._foot_site_ids],
            "contact_state": contact_state,
            "airborne": jp.zeros(6, dtype=jp.bool_),
            "support_height": support_height,
            "terrain_patch": terrain_patch,
            "start_root_position": data.qpos[:3],
            "start_forward": _quat_rotate(data.qpos[3:7], MODEL_FORWARD),
            "velocity_error_integral": jp.zeros(()),
        }
        metrics = {f"reward/{name}": jp.zeros(()) for name in self._config.reward.keys()}
        metrics.update(
            workspace_error=jp.zeros(()), projection_cost=jp.zeros(()),
            support_height=jp.zeros(()), body_clearance=jp.zeros(()),
            terrain_patch=terrain_patch.astype(jp.float32),
            terrain_success=jp.zeros(()),
            velocity_error_mps=jp.zeros(()), yaw_error_rps=jp.zeros(()),
            torque_rms_nm=jp.zeros(()), torque_saturation=jp.zeros(()),
            self_collision=jp.zeros(()), effective_stride_m=jp.zeros(()),
            applied_step_scale=jp.ones(()),
            applied_frequency_scale=jp.ones(()),
            applied_swing_height_m=jp.array(
                self._config.controller.nominal.base_swing_height
            ),
            applied_radial_offset_m=jp.array(
                self._config.controller.nominal.base_radial_offset
            ),
            gait_filter_error=jp.zeros(()),
            body_residual_filter_error=jp.zeros(()),
            posture_command_accepted=jp.ones(()),
            applied_body_forward_m=jp.zeros(()),
            applied_body_lateral_m=jp.zeros(()),
            applied_body_height_m=jp.zeros(()),
            applied_body_roll_rad=jp.zeros(()),
            applied_body_pitch_rad=jp.zeros(()),
            applied_body_yaw_rad=jp.zeros(()),
            position_error_m=jp.zeros(()),
            heading_error_rad=jp.zeros(()),
            body_position_estimator_error_m=jp.zeros(()),
            slip_leg_fraction=jp.zeros(()),
            curriculum_stage=curriculum_stage.astype(jp.float32),
            contact_early_landing=jp.zeros(()), contact_lost=jp.zeros(()),
            phase_late_landing=jp.zeros(()),
        )
        return mjx_env.State(data, self._get_obs(data, info), jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        previous_action = state.info["last_action"]
        filtered_body_action = smooth_gait_action(
            state.info["body_pose_filter"],
            action[FOOT_RESIDUAL_SIZE:],
            control_dt=self.dt,
            time_constant=self._config.controller.residual.body_filter_time_constant,
        )
        applied_action = action.at[FOOT_RESIDUAL_SIZE:].set(filtered_body_action)
        blend = jp.clip(state.info["steps"] * self.dt / 0.75, 0.0, 1.0)

        command_config = self._config.controller.command
        (
            applied_twist,
            filtered_command,
            desired_position_xy,
            desired_heading,
            position_integral,
            heading_integral,
            position_error,
            heading_error,
        ) = classical_body_twist(
            command_target=state.info["command"],
            filtered_command=state.info["filtered_command"],
            desired_position_xy=state.info["desired_position_xy"],
            desired_heading=state.info["desired_heading"],
            position_integral=state.info["position_integral"],
            heading_integral=state.info["heading_integral"],
            applied_twist=state.info["applied_twist"],
            body_position_xy=state.info["body_position_estimate"][:2],
            body_heading=_body_heading(state.data.qpos[3:7]),
            dt=self.dt,
            command_deadzone=command_config.deadzone,
            command_rate_limit=jp.asarray(command_config.rate_limit),
            position_kp=jp.asarray(command_config.position_kp),
            position_ki=jp.asarray(command_config.position_ki),
            position_integral_limit=jp.asarray(
                command_config.position_integral_limit
            ),
            position_feedback_limit=jp.asarray(
                command_config.position_feedback_limit
            ),
            heading_kp=command_config.heading_kp,
            heading_ki=command_config.heading_ki,
            heading_integral_limit=command_config.heading_integral_limit,
            heading_feedback_limit=command_config.heading_feedback_limit,
            twist_limit=jp.asarray(command_config.twist_limit),
            twist_rate_limit=jp.asarray(command_config.twist_rate_limit),
            position_valid=jp.any(state.info["stance_anchor_valid"]),
        )

        body_pose_target = blend * self._physical_body_residual(
            filtered_body_action
        )
        nominal_config = self._config.controller.nominal
        motion_request = (
            jp.linalg.norm(applied_twist[:2]) >= nominal_config.gait_enable_speed
        ) | (jp.abs(applied_twist[2]) >= nominal_config.gait_enable_yaw_rate)
        gait_running = motion_request | state.info["gait_running"]
        stop_pending = (~motion_request) & state.info["gait_running"]
        posture_config = self._config.controller.posture
        measured_heading = _body_heading(state.data.qpos[3:7])
        measured_roll_pitch = _controller_roll_pitch(state.data.qpos[3:7])
        posture_candidate, posture_integral_candidate, posture_error = (
            posture_pi_candidate(
                target_rpy=body_pose_target[3:],
                measured_rpy=jp.array(
                    (measured_roll_pitch[0], measured_roll_pitch[1], 0.0)
                ),
                desired_heading=desired_heading,
                measured_heading=measured_heading,
                posture_integral=state.info["posture_integral"],
                posture_command=state.info["body_pose_command"][3:],
                dt=self.dt,
                kp=jp.asarray(posture_config.kp),
                ki=jp.asarray(posture_config.ki),
                integral_limit=jp.asarray(posture_config.integral_limit),
                angular_rate_limit=jp.asarray(posture_config.angular_rate_limit),
                posture_limit=jp.asarray(posture_config.angle_limit),
            )
        )
        targets, controller = self._controller_targets(
            applied_action, state.info["phase"], applied_twist, blend,
            state.info["contact_state"], state.info["airborne"],
            self._feet_leg_local(state.data),
            body_pose_target,
            posture_candidate,
            state.info["body_pose_command"],
            gait_running,
        )
        posture_integral = jp.where(
            controller["posture_accepted"],
            posture_integral_candidate,
            state.info["posture_integral"],
        )
        # A discontinuous IK candidate is held before the permanent servo-rate
        # limiter, matching the updated source safety ordering.
        jump = jp.abs(targets - state.data.ctrl) > self._config.controller.safety.max_joint_jump
        targets = jp.where(jump, state.data.ctrl, targets)
        max_delta = self._config.controller.safety.max_joint_speed * self.dt
        targets = state.data.ctrl + jp.clip(targets - state.data.ctrl, -max_delta, max_delta)
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)

        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        forward_velocity = jp.dot(local_velocity, MODEL_FORWARD)
        lateral_velocity = jp.dot(local_velocity, MODEL_LATERAL)
        velocity_error_mps = jp.linalg.norm(
            jp.array((forward_velocity, lateral_velocity))
            - state.info["command"][:2]
        )
        yaw_error_rps = jp.abs(data.qvel[5] - state.info["command"][2])
        up_z = _quat_rotate(quat, jp.array((0.0, 0.0, 1.0)))[2]
        measured_roll_pitch = _controller_roll_pitch(quat)
        contacts = self._foot_contacts(data, state.info["contact_state"])
        support_height = self._support_height(data, contacts, controller["swing"])
        (
            body_position_estimate,
            stance_anchors_world,
            stance_anchor_valid,
            slip_counts,
            slip_mask,
        ) = self._update_body_position_estimate(
            data, contacts, controller["swing"], state.info
        )
        clearance = data.qpos[2] - support_height
        rollover = jp.any(
            jp.abs(measured_roll_pitch)
            >= self._config.controller.safety.rollover_angle
        )
        terminated = rollover | (clearance < 0.14) | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        foot_velocity = (data.site_xpos[self._foot_site_ids] - state.info["previous_foot_positions"]) / self.dt
        contact_count = jp.maximum(jp.sum(contacts.astype(jp.float32)), 1.0)
        slip_cost = jp.sum(contacts.astype(jp.float32) * jp.sum(jp.square(foot_velocity[:, :2] / 0.30), axis=-1)) / contact_count
        previous_policy_action = state.info["last_policy_action"]
        swing = controller["swing"]
        foot_action = applied_action[:FOOT_RESIDUAL_SIZE].reshape(6, 3)
        previous_foot_action = previous_action[:FOOT_RESIDUAL_SIZE].reshape(6, 3)
        swing_count = jp.maximum(jp.sum(swing.astype(jp.float32)) * 3.0, 1.0)
        stance_count = jp.maximum(jp.sum((~swing).astype(jp.float32)), 1.0)
        swing_residual_cost = jp.sum(jp.square(foot_action) * swing[:, None]) / swing_count
        stance_residual_cost = jp.sum(jp.square(foot_action[:, 2]) * (~swing)) / stance_count
        projection_cost = controller["projection_cost"] / jp.square(self._config.controller.safety.projection_reference)
        force_limit = self._config.controller.safety.actuator_force_limit
        torque_rms_nm = jp.sqrt(jp.mean(jp.square(data.actuator_force)))
        saturation_cost = torque_saturation_cost(
            data.actuator_force, force_limit=force_limit
        )
        self_collision = self._self_collision(data).astype(jp.float32)
        target_roll_pitch = body_pose_target[3:5]
        attitude_error = jp.array(
            (
                wrap_angle(target_roll_pitch[0] - measured_roll_pitch[0]),
                wrap_angle(target_roll_pitch[1] - measured_roll_pitch[1]),
            )
        )

        reward_terms = {
            "velocity": jp.exp(-jp.square(velocity_error_mps) / 0.02),
            "yaw": jp.exp(-jp.square(data.qvel[5] - state.info["command"][2]) / 0.16),
            # Track the controller-approved body bend instead of penalizing it
            # as non-upright; this is what makes stair pitching learnable.
            "upright": jp.exp(-jp.sum(jp.square(attitude_error)) / 0.20),
            "height": jp.exp(
                -jp.square(clearance - (0.33 + controller["body_pose"][2]))
                / 0.015
            ),
            "progress": jp.clip(forward_velocity, -0.2, 0.3),
            "swing_residual": swing_residual_cost,
            "stance_residual": stance_residual_cost,
            "body_translation_residual": jp.mean(
                jp.square(applied_action[FOOT_RESIDUAL_SIZE:21])
            ),
            "body_rotation_residual": jp.mean(
                jp.square(applied_action[21:ACTION_SIZE])
            ),
            "foot_action_rate": jp.mean(jp.square(foot_action - previous_foot_action)),
            "body_action_rate": jp.mean(
                jp.square(
                    action[FOOT_RESIDUAL_SIZE:]
                    - previous_policy_action[FOOT_RESIDUAL_SIZE:]
                )
            ),
            "vertical_velocity": jp.square(data.qvel[2]),
            "lateral_velocity": jp.square(
                lateral_velocity - state.info["command"][1]
            ),
            "joint_velocity": jp.mean(jp.square(data.qvel[self._joint_qvel_ids] / 10.0)),
            "torque": jp.mean(jp.square(data.actuator_force / force_limit)),
            "torque_saturation": saturation_cost,
            "slip": slip_cost,
            "projection": projection_cost,
            "body_contact": self._torso_contact(data).astype(jp.float32),
            "self_collision": self_collision,
            "termination": terminated.astype(jp.float32),
        }
        scaled = {name: value * self._config.reward[name] for name, value in reward_terms.items()}
        reward = jp.clip(sum(scaled.values()) * self.dt, -10.0, 10.0)

        airborne = update_airborne_state(
            state.info["airborne"], controller["phase_swing"], contacts
        )
        next_phase, phase_late_landing, phase_completed = advance_contact_gated_phase(
            phase=state.info["phase"],
            gait_enabled=controller["gait_enabled"],
            swing=controller["phase_swing"],
            contacts=contacts,
            airborne=airborne,
            dt=self.dt,
            phase_time=self._config.controller.nominal.phase_time,
        )
        stop_completed = stop_pending & phase_completed
        gait_running = gait_running & (~stop_completed)
        stop_pending = stop_pending & (~stop_completed)
        next_phase = jp.where(stop_completed, 0.0, next_phase)
        state.info["phase"] = next_phase
        next_steps = state.info["steps"] + 1
        forward_distance = jp.dot(
            data.qpos[:3] - state.info["start_root_position"],
            state.info["start_forward"],
        )
        velocity_error_integral = (
            state.info["velocity_error_integral"] + velocity_error_mps * self.dt
        )
        mean_velocity_error = velocity_error_integral / jp.maximum(
            next_steps * self.dt, self.dt
        )
        timed_out = next_steps >= int(self._config.episode_length)
        terrain_success = (
            timed_out
            & (~terminated)
            & (mean_velocity_error < 0.08)
            & (forward_distance > 0.5)
        )
        next_stage = self._curriculum_stage(next_steps)
        if self._scripted_commands:
            state.info["command"] = self._scripted_command(next_steps, next_stage)
            state.info["command_steps_remaining"] = jp.zeros((), dtype=jp.int32)
        else:
            next_rng, command_key, interval_key = jax.random.split(
                state.info["rng"], 3
            )
            stage_changed = next_stage != state.info["curriculum_stage"]
            interval_elapsed = state.info["command_steps_remaining"] <= 1
            resample_command = stage_changed | interval_elapsed
            sampled_command = self._sample_command(command_key, next_stage)
            sampled_interval = self._sample_command_interval(interval_key)
            state.info["rng"] = next_rng
            state.info["command"] = jp.where(
                resample_command, sampled_command, state.info["command"]
            )
            state.info["command_steps_remaining"] = jp.where(
                resample_command,
                sampled_interval,
                state.info["command_steps_remaining"] - 1,
            )
        state.info["curriculum_stage"] = next_stage
        state.info["steps"] = next_steps
        state.info["velocity_error_integral"] = velocity_error_integral
        state.info["filtered_command"] = filtered_command
        state.info["applied_twist"] = applied_twist
        state.info["desired_position_xy"] = desired_position_xy
        state.info["desired_heading"] = desired_heading
        state.info["position_integral"] = position_integral
        state.info["heading_integral"] = heading_integral
        state.info["body_position_estimate"] = body_position_estimate
        state.info["stance_anchors_world"] = stance_anchors_world
        state.info["stance_anchor_valid"] = stance_anchor_valid
        state.info["slip_counts"] = slip_counts
        state.info["slip_mask"] = slip_mask
        state.info["body_pose_filter"] = filtered_body_action
        state.info["body_pose_command"] = controller["body_pose"]
        state.info["posture_integral"] = posture_integral
        state.info["gait_running"] = gait_running
        state.info["stop_pending"] = stop_pending
        state.info["last_action"] = applied_action
        state.info["last_policy_action"] = action
        state.info["previous_foot_positions"] = data.site_xpos[self._foot_site_ids]
        state.info["contact_state"] = contacts
        state.info["support_height"] = support_height
        state.info["airborne"] = airborne
        for name, value in scaled.items():
            state.metrics[f"reward/{name}"] = value
        state.metrics["workspace_error"] = controller["projection_cost"]
        state.metrics["projection_cost"] = controller["projection_cost"]
        state.metrics["support_height"] = support_height
        state.metrics["body_clearance"] = clearance
        state.metrics["terrain_patch"] = state.info["terrain_patch"].astype(jp.float32)
        state.metrics["terrain_success"] = terrain_success.astype(jp.float32)
        state.metrics["velocity_error_mps"] = velocity_error_mps
        state.metrics["yaw_error_rps"] = yaw_error_rps
        state.metrics["torque_rms_nm"] = torque_rms_nm
        state.metrics["torque_saturation"] = saturation_cost
        state.metrics["self_collision"] = self_collision
        state.metrics["effective_stride_m"] = controller["effective_stride"]
        state.metrics["applied_step_scale"] = controller["step_scale"]
        state.metrics["applied_frequency_scale"] = controller["frequency_scale"]
        state.metrics["applied_swing_height_m"] = controller["swing_height"]
        state.metrics["applied_radial_offset_m"] = controller["radial_offset"]
        state.metrics["gait_filter_error"] = jp.zeros(())
        state.metrics["body_residual_filter_error"] = jp.sqrt(
            jp.mean(
                jp.square(
                    action[FOOT_RESIDUAL_SIZE:]
                    - applied_action[FOOT_RESIDUAL_SIZE:]
                )
            )
        )
        state.metrics["posture_command_accepted"] = controller[
            "posture_accepted"
        ].astype(jp.float32)
        state.metrics["applied_body_forward_m"] = controller["body_pose"][0]
        state.metrics["applied_body_lateral_m"] = controller["body_pose"][1]
        state.metrics["applied_body_height_m"] = controller["body_pose"][2]
        state.metrics["applied_body_roll_rad"] = controller["body_pose"][3]
        state.metrics["applied_body_pitch_rad"] = controller["body_pose"][4]
        state.metrics["applied_body_yaw_rad"] = controller["body_pose"][5]
        state.metrics["position_error_m"] = jp.linalg.norm(position_error)
        state.metrics["heading_error_rad"] = jp.abs(heading_error)
        state.metrics["body_position_estimator_error_m"] = jp.linalg.norm(
            body_position_estimate - data.qpos[:3]
        )
        state.metrics["slip_leg_fraction"] = jp.mean(
            slip_mask.astype(jp.float32)
        )
        state.metrics["curriculum_stage"] = state.info["curriculum_stage"].astype(jp.float32)
        state.metrics["contact_early_landing"] = jp.mean(controller["early_landing"].astype(jp.float32))
        state.metrics["contact_lost"] = jp.mean(controller["lost_contact"].astype(jp.float32))
        state.metrics["phase_late_landing"] = phase_late_landing.astype(jp.float32)
        return state.replace(data=data, obs=self._get_obs(data, state.info), reward=reward, done=terminated.astype(jp.float32))

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def observation_size(self) -> int:
        return OBSERVATION_SIZE

    @property
    def action_contract_version(self) -> str:
        return ACTION_CONTRACT_VERSION

    @property
    def observation_contract_version(self) -> str:
        return OBSERVATION_CONTRACT_VERSION

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
