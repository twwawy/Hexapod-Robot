"""MJX flat/mixed-terrain environments with a classical-first 22-D contract."""

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
    MODEL_FORWARD,
    analytical_ik,
    contact_adapt_targets,
    heading_aligned_points,
    hysteretic_clearance_contact,
    limit_effective_stride,
    median_support_height,
    nominal_foot_targets,
    nominal_touchdown_body_targets,
    phase_masks,
    phase_masked_residual,
    project_workspace,
    scale_asymmetric,
    self_collision_detected,
    smooth_gait_action,
    torque_saturation_cost,
    update_airborne_state,
)


ACTION_SIZE = 22
OBSERVATION_SIZE = 110
ACTION_CONTRACT_VERSION = "cartesian_gait_residual_v2"
OBSERVATION_CONTRACT_VERSION = "body_state_coarse9_touchdown6_v1"


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
        command_max_yaw_rate=0.35,
        command_curriculum=config_dict.create(
            forward_only_steps=250,
            limited_yaw_steps=250,
            speed_min=(0.03, 0.05, 0.03),
            speed_max=(0.08, 0.12, 0.21),
            yaw_limit=(0.00, 0.15, 0.35),
            resample_seconds=(1.5, 4.0),
        ),
        controller=config_dict.create(
            nominal=config_dict.create(
                phase_time=0.5,
                base_swing_height=0.07,
                base_radial_offset=0.01,
            ),
            residual=config_dict.create(
                gait_filter_time_constant=0.15,
                command=config_dict.create(
                    # Physical semantics are identical to terrain so a flat
                    # checkpoint can transfer without remapping its outputs.
                    swing_x=0.025,
                    swing_y=0.015,
                    swing_z_low=-0.010,
                    swing_z_high=0.050,
                    stance_z=0.008,
                    stride_half_range=0.20,
                    frequency_half_range=0.15,
                    swing_height_min=0.050,
                    swing_height_max=0.110,
                    radial_min=0.005,
                    radial_max=0.025,
                ),
                terrain=config_dict.create(
                    swing_x=0.025,
                    swing_y=0.015,
                    swing_z_low=-0.010,
                    swing_z_high=0.050,
                    stance_z=0.008,
                    stride_half_range=0.20,
                    frequency_half_range=0.15,
                    swing_height_min=0.050,
                    swing_height_max=0.110,
                    radial_min=0.005,
                    radial_max=0.025,
                ),
            ),
            contact=config_dict.create(
                contact_enter_clearance=0.035,
                contact_release_clearance=0.045,
                lost_contact_search=0.010,
            ),
            safety=config_dict.create(
                workspace_min_distance=0.112,
                workspace_max_distance=0.345,
                projection_reference=0.050,
                joint_limit=2.356194,
                max_joint_speed=4.1887902047863905,
                max_effective_stride=0.120,
                actuator_force_limit=8.0,
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
            friction=1.0,
            flat_friction=DRY_ASPHALT_FRICTION,
            patch_probabilities=(0.10, 0.15, 0.20, 0.20, 0.20, 0.15),
        ),
        # Ranges are consumed by the optional per-env randomization wrapper.
        randomization=config_dict.create(
            enabled=False,
            step_height_range=(0.020, 0.060),
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
            gait_residual=-0.060,
            foot_action_rate=-0.005,
            gait_action_rate=-0.020,
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


class HexapodRoughTerrainEnv(mjx_env.MjxEnv):
    """Classical tripod owns phase; RL has masked foot/gait authority.

    Action remains 22-D, but v2 maps ``[0:18]`` to swing XYZ and stance Z
    only.  Existing v1 22-D checkpoints therefore must not be resumed.
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
        self._fixed_curriculum_stage = fixed_curriculum_stage
        self._scripted_commands = scripted_commands
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

        origins, outward, raw_signs, tripod_a = [], [], [], []
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
        self._origins = jp.array(origins)
        self._outward = jp.array(outward)
        self._raw_signs = jp.array(raw_signs)
        self._tripod_a = jp.array(tripod_a)

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

            ramp_start, ramp_length, ramp_rise = MIXED_RAMP
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
            stair_inside = (
                (jp.abs(stair_x - jp.array([0.50 + 0.28 * i for i in range(6)])) <= 0.14)
                & (jp.abs(y[..., None] - MIXED_PATCH_Y[4]) <= MIXED_LANE_HALF_WIDTH)
            )
            stair_height = jp.max(
                jp.where(stair_inside, jp.array([0.035 * (i + 1) for i in range(6)]), 0.0),
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

        stage0 = jp.array((0.06, 0.0))
        stage1_index = jp.clip((local_seconds / 3.0).astype(jp.int32), 0, 2)
        stage1 = jp.array(
            ((0.09, 0.12), (0.09, -0.12), (0.09, 0.08))
        )[stage1_index]
        stage2_index = jp.clip((local_seconds / 3.0).astype(jp.int32), 0, 3)
        stage2 = jp.array(
            ((0.08, 0.0), (0.14, 0.30), (0.10, -0.30), (0.21, 0.15))
        )[stage2_index]
        return jp.where(stage == 0, stage0, jp.where(stage == 1, stage1, stage2))

    def _sample_command(self, rng: jax.Array, stage: jax.Array) -> jax.Array:
        speed_key, yaw_key = jax.random.split(rng)
        if self._command_curriculum:
            speed_min = jp.array(self._config.command_curriculum.speed_min)[stage]
            speed_max = jp.array(self._config.command_curriculum.speed_max)[stage]
            yaw_limit = jp.array(self._config.command_curriculum.yaw_limit)[stage]
        else:
            speed_min = self._config.command_min_speed
            speed_max = self._config.command_max_speed
            yaw_limit = self._config.command_max_yaw_rate
        speed = jax.random.uniform(speed_key, (), minval=speed_min, maxval=speed_max)
        yaw_rate = jax.random.uniform(yaw_key, (), minval=-yaw_limit, maxval=yaw_limit)
        return jp.array((speed, yaw_rate))

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
            command=info["command"],
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
        self, action: jax.Array, command: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        authority = self._authority()
        nominal = self._config.controller.nominal
        requested_step_scale = 1.0 + authority.stride_half_range * action[18]
        step_scale, effective_stride = limit_effective_stride(
            requested_scale=requested_step_scale,
            command=command,
            origins=self._origins,
            outward=self._outward,
            phase_time=nominal.phase_time,
            max_stride=self._config.controller.safety.max_effective_stride,
        )
        frequency_scale = 1.0 + authority.frequency_half_range * action[19]
        swing_height = nominal.base_swing_height + scale_asymmetric(
            action[20],
            authority.swing_height_min - nominal.base_swing_height,
            authority.swing_height_max - nominal.base_swing_height,
        )
        radial_offset = nominal.base_radial_offset + scale_asymmetric(
            action[21], authority.radial_min - nominal.base_radial_offset,
            authority.radial_max - nominal.base_radial_offset,
        )
        return (
            step_scale,
            frequency_scale,
            swing_height,
            radial_offset,
            effective_stride,
        )

    def _controller_targets(
        self,
        action: jax.Array,
        phase: jax.Array,
        command: jax.Array,
        blend: jax.Array,
        contacts: jax.Array,
        airborne: jax.Array,
        current_feet_local: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Nominal -> phase-masked RL -> contact -> workspace -> IK."""
        action = jp.clip(action, -1.0, 1.0)
        authority = self._authority()
        (
            step_scale,
            frequency_scale,
            swing_height,
            radial_offset,
            effective_stride,
        ) = self._gait_terms(action, command)
        nominal, swing = nominal_foot_targets(
            origins=self._origins,
            outward=self._outward,
            tripod_a=self._tripod_a,
            phase=phase,
            command=command,
            phase_time=self._config.controller.nominal.phase_time,
            step_scale=step_scale,
            swing_height=swing_height,
            radial_offset=radial_offset,
        )
        residual = phase_masked_residual(
            action[:18].reshape(6, 3), swing,
            swing_x=authority.swing_x,
            swing_y=authority.swing_y,
            swing_z_low=authority.swing_z_low,
            swing_z_high=authority.swing_z_high,
            stance_z=authority.stance_z,
        )
        requested = nominal + blend * residual
        adapted, early_landing, lost_contact = contact_adapt_targets(
            requested, current_feet_local, swing, contacts, airborne,
            lost_contact_search=self._config.controller.contact.lost_contact_search,
        )
        safe_feet, projection_cost = project_workspace(
            adapted,
            min_distance=self._config.controller.safety.workspace_min_distance,
            max_distance=self._config.controller.safety.workspace_max_distance,
        )
        raw = analytical_ik(safe_feet) * self._raw_signs
        targets = self._home_ctrl.at[self._actuator_ids].set(raw)
        targets = jp.clip(targets, -self._config.controller.safety.joint_limit, self._config.controller.safety.joint_limit)
        targets = self._home_ctrl + blend * (targets - self._home_ctrl)
        return targets, {
            "frequency_scale": frequency_scale,
            "step_scale": step_scale,
            "swing_height": swing_height,
            "radial_offset": radial_offset,
            "effective_stride": effective_stride,
            "swing": swing,
            "residual": residual,
            "early_landing": early_landing,
            "lost_contact": lost_contact,
            "projection_cost": projection_cost,
            "requested_feet": requested,
            "safe_feet": safe_feet,
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
            probabilities = jp.array(self._config.terrain.patch_probabilities)
            probabilities = probabilities / jp.sum(probabilities)
            terrain_patch = jax.random.choice(
                patch_key, len(MIXED_PATCH_NAMES), (), p=probabilities
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
        info = {
            "rng": rng,
            "command": command,
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
            curriculum_stage=curriculum_stage.astype(jp.float32),
            contact_early_landing=jp.zeros(()), contact_lost=jp.zeros(()),
        )
        return mjx_env.State(data, self._get_obs(data, info), jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        previous_action = state.info["last_action"]
        applied_gait = smooth_gait_action(
            previous_action[18:],
            action[18:],
            control_dt=self.dt,
            time_constant=self._config.controller.residual.gait_filter_time_constant,
        )
        applied_action = action.at[18:].set(applied_gait)
        blend = jp.clip(state.info["steps"] * self.dt / 0.75, 0.0, 1.0)
        targets, controller = self._controller_targets(
            applied_action, state.info["phase"], state.info["command"], blend,
            state.info["contact_state"], state.info["airborne"],
            self._feet_leg_local(state.data),
        )
        max_delta = self._config.controller.safety.max_joint_speed * self.dt
        targets = state.data.ctrl + jp.clip(targets - state.data.ctrl, -max_delta, max_delta)
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)

        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        forward_velocity = jp.dot(local_velocity, MODEL_FORWARD)
        velocity_error_mps = jp.abs(
            forward_velocity - state.info["command"][0]
        )
        yaw_error_rps = jp.abs(data.qvel[5] - state.info["command"][1])
        up_z = _quat_rotate(quat, jp.array((0.0, 0.0, 1.0)))[2]
        contacts = self._foot_contacts(data, state.info["contact_state"])
        support_height = self._support_height(data, contacts, controller["swing"])
        clearance = data.qpos[2] - support_height
        terminated = (up_z < 0.35) | (clearance < 0.14) | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        foot_velocity = (data.site_xpos[self._foot_site_ids] - state.info["previous_foot_positions"]) / self.dt
        contact_count = jp.maximum(jp.sum(contacts.astype(jp.float32)), 1.0)
        slip_cost = jp.sum(contacts.astype(jp.float32) * jp.sum(jp.square(foot_velocity[:, :2] / 0.30), axis=-1)) / contact_count
        previous_policy_action = state.info["last_policy_action"]
        swing = controller["swing"]
        foot_action = applied_action[:18].reshape(6, 3)
        previous_foot_action = previous_action[:18].reshape(6, 3)
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

        reward_terms = {
            "velocity": jp.exp(-jp.square(forward_velocity - state.info["command"][0]) / 0.02),
            "yaw": jp.exp(-jp.square(data.qvel[5] - state.info["command"][1]) / 0.16),
            "upright": jp.clip(up_z, 0.0, 1.0),
            "height": jp.exp(-jp.square(clearance - 0.33) / 0.015),
            "progress": jp.clip(forward_velocity, -0.2, 0.3),
            "swing_residual": swing_residual_cost,
            "stance_residual": stance_residual_cost,
            "gait_residual": jp.mean(jp.square(applied_action[18:])),
            "foot_action_rate": jp.mean(jp.square(foot_action - previous_foot_action)),
            "gait_action_rate": jp.mean(
                jp.square(action[18:] - previous_policy_action[18:])
            ),
            "vertical_velocity": jp.square(data.qvel[2]),
            "lateral_velocity": jp.square(local_velocity[0]),
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

        phase_increment = self.dt / (2.0 * self._config.controller.nominal.phase_time) * controller["frequency_scale"]
        state.info["phase"] = jp.mod(state.info["phase"] + phase_increment, 1.0)
        next_steps = state.info["steps"] + 1
        forward_distance = data.qpos[0] - state.info["start_root_position"][0]
        timed_out = next_steps >= int(self._config.episode_length)
        terrain_success = (
            timed_out
            & (~terminated)
            & (jp.abs(forward_velocity - state.info["command"][0]) < 0.08)
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
        state.info["last_action"] = applied_action
        state.info["last_policy_action"] = action
        state.info["previous_foot_positions"] = data.site_xpos[self._foot_site_ids]
        state.info["contact_state"] = contacts
        state.info["support_height"] = support_height
        state.info["airborne"] = update_airborne_state(
            state.info["airborne"], controller["swing"], contacts
        )
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
        state.metrics["gait_filter_error"] = jp.sqrt(
            jp.mean(jp.square(action[18:] - applied_action[18:]))
        )
        state.metrics["curriculum_stage"] = state.info["curriculum_stage"].astype(jp.float32)
        state.metrics["contact_early_landing"] = jp.mean(controller["early_landing"].astype(jp.float32))
        state.metrics["contact_lost"] = jp.mean(controller["lost_contact"].astype(jp.float32))
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
