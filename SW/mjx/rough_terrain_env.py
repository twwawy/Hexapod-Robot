"""MJX flat/stair environments with a classical-first 22-D residual contract."""

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
    FLAT_RL_SCENE_OUTPUT,
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
    nominal_foot_targets,
    phase_masked_residual,
    project_workspace,
    scale_asymmetric,
)


ACTION_SIZE = 22
OBSERVATION_SIZE = 110
ACTION_CONTRACT_VERSION = "cartesian_gait_residual_v2"


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
            speed_max=(0.08, 0.12, 0.18),
            yaw_limit=(0.00, 0.15, 0.35),
        ),
        controller=config_dict.create(
            nominal=config_dict.create(
                phase_time=0.5,
                base_swing_height=0.07,
                base_radial_offset=0.01,
            ),
            residual=config_dict.create(
                command=config_dict.create(
                    swing_x=0.010,
                    swing_y=0.008,
                    swing_z_low=-0.005,
                    swing_z_high=0.020,
                    stance_z=0.003,
                    stride_half_range=0.20,
                    frequency_half_range=0.15,
                    swing_height_min=0.065,
                    swing_height_max=0.090,
                    radial_min=0.005,
                    radial_max=0.018,
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
                foot_clearance=0.038,
                lost_contact_search=0.010,
            ),
            safety=config_dict.create(
                workspace_min_distance=0.112,
                workspace_max_distance=0.345,
                projection_reference=0.050,
                joint_limit=2.356194,
                max_joint_speed=4.1887902047863905,
                actuator_force_limit=8.0,
            ),
        ),
        observation=config_dict.create(
            forward_offsets=(-0.15, 0.05, 0.25, 0.45, 0.65),
            lateral_offsets=(-0.22, 0.0, 0.22),
        ),
        terrain=config_dict.create(
            step_start_x=STEP_START_X,
            step_depth=STEP_DEPTH,
            step_height=STEP_HEIGHT,
            step_count=STEP_COUNT,
            friction=1.0,
        ),
        # Values are recorded now; deterministic v2 is trained before this
        # domain-randomization schedule is switched on.
        randomization=config_dict.create(
            enabled=False,
            step_height_range=(0.020, 0.060),
            step_depth_range=(0.180, 0.350),
            friction_range=(0.6, 1.3),
            mass_scale_range=(0.90, 1.10),
            actuator_strength_range=(0.85, 1.15),
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
            swing_residual=-0.010,
            stance_residual=-0.040,
            gait_residual=-0.020,
            foot_action_rate=-0.005,
            gait_action_rate=-0.020,
            vertical_velocity=-0.10,
            lateral_velocity=-0.10,
            joint_velocity=-0.002,
            torque=-0.010,
            slip=-0.050,
            projection=-0.50,
            body_contact=-1.0,
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
        terrain: str = "stairs",
        command_curriculum: bool = False,
    ) -> None:
        if terrain not in {"flat", "stairs"}:
            raise ValueError(f"terrain must be 'flat' or 'stairs', got {terrain!r}")
        self._terrain = terrain
        self._command_curriculum = command_curriculum
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
        else:
            prepare_flat_rl_scene(
                FLAT_RL_SCENE_OUTPUT, friction=self._config.terrain.friction
            )
            scene_path = FLAT_RL_SCENE_OUTPUT
        self._xml_path = str(scene_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        # This is intentional run-level randomization: every experiment gets
        # one coherent model, and the sampled values are preserved in metadata.
        randomization = self._config.randomization
        self._mj_model.body_mass[1:] *= randomization.mass_scale
        self._mj_model.body_inertia[1:] *= randomization.mass_scale
        self._mj_model.actuator_forcerange *= randomization.actuator_strength_scale
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
        x = xy[..., 0, None]
        y = xy[..., 1, None]
        inside = (
            (jp.abs(x - self._step_centers) <= self._config.terrain.step_depth / 2.0)
            & (jp.abs(y) <= 1.0)
        )
        return jp.max(jp.where(inside, self._step_heights, 0.0), axis=-1)

    def _curriculum_stage(self, steps: jax.Array) -> jax.Array:
        if not self._command_curriculum:
            return jp.zeros((), dtype=jp.int32)
        first_end = int(self._config.command_curriculum.forward_only_steps)
        second_end = first_end + int(self._config.command_curriculum.limited_yaw_steps)
        return jp.where(steps < first_end, 0, jp.where(steps < second_end, 1, 2)).astype(jp.int32)

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

    def _foot_contacts(self, data: mjx.Data) -> jax.Array:
        """Actual foot collision with a clearance fallback for robust sensing."""
        contact = data._impl.contact
        active = contact.dist < 0.0
        foot_match = jp.any(contact.geom[:, :, None] == self._foot_geom_ids[None, None, :], axis=1)
        physical = jp.any(active[:, None] & foot_match, axis=0)
        feet = data.site_xpos[self._foot_site_ids]
        clearance = feet[:, 2] - self._terrain_height(feet[:, :2])
        return physical | (clearance < self._config.controller.contact.foot_clearance)

    def _torso_contact(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        return jp.any((contact.dist < 0.0) & jp.any(contact.geom == self._torso_geom_id, axis=-1))

    def _heading_terrain_samples(self, data: mjx.Data) -> jax.Array:
        """Horizontal forward/lateral terrain samples: yaw aligned, no pitch tilt."""
        forward = _quat_rotate(data.qpos[3:7], MODEL_FORWARD)[:2]
        sample_world = heading_aligned_points(data.qpos[:2], forward, self._terrain_offsets)
        return self._terrain_height(sample_world) - data.qpos[2]

    def _gait_terms(self, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        authority = self._authority()
        nominal = self._config.controller.nominal
        step_scale = 1.0 + authority.stride_half_range * action[18]
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
        return step_scale, frequency_scale, swing_height, radial_offset

    def _controller_targets(
        self,
        action: jax.Array,
        phase: jax.Array,
        command: jax.Array,
        blend: jax.Array,
        contacts: jax.Array,
        current_feet_local: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Nominal -> phase-masked RL -> contact -> workspace -> IK."""
        action = jp.clip(action, -1.0, 1.0)
        authority = self._authority()
        step_scale, frequency_scale, swing_height, radial_offset = self._gait_terms(action)
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
            requested, current_feet_local, swing, contacts,
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
                jp.ravel(self._feet_body(data)), self._foot_contacts(data).astype(jp.float32),
                self._heading_terrain_samples(data),
                jp.array((jp.sin(2 * jp.pi * info["phase"]), jp.cos(2 * jp.pi * info["phase"]))),
                info["last_action"],
            )
        )
        return obs

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, q_key, vel_key, command_key = jax.random.split(rng, 4)
        qpos = self._home_qpos.at[self._joint_qpos_ids].add(
            jax.random.uniform(q_key, (18,), minval=-0.015, maxval=0.015)
        )
        qvel = jp.zeros(self.mjx_model.nv).at[:6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.02, maxval=0.02)
        )
        curriculum_stage = self._curriculum_stage(jp.zeros((), dtype=jp.int32))
        data = mjx.make_data(self.mj_model, impl=self.mjx_model.impl.value, naconmax=self._config.nconmax, njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data.replace(qpos=qpos, qvel=qvel, ctrl=self._home_ctrl))
        info = {
            "rng": rng,
            "command": self._sample_command(command_key, curriculum_stage),
            "phase": jp.zeros(()),
            "steps": jp.zeros((), dtype=jp.int32),
            "curriculum_stage": curriculum_stage,
            "last_action": jp.zeros(ACTION_SIZE),
            "previous_foot_positions": data.site_xpos[self._foot_site_ids],
        }
        metrics = {f"reward/{name}": jp.zeros(()) for name in self._config.reward.keys()}
        metrics.update(
            workspace_error=jp.zeros(()), projection_cost=jp.zeros(()),
            curriculum_stage=curriculum_stage.astype(jp.float32),
            contact_early_landing=jp.zeros(()), contact_lost=jp.zeros(()),
        )
        return mjx_env.State(data, self._get_obs(data, info), jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        blend = jp.clip(state.info["steps"] * self.dt / 0.75, 0.0, 1.0)
        targets, controller = self._controller_targets(
            action, state.info["phase"], state.info["command"], blend,
            self._foot_contacts(state.data), self._feet_leg_local(state.data),
        )
        max_delta = self._config.controller.safety.max_joint_speed * self.dt
        targets = state.data.ctrl + jp.clip(targets - state.data.ctrl, -max_delta, max_delta)
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)

        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        forward_velocity = jp.dot(local_velocity, MODEL_FORWARD)
        up_z = _quat_rotate(quat, jp.array((0.0, 0.0, 1.0)))[2]
        clearance = data.qpos[2] - self._terrain_height(data.qpos[:2])
        terminated = (up_z < 0.35) | (clearance < 0.14) | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        contacts = self._foot_contacts(data)
        foot_velocity = (data.site_xpos[self._foot_site_ids] - state.info["previous_foot_positions"]) / self.dt
        contact_count = jp.maximum(jp.sum(contacts.astype(jp.float32)), 1.0)
        slip_cost = jp.sum(contacts.astype(jp.float32) * jp.sum(jp.square(foot_velocity[:, :2] / 0.30), axis=-1)) / contact_count
        previous_action = state.info["last_action"]
        swing = controller["swing"]
        foot_action = action[:18].reshape(6, 3)
        previous_foot_action = previous_action[:18].reshape(6, 3)
        swing_count = jp.maximum(jp.sum(swing.astype(jp.float32)) * 3.0, 1.0)
        stance_count = jp.maximum(jp.sum((~swing).astype(jp.float32)), 1.0)
        swing_residual_cost = jp.sum(jp.square(foot_action) * swing[:, None]) / swing_count
        stance_residual_cost = jp.sum(jp.square(foot_action[:, 2]) * (~swing)) / stance_count
        projection_cost = controller["projection_cost"] / jp.square(self._config.controller.safety.projection_reference)

        reward_terms = {
            "velocity": jp.exp(-jp.square(forward_velocity - state.info["command"][0]) / 0.02),
            "yaw": jp.exp(-jp.square(data.qvel[5] - state.info["command"][1]) / 0.16),
            "upright": jp.clip(up_z, 0.0, 1.0),
            "height": jp.exp(-jp.square(clearance - 0.33) / 0.015),
            "progress": jp.clip(forward_velocity, -0.2, 0.3),
            "swing_residual": swing_residual_cost,
            "stance_residual": stance_residual_cost,
            "gait_residual": jp.mean(jp.square(action[18:])),
            "foot_action_rate": jp.mean(jp.square(foot_action - previous_foot_action)),
            "gait_action_rate": jp.mean(jp.square(action[18:] - previous_action[18:])),
            "vertical_velocity": jp.square(data.qvel[2]),
            "lateral_velocity": jp.square(local_velocity[0]),
            "joint_velocity": jp.mean(jp.square(data.qvel[self._joint_qvel_ids] / 10.0)),
            "torque": jp.mean(jp.square(data.actuator_force / self._config.controller.safety.actuator_force_limit)),
            "slip": slip_cost,
            "projection": projection_cost,
            "body_contact": self._torso_contact(data).astype(jp.float32),
            "termination": terminated.astype(jp.float32),
        }
        scaled = {name: value * self._config.reward[name] for name, value in reward_terms.items()}
        reward = jp.clip(sum(scaled.values()) * self.dt, -10.0, 10.0)

        phase_increment = self.dt / (2.0 * self._config.controller.nominal.phase_time) * controller["frequency_scale"]
        state.info["phase"] = jp.mod(state.info["phase"] + phase_increment, 1.0)
        next_steps = state.info["steps"] + 1
        if self._command_curriculum:
            next_stage = self._curriculum_stage(next_steps)
            next_rng, command_key = jax.random.split(state.info["rng"])
            changed = next_stage != state.info["curriculum_stage"]
            state.info["rng"] = next_rng
            state.info["command"] = jp.where(changed, self._sample_command(command_key, next_stage), state.info["command"])
            state.info["curriculum_stage"] = next_stage
        state.info["steps"] = next_steps
        state.info["last_action"] = action
        state.info["previous_foot_positions"] = data.site_xpos[self._foot_site_ids]
        for name, value in scaled.items():
            state.metrics[f"reward/{name}"] = value
        state.metrics["workspace_error"] = controller["projection_cost"]
        state.metrics["projection_cost"] = controller["projection_cost"]
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
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
