"""Firmware-based MJX residual environment for staircase locomotion."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

import firmware_mjx_controller as firmware
from prepare_rl_scene import (
    RL_SCENE_OUTPUT,
    STAIR_TOTAL_RISE,
    STEP_COUNT,
    STEP_DEPTH,
    STEP_START_X,
    prepare_rl_scene,
)
from tripod_controller import LEG_PREFIXES


ACTION_SIZE = 18
OBSERVATION_SIZE = 142
ACTION_CONTRACT_VERSION = "stm32_firmware_cartesian_foot_residual_v1"
OBSERVATION_CONTRACT_VERSION = "firmware_state_collision_contact_stairs_v1"
MODEL_FORWARD = jp.array((0.0, -1.0, 0.0))
MODEL_LATERAL = jp.array((1.0, 0.0, 0.0))
STAIR_TOP_X = STEP_START_X + STEP_DEPTH * (STEP_COUNT - 1) + STEP_DEPTH / 2.0
# Difficulty is defined by the height of the complete seven-step staircase,
# not by one riser.  Level 4 therefore ends exactly 20 cm above the floor.
CURRICULUM_TOTAL_RISES = (0.0, 0.05, 0.10, 0.15, STAIR_TOTAL_RISE)
CURRICULUM_STEP_HEIGHTS = tuple(
    total_rise / STEP_COUNT for total_rise in CURRICULUM_TOTAL_RISES
)


def default_config() -> config_dict.ConfigDict:
    """Training defaults biased toward a stable complete staircase ascent."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.0025,
        episode_length=2000,
        impl="jax",
        nconmax=256,
        njmax=128,
        command_min_speed=0.06,
        command_max_speed=0.12,
        command_max_yaw_rate=0.0,
        command_delay=1.0,
        target_clearance=0.316,
        safety=config_dict.create(
            max_tilt=0.7853981633974483,
            min_clearance=0.14,
            max_root_linear_speed=1.5,
            max_root_angular_speed=6.0,
            max_joint_speed=20.0,
            joint_limit_margin=0.017453292519943295,
        ),
        reward=config_dict.create(
            velocity=2.5,
            yaw=0.25,
            upright=1.0,
            height=0.6,
            progress=1.0,
            stability=0.5,
            joint_margin=0.3,
            action_rate=-0.02,
            residual=-0.005,
            vertical_velocity=-0.10,
            lateral_velocity=-0.10,
            joint_velocity=-0.04,
            torque=-0.03,
            torque_saturation=-0.25,
            gait_rejected=-0.20,
            posture_rejected=-0.20,
            policy_rejected=-0.50,
            foot_limited=-0.25,
            body_contact=-2.0,
            self_collision=-0.5,
        ),
        ascent_bonus=8.0,
        success_bonus=30.0,
        failure_penalty=-30.0,
    )


def _quat_conjugate(quaternion: jax.Array) -> jax.Array:
    return quaternion * jp.array((1.0, -1.0, -1.0, -1.0))


def _quat_multiply(left: jax.Array, right: jax.Array) -> jax.Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jp.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quat_rotate(quat: jax.Array, vector: jax.Array) -> jax.Array:
    scalar = quat[0]
    xyz = quat[1:]
    temp = 2.0 * jp.cross(xyz, vector)
    return vector + scalar * temp + jp.cross(xyz, temp)


def _quat_rotate_inverse(quat: jax.Array, vector: jax.Array) -> jax.Array:
    return _quat_rotate(_quat_conjugate(quat), vector)


def _quat_to_euler(quaternion: jax.Array) -> jax.Array:
    quaternion = quaternion / jp.maximum(jp.linalg.norm(quaternion), 1.0e-8)
    w, x, y, z = quaternion
    return jp.array(
        (
            jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        )
    )


def mujoco_terrain_foot_contacts(
    contact_geom: jax.Array,
    contact_distance: jax.Array,
    foot_geom_ids: jax.Array,
    geom_body_ids: jax.Array,
) -> jax.Array:
    """Use only real foot-to-world collisions as firmware contact input."""
    safe_geom = jp.maximum(contact_geom, 0)
    first_geom, second_geom = safe_geom[:, 0], safe_geom[:, 1]
    first_world = geom_body_ids[first_geom] == 0
    second_world = geom_body_ids[second_geom] == 0
    first_foot = first_geom[:, None] == foot_geom_ids[None, :]
    second_foot = second_geom[:, None] == foot_geom_ids[None, :]
    pair = (first_foot & second_world[:, None]) | (
        second_foot & first_world[:, None]
    )
    active = (
        (contact_geom[:, 0] >= 0)
        & (contact_geom[:, 1] >= 0)
        & (contact_distance <= 0.0)
    )
    return jp.any(active[:, None] & pair, axis=0)


class HexapodRoughTerrainEnv(mjx_env.MjxEnv):
    """Exact STM32 base controller plus a safety-gated 18-D foot residual.

    The firmware owns command filtering, pose feedback, gait phase, contact
    adaptation, swing/stance trajectories, posture feedback, IK and joint
    rate limiting.  The policy may adjust swing XYZ and stance Z only.  An
    unreachable policy request is rejected before it can reach an actuator.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[dict[str, Any]] = None,
        *,
        terrain_level: int = 4,
    ) -> None:
        if terrain_level not in range(len(CURRICULUM_STEP_HEIGHTS)):
            raise ValueError(
                f"terrain_level must be 0..{len(CURRICULUM_STEP_HEIGHTS) - 1}"
            )
        self._terrain_level = terrain_level
        self._terrain_total_rise = CURRICULUM_TOTAL_RISES[terrain_level]
        self._terrain_step_height = self._terrain_total_rise / STEP_COUNT
        super().__init__(config, config_overrides)
        ratio = self.dt / firmware.FIRMWARE_CONTROL_DT
        if abs(ratio - round(ratio)) > 1.0e-9:
            raise ValueError("ctrl_dt must be an integer multiple of the 5 ms firmware tick")
        self._firmware_steps = int(round(ratio))

        prepare_rl_scene(RL_SCENE_OUTPUT)
        self._xml_path = str(RL_SCENE_OUTPUT)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        for index in range(STEP_COUNT):
            geom = self._mj_model.geom(f"stair_{index + 1}")
            height = self._terrain_step_height * (index + 1)
            if terrain_level == 0:
                self._mj_model.geom_contype[geom.id] = 0
                self._mj_model.geom_conaffinity[geom.id] = 0
                self._mj_model.geom_rgba[geom.id, 3] = 0.0
            else:
                self._mj_model.geom_pos[geom.id, 2] = height / 2.0
                self._mj_model.geom_size[geom.id, 2] = height / 2.0
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        home_id = self._mj_model.key("home").id
        self._home_qpos = jp.array(self._mj_model.key_qpos[home_id])
        self._home_quaternion = self._home_qpos[3:7]
        self._home_ctrl = jp.array(self._mj_model.key_ctrl[home_id])
        self._joint_qpos_ids = jp.array(
            [
                self._mj_model.joint(f"{prefix}_{joint}").qposadr[0]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._joint_qvel_ids = jp.array(
            [
                self._mj_model.joint(f"{prefix}_{joint}").dofadr[0]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._actuator_ids = jp.array(
            [
                self._mj_model.actuator(f"{prefix}_{joint}_position").id
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._foot_site_ids = jp.array(
            [self._mj_model.site(f"{prefix}_foot_site").id for prefix in LEG_PREFIXES]
        )
        self._foot_geom_ids = jp.array(
            [
                self._mj_model.geom(f"{prefix}_foot_collision").id
                for prefix in LEG_PREFIXES
            ]
        )
        self._geom_body_ids = jp.array(self._mj_model.geom_bodyid)
        self._torso_geom_id = self._mj_model.geom("torso_collision").id
        self._step_centers = jp.array(
            [STEP_START_X + STEP_DEPTH * index for index in range(STEP_COUNT)]
        )
        self._step_heights = jp.array(
            [self._terrain_step_height * (index + 1) for index in range(STEP_COUNT)]
        )
        sample_x, sample_y = np.meshgrid(
            np.array((-0.10, 0.15, 0.40, 0.65, 0.90)),
            np.array((-0.24, 0.0, 0.24)),
        )
        self._height_samples = jp.array(
            np.stack((sample_x.ravel(), sample_y.ravel()), axis=-1)
        )

    def _relative_attitude(self, data: mjx.Data) -> jax.Array:
        relative = _quat_multiply(
            data.qpos[3:7], _quat_conjugate(self._home_quaternion)
        )
        return _quat_to_euler(relative)

    def _terrain_height(self, xy: jax.Array) -> jax.Array:
        x, y = xy[..., 0, None], xy[..., 1, None]
        inside = (
            (jp.abs(x - self._step_centers) <= STEP_DEPTH / 2.0)
            & (jp.abs(y) <= 1.0)
        )
        return jp.max(jp.where(inside, self._step_heights, 0.0), axis=-1)

    def _foot_contacts(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        return mujoco_terrain_foot_contacts(
            contact.geom, contact.dist, self._foot_geom_ids, self._geom_body_ids
        )

    def _body_contact(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        return jp.any(
            (contact.dist < 0.0)
            & jp.any(contact.geom == self._torso_geom_id, axis=-1)
        )

    def _self_collision(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        safe_geom = jp.maximum(contact.geom, 0)
        first_body = self._geom_body_ids[safe_geom[:, 0]]
        second_body = self._geom_body_ids[safe_geom[:, 1]]
        active = (
            (contact.geom[:, 0] >= 0)
            & (contact.geom[:, 1] >= 0)
            & (contact.dist < 0.0)
        )
        return jp.any(active & (first_body > 0) & (second_body > 0))

    def _support_height(self, data: mjx.Data, contacts: jax.Array) -> jax.Array:
        feet = data.site_xpos[self._foot_site_ids]
        heights = self._terrain_height(feet[:, :2])
        count = jp.sum(contacts.astype(jp.int32))
        ordered = jp.sort(jp.where(contacts, heights, jp.inf))
        median_index = jp.maximum((count - 1) // 2, 0)
        median = ordered[median_index]
        fallback = self._terrain_height(data.qpos[:2])
        return jp.where(count > 0, median, fallback)

    def _feet_controller_body(self, data: mjx.Data) -> jax.Array:
        relative_world = data.site_xpos[self._foot_site_ids] - data.qpos[None, :3]
        model_body = jax.vmap(_quat_rotate_inverse, in_axes=(None, 0))(
            data.qpos[3:7], relative_world
        )
        return jp.stack(
            (
                model_body @ MODEL_FORWARD,
                model_body @ MODEL_LATERAL,
                model_body[:, 2],
            ),
            axis=-1,
        )

    def _terrain_features(self, data: mjx.Data, support_height: jax.Array) -> jax.Array:
        attitude = self._relative_attitude(data)
        cosine, sine = jp.cos(attitude[2]), jp.sin(attitude[2])
        rotation = jp.array(((cosine, -sine), (sine, cosine)))
        sample_world = data.qpos[None, :2] + self._height_samples @ rotation.T
        return self._terrain_height(sample_world) - support_height

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        local_gravity = _quat_rotate_inverse(quat, jp.array((0.0, 0.0, -1.0)))
        output = info["controller_output"]
        observation = jp.concatenate(
            (
                info["command"],
                local_velocity,
                0.2 * data.qvel[3:6],
                local_gravity,
                self._relative_attitude(data)[:2],
                data.qpos[self._joint_qpos_ids]
                - self._home_qpos[self._joint_qpos_ids],
                0.1 * data.qvel[self._joint_qvel_ids],
                jp.ravel(self._feet_controller_body(data)),
                info["contact_state"].astype(jp.float32),
                self._terrain_features(data, info["support_height"]),
                output.gait_progress,
                output.gait_state.astype(jp.float32) / float(firmware.LEG_LATE_LANDING),
                output.applied_twist,
                output.ik_valid.astype(jp.float32),
                output.policy_valid.astype(jp.float32),
                output.foot_limited.astype(jp.float32),
                jp.array(
                    (
                        output.gait_accepted.astype(jp.float32),
                        output.posture_accepted.astype(jp.float32),
                    )
                ),
                info["last_action"],
            )
        )
        return observation

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, q_key, vel_key, cmd_key, yaw_key = jax.random.split(rng, 5)
        qpos = self._home_qpos.at[self._joint_qpos_ids].add(
            jax.random.uniform(q_key, (18,), minval=-0.01, maxval=0.01)
        )
        qvel = jp.zeros(self.mjx_model.nv).at[:6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.01, maxval=0.01)
        )
        speed = jax.random.uniform(
            cmd_key,
            (),
            minval=self._config.command_min_speed,
            maxval=self._config.command_max_speed,
        )
        yaw_rate = jax.random.uniform(
            yaw_key,
            (),
            minval=-self._config.command_max_yaw_rate,
            maxval=self._config.command_max_yaw_rate,
        )
        data = mjx.make_data(
            self.mj_model,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._home_ctrl)
        data = mjx.forward(self.mjx_model, data)
        contacts = self._foot_contacts(data)
        support_height = self._support_height(data, contacts)
        controller_output = firmware.initial_output()
        info = {
            "rng": rng,
            "command": jp.array((speed, yaw_rate)),
            "steps": jp.zeros((), dtype=jp.int32),
            "last_action": jp.zeros(ACTION_SIZE),
            "controller_state": firmware.initial_state(),
            "controller_output": controller_output,
            "contact_state": contacts,
            "support_height": support_height,
            "max_terrain_height": support_height,
            "previous_root_x": data.qpos[0],
            "controller_rejection_steps": jp.zeros((), dtype=jp.int32),
        }
        metrics = {
            f"reward/{name}": jp.zeros(()) for name in self._config.reward.keys()
        }
        metrics.update(
            {
                "reward/ascent": jp.zeros(()),
                "reward/success": jp.zeros(()),
                "reward/failure": jp.zeros(()),
                "terrain_level": jp.zeros(()),
                "terrain_success": jp.zeros(()),
                "controller_rejection_steps": jp.zeros(()),
                "policy_rejection_fraction": jp.zeros(()),
                "foot_limited_fraction": jp.zeros(()),
                "joint_limit_margin_rad": jp.zeros(()),
                "root_linear_speed": jp.zeros(()),
                "root_angular_speed": jp.zeros(()),
                "joint_speed_max": jp.zeros(()),
                "termination/controller_invalid": jp.zeros(()),
                "termination/joint_limit": jp.zeros(()),
                "termination/dynamics": jp.zeros(()),
                "termination/tilt": jp.zeros(()),
                "termination/clearance": jp.zeros(()),
                "termination/body_contact": jp.zeros(()),
                "termination/nonfinite": jp.zeros(()),
            }
        )
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        contacts_before = self._foot_contacts(state.data)
        attitude_before = self._relative_attitude(state.data)
        command_active = state.info["steps"] * self.dt >= self._config.command_delay
        firmware_command = jp.where(command_active, state.info["command"], jp.zeros(2))

        def controller_tick(controller_state, _):
            return firmware.step(
                controller_state,
                target_velocity=firmware_command,
                body_position_world=state.data.qpos[:3],
                attitude_rpy=attitude_before,
                contacts=contacts_before,
                policy_action=action,
            )

        controller_state, controller_history = jax.lax.scan(
            controller_tick,
            state.info["controller_state"],
            xs=None,
            length=self._firmware_steps,
        )
        controller = jax.tree_util.tree_map(lambda value: value[-1], controller_history)
        targets = self._home_ctrl.at[self._actuator_ids].set(
            controller.model_joint_targets.reshape(18)
        )
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)

        attitude = self._relative_attitude(data)
        raw_local_velocity = _quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
        forward_velocity = jp.dot(raw_local_velocity, MODEL_FORWARD)
        lateral_velocity = jp.dot(raw_local_velocity, MODEL_LATERAL)
        contacts = self._foot_contacts(data)
        support_height = self._support_height(data, contacts)
        clearance = data.qpos[2] - support_height
        body_contact = self._body_contact(data)
        self_collision = self._self_collision(data)

        root_linear_speed = jp.linalg.norm(data.qvel[:3])
        root_angular_speed = jp.linalg.norm(data.qvel[3:6])
        joint_speed_max = jp.max(jp.abs(data.qvel[self._joint_qvel_ids]))
        joint_limit_margin = firmware.JOINT_LIMIT - jp.max(
            jp.abs(controller.servo_joint_targets)
        )
        controller_invalid = jp.any(~controller.ik_valid)
        joint_limit_failure = joint_limit_margin <= self._config.safety.joint_limit_margin
        dynamics_failure = (
            (root_linear_speed > self._config.safety.max_root_linear_speed)
            | (root_angular_speed > self._config.safety.max_root_angular_speed)
            | (joint_speed_max > self._config.safety.max_joint_speed)
        )
        tilt_failure = jp.any(jp.abs(attitude[:2]) > self._config.safety.max_tilt)
        clearance_failure = clearance < self._config.safety.min_clearance
        finite = (
            jp.all(jp.isfinite(data.qpos))
            & jp.all(jp.isfinite(data.qvel))
            & jp.all(jp.isfinite(controller.servo_joint_targets))
        )
        failure = (
            controller_invalid
            | joint_limit_failure
            | dynamics_failure
            | tilt_failure
            | clearance_failure
            | body_contact
            | (~finite)
        )
        success = (
            (data.qpos[0] >= STAIR_TOP_X + 0.10)
            & (
                support_height >= self._terrain_total_rise - 1.0e-3
            )
            & (jp.max(jp.abs(attitude[:2])) < jp.deg2rad(20.0))
            & (~failure)
        )
        terminated = failure | success

        torque_limit = 8.0
        torque_saturation = jp.mean(
            jp.square(jp.maximum(jp.abs(data.actuator_force) - 6.8, 0.0) / 1.2)
        )
        joint_proximity = jp.clip(
            (jp.abs(controller.servo_joint_targets) - jp.deg2rad(105.0))
            / jp.deg2rad(30.0),
            0.0,
            1.0,
        )
        rejection = (~controller.gait_accepted) | (~controller.posture_accepted)
        rejection_steps = jp.where(
            rejection, state.info["controller_rejection_steps"] + 1, 0
        )
        policy_rejected = jp.mean((~controller.policy_valid).astype(jp.float32))
        foot_limited = jp.mean(controller.foot_limited.astype(jp.float32))
        reward_terms = {
            "velocity": jp.exp(
                -jp.square(forward_velocity - state.info["command"][0]) / 0.01
            ),
            "yaw": jp.exp(
                -jp.square(data.qvel[5] - state.info["command"][1]) / 0.09
            ),
            "upright": jp.exp(-jp.sum(jp.square(attitude[:2])) / 0.12),
            "height": jp.exp(
                -jp.square(clearance - self._config.target_clearance) / 0.01
            ),
            "progress": jp.clip(forward_velocity, -0.20, 0.25),
            "stability": jp.exp(-jp.square(root_angular_speed / 2.0)),
            "joint_margin": 1.0 - jp.mean(joint_proximity),
            "action_rate": jp.mean(jp.square(action - state.info["last_action"])),
            "residual": jp.mean(jp.square(action)),
            "vertical_velocity": jp.square(data.qvel[2]),
            "lateral_velocity": jp.square(lateral_velocity),
            "joint_velocity": jp.mean(
                jp.square(data.qvel[self._joint_qvel_ids] / 10.0)
            ),
            "torque": jp.mean(jp.square(data.actuator_force / torque_limit)),
            "torque_saturation": torque_saturation,
            "gait_rejected": (~controller.gait_accepted).astype(jp.float32),
            "posture_rejected": (~controller.posture_accepted).astype(jp.float32),
            "policy_rejected": policy_rejected,
            "foot_limited": foot_limited,
            "body_contact": body_contact.astype(jp.float32),
            "self_collision": self_collision.astype(jp.float32),
        }
        scaled = {
            name: value * self._config.reward[name]
            for name, value in reward_terms.items()
        }
        running_reward = sum(scaled.values()) * self.dt
        positive_ascent = jp.maximum(
            support_height - state.info["max_terrain_height"], 0.0
        )
        max_terrain_height = jp.maximum(
            state.info["max_terrain_height"], support_height
        )
        ascent_bonus = (
            self._config.ascent_bonus
            * positive_ascent
            / max(self._terrain_total_rise, 1.0e-3)
        )
        success_bonus = jp.where(success, self._config.success_bonus, 0.0)
        failure_penalty = jp.where(failure, self._config.failure_penalty, 0.0)
        reward = jp.clip(
            running_reward + ascent_bonus + success_bonus + failure_penalty,
            -50.0,
            50.0,
        )

        state.info["steps"] += 1
        state.info["last_action"] = action
        state.info["controller_state"] = controller_state
        state.info["controller_output"] = controller
        state.info["contact_state"] = contacts
        state.info["support_height"] = support_height
        state.info["max_terrain_height"] = max_terrain_height
        state.info["previous_root_x"] = data.qpos[0]
        state.info["controller_rejection_steps"] = rejection_steps
        obs = self._get_obs(data, state.info)
        for name, value in scaled.items():
            state.metrics[f"reward/{name}"] = value
        state.metrics["reward/ascent"] = ascent_bonus
        state.metrics["reward/success"] = success_bonus
        state.metrics["reward/failure"] = failure_penalty
        state.metrics["terrain_level"] = max_terrain_height / max(
            self._terrain_step_height, 1.0e-3
        )
        state.metrics["terrain_success"] = success.astype(jp.float32)
        state.metrics["controller_rejection_steps"] = rejection_steps.astype(jp.float32)
        state.metrics["policy_rejection_fraction"] = policy_rejected
        state.metrics["foot_limited_fraction"] = foot_limited
        state.metrics["joint_limit_margin_rad"] = joint_limit_margin
        state.metrics["root_linear_speed"] = root_linear_speed
        state.metrics["root_angular_speed"] = root_angular_speed
        state.metrics["joint_speed_max"] = joint_speed_max
        state.metrics["termination/controller_invalid"] = controller_invalid.astype(jp.float32)
        state.metrics["termination/joint_limit"] = joint_limit_failure.astype(jp.float32)
        state.metrics["termination/dynamics"] = dynamics_failure.astype(jp.float32)
        state.metrics["termination/tilt"] = tilt_failure.astype(jp.float32)
        state.metrics["termination/clearance"] = clearance_failure.astype(jp.float32)
        state.metrics["termination/body_contact"] = body_contact.astype(jp.float32)
        state.metrics["termination/nonfinite"] = (~finite).astype(jp.float32)
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=terminated.astype(jp.float32),
        )

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def observation_size(self) -> int:
        return OBSERVATION_SIZE

    @property
    def episode_length(self) -> int:
        return int(self._config.episode_length)

    @property
    def curriculum_level(self) -> int:
        return self._terrain_level

    @property
    def terrain_step_height(self) -> float:
        return self._terrain_step_height

    @property
    def terrain_total_rise(self) -> float:
        return self._terrain_total_rise

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
