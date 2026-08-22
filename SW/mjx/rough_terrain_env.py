"""Shared MJX residual-learning environment for flat and stair terrain."""

from __future__ import annotations

from pathlib import Path
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


ACTION_SIZE = 22
MODEL_FORWARD = jp.array((0.0, -1.0, 0.0))


def default_config() -> config_dict.ConfigDict:
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
            # One 1,000-step episode is ordered as walk -> gentle turn ->
            # full walk-and-turn.  This is deliberately a *single* task: the
            # action/observation/controller contract never changes.
            forward_only_steps=250,
            limited_yaw_steps=250,
            speed_min=(0.03, 0.05, 0.03),
            speed_max=(0.08, 0.12, 0.18),
            yaw_limit=(0.00, 0.15, 0.35),
        ),
        phase_time=0.5,
        base_swing_height=0.07,
        base_radial_offset=0.01,
        residual_scale=[0.04, 0.03, 0.09],
        reward=config_dict.create(
            velocity=2.0,
            yaw=0.5,
            upright=0.5,
            height=0.3,
            progress=0.2,
            action_rate=-0.02,
            residual=-0.005,
            vertical_velocity=-0.10,
            lateral_velocity=-0.10,
            joint_velocity=-0.0002,
            workspace=-1.0,
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
    """Learn residuals on a mesh-free flat or staircase training scene.

    Action layout:
      [0:18]  six per-leg foot residuals (x, y, z in each leg frame)
      [18]    stride-length scale
      [19]    gait-frequency scale
      [20]    swing-height residual
      [21]    radial/stance-width residual
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
            prepare_rl_scene(RL_SCENE_OUTPUT, terrain="stairs")
            scene_path = RL_SCENE_OUTPUT
        else:
            prepare_flat_rl_scene(FLAT_RL_SCENE_OUTPUT)
            scene_path = FLAT_RL_SCENE_OUTPUT
        self._xml_path = str(scene_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        home_id = self._mj_model.key("home").id
        self._home_qpos = jp.array(self._mj_model.key_qpos[home_id])
        self._home_ctrl = jp.array(self._mj_model.key_ctrl[home_id])
        self._joint_qpos_ids = jp.array(
            [
                self._mj_model.jnt_qposadr[
                    self._mj_model.joint(f"{prefix}_{joint}").id
                ]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._joint_qvel_ids = jp.array(
            [
                self._mj_model.jnt_dofadr[
                    self._mj_model.joint(f"{prefix}_{joint}").id
                ]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._actuator_ids = jp.array(
            [
                [
                    self._mj_model.actuator(
                        f"{prefix}_{joint}_position"
                    ).id
                    for joint in (1, 2, 3)
                ]
                for prefix in LEG_PREFIXES
            ]
        )
        self._foot_site_ids = jp.array(
            [self._mj_model.site(f"{prefix}_foot_site").id for prefix in LEG_PREFIXES]
        )

        origins = []
        outward = []
        raw_signs = []
        tripod_a = []
        for prefix in LEG_PREFIXES:
            body_id = self._mj_model.body(f"{prefix}_motor_horn_1_1").id
            origin = np.asarray(self._mj_model.body_pos[body_id], dtype=float)
            direction = origin.copy()
            direction[2] = 0.0
            direction /= np.linalg.norm(direction)
            origins.append(origin)
            outward.append(direction)
            raw_signs.append(
                (1.0, -1.0, 1.0)
                if prefix in RIGHT_LEGS
                else (1.0, 1.0, -1.0)
            )
            tripod_a.append(prefix in TRIPOD_A)
        self._origins = jp.array(origins)
        self._outward = jp.array(outward)
        self._raw_signs = jp.array(raw_signs)
        self._tripod_a = jp.array(tripod_a)

        self._step_centers = jp.array(
            [STEP_START_X + STEP_DEPTH * i for i in range(STEP_COUNT)]
        )
        self._step_heights = jp.array(
            [STEP_HEIGHT * (i + 1) for i in range(STEP_COUNT)]
        )
        sample_x, sample_y = np.meshgrid(
            np.array((-0.15, 0.05, 0.25, 0.45, 0.65)),
            np.array((-0.22, 0.0, 0.22)),
        )
        self._height_samples = jp.array(
            np.stack((sample_x.ravel(), sample_y.ravel()), axis=-1)
        )

    def _terrain_height(self, xy: jax.Array) -> jax.Array:
        if self._terrain == "flat":
            return jp.zeros(xy.shape[:-1])
        x = xy[..., 0, None]
        y = xy[..., 1, None]
        inside = (
            (jp.abs(x - self._step_centers) <= STEP_DEPTH / 2.0)
            & (jp.abs(y) <= 1.0)
        )
        return jp.max(jp.where(inside, self._step_heights, 0.0), axis=-1)

    def _curriculum_stage(self, steps: jax.Array) -> jax.Array:
        """Return flat-task stage 0/1/2; non-curriculum tasks stay at 0."""
        if not self._command_curriculum:
            return jp.zeros((), dtype=jp.int32)
        first_end = int(self._config.command_curriculum.forward_only_steps)
        second_end = first_end + int(
            self._config.command_curriculum.limited_yaw_steps
        )
        return jp.where(
            steps < first_end,
            0,
            jp.where(steps < second_end, 1, 2),
        ).astype(jp.int32)

    def _sample_command(self, rng: jax.Array, stage: jax.Array) -> jax.Array:
        """Sample a speed/yaw command for the active task or curriculum stage."""
        speed_key, yaw_key = jax.random.split(rng)
        if self._command_curriculum:
            speed_min = jp.array(self._config.command_curriculum.speed_min)[stage]
            speed_max = jp.array(self._config.command_curriculum.speed_max)[stage]
            yaw_limit = jp.array(self._config.command_curriculum.yaw_limit)[stage]
        else:
            speed_min = self._config.command_min_speed
            speed_max = self._config.command_max_speed
            yaw_limit = self._config.command_max_yaw_rate
        speed = jax.random.uniform(
            speed_key, (), minval=speed_min, maxval=speed_max
        )
        yaw_rate = jax.random.uniform(
            yaw_key, (), minval=-yaw_limit, maxval=yaw_limit
        )
        return jp.array((speed, yaw_rate))

    @staticmethod
    def _quintic(tau: jax.Array) -> jax.Array:
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    @staticmethod
    def _ik(feet: jax.Array) -> tuple[jax.Array, jax.Array]:
        link1, link2, link3 = 0.074, 0.121, 0.230
        x, y, z = feet[:, 0], feet[:, 1], feet[:, 2]
        theta1 = jp.arctan2(y, x)
        rho = jp.sqrt(x * x + y * y) - link1
        cosine3_raw = (
            rho * rho + z * z - link2**2 - link3**2
        ) / (2.0 * link2 * link3)
        cosine3 = jp.clip(cosine3_raw, -1.0, 1.0)
        theta3 = jp.arctan2(-jp.sqrt(jp.maximum(0.0, 1.0 - cosine3**2)), cosine3)
        theta2 = jp.arctan2(z, rho) - jp.arctan2(
            link3 * jp.sin(theta3), link2 + link3 * jp.cos(theta3)
        )
        servo = jp.stack((theta1, -theta2, -theta3), axis=-1)
        workspace_error = jp.mean(jp.maximum(jp.abs(cosine3_raw) - 1.0, 0.0))
        return servo, workspace_error

    def _controller_targets(
        self, action: jax.Array, phase: jax.Array, command: jax.Array, blend: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        action = jp.clip(action, -1.0, 1.0)
        residual = action[:18].reshape(6, 3) * jp.array(self._config.residual_scale)
        step_scale = 1.0 + 0.5 * action[18]
        frequency_scale = 1.0 + 0.35 * action[19]
        swing_height = jp.clip(
            self._config.base_swing_height + 0.08 * action[20], 0.025, 0.17
        )
        radial_offset = jp.clip(
            self._config.base_radial_offset + 0.035 * action[21], 0.0, 0.06
        )

        first_half = phase < 0.5
        tau = jp.where(first_half, phase * 2.0, (phase - 0.5) * 2.0)
        smooth = self._quintic(tau)
        swing = self._tripod_a == first_half
        phase_position = jp.where(swing, smooth - 0.5, 0.5 - tau)
        lift = jp.where(swing, 4.0 * swing_height * smooth * (1.0 - smooth), 0.0)
        radial = jp.where(swing, 4.0 * radial_offset * smooth * (1.0 - smooth), 0.0)

        nominal = self._origins + self._outward * 0.218728
        nominal = nominal.at[:, 2].set(self._origins[:, 2] - 0.287006)
        yaw_velocity = jp.cross(
            jp.tile(jp.array((0.0, 0.0, command[1])), (6, 1)), nominal
        )
        foot_velocity = MODEL_FORWARD * command[0] + yaw_velocity
        target_body = (
            nominal
            + foot_velocity * (self._config.phase_time * step_scale * phase_position)[:, None]
            + self._outward * radial[:, None]
            + jp.stack((jp.zeros(6), jp.zeros(6), lift), axis=-1)
        )

        tangent = jp.stack(
            (-self._outward[:, 1], self._outward[:, 0], jp.zeros(6)), axis=-1
        )
        relative = target_body - self._origins
        feet_local = jp.stack(
            (
                jp.sum(relative * self._outward, axis=-1),
                jp.sum(relative * tangent, axis=-1),
                relative[:, 2],
            ),
            axis=-1,
        )
        feet_local = feet_local + residual * blend
        servo, workspace_error = self._ik(feet_local)
        raw = servo * self._raw_signs
        targets = self._home_ctrl.at[self._actuator_ids].set(raw)
        targets = jp.clip(targets, -2.356194, 2.356194)
        targets = self._home_ctrl + blend * (targets - self._home_ctrl)
        info = {
            "frequency_scale": frequency_scale,
            "workspace_error": workspace_error,
            "swing": swing,
            "feet_local": feet_local,
        }
        return targets, info

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        local_gravity = _quat_rotate_inverse(quat, jp.array((0.0, 0.0, -1.0)))
        sample_world = self._height_samples + data.qpos[None, :2]
        terrain = self._terrain_height(sample_world) - data.qpos[2]
        foot_positions = data.site_xpos[self._foot_site_ids]
        feet_relative = foot_positions - data.qpos[None, :3]
        foot_clearance = foot_positions[:, 2] - self._terrain_height(
            foot_positions[:, :2]
        )
        contacts = (foot_clearance < 0.038).astype(jp.float32)
        return jp.concatenate(
            (
                info["command"],
                local_velocity,
                data.qvel[3:6],
                local_gravity,
                data.qpos[self._joint_qpos_ids] - self._home_qpos[self._joint_qpos_ids],
                0.1 * data.qvel[self._joint_qvel_ids],
                jp.ravel(feet_relative),
                contacts,
                terrain,
                jp.array((jp.sin(2 * jp.pi * info["phase"]), jp.cos(2 * jp.pi * info["phase"]))),
                info["last_action"],
            )
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, q_key, vel_key, command_key = jax.random.split(rng, 4)
        qpos = self._home_qpos
        qpos = qpos.at[self._joint_qpos_ids].add(
            jax.random.uniform(q_key, (18,), minval=-0.015, maxval=0.015)
        )
        qvel = jp.zeros(self.mjx_model.nv)
        qvel = qvel.at[:6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.02, maxval=0.02)
        )
        curriculum_stage = self._curriculum_stage(jp.zeros((), dtype=jp.int32))
        data = mjx.make_data(
            self.mj_model,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._home_ctrl)
        data = mjx.forward(self.mjx_model, data)
        info = {
            "rng": rng,
            "command": self._sample_command(command_key, curriculum_stage),
            "phase": jp.zeros(()),
            "steps": jp.zeros((), dtype=jp.int32),
            "curriculum_stage": curriculum_stage,
            "last_action": jp.zeros(ACTION_SIZE),
        }
        metrics = {
            f"reward/{name}": jp.zeros(()) for name in self._config.reward.keys()
        }
        metrics["workspace_error"] = jp.zeros(())
        metrics["curriculum_stage"] = curriculum_stage.astype(jp.float32)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        blend = jp.clip(state.info["steps"] * self.dt / 0.75, 0.0, 1.0)
        targets, controller_info = self._controller_targets(
            action, state.info["phase"], state.info["command"], blend
        )
        max_delta = jp.deg2rad(240.0) * self.dt
        targets = state.data.ctrl + jp.clip(
            targets - state.data.ctrl, -max_delta, max_delta
        )
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)

        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        forward_velocity = jp.dot(local_velocity, MODEL_FORWARD)
        up_z = _quat_rotate(quat, jp.array((0.0, 0.0, 1.0)))[2]
        ground_height = self._terrain_height(data.qpos[:2])
        clearance = data.qpos[2] - ground_height
        terminated = (
            (up_z < 0.35)
            | (clearance < 0.14)
            | jp.any(jp.isnan(data.qpos))
            | jp.any(jp.isnan(data.qvel))
        )

        velocity_reward = jp.exp(
            -jp.square(forward_velocity - state.info["command"][0]) / 0.02
        )
        yaw_reward = jp.exp(
            -jp.square(data.qvel[5] - state.info["command"][1]) / 0.16
        )
        height_reward = jp.exp(-jp.square(clearance - 0.33) / 0.015)
        reward_terms = {
            "velocity": velocity_reward,
            "yaw": yaw_reward,
            "upright": jp.clip(up_z, 0.0, 1.0),
            "height": height_reward,
            "progress": jp.clip(forward_velocity, -0.2, 0.3),
            "action_rate": jp.sum(jp.square(action - state.info["last_action"])),
            "residual": jp.sum(jp.square(action)),
            "vertical_velocity": jp.square(data.qvel[2]),
            "lateral_velocity": jp.square(local_velocity[0]),
            "joint_velocity": jp.sum(jp.square(data.qvel[self._joint_qvel_ids])),
            "workspace": controller_info["workspace_error"],
            "termination": terminated.astype(jp.float32),
        }
        scaled = {
            name: value * self._config.reward[name]
            for name, value in reward_terms.items()
        }
        reward = jp.clip(sum(scaled.values()) * self.dt, -10.0, 10.0)

        phase_increment = (
            self.dt
            / (2.0 * self._config.phase_time)
            * controller_info["frequency_scale"]
        )
        state.info["phase"] = jp.mod(state.info["phase"] + phase_increment, 1.0)
        next_steps = state.info["steps"] + 1
        if self._command_curriculum:
            next_stage = self._curriculum_stage(next_steps)
            next_rng, command_key = jax.random.split(state.info["rng"])
            next_command = self._sample_command(command_key, next_stage)
            changed_stage = next_stage != state.info["curriculum_stage"]
            state.info["rng"] = next_rng
            state.info["command"] = jp.where(
                changed_stage, next_command, state.info["command"]
            )
            state.info["curriculum_stage"] = next_stage
        state.info["steps"] = next_steps
        state.info["last_action"] = action
        obs = self._get_obs(data, state.info)
        for name, value in scaled.items():
            state.metrics[f"reward/{name}"] = value
        state.metrics["workspace_error"] = controller_info["workspace_error"]
        state.metrics["curriculum_stage"] = state.info["curriculum_stage"].astype(
            jp.float32
        )
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
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
