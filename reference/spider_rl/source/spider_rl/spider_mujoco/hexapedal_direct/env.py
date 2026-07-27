from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - local import fallback only
    class _Env:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)
            return None

    class _Box:
        def __init__(self, low: Any, high: Any, shape: tuple[int, ...], dtype: np.dtype | type[np.floating]):
            self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
            self.shape = shape
            self.dtype = np.dtype(dtype)

    gym = SimpleNamespace(Env=_Env)
    spaces = SimpleNamespace(Box=_Box)

from .env_cfg import HexapedalDirectEnvCfg


class HexapedalDirectEnv(gym.Env):
    metadata = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(self, cfg: HexapedalDirectEnvCfg | None = None, render_mode: str | None = None, **_: Any):
        self.cfg = cfg if cfg is not None else HexapedalDirectEnvCfg()
        self.render_mode = render_mode
        self.metadata["render_fps"] = int(round(1.0 / self.cfg.sim.policy_dt))

        self.model_path = self._resolve_model_path(self.cfg.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.observation_dim,),
            dtype=np.float32,
        )

        self.np_random = np.random.default_rng()
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None
        self._elapsed_steps = 0

        self._bind_model()
        self._init_buffers()
        self._forward()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super_reset = getattr(super(), "reset", None)
        if callable(super_reset):
            super_reset(seed=seed)
        elif seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._elapsed_steps = 0
        self._init_buffers()
        self._apply_reset_state()
        self._resample_command(force=True)
        self._forward()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float64),
            -self.cfg.pd.action_clip,
            self.cfg.pd.action_clip,
        )
        if clipped_action.shape != (self.cfg.action_dim,):
            raise ValueError(f"Expected action shape {(self.cfg.action_dim,)}, got {clipped_action.shape}.")

        self._action[:] = clipped_action
        self._pd_targets[:] = np.clip(
            self._default_joint_qpos + clipped_action * self.cfg.pd.action_scale,
            self._joint_limit_low,
            self._joint_limit_high,
        )

        for _ in range(self.cfg.sim.frame_skip):
            self._apply_pd_targets()
            mujoco.mj_step(self.model, self.data)
            self._command_time_left -= self.cfg.sim.dt

        self._elapsed_steps += 1
        if self._command_time_left <= 0.0:
            self._resample_command(force=True)

        reward = float(self._compute_reward())
        diagnostics = self._step_diagnostics()
        self._episode_reward += reward
        self._episode_tracking_lin_vel_error += diagnostics["tracking_lin_vel_error"]
        self._episode_tracking_ang_vel_error += diagnostics["tracking_ang_vel_error"]
        self._episode_desired_contact_count += diagnostics["desired_contact_count"]
        self._episode_undesired_contact_count += diagnostics["undesired_contact_count"]
        terminated = self._is_terminated()
        truncated = self._elapsed_steps >= self.cfg.sim.max_episode_steps
        obs = self._get_obs()
        info = self._get_info()
        info.update(diagnostics)
        if terminated or truncated:
            termination_reason = self._termination_reason(terminated, truncated)
            info["termination_reason"] = termination_reason
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._elapsed_steps,
                "tracking_lin_vel_error": self._episode_tracking_lin_vel_error / max(self._elapsed_steps, 1),
                "tracking_ang_vel_error": self._episode_tracking_ang_vel_error / max(self._elapsed_steps, 1),
                "undesired_contact_count": self._episode_undesired_contact_count / max(self._elapsed_steps, 1),
                "desired_contact_count": self._episode_desired_contact_count / max(self._elapsed_steps, 1),
                "fall_count": 1 if termination_reason in {"fall", "height_violation"} else 0,
                "command_vx": float(self._command[0]),
                "command_vy": float(self._command[1]),
                "command_wz": float(self._command[2]),
            }

        self._prev_action[:] = self._action
        self._last_joint_vel[:] = self.data.qvel[self._joint_dof_indices]
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    height=self.cfg.render.height,
                    width=self.cfg.render.width,
                )
            self._renderer.update_scene(self.data, camera=self.cfg.render.camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @staticmethod
    def _resolve_model_path(model_path: Path) -> Path:
        resolved = Path(model_path)
        if resolved.exists():
            return resolved

        try:
            builder = importlib.import_module("spider_mujoco.hexapedal_direct.model_builder")
        except ModuleNotFoundError:
            return resolved

        load_metadata = getattr(builder, "load_hexapedal_model", None)
        if callable(load_metadata):
            try:
                metadata = load_metadata(validate_generated=True)
            except (FileNotFoundError, RuntimeError):
                metadata = None
            if metadata is not None:
                xml_path = getattr(metadata, "xml_path", None)
                if isinstance(xml_path, (str, Path)) and Path(xml_path).exists():
                    return Path(xml_path)

        for method_name in ("write_hexapedal_assets", "build_hexapedal_assets"):
            method = getattr(builder, method_name, None)
            if callable(method):
                built = method()
                if isinstance(built, (str, Path)) and Path(built).exists():
                    return Path(built)
                xml_path = getattr(built, "xml_path", None)
                if isinstance(xml_path, (str, Path)) and Path(xml_path).exists():
                    return Path(xml_path)
                xml_path = getattr(builder, "XML_PATH", None)
                if isinstance(xml_path, (str, Path)) and Path(xml_path).exists():
                    return Path(xml_path)
        return resolved

    def _bind_model(self) -> None:
        self._base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.cfg.base_body_name,
        )
        if self._base_body_id < 0:
            raise ValueError(f"Body '{self.cfg.base_body_name}' not found in {self.model_path}.")

        joint_qpos_indices: list[int] = []
        joint_dof_indices: list[int] = []
        joint_ids: list[int] = []
        joint_body_ids: list[int] = []
        actuator_indices: list[int] = []
        actuator_ctrl_low: list[float] = []
        actuator_ctrl_high: list[float] = []

        for joint_name in self.cfg.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in {self.model_path}.")
            joint_ids.append(int(joint_id))
            joint_qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            joint_dof_indices.append(int(self.model.jnt_dofadr[joint_id]))
            joint_body_ids.append(int(self.model.jnt_bodyid[joint_id]))

            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id >= 0:
                actuator_indices.append(int(actuator_id))
                if self.model.actuator_ctrllimited[actuator_id]:
                    actuator_ctrl_low.append(float(self.model.actuator_ctrlrange[actuator_id, 0]))
                    actuator_ctrl_high.append(float(self.model.actuator_ctrlrange[actuator_id, 1]))
                else:
                    actuator_ctrl_low.append(-np.inf)
                    actuator_ctrl_high.append(np.inf)

        self._joint_id_buffer = np.asarray(joint_ids, dtype=np.int32)
        self._joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=np.int32)
        self._joint_dof_indices = np.asarray(joint_dof_indices, dtype=np.int32)
        self._actuator_indices = np.asarray(actuator_indices, dtype=np.int32)
        self._actuator_ctrl_low = np.asarray(actuator_ctrl_low, dtype=np.float64)
        self._actuator_ctrl_high = np.asarray(actuator_ctrl_high, dtype=np.float64)
        self._default_joint_qpos = np.asarray(
            [self.cfg.default_joint_positions[name] for name in self.cfg.joint_names],
            dtype=np.float64,
        )
        self._joint_limit_low = self.model.jnt_range[self._joint_id_buffer, 0].astype(np.float64)
        self._joint_limit_high = self.model.jnt_range[self._joint_id_buffer, 1].astype(np.float64)

        self._desired_contact_body_ids = []
        self._desired_contact_body_id_to_leg: dict[int, int] = {}
        for leg_index, site_name in enumerate(self.cfg.contact_site_names):
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise ValueError(f"Site '{site_name}' not found in {self.model_path}.")
            body_id = int(self.model.site_bodyid[site_id])
            self._desired_contact_body_ids.append(body_id)
            self._desired_contact_body_id_to_leg[body_id] = leg_index

        self._undesired_contact_body_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                for body_name in self.cfg.undesired_contact_body_names
            ],
            dtype=np.int32,
        )
        if np.any(self._undesired_contact_body_ids < 0):
            missing = [
                name
                for name, body_id in zip(self.cfg.undesired_contact_body_names, self._undesired_contact_body_ids)
                if body_id < 0
            ]
            raise ValueError(f"Undesired contact bodies missing from {self.model_path}: {missing}")

        self._robot_body_ids = np.unique(
            np.concatenate(
                (
                    np.asarray([self._base_body_id], dtype=np.int32),
                    np.asarray(joint_body_ids, dtype=np.int32),
                    np.asarray(self._desired_contact_body_ids, dtype=np.int32),
                    self._undesired_contact_body_ids,
                )
            )
        )


    def _init_buffers(self) -> None:
        self._action = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self._prev_action = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self._pd_targets = self._default_joint_qpos.copy()
        self._command = np.zeros(self.cfg.command_dim, dtype=np.float64)
        self._command_time_left = 0.0
        self._last_joint_vel = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self._feet_air_time = np.zeros(len(self.cfg.contact_site_names), dtype=np.float64)
        self._episode_reward = 0.0
        self._episode_tracking_lin_vel_error = 0.0
        self._episode_tracking_ang_vel_error = 0.0
        self._episode_desired_contact_count = 0.0
        self._episode_undesired_contact_count = 0.0

    def _apply_reset_state(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        if self.model.na:
            self.data.act[:] = 0.0

        self.data.qpos[0:3] = np.asarray(self.cfg.default_base_position, dtype=np.float64)
        if self.cfg.reset_noise.root_xy > 0.0:
            self.data.qpos[0:2] += self.np_random.uniform(
                -self.cfg.reset_noise.root_xy,
                self.cfg.reset_noise.root_xy,
                size=2,
            )
        if self.cfg.reset_noise.root_yaw > 0.0:
            half_yaw = self.np_random.uniform(-self.cfg.reset_noise.root_yaw, self.cfg.reset_noise.root_yaw) * 0.5
            self.data.qpos[3:7] = np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float64)

        joint_noise = self.np_random.uniform(
            -self.cfg.reset_noise.joint_position,
            self.cfg.reset_noise.joint_position,
            size=self.cfg.action_dim,
        )
        self.data.qpos[self._joint_qpos_indices] = np.clip(
            self._default_joint_qpos + joint_noise,
            self._joint_limit_low,
            self._joint_limit_high,
        )
        self.data.qvel[self._joint_dof_indices] = self.np_random.uniform(
            -self.cfg.reset_noise.joint_velocity,
            self.cfg.reset_noise.joint_velocity,
            size=self.cfg.action_dim,
        )
        self._pd_targets[:] = self.data.qpos[self._joint_qpos_indices]

    def _forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self._last_joint_vel[:] = self.data.qvel[self._joint_dof_indices]

    def _resample_command(self, force: bool = False) -> None:
        if not force and self._command_time_left > 0.0:
            return
        self._command[0] = self.np_random.uniform(*self.cfg.commands.vx_range)
        self._command[1] = self.np_random.uniform(*self.cfg.commands.vy_range)
        self._command[2] = self.np_random.uniform(*self.cfg.commands.wz_range)
        self._command_time_left = self.np_random.uniform(*self.cfg.commands.resample_time_range_s)

    def _apply_pd_targets(self) -> None:
        joint_qpos = self.data.qpos[self._joint_qpos_indices]
        joint_qvel = self.data.qvel[self._joint_dof_indices]

        self.data.qfrc_applied[:] = 0.0
        if self.cfg.pd.use_mujoco_position_actuators and len(self._actuator_indices) == self.cfg.action_dim:
            ctrl_targets = np.clip(self._pd_targets, self._actuator_ctrl_low, self._actuator_ctrl_high)
            self.data.ctrl[self._actuator_indices] = ctrl_targets
            return

        torques = self.cfg.pd.stiffness * (self._pd_targets - joint_qpos)
        torques += self.cfg.pd.damping * (self.cfg.pd.target_velocity - joint_qvel)
        if self.cfg.pd.torque_limit is not None:
            torques = np.clip(torques, -self.cfg.pd.torque_limit, self.cfg.pd.torque_limit)
        self.data.qfrc_applied[self._joint_dof_indices] = torques

    def _get_obs(self) -> np.ndarray:
        lin_vel_b, ang_vel_b = self._body_velocities()
        proj_gravity_b = self._projected_gravity()
        joint_pos = self.data.qpos[self._joint_qpos_indices] - self._default_joint_qpos
        joint_vel = self.data.qvel[self._joint_dof_indices]
        obs = np.concatenate(
            (
                self._rotate_body_frame(lin_vel_b),
                self._rotate_body_frame(ang_vel_b),
                self._rotate_body_frame(proj_gravity_b),
                self._command,
                joint_pos,
                joint_vel,
            )
        )
        return obs.astype(np.float32, copy=False)

    def _body_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._base_body_id,
            velocity,
            1,
        )
        return velocity[3:].copy(), velocity[:3].copy()

    def _projected_gravity(self) -> np.ndarray:
        body_rot = self.data.xmat[self._base_body_id].reshape(3, 3)
        gravity_world = np.asarray(self.cfg.sim.gravity, dtype=np.float64)
        gravity_dir = gravity_world / np.linalg.norm(gravity_world)
        return body_rot.T @ gravity_dir

    @staticmethod
    def _rotate_body_frame(vector: np.ndarray) -> np.ndarray:
        rotated = np.array(vector, dtype=np.float64, copy=True)
        rotated[0] = -vector[1]
        rotated[1] = vector[0]
        return rotated

    def _compute_reward(self) -> float:
        lin_vel_b, ang_vel_b = self._body_velocities()
        lin_vel = self._rotate_body_frame(lin_vel_b)
        ang_vel = self._rotate_body_frame(ang_vel_b)
        proj_gravity = self._rotate_body_frame(self._projected_gravity())
        root_height = float(self.data.xpos[self._base_body_id, 2])
        joint_vel = self.data.qvel[self._joint_dof_indices].copy()

        action_rate = self._action - self._prev_action
        joint_acc = (joint_vel - self._last_joint_vel) / self.cfg.sim.policy_dt
        desired_contacts, undesired_contacts, desired_contact_count, undesired_contact_count = self._contact_state()

        tracking_sigma = self.cfg.rewards.tracking_sigma
        lin_vel_error = np.square(self._command[:2] - lin_vel[:2]).sum()
        rew_lin_vel_xy = self.cfg.rewards.lin_vel_xy * math.exp(-lin_vel_error / tracking_sigma)
        ang_vel_error = (self._command[2] - ang_vel[2]) ** 2
        rew_ang_vel_z = self.cfg.rewards.ang_vel_z * math.exp(-ang_vel_error / tracking_sigma)

        applied_joint_effort = self._applied_joint_effort()
        rew_lin_vel_z = self.cfg.rewards.lin_vel_z * (lin_vel[2] ** 2)
        rew_ang_vel_xy = self.cfg.rewards.ang_vel_xy * np.square(ang_vel[:2]).sum()
        rew_joint_torques = self.cfg.rewards.joint_torques * np.square(applied_joint_effort).sum()
        rew_joint_acc = self.cfg.rewards.joint_acc * np.square(joint_acc).sum()
        rew_action_rate = self.cfg.rewards.action_rate * np.square(action_rate).sum()
        rew_action_l2 = self.cfg.rewards.action_l2 * np.square(self._action).sum()
        rew_energy = self.cfg.rewards.energy * np.abs(applied_joint_effort * joint_vel).sum()
        rew_flat_orientation = self.cfg.rewards.flat_orientation * np.square(proj_gravity[:2]).sum()
        rew_base_height = self.cfg.rewards.base_height * ((root_height - self.cfg.target_base_height) ** 2)

        command_norm = float(np.linalg.norm(self._command))
        rew_stand_still = self.cfg.rewards.stand_still * float(command_norm < 0.1) * (
            np.linalg.norm(lin_vel[:2]) + abs(ang_vel[2])
        )
        rew_desired_contact = self.cfg.rewards.desired_contact * desired_contact_count / len(self.cfg.contact_site_names)
        rew_undesired_contact = self.cfg.rewards.undesired_contact * undesired_contact_count

        first_contact = (self._feet_air_time > 0.0) & desired_contacts
        self._feet_air_time += self.cfg.sim.policy_dt
        rew_feet_air_time = self.cfg.rewards.feet_air_time * np.sum((self._feet_air_time - 0.5) * first_contact)
        self._feet_air_time *= (~desired_contacts).astype(np.float64)

        return (
            rew_lin_vel_xy
            + rew_ang_vel_z
            + rew_lin_vel_z
            + rew_ang_vel_xy
            + rew_joint_torques
            + rew_joint_acc
            + rew_action_rate
            + rew_action_l2
            + rew_energy
            + rew_flat_orientation
            + rew_base_height
            + rew_stand_still
            + rew_desired_contact
            + rew_undesired_contact
            + rew_feet_air_time
        )

    def _contact_state(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        desired_contacts = np.zeros(len(self.cfg.contact_site_names), dtype=bool)
        undesired_contacts = np.zeros(len(self.cfg.undesired_contact_body_names), dtype=bool)

        force = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            if np.linalg.norm(force[:3]) <= self.cfg.contact_force_threshold:
                continue

            body_1 = int(self.model.geom_bodyid[contact.geom1])
            body_2 = int(self.model.geom_bodyid[contact.geom2])
            if body_1 in self._robot_body_ids and body_2 in self._robot_body_ids:
                continue

            for body_id in (body_1, body_2):
                leg_index = self._desired_contact_body_id_to_leg.get(body_id)
                if leg_index is not None:
                    desired_contacts[leg_index] = True
                matches = np.where(self._undesired_contact_body_ids == body_id)[0]
                if matches.size:
                    undesired_contacts[matches[0]] = True

        return (
            desired_contacts,
            undesired_contacts,
            float(desired_contacts.sum()),
            float(undesired_contacts.sum()),
        )

    def _applied_joint_effort(self) -> np.ndarray:
        if len(self._actuator_indices) == self.cfg.action_dim:
            return self.data.actuator_force[self._actuator_indices].copy()
        return self.data.qfrc_applied[self._joint_dof_indices].copy()

    def _is_terminated(self) -> bool:
        root_height = float(self.data.xpos[self._base_body_id, 2])
        gravity_z = float(self._projected_gravity()[2])
        return (
            root_height < self.cfg.termination.min_base_height
            or gravity_z > -self.cfg.termination.upright_threshold
        )

    def _termination_reason(self, terminated: bool, truncated: bool) -> str:
        if truncated:
            return "timeout"
        if not terminated:
            return "running"
        root_height = float(self.data.xpos[self._base_body_id, 2])
        if root_height < self.cfg.termination.min_base_height:
            return "height_violation"
        return "fall"

    def _step_diagnostics(self) -> dict[str, float]:
        lin_vel_b = self.data.cvel[self._base_body_id, 3:].copy()
        ang_vel_b = self.data.cvel[self._base_body_id, :3].copy()
        lin_vel = self._rotate_body_frame(lin_vel_b)
        ang_vel = self._rotate_body_frame(ang_vel_b)
        _, _, desired_contact_count, undesired_contact_count = self._contact_state()
        return {
            "tracking_lin_vel_error": float(np.sqrt(np.square(self._command[:2] - lin_vel[:2]).sum())),
            "tracking_ang_vel_error": float(abs(self._command[2] - ang_vel[2])),
            "desired_contact_count": float(desired_contact_count),
            "undesired_contact_count": float(undesired_contact_count),
            "command_vx": float(self._command[0]),
            "command_vy": float(self._command[1]),
            "command_wz": float(self._command[2]),
            "base_x": float(self.data.qpos[0]),
        }

    def _get_info(self) -> dict[str, Any]:
        desired_contacts, undesired_contacts, desired_contact_count, undesired_contact_count = self._contact_state()
        return {
            "command": self._command.astype(np.float32, copy=True),
            "pd_targets": self._pd_targets.astype(np.float32, copy=True),
            "desired_contacts": desired_contacts.copy(),
            "undesired_contacts": undesired_contacts.copy(),
            "desired_contact_count": desired_contact_count,
            "undesired_contact_count": undesired_contact_count,
            "model_path": str(self.model_path),
        }
