"""Isaac Lab Hexapod task with frozen replay and online Torch modes.

Golden mode validates the asset, joint mapping, 2.5 ms / decimation-8 timing,
and the 500-step command trace.  Perceptive mode connects the online controller
and sensor observation scaffold used by the registered development task.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu, RayCaster, RayCasterCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_conjugate, quat_mul

from ....assets import FOOT_BODY_ORDER, HOME_ROOT_QUAT_WXYZ, JOINT_ORDER
from ....contracts import (
    build_actor_observation,
    build_critic_observation,
    build_legacy_observation,
)
from ....controllers import initial_output, initial_state, step as firmware_step
from ....perception import TerrainEncoder, build_elevation_map
from .hexapod_env_cfg import HexapodEnvCfg


class HexapodEnv(DirectRLEnv):
    cfg: HexapodEnvCfg

    def __init__(self, cfg: HexapodEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, names = self.robot.find_joints(list(JOINT_ORDER), preserve_order=True)
        if tuple(names) != JOINT_ORDER:
            raise RuntimeError(f"joint contract mismatch: {names}")
        repo_root = Path(__file__).resolve().parents[7]
        golden_path = repo_root / "mjx/golden/isaac_contract_v1_flat_seed0.npz"
        with np.load(golden_path) as trace:
            q_des = np.asarray(trace["q_des"], dtype=np.float32)
            command = np.asarray(trace["command"], dtype=np.float32)
        self.q_des_trace = torch.as_tensor(q_des, device=self.device)
        self.command_trace = torch.as_tensor(command, device=self.device)
        self.actions = torch.zeros((self.num_envs, 18), device=self.device)
        self.q_des = self.q_des_trace[0].repeat(self.num_envs, 1)
        self.command = torch.as_tensor(
            cfg.default_command, device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        self.controller_state = initial_state(self.num_envs, self.device)
        self.controller_output = initial_output(self.controller_state)
        self.home_quaternion = torch.as_tensor(
            HOME_ROOT_QUAT_WXYZ, device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        if not cfg.golden_replay:
            self.foot_body_ids, foot_names = self.contact_sensor.find_bodies(
                list(FOOT_BODY_ORDER), preserve_order=True
            )
            if tuple(foot_names) != FOOT_BODY_ORDER:
                raise RuntimeError(f"foot contact contract mismatch: {foot_names}")
            self.terrain_encoder = TerrainEncoder().to(self.device).eval()

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        if not self.cfg.golden_replay:
            self.contact_sensor = ContactSensor(self.cfg.foot_contact)
            self.legacy_height_scanner = RayCaster(self.cfg.legacy_height_scanner)
            self.lidar = RayCaster(self.cfg.lidar)
            self.imu = Imu(self.cfg.imu)
            self.scene.sensors["foot_contact"] = self.contact_sensor
            self.scene.sensors["legacy_height_scanner"] = self.legacy_height_scanner
            self.scene.sensors["lidar"] = self.lidar
            self.scene.sensors["imu"] = self.imu
            if self.cfg.enable_depth_sensor:
                self.depth = RayCasterCamera(self.cfg.depth)
                self.scene.sensors["depth"] = self.depth
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.clone(), -1.0, 1.0)
        if self.cfg.golden_replay:
            index = torch.clamp(self.episode_length_buf, max=self.q_des_trace.shape[0] - 1)
            self.q_des = self.q_des_trace[index]
            return

        root_position = self.robot.data.root_pos_w
        relative_quaternion = quat_mul(
            self.robot.data.root_quat_w, quat_conjugate(self.home_quaternion)
        )
        attitude = torch.stack(euler_xyz_from_quat(relative_quaternion), dim=-1)
        contacts = (
            torch.linalg.vector_norm(
                self.contact_sensor.data.net_forces_w[:, self.foot_body_ids], dim=-1
            )
            >= self.cfg.contact_force_threshold
        )
        for _ in range(self.cfg.firmware_ticks_per_policy_step):
            self.controller_state, self.controller_output = firmware_step(
                self.controller_state,
                target_velocity=self.command[:, :2],
                body_position_world=root_position,
                attitude_rpy=attitude,
                contacts=contacts,
                policy_action=self.actions,
                roll_cmd=self.command[:, 4],
                pitch_cmd=self.command[:, 3],
                height_offset=self.command[:, 2],
            )
        self.q_des = self.controller_output.model_joint_targets.reshape(self.num_envs, 18)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.q_des, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if not self.cfg.golden_replay:
            return self._get_perceptive_observations()
        obs = torch.zeros((self.num_envs, 146), device=self.device)
        index = torch.clamp(self.episode_length_buf, max=self.command_trace.shape[0] - 1)
        obs[:, 0:5] = self.command_trace[index]
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]
        default_pos = self.robot.data.default_joint_pos[:, self.joint_ids]
        obs[:, 16:34] = joint_pos - default_pos
        obs[:, 34:52] = 0.1 * joint_vel
        obs[:, 127:145] = self.actions
        return {"policy": obs}

    def _get_perceptive_observations(self) -> dict[str, torch.Tensor]:
        relative_quaternion = quat_mul(
            self.robot.data.root_quat_w, quat_conjugate(self.home_quaternion)
        )
        attitude = torch.stack(euler_xyz_from_quat(relative_quaternion), dim=-1)
        contacts = (
            torch.linalg.vector_norm(
                self.contact_sensor.data.net_forces_w[:, self.foot_body_ids], dim=-1
            )
            >= self.cfg.contact_force_threshold
        )
        terrain_gt = (
            self.legacy_height_scanner.data.ray_hits_w[..., 2]
            - self.legacy_height_scanner.data.ray_hits_w[..., 2:3].amin(dim=1)
        ).nan_to_num(0.0, posinf=0.0, neginf=0.0)
        lidar_points_body = quat_apply_inverse(
            self.robot.data.root_quat_w[:, None].expand(-1, self.lidar.data.ray_hits_w.shape[1], -1),
            self.lidar.data.ray_hits_w - self.robot.data.root_pos_w[:, None],
        )
        depth_points_body = None
        if self.cfg.enable_depth_sensor:
            depth_hits_w = self.depth.data.ray_hits_w.reshape(self.num_envs, -1, 3)
            depth_points_body = quat_apply_inverse(
                self.robot.data.root_quat_w[:, None].expand(-1, depth_hits_w.shape[1], -1),
                depth_hits_w - self.robot.data.root_pos_w[:, None],
            )
        elevation = build_elevation_map(
            lidar_points_body=lidar_points_body,
            depth_points_body=depth_points_body,
            body_quaternion_wxyz=self.robot.data.root_quat_w,
        )
        with torch.no_grad():
            terrain_latent = self.terrain_encoder(elevation)
        legacy = build_legacy_observation(
            command=self.command,
            root_local_linear_velocity=self.robot.data.root_lin_vel_b,
            root_world_angular_velocity=self.robot.data.root_ang_vel_w,
            projected_gravity=self.robot.data.projected_gravity_b,
            relative_roll_pitch=attitude[:, :2],
            joint_position_error=self.robot.data.joint_pos[:, self.joint_ids] - self.robot.data.default_joint_pos[:, self.joint_ids],
            joint_velocity=self.robot.data.joint_vel[:, self.joint_ids],
            foot_position_controller=self.controller_output.foot_targets_body,
            foot_contact=contacts,
            terrain_height_gt=terrain_gt,
            gait_progress=self.controller_output.gait_progress,
            gait_state=self.controller_output.gait_state,
            applied_twist=self.controller_output.applied_twist,
            ik_valid=self.controller_output.ik_valid,
            policy_valid=self.controller_output.policy_valid,
            foot_limited=self.controller_output.foot_limited,
            gait_accepted=self.controller_output.gait_accepted[:, None],
            posture_accepted=self.controller_output.posture_accepted[:, None],
            last_action=self.actions,
            pitch_feedforward=torch.zeros((self.num_envs, 1), device=self.device),
        )
        actor = build_actor_observation(legacy, terrain_latent)
        contact_forces = torch.linalg.vector_norm(
            self.contact_sensor.data.net_forces_w[:, self.foot_body_ids], dim=-1
        )
        privileged = torch.cat(
            (
                terrain_gt,
                self.robot.data.root_lin_vel_w,
                self.robot.data.root_ang_vel_w,
                contact_forces,
                self.robot.data.root_pos_w[:, 2:3],
                attitude[:, :2],
            ),
            dim=-1,
        )
        return {"policy": actor, "critic": build_critic_observation(actor, privileged)}

    def _get_rewards(self) -> torch.Tensor:
        error = self.robot.data.joint_pos[:, self.joint_ids] - self.q_des
        return torch.exp(-torch.mean(torch.square(error), dim=1) / 0.01)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        failed = ~torch.isfinite(self.robot.data.joint_pos[:, self.joint_ids]).all(dim=1)
        return failed, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        reset_state = initial_state(len(env_ids), self.device)
        for current, reset in zip(self.controller_state, reset_state):
            current[env_ids] = reset
        self.controller_output = initial_output(self.controller_state)
