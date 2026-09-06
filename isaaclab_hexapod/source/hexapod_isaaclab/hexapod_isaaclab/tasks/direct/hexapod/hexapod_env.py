"""Isaac Lab Hexapod task with frozen replay and online Torch modes.

Golden mode validates the asset, joint mapping, 2.5 ms / decimation-8 timing,
and the 500-step command trace.  Perceptive mode connects the online controller
and sensor observation scaffold used by the registered development task.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu, RayCaster, RayCasterCamera
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
from isaaclab.utils.warp import raycast_mesh

from ....assets import FOOT_BODY_ORDER, HOME_ROOT_QUAT_WXYZ, JOINT_ORDER, MODEL_FORWARD
from ....contracts import (
    build_actor_observation,
    build_critic_observation,
    build_legacy_observation,
)
from ....controllers import initial_output, initial_state, step as firmware_step
from ....perception import build_elevation_map, deterministic_terrain_features
from ....terrains import MAX_TRAINING_TERRAIN_LEVEL, TERRAIN_GOAL_DISTANCE_M
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
        self.previous_actions = torch.zeros_like(self.actions)
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
            self.episode_start_position = self.robot.data.root_pos_w.clone()
            model_forward = torch.as_tensor(MODEL_FORWARD, device=self.device, dtype=torch.float32)
            self.episode_forward_world = quat_apply(
                self.robot.data.root_quat_w,
                model_forward.repeat(self.num_envs, 1),
            )
            self.episode_forward_world[:, 2] = 0.0
            self.episode_forward_world /= torch.linalg.vector_norm(
                self.episode_forward_world, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            # ``None`` keeps the normal per-environment curriculum behavior.
            # The staged trainer sets this to 0..9 and advances only after it
            # has saved and rendered that stage's best policy.
            self.fixed_training_level: int | None = None
            self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_failure = torch.zeros_like(self.last_success)
            self.cached_lidar_points_body = None
            self.cached_depth_points_body = None
            self.episode_reward_sums = {
                name: torch.zeros(self.num_envs, device=self.device)
                for name in (
                    "velocity", "progress", "under_speed", "upright", "height",
                    "stability", "action_rate", "residual", "joint_velocity",
                    "controller_rejection", "success", "failure",
                )
            }

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
        if self.cfg.golden_replay:
            from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

            spawn_ground_plane("/World/ground", GroundPlaneCfg())
        else:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            global_paths = [] if self.cfg.golden_replay else [self.cfg.terrain.prim_path]
            self.scene.filter_collisions(global_prim_paths=global_paths)
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
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
        lidar_period = max(1, math.ceil(self.cfg.lidar.update_period / self.step_dt))
        if self.cached_lidar_points_body is None or self.common_step_counter % lidar_period == 0:
            self.cached_lidar_points_body = self._batched_raycast_points_body(
                self.lidar, self.cfg.lidar.max_distance, camera=False
            )
        lidar_points_body = self.cached_lidar_points_body
        depth_points_body = None
        if self.cfg.enable_depth_sensor:
            depth_period = max(1, math.ceil(self.cfg.depth.update_period / self.step_dt))
            if self.cached_depth_points_body is None or self.common_step_counter % depth_period == 0:
                self.cached_depth_points_body = self._batched_raycast_points_body(
                    self.depth, self.cfg.depth.max_distance, camera=True
                )
            depth_points_body = self.cached_depth_points_body
        elevation = build_elevation_map(
            lidar_points_body=lidar_points_body,
            depth_points_body=depth_points_body,
            body_quaternion_wxyz=self.robot.data.root_quat_w,
        )
        terrain_latent = deterministic_terrain_features(elevation)
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

    def _batched_raycast_points_body(
        self, sensor: RayCaster | RayCasterCamera, max_distance: float, *, camera: bool
    ) -> torch.Tensor:
        """Ray-cast every environment from robot root poses in one Warp batch.

        Referenced USD internals only expose the source prim to SensorBase, so
        relying on its prim view would reuse env_0.  Explicit root-pose batching
        is both correct for cloned environments and faster than XForm lookups.
        """
        starts_body = sensor.ray_starts[0]
        directions_body = sensor.ray_directions[0]
        if camera:
            offset_quat = sensor._offset_quat[0].expand(starts_body.shape[0], -1)
            starts_body = quat_apply(offset_quat, starts_body) + sensor._offset_pos[0]
            directions_body = quat_apply(offset_quat, directions_body)

        point_count = starts_body.shape[0]
        root_quat = self.robot.data.root_quat_w[:, None].expand(-1, point_count, -1)
        starts_body = starts_body[None].expand(self.num_envs, -1, -1)
        directions_body = directions_body[None].expand(self.num_envs, -1, -1)
        starts_w = quat_apply(root_quat, starts_body) + self.robot.data.root_pos_w[:, None]
        directions_w = quat_apply(root_quat, directions_body)
        hits_w, _, _, _ = raycast_mesh(
            starts_w,
            directions_w,
            mesh=RayCaster.meshes[self.cfg.lidar.mesh_prim_paths[0]],
            max_dist=max_distance,
        )
        return quat_apply_inverse(
            root_quat,
            hits_w - self.robot.data.root_pos_w[:, None],
        )

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.golden_replay:
            error = self.robot.data.joint_pos[:, self.joint_ids] - self.q_des
            return torch.exp(-torch.mean(torch.square(error), dim=1) / 0.01)

        forward_velocity = -self.robot.data.root_lin_vel_b[:, 1]
        lateral_velocity = self.robot.data.root_lin_vel_b[:, 0]
        speed_command = torch.clamp(self.command[:, 0], min=0.05)
        motion_gate = torch.clamp(forward_velocity / (0.5 * speed_command), 0.0, 1.0)
        ground_height = self.legacy_height_scanner.data.ray_hits_w[:, 5, 2].nan_to_num(0.0)
        clearance = self.robot.data.root_pos_w[:, 2] - ground_height
        projected_gravity_xy = self.robot.data.projected_gravity_b[:, :2]
        root_angular_speed = torch.linalg.vector_norm(self.robot.data.root_ang_vel_b, dim=-1)
        policy_rejection = 1.0 - self.controller_output.policy_valid.float().mean(dim=1)
        foot_limited = self.controller_output.foot_limited.float().mean(dim=1)

        terms = {
            "velocity": 2.5 * torch.exp(-torch.square((forward_velocity - self.command[:, 0]) / 0.05)),
            "progress": 2.0 * torch.clamp(forward_velocity / speed_command, -1.0, 1.5),
            "under_speed": -2.0 * torch.square(torch.clamp(speed_command - forward_velocity, min=0.0) / speed_command),
            "upright": motion_gate * torch.exp(-torch.sum(torch.square(projected_gravity_xy), dim=1) / 0.10),
            "height": 0.6 * motion_gate * torch.exp(-torch.square(clearance - self.cfg.target_clearance) / 0.01),
            "stability": 0.5 * motion_gate * torch.exp(-torch.square(root_angular_speed / 2.0)),
            "action_rate": -0.02 * torch.mean(torch.square(self.actions - self.previous_actions), dim=1),
            "residual": -0.005 * torch.mean(torch.square(self.actions), dim=1),
            "joint_velocity": -0.04 * torch.mean(torch.square(self.robot.data.joint_vel[:, self.joint_ids] / 10.0), dim=1),
            "controller_rejection": -2.0 * policy_rejection - 2.0 * foot_limited,
            "success": 30.0 * self.last_success.float() / self.step_dt,
            "failure": -30.0 * self.last_failure.float() / self.step_dt,
        }
        for name, value in terms.items():
            self.episode_reward_sums[name] += value * self.step_dt
        return torch.stack(tuple(terms.values()), dim=0).sum(dim=0) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.golden_replay:
            failed = ~torch.isfinite(self.robot.data.joint_pos[:, self.joint_ids]).all(dim=1)
            return failed, timeout

        progress = self._episode_progress()
        goals = torch.as_tensor(TERRAIN_GOAL_DISTANCE_M, device=self.device)[self.terrain.terrain_levels]
        success = progress >= goals
        gravity_xy = torch.linalg.vector_norm(self.robot.data.projected_gravity_b[:, :2], dim=-1)
        tilt = torch.asin(torch.clamp(gravity_xy, 0.0, 1.0))
        ground_height = self.legacy_height_scanner.data.ray_hits_w[:, 5, 2].nan_to_num(0.0)
        clearance = self.robot.data.root_pos_w[:, 2] - ground_height
        finite = (
            torch.isfinite(self.robot.data.root_state_w).all(dim=1)
            & torch.isfinite(self.robot.data.joint_pos[:, self.joint_ids]).all(dim=1)
        )
        failure = (tilt > self.cfg.maximum_tilt_rad) | (clearance < self.cfg.minimum_clearance) | (~finite)
        self.last_success = success
        self.last_failure = failure
        return success | failure, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if not self.cfg.golden_replay and hasattr(self, "episode_start_position"):
            progress = self._episode_progress(env_ids)
            levels = self.terrain.terrain_levels[env_ids]
            goals = torch.as_tensor(TERRAIN_GOAL_DISTANCE_M, device=self.device)[levels]
            if self.fixed_training_level is None:
                promote = self.last_success[env_ids] & (levels < MAX_TRAINING_TERRAIN_LEVEL)
                demote = self.last_failure[env_ids] | (
                    self.reset_time_outs[env_ids] & (progress < 0.35 * goals)
                )
                demote &= ~promote
                self.terrain.update_env_origins(env_ids, promote, demote)
            else:
                self.terrain.terrain_levels[env_ids] = self.fixed_training_level
                self.terrain.env_origins[env_ids] = self.terrain.terrain_origins[
                    self.fixed_training_level, self.terrain.terrain_types[env_ids]
                ]

            self.extras["log"] = {
                "Curriculum/mean_level": self.terrain.terrain_levels.float().mean(),
                "Curriculum/max_level": self.terrain.terrain_levels.max().float(),
                "Episode/progress_m": progress.mean(),
                "Episode/progress_ratio": (progress / goals.clamp_min(1.0e-6)).mean(),
                "Episode/success_rate": self.last_success[env_ids].float().mean(),
                "Episode/failure_rate": self.last_failure[env_ids].float().mean(),
                "Sensors/depth_enabled": float(self.cfg.enable_depth_sensor),
            }
            for name, value in self.episode_reward_sums.items():
                self.extras["log"][f"Episode_Reward/{name}"] = value[env_ids].mean()
                value[env_ids] = 0.0
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        env_origins = self.scene.env_origins[env_ids] if self.cfg.golden_replay else self.terrain.env_origins[env_ids]
        root_state[:, :3] += env_origins
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        reset_state = initial_state(len(env_ids), self.device)
        for current, reset in zip(self.controller_state, reset_state):
            current[env_ids] = reset
        self.controller_output = initial_output(self.controller_state)
        self.previous_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        if not self.cfg.golden_replay:
            self.command[env_ids] = 0.0
            self.command[env_ids, 0].uniform_(*self.cfg.command_speed_range)
            self.episode_start_position[env_ids] = root_state[:, :3]
            model_forward = torch.as_tensor(
                MODEL_FORWARD, device=self.device, dtype=torch.float32
            ).repeat(len(env_ids), 1)
            self.episode_forward_world[env_ids] = quat_apply(
                root_state[:, 3:7], model_forward
            )
            self.episode_forward_world[env_ids, 2] = 0.0
            self.episode_forward_world[env_ids] /= torch.linalg.vector_norm(
                self.episode_forward_world[env_ids], dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            self.last_success[env_ids] = False
            self.last_failure[env_ids] = False
            # Force fresh sensor frames after robots move between terrain rows.
            self.cached_lidar_points_body = None
            self.cached_depth_points_body = None

    def _episode_progress(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return displacement along each episode's actual initial body-forward axis."""
        if env_ids is None:
            displacement = self.robot.data.root_pos_w - self.episode_start_position
            return torch.sum(displacement * self.episode_forward_world, dim=1)
        displacement = (
            self.robot.data.root_pos_w[env_ids] - self.episode_start_position[env_ids]
        )
        return torch.sum(displacement * self.episode_forward_world[env_ids], dim=1)

    def set_training_level(self, level: int) -> None:
        """Pin every environment to one curriculum level and reset it safely."""
        if self.cfg.golden_replay:
            raise RuntimeError("training levels are unavailable in golden replay mode")
        if not 0 <= level <= MAX_TRAINING_TERRAIN_LEVEL:
            raise ValueError(
                f"training level must be in [0, {MAX_TRAINING_TERRAIN_LEVEL}], got {level}"
            )
        self.fixed_training_level = int(level)
        env_ids = self.robot._ALL_INDICES
        self.terrain.terrain_levels[env_ids] = self.fixed_training_level
        self.terrain.env_origins[env_ids] = self.terrain.terrain_origins[
            self.fixed_training_level, self.terrain.terrain_types[env_ids]
        ]
        self._reset_idx(env_ids)
