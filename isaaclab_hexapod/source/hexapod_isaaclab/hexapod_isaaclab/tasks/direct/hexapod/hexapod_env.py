"""Isaac Lab flat task that replays the frozen MJX q_des reference trace.

This environment intentionally keeps PPO disabled.  It validates the asset,
joint mapping, 2.5 ms / decimation-8 timing, and the 500-step command trace
before the online Torch firmware controller is connected.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ....assets import JOINT_ORDER
from .hexapod_env_cfg import HexapodEnvCfg


class HexapodEnv(DirectRLEnv):
    cfg: HexapodEnvCfg

    def __init__(self, cfg: HexapodEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, names = self.robot.find_joints(JOINT_ORDER, preserve_order=True)
        if names != JOINT_ORDER:
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

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.clone(), -1.0, 1.0)
        index = torch.clamp(self.episode_length_buf, max=self.q_des_trace.shape[0] - 1)
        self.q_des = self.q_des_trace[index]

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.q_des, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
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
