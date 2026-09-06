#!/usr/bin/env python3
"""Import and shape-check Hexapod configs without creating a physics scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import gymnasium as gym

from hexapod_isaaclab.assets import USD_PATH
from hexapod_isaaclab.contracts.observation_contract import (
    ACTOR_OBSERVATION_SIZE,
    CRITIC_OBSERVATION_SIZE,
    LEGACY_SIZE,
    TERRAIN_HEIGHT_SLICE,
    actor_proprioception,
    build_actor_observation,
    build_critic_observation,
)
from hexapod_isaaclab.contracts.training_handoff import load_training_handoff
from hexapod_isaaclab.perception import DEFAULT_ELEVATION_MAP_CFG, TerrainEncoder, rasterize_elevation_map
from hexapod_isaaclab.sensors import (
    DEPTH_EXTRINSIC_CONFIRMED,
    Hexapod15PointPatternCfg,
    MID360_CHANNELS,
    MID360_FRAME_RATE_HZ,
    MID360_MAX_RANGE_M,
    MID360_POINT_RATE_HZ,
    MID360_POSITION_BODY,
    MID360_VERTICAL_FOV_DEG,
    body_imu_cfg,
    depth_raycast_cfg,
    foot_contact_cfg,
    legacy_height_scanner_cfg,
    mid360_raycast_cfg,
)
from hexapod_isaaclab.sensors.ray_patterns import hexapod_15_point_pattern
from hexapod_isaaclab.tasks.direct.hexapod.agents.rsl_rl_ppo_cfg import HexapodPerceptivePPORunnerCfg
from hexapod_isaaclab.tasks.direct.hexapod.hexapod_env_cfg import HexapodEnvCfg
from hexapod_isaaclab.tasks.direct.hexapod.perceptive_env_cfg import HexapodPerceptiveEnvCfg


def main() -> None:
    batch = 2
    legacy = torch.zeros(batch, LEGACY_SIZE)
    legacy[:, TERRAIN_HEIGHT_SLICE] = 123.0
    proprio = actor_proprioception(legacy)
    actor = build_actor_observation(legacy, torch.zeros(batch, 64))
    critic = build_critic_observation(actor, torch.zeros(batch, 30))
    if torch.any(proprio == 123.0):
        raise AssertionError("ground-truth terrain leaked into actor proprioception")

    map_cfg = DEFAULT_ELEVATION_MAP_CFG
    points = torch.tensor([[[0.10, 0.10, 0.04], [0.11, 0.11, 0.06]]]).repeat(batch, 1, 1)
    elevation = rasterize_elevation_map(points, map_cfg)
    encoder = TerrainEncoder()
    latent = encoder(elevation)
    starts, directions = hexapod_15_point_pattern(Hexapod15PointPatternCfg(), "cpu")

    flat_cfg = HexapodEnvCfg()
    perceptive_cfg = HexapodPerceptiveEnvCfg()
    runner_cfg = HexapodPerceptivePPORunnerCfg()
    handoff = load_training_handoff()
    handoff_contracts = handoff["contracts"]
    sensor_cfgs = {
        "legacy_height_scanner": legacy_height_scanner_cfg(),
        "lidar": mid360_raycast_cfg(),
        "depth": depth_raycast_cfg(),
        "imu": body_imu_cfg(),
        "foot_contact": foot_contact_cfg(),
    }
    lidar_cfg = sensor_cfgs["lidar"]
    report = {
        "schema_version": 1,
        "checks": {
            "flat_action_space": flat_cfg.action_space == 18,
            "flat_observation_space": flat_cfg.observation_space == 146,
            "timing_400hz_decimation8": flat_cfg.sim.dt == 0.0025 and flat_cfg.decimation == 8,
            "actor_shape": list(actor.shape) == [batch, ACTOR_OBSERVATION_SIZE],
            "critic_shape": list(critic.shape) == [batch, CRITIC_OBSERVATION_SIZE],
            "terrain_gt_removed_from_actor": not bool(torch.any(proprio == 123.0)),
            "elevation_map_shape": list(elevation.shape) == [batch, 32, 24, 3],
            "terrain_latent_shape": list(latent.shape) == [batch, 64],
            "legacy_ray_pattern_shape": list(starts.shape) == [15, 3] and list(directions.shape) == [15, 3],
            "perceptive_observation_groups": perceptive_cfg.observation_space == {"policy": 195, "critic": 225},
            "rsl_asymmetric_groups": runner_cfg.obs_groups == {"policy": ["policy"], "critic": ["critic"]},
            "sensor_configs_construct": len(sensor_cfgs) == 5,
            "mid360_current_urdf_mount": tuple(lidar_cfg.offset.pos)
            == MID360_POSITION_BODY,
            "mid360_published_fov": (
                lidar_cfg.pattern_cfg.channels == MID360_CHANNELS
                and tuple(lidar_cfg.pattern_cfg.vertical_fov_range)
                == MID360_VERTICAL_FOV_DEG
            ),
            "mid360_published_rate_range": (
                lidar_cfg.update_period == 1.0 / MID360_FRAME_RATE_HZ
                and lidar_cfg.max_distance == MID360_MAX_RANGE_M
            ),
            "mid360_rtx_proxy_contract": (
                MID360_CHANNELS == 40
                and MID360_POINT_RATE_HZ == 200_000
                and MID360_MAX_RANGE_M == 40.0
            ),
            "depth_extrinsic_safety_gate": DEPTH_EXTRINSIC_CONFIRMED is False,
            "full_mesh_usd_selected": (
                USD_PATH.name == "hexapod_full_mesh_mjx_parity.usd"
                and USD_PATH.is_file()
            ),
            "mjx_handoff_action_contract": (
                handoff_contracts["action"]["size"] == 18
                and handoff_contracts["action"]["residual_scale_m"]
                == [0.1, 0.1, 0.1]
            ),
            "mjx_handoff_observation_contract": (
                handoff_contracts["observation"]["size"] == LEGACY_SIZE
            ),
            "mjx_handoff_reward_contract": (
                handoff_contracts["reward"]["version"]
                == "commanded_progress_motion_gate_v1"
            ),
            "unsafe_latest_weights_not_autoloaded": (
                handoff["isaac_transfer_gate"]["load_mjx_weights_by_default"]
                is False
            ),
            "flat_task_registered": "Hexapod-Firmware-Flat-Direct-v0" in gym.registry,
            "perceptive_task_registered": "Hexapod-Perceptive-Direct-v0" in gym.registry,
        },
        "dimensions": {
            "legacy": LEGACY_SIZE,
            "actor": ACTOR_OBSERVATION_SIZE,
            "critic": CRITIC_OBSERVATION_SIZE,
            "elevation_map": list(map_cfg.shape),
            "terrain_latent": int(latent.shape[-1]),
            "latest_mjx_terrain_level": handoff["latest_attempt"]["terrain_level"],
        },
    }
    report["passed"] = all(report["checks"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"configuration validation failed: {args.report}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
