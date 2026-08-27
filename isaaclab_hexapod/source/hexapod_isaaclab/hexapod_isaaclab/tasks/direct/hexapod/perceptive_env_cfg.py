"""Experimental sensor-actor contract registered for integration work."""

from isaaclab.utils import configclass

from ....sensors import (
    body_imu_cfg,
    depth_raycast_cfg,
    foot_contact_cfg,
    legacy_height_scanner_cfg,
    mid360_raycast_cfg,
)
from .hexapod_env_cfg import HexapodEnvCfg


@configclass
class HexapodPerceptiveEnvCfg(HexapodEnvCfg):
    observation_space = {"policy": 195, "critic": 225}
    state_space = 0
    golden_replay = False
    use_ground_truth_terrain_in_actor = False
    enable_depth_sensor = False
    legacy_height_scanner = legacy_height_scanner_cfg()
    lidar = mid360_raycast_cfg()
    depth = depth_raycast_cfg()
    imu = body_imu_cfg()
    foot_contact = foot_contact_cfg()
