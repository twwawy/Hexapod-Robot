"""Experimental sensor-actor contract registered for integration work."""

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

from ....sensors import (
    body_imu_cfg,
    depth_raycast_cfg,
    foot_contact_cfg,
    legacy_height_scanner_cfg,
    mid360_raycast_cfg,
)
from ....terrains import mjx_curriculum_terrain_cfg
from .hexapod_env_cfg import HexapodEnvCfg


@configclass
class HexapodPerceptiveEnvCfg(HexapodEnvCfg):
    episode_length_s = 40.0
    observation_space = {"policy": 195, "critic": 225}
    state_space = 0
    golden_replay = False
    # Headless training default for a 24 GB RTX 3090.  CLI --num_envs remains
    # authoritative, so real-time checks can use one environment.
    scene = InteractiveSceneCfg(
        num_envs=512, env_spacing=4.0, replicate_physics=True
    )
    terrain = mjx_curriculum_terrain_cfg()
    use_ground_truth_terrain_in_actor = False
    # The Xacro has no confirmed physical depth-camera extrinsic.  Training
    # uses the explicitly provisional synthetic camera from perception_cfg.
    enable_depth_sensor = True
    legacy_height_scanner = legacy_height_scanner_cfg()
    lidar = mid360_raycast_cfg()
    depth = depth_raycast_cfg()
    imu = body_imu_cfg()
    foot_contact = foot_contact_cfg()
    command_speed_range = (0.06, 0.12)
    target_clearance = 0.316
    maximum_tilt_rad = 0.7853981633974483
    minimum_clearance = 0.14
