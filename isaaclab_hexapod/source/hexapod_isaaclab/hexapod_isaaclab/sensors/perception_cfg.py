"""Pinned Isaac Lab 2.3 sensor setup for the perceptive locomotion track."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, ContactSensorCfg, ImuCfg, RayCasterCameraCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

from .ray_patterns import Hexapod15PointPatternCfg
from .mid360 import (
    MID360_CHANNELS,
    MID360_FRAME_RATE_HZ,
    MID360_HORIZONTAL_FOV_DEG,
    MID360_MAX_RANGE_M,
    MID360_POSITION_BODY,
    MID360_ROTATION_BODY_WXYZ,
    MID360_TRAINING_HORIZONTAL_RES_DEG,
    MID360_VERTICAL_FOV_DEG,
)


# Sensors are instantiated manually by DirectRLEnv, so use the expanded global
# regex rather than the InteractiveScene-only ``{ENV_REGEX_NS}`` placeholder.
ROBOT_BODY_PRIM = "/World/envs/env_.*/Robot/hexapod/hexapod"
GROUND_MESH_PRIM = "/World/ground"

MODEL_TO_SENSOR_FORWARD_QUAT_WXYZ = MID360_ROTATION_BODY_WXYZ

# No unambiguous depth-camera link exists in the checked-in Xacro.  This
# provisional transform keeps the pipeline configurable but is not authorized
# for sensor-policy training until measured on the physical robot.
DEPTH_EXTRINSIC_CONFIRMED = False
DEPTH_POSITION_BODY_PROVISIONAL = (0.0, -0.18, 0.10)
# Synthetic camera: body -Y forward, 15 degrees downward.  This is deliberately
# independent from the inverted LiDAR optical frame.
DEPTH_ROTATION_BODY_PROVISIONAL_WXYZ = (
    0.7010573846499779,
    0.09229595564125724,
    0.09229595564125725,
    -0.7010573846499778,
)


def legacy_height_scanner_cfg() -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=ROBOT_BODY_PRIM,
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(),
        ray_alignment="yaw",
        pattern_cfg=Hexapod15PointPatternCfg(start_height=1.0),
        max_distance=2.0,
        mesh_prim_paths=[GROUND_MESH_PRIM],
        debug_vis=False,
    )


def mid360_raycast_cfg() -> RayCasterCfg:
    """Batched geometric MID-360 proxy for locomotion training."""
    return RayCasterCfg(
        prim_path=ROBOT_BODY_PRIM,
        update_period=1.0 / MID360_FRAME_RATE_HZ,
        offset=RayCasterCfg.OffsetCfg(
            pos=MID360_POSITION_BODY,
            rot=MODEL_TO_SENSOR_FORWARD_QUAT_WXYZ,
        ),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=MID360_CHANNELS,
            vertical_fov_range=MID360_VERTICAL_FOV_DEG,
            horizontal_fov_range=MID360_HORIZONTAL_FOV_DEG,
            horizontal_res=MID360_TRAINING_HORIZONTAL_RES_DEG,
        ),
        max_distance=MID360_MAX_RANGE_M,
        mesh_prim_paths=[GROUND_MESH_PRIM],
        debug_vis=False,
    )


def depth_raycast_cfg() -> RayCasterCameraCfg:
    return RayCasterCameraCfg(
        prim_path=ROBOT_BODY_PRIM,
        update_period=0.05,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=DEPTH_POSITION_BODY_PROVISIONAL,
            rot=DEPTH_ROTATION_BODY_PROVISIONAL_WXYZ,
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="none",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.93,
            horizontal_aperture=3.68,
            width=32,
            height=24,
        ),
        max_distance=3.0,
        mesh_prim_paths=[GROUND_MESH_PRIM],
        debug_vis=False,
    )


def depth_camera_cfg() -> CameraCfg:
    """RTX depth alternative for the later sensor-fidelity phase."""
    return CameraCfg(
        prim_path=f"{ROBOT_BODY_PRIM}/depth_camera",
        update_period=0.05,
        offset=CameraCfg.OffsetCfg(
            pos=DEPTH_POSITION_BODY_PROVISIONAL,
            rot=DEPTH_ROTATION_BODY_PROVISIONAL_WXYZ,
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=3.68,
            clipping_range=(0.10, 3.0),
        ),
        data_types=["distance_to_image_plane"],
        width=64,
        height=48,
    )


def body_imu_cfg() -> ImuCfg:
    return ImuCfg(
        prim_path=ROBOT_BODY_PRIM,
        update_period=0.005,
        offset=ImuCfg.OffsetCfg(),
        gravity_bias=(0.0, 0.0, 9.81),
        debug_vis=False,
    )


def foot_contact_cfg() -> ContactSensorCfg:
    """Net forces on the six distal-leg bodies in controller leg order."""
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/hexapod/.*_motor_horn_3_1",
        update_period=0.005,
        history_length=1,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
