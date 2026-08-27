"""Pinned Isaac Lab 2.3 sensor setup for the perceptive locomotion track."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, ContactSensorCfg, ImuCfg, RayCasterCameraCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

from .ray_patterns import Hexapod15PointPatternCfg


ROBOT_BODY_PRIM = "{ENV_REGEX_NS}/Robot/hexapod/hexapod"
GROUND_MESH_PRIM = "/World/ground"

# Full fixed-joint chain base_link -> MID-360 from the main Xacro.  All joints
# in that chain have zero RPY.  The simplified dynamics USD intentionally does
# not carry the MID-360 CAD mass/mesh, so the simulated sensor uses this offset.
MID360_POSITION_BODY = (-0.017929, 0.004714, 0.393473)
MODEL_TO_SENSOR_FORWARD_QUAT_WXYZ = (
    math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)
)

# No unambiguous depth-camera link exists in the checked-in Xacro.  This
# provisional transform keeps the pipeline configurable but is not authorized
# for sensor-policy training until measured on the physical robot.
DEPTH_EXTRINSIC_CONFIRMED = False
DEPTH_POSITION_BODY_PROVISIONAL = (0.0, -0.18, 0.10)
DEPTH_ROTATION_BODY_PROVISIONAL_WXYZ = MODEL_TO_SENSOR_FORWARD_QUAT_WXYZ


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
    """Phase-1 sparse geometric LiDAR, not an optical MID-360 model."""
    return RayCasterCfg(
        prim_path=ROBOT_BODY_PRIM,
        update_period=0.05,
        offset=RayCasterCfg.OffsetCfg(
            pos=MID360_POSITION_BODY,
            rot=MODEL_TO_SENSOR_FORWARD_QUAT_WXYZ,
        ),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-7.0, 52.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=4.0,
        ),
        max_distance=8.0,
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
            width=64,
            height=48,
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
        prim_path="{ENV_REGEX_NS}/Robot/hexapod/.*_motor_horn_3_1",
        update_period=0.005,
        history_length=1,
        track_air_time=True,
    )
