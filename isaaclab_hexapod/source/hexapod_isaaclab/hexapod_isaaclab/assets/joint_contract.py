"""Canonical Hexapod joint, frame, and actuator contract."""

from __future__ import annotations

import math


LEG_ORDER = ("RB", "RM", "RF", "LB", "LM", "LF")
JOINT_ORDER = tuple(f"{leg}_{joint}" for leg in LEG_ORDER for joint in (1, 2, 3))
FOOT_BODY_ORDER = tuple(f"{leg}_motor_horn_3_1" for leg in LEG_ORDER)
FOOT_SITE_ORDER = tuple(f"{leg}_foot_site" for leg in LEG_ORDER)
FOOT_SITE_LOCAL_POS = (
    (-0.135623225, 0.182433092, 0.0),
    (-0.224900024, 0.033100001, 0.0),
    (-0.182433092, -0.135623225, 0.0),
    (0.135623225, 0.182433092, 0.0),
    (0.224900024, 0.033100001, 0.0),
    (0.182433092, -0.135623225, 0.0),
)
"""Exact outer support points extracted from the current CAD foot meshes."""

FOOT_COLLISION_RADIUS_M = 0.032
FOOT_COLLISION_CENTER_LOCAL_POS = (
    (-0.112995808, 0.159805675, 0.0),
    (-0.192900024, 0.033100001, 0.0),
    (-0.159805675, -0.112995808, 0.0),
    (0.112995808, 0.159805675, 0.0),
    (0.192900024, 0.033100001, 0.0),
    (0.159805675, -0.112995808, 0.0),
)
"""Sphere centers whose outward surfaces pass through the CAD support points."""

MODEL_FORWARD = (0.0, -1.0, 0.0)
MODEL_LATERAL = (1.0, 0.0, 0.0)
HOME_ROOT_POS = (0.0, 0.0, 0.287006)
HOME_ROOT_QUAT_WXYZ = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
HOME_JOINT_POS = (
    0.0, -math.radians(30.0), math.radians(50.0),
    0.0, -math.radians(30.0), math.radians(50.0),
    0.0, -math.radians(30.0), math.radians(50.0),
    0.0, math.radians(30.0), -math.radians(50.0),
    0.0, math.radians(30.0), -math.radians(50.0),
    0.0, math.radians(30.0), -math.radians(50.0),
)

ACTION_SIZE = 18
LEGACY_OBSERVATION_SIZE = 146
FIRMWARE_TICKS_PER_POLICY_STEP = 4
FIRMWARE_DT = 0.005
PHYSICS_DT = 0.0025
POLICY_DT = 0.0200
DECIMATION = 8

ACTION_CONTRACT_VERSION = "stm32_firmware_adaptive_swing_residual_100mm_v4"
OBSERVATION_CONTRACT_VERSION = "firmware_state_collision_terrain_command5_pitch_v3"

__all__ = [
    "ACTION_CONTRACT_VERSION",
    "ACTION_SIZE",
    "DECIMATION",
    "FIRMWARE_DT",
    "FIRMWARE_TICKS_PER_POLICY_STEP",
    "FOOT_BODY_ORDER",
    "FOOT_COLLISION_CENTER_LOCAL_POS",
    "FOOT_COLLISION_RADIUS_M",
    "FOOT_SITE_LOCAL_POS",
    "FOOT_SITE_ORDER",
    "HOME_JOINT_POS",
    "HOME_ROOT_POS",
    "HOME_ROOT_QUAT_WXYZ",
    "JOINT_ORDER",
    "LEGACY_OBSERVATION_SIZE",
    "LEG_ORDER",
    "MODEL_FORWARD",
    "MODEL_LATERAL",
    "OBSERVATION_CONTRACT_VERSION",
    "PHYSICS_DT",
    "POLICY_DT",
]
