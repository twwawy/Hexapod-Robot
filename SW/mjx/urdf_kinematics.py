"""Kinematic constants measured from the source HEXAPEDAL URDF.

All values are expressed in a leg frame whose +X axis points away from the
body, +Y is tangent counter-clockwise around the body, and +Z points up.  The
right legs have a positive shoulder lateral offset; the left legs have the
same offset with the opposite sign.
"""

from __future__ import annotations

import math


SHOULDER_RADIAL_OFFSET = 0.074
SHOULDER_LATERAL_OFFSET = 0.0329
SHOULDER_VERTICAL_OFFSET = -0.0329
FEMUR_LENGTH = 0.124
TIBIA_LENGTH = 0.230

HOME_HIP_ANGLE = 0.0
HOME_KNEE_ANGLE = math.radians(30.0)
HOME_ANKLE_ANGLE = math.radians(50.0)

# FK of the URDF toe point at the documented 0/30/50 degree home pose,
# relative to joint 1.  Lateral sign is applied per side by the callers.
NOMINAL_FOOT_RADIAL = (
    SHOULDER_RADIAL_OFFSET
    + FEMUR_LENGTH * math.cos(HOME_KNEE_ANGLE)
    + TIBIA_LENGTH * math.cos(HOME_KNEE_ANGLE + HOME_ANKLE_ANGLE)
)
NOMINAL_FOOT_VERTICAL = (
    SHOULDER_VERTICAL_OFFSET
    - FEMUR_LENGTH * math.sin(HOME_KNEE_ANGLE)
    - TIBIA_LENGTH * math.sin(HOME_KNEE_ANGLE + HOME_ANKLE_ANGLE)
)

# Joint 1 is this far above the floating-root origin in the converted model.
HIP_JOINT_ROOT_HEIGHT = 0.0329
FOOT_COLLISION_RADIUS = 0.032
STAND_ROOT_HEIGHT = (
    -(HIP_JOINT_ROOT_HEIGHT + NOMINAL_FOOT_VERTICAL) + FOOT_COLLISION_RADIUS
)


def shoulder_lateral_offset(*, right: bool) -> float:
    return SHOULDER_LATERAL_OFFSET if right else -SHOULDER_LATERAL_OFFSET
