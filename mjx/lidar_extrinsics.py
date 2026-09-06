"""User-specified LiDAR mounting transform, 2026-09-06.

Measurement datum: centre of the chassis underside, not the feet/ground plane.
The current base_link STL underside is z=-50.8 mm with zero visual origin.
Controller axes at that datum are forward/left/up; CAD forward is base_link -Y.
The sensor +Z axis points upward and is tilted forward 38 deg (upright mounting).
This raises the local horizontal scan plane by 7 deg from the previous 45 deg mount.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

BOTTOM_FRAME = 'robot_bottom_center'
SENSOR_FRAME = 'lidar_sensor_frame'
BASE_TO_BOTTOM_XYZ_M = (0.0, 0.0, -0.0508)
BASE_TO_BOTTOM_RPY_RAD = (0.0, 0.0, -math.pi / 2)
BOTTOM_TO_LIDAR_XYZ_M = (0.013529, 0.0, 0.215)
BOTTOM_TO_LIDAR_RPY_RAD = (0.0, math.radians(38.0), 0.0)


def measured_transform():
    """Return T_base_link_lidar_sensor_frame without touching the CAD visuals."""
    angle = BOTTOM_TO_LIDAR_RPY_RAD[1]
    c, s = math.cos(angle), math.sin(angle)
    transform = np.eye(4)
    # Rz(-90 deg) @ Ry(+38 deg): sensor +Z points up and forward.  Decreasing
    # this pitch from 45 deg raises the local horizontal scan rays by 7 deg.
    transform[:3, :3] = ((0.0, 1.0, 0.0), (-c, 0.0, -s), (-s, 0.0, c))
    transform[:3, 3] = (0.0, -BOTTOM_TO_LIDAR_XYZ_M[0],
                         BASE_TO_BOTTOM_XYZ_M[2] + BOTTOM_TO_LIDAR_XYZ_M[2])
    return transform


def add_sensor_frames(robot):
    """Add explicit measurement frames to the standalone generated URDF."""
    for parent, child, xyz, rpy in (
        ('base_link', BOTTOM_FRAME, BASE_TO_BOTTOM_XYZ_M, BASE_TO_BOTTOM_RPY_RAD),
        (BOTTOM_FRAME, SENSOR_FRAME, BOTTOM_TO_LIDAR_XYZ_M, BOTTOM_TO_LIDAR_RPY_RAD),
    ):
        if robot.find(f"link[@name='{child}']") is not None:
            raise ValueError(f'Measurement frame already exists: {child}')
        ET.SubElement(robot, 'link', name=child)
        joint = ET.SubElement(robot, 'joint', name=f'{child}_fixed', type='fixed')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin',
                      xyz=' '.join(f'{value:.12g}' for value in xyz),
                      rpy=' '.join(f'{value:.12g}' for value in rpy))


def measurement_metadata():
    return {
        'measurement_date': '2026-09-06',
        'source': 'user dimensions; runtime sensor validation left to user',
        'reference': 'chassis underside centre, not ground contact plane',
        'reference_axes': '+X forward, +Y left, +Z up',
        'T_base_bottom_xyz_m': BASE_TO_BOTTOM_XYZ_M,
        'T_base_bottom_rpy_rad': BASE_TO_BOTTOM_RPY_RAD,
        'T_bottom_lidar_xyz_m': BOTTOM_TO_LIDAR_XYZ_M,
        'T_bottom_lidar_rpy_rad': BOTTOM_TO_LIDAR_RPY_RAD,
        'pitch_description': 'upright: sensor +Z tilted 38 deg forward from vertical; scan raised 7 deg from previous 45 deg; roll=0',
    }
