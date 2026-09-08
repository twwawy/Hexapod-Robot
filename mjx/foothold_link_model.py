"""Legacy MJX training primitives for the foothold preview.

Geometry comes from mjx/prepare_rl_scene.py at 3a817c4 (_add_robot_colliders),
also used by SW/mjx/prepare_rl_scene.py: box torso, three capsules per leg,
230 mm distal segment, 32 mm spherical foot. Keep this adapter independent
of terrain/training dependencies and local changes to the training pipeline.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tripod_controller import LEG_PREFIXES

FOOT_RADIUS = 0.032


def numbers(values):
    return ' '.join(f'{float(value):.12g}' for value in values)


def replace_with_training_links(root, source_model):
    """Use compiled body transforms, joints and inertias, with training geometry."""
    world = root.find('worldbody')
    old_robot = world.find('body[@name="hexapod"]')
    if old_robot is None:
        raise ValueError('Missing hexapod body')
    robot_id = source_model.body('hexapod').id
    bodies_by_id = {}
    # Rebuild from the compiled tree so fixed CAD offsets are already composed;
    # saved source MJCF can still contain the pre-fusion fixed attachment bodies.
    for body_id in range(robot_id, source_model.nbody):
        parent = int(source_model.body_parentid[body_id])
        if body_id != robot_id and parent not in bodies_by_id:
            continue
        body = ET.Element('body', name=source_model.body(body_id).name,
                          pos=numbers(source_model.body_pos[body_id]),
                          quat=numbers(source_model.body_quat[body_id]))
        if source_model.body_mass[body_id] > 0:
            ET.SubElement(body, 'inertial',
                pos=numbers(source_model.body_ipos[body_id]),
                quat=numbers(source_model.body_iquat[body_id]),
                mass=str(float(source_model.body_mass[body_id])),
                diaginertia=numbers(source_model.body_inertia[body_id]))
        start = int(source_model.body_jntadr[body_id])
        for joint_id in range(start, start + int(source_model.body_jntnum[body_id])):
            joint_type = source_model.jnt_type[joint_id]
            name = source_model.joint(joint_id).name
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                ET.SubElement(body, 'freejoint', name=name)
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                dof = int(source_model.jnt_dofadr[joint_id])
                ET.SubElement(body, 'joint', name=name, type='hinge',
                    pos=numbers(source_model.jnt_pos[joint_id]),
                    axis=numbers(source_model.jnt_axis[joint_id]),
                    limited='true' if source_model.jnt_limited[joint_id] else 'false',
                    range=numbers(source_model.jnt_range[joint_id]),
                    ref=str(float(source_model.qpos0[source_model.jnt_qposadr[joint_id]])),
                    armature=str(float(source_model.dof_armature[dof])),
                    damping=str(float(source_model.dof_damping[dof])),
                    frictionloss=str(float(source_model.dof_frictionloss[dof])))
            else:
                raise ValueError(f'Unsupported training-link joint type: {name}')
        if body_id != robot_id:
            bodies_by_id[parent].append(body)
        bodies_by_id[body_id] = body
    robot = bodies_by_id[robot_id]
    sensor = source_model.site('lidar_origin').id
    ET.SubElement(bodies_by_id[int(source_model.site_bodyid[sensor])], 'site',
                  name='lidar_origin', pos=numbers(source_model.site_pos[sensor]),
                  quat=numbers(source_model.site_quat[sensor]), size='0.008', rgba='0 0.8 1 1')
    world.remove(old_robot)
    world.append(robot)
    asset = root.find('asset')
    if asset is not None:
        for mesh in list(asset.findall('mesh')):
            asset.remove(mesh)

    def geom(body, **attributes):
        # Group 1 is also the LiDAR robot-occlusion group.
        return ET.SubElement(body, 'geom', group='1', contype='0', conaffinity='0',
                             **attributes)

    geom(robot, name='torso_collision', type='box', size='0.17 0.15 0.045',
         rgba='0.22 0.30 0.38 1')
    for leg in LEG_PREFIXES:
        bodies = [root.find(f'.//body[@name="{leg}_{suffix}"]') for suffix in
                  ('motor_horn_1_1', 'DS51150_270_2_1', 'motor_horn_3_1')]
        if any(body is None for body in bodies):
            raise ValueError(f'Missing training-model articulated bodies for {leg}')
        body1, body2, body3 = bodies
        outward = np.fromstring(body1.get('pos'), sep=' ')
        outward[2] = 0
        outward /= np.linalg.norm(outward)
        foot_center = 0.230 * outward
        ends = (np.fromstring(body2.get('pos'), sep=' '),
                np.fromstring(body3.get('pos'), sep=' '), foot_center)
        for body, end, name, radius, color in zip(
                bodies, ends, ('coxa', 'femur', 'tibia'), (0.028, 0.026, 0.023),
                ('0.30 0.38 0.45 1', '0.34 0.43 0.52 1', '0.40 0.50 0.60 1')):
            geom(body, name=f'{leg}_{name}_collision', type='capsule',
                 fromto=f'0 0 0 {numbers(end)}', size=str(radius), rgba=color)
        geom(body3, name=f'{leg}_foot_collision', type='sphere',
             pos=numbers(foot_center), size=str(FOOT_RADIUS), rgba='0.85 0.45 0.12 1')
        # The site is the sphere centre. SiteIK subtracts its radius along world
        # Z for the ground contact, rather than rotating a fake sole with the leg.
        ET.SubElement(body3, 'site', name=f'{leg}_sole', pos=numbers(foot_center),
                      size='0.004', rgba='0 0 0 0')
