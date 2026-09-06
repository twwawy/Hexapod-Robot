"""Build an isolated training-link or CAD scene from a committed URDF snapshot."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from prepare_scene import _add_scene_elements, _standing_angle, ROOT_QUATERNION, STAND_HEIGHT
from prepare_urdf import _has_positive_inertia
from tripod_controller import LEG_PREFIXES
from mid360_profile import metadata as mid360_metadata
from foothold_link_model import replace_with_training_links, FOOT_RADIUS
from lidar_extrinsics import (
    SENSOR_FRAME, add_sensor_frames, measured_transform, measurement_metadata,
)

REPO = Path(__file__).resolve().parents[1]


def numbers(values):
    return ' '.join(f'{float(value):.12g}' for value in values)


def fixed_transform(robot, child):
    """Compose the actual fixed URDF chain from base_link, including RPY."""
    if child == 'base_link':
        return np.eye(4)
    joint = next((j for j in robot.findall('joint')
                  if j.find('child').get('link') == child), None)
    if joint is None or joint.get('type') != 'fixed':
        raise ValueError(f'No fixed base_link chain to sensor frame {child}')
    origin = joint.find('origin')
    r, p, y = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    transform = np.eye(4)
    transform[:3, :3] = ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                         (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
                         (-sp, cp*sr, cp*cr))
    transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return fixed_transform(robot, joint.find('parent').get('link')) @ transform


def snapshot(revision, output):
    commit = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', '--verify', f'{revision}^{{commit}}'],
        text=True).strip()
    destination = output / f'asset-{commit[:12]}'
    destination.mkdir(parents=True, exist_ok=True)
    archive = output / 'urdf_snapshot.tar'
    subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                    f'--output={archive}', commit, 'HW/urdf'], check=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unexpected archive entry: {member.name}')
        bundle.extractall(destination, members=members)
    return commit, destination


def add_obstacle_course(world, terrain):
    """Twelve-metre inspection area with distinct approach lanes."""
    floor = world.find("geom[@name='floor']")
    floor.set('size', '6 6 0.1')
    if terrain == 'flat':
        return []
    obstacles = []

    def box(name, xyz, half_size, color=(0.57, 0.43, 0.26, 1), euler=None):
        values = dict(name=name, type='box', pos=numbers(xyz), size=numbers(half_size),
                      rgba=numbers(color), group='0')
        if euler is not None:
            values['euler'] = numbers(euler)
        ET.SubElement(world, 'geom', **values)
        obstacles.append({'name': name, 'center': list(xyz), 'half_size': list(half_size)})

    # The initial pose is in a clear area. Stairs are visible before reaching them.
    for i in range(6):
        h = 0.04 * (i + 1)
        box(f'stair_{i+1}', (1.4 + i * 0.32, 0, h / 2), (0.16, 0.70, h / 2))
    box('stair_landing', (3.55, 0, 0.12), (0.39, 0.70, 0.12))
    # Left lane: broad low platforms and isolated stepping stones.
    for i in range(3):
        h = (0.04, 0.08, 0.12)[i]
        box(f'platform_{i}', (0.9 + i * 0.95, 2.0, h / 2), (0.30, 0.40, h / 2),
            (0.30, 0.47, 0.58, 1))
    for i in range(4):
        for j in range(3):
            h = 0.03 + 0.02 * ((i + j) % 3)
            box(f'stone_{i}_{j}', (-2.8 + i * 0.42, 1.4 + j * 0.42, h / 2),
                (0.14, 0.14, h / 2), (0.43, 0.49, 0.39, 1))
    # Right lane: continuous ramp and narrow ridges with flat approaches.
    angle = np.deg2rad(8.0)
    box('ramp', (1.8, -2.0, 0.70 * np.sin(angle) + 0.025 * np.cos(angle)),
        (0.70, 0.55, 0.025), (0.43, 0.51, 0.34, 1), (0, -angle, 0))
    for i, h in enumerate((0.025, 0.05, 0.075, 0.10)):
        box(f'ridge_{i}', (0.7 + i * 0.55, -3.7, h / 2), (0.025, 0.50, h / 2),
            (0.62, 0.34, 0.27, 1))
    # Rear lane: deterministic uneven patches and rounded obstacles.
    rng = np.random.default_rng(6)
    for i in range(5):
        for j in range(4):
            h = float(rng.uniform(0.015, 0.085))
            box(f'rough_{i}_{j}', (-3.2 + i * 0.31, -1.7 + j * 0.31, h / 2),
                (0.153, 0.153, h / 2), (0.46, 0.42, 0.37, 1))
    for i in range(5):
        ET.SubElement(world, 'geom', name=f'rock_{i}', type='ellipsoid',
                      pos=numbers((-3.5 + i * 0.50, -3.4 + 0.12 * (i % 2), 0.015)),
                      size=numbers((0.16, 0.13, 0.05 + 0.01 * i)),
                      rgba='0.44 0.45 0.48 1', group='0')
    return obstacles


def build_scene(output: Path, revision: str, terrain: str, lidar_frame: str,
                lidar_tf_source: str = 'measured', robot_model: str = 'skeleton'):
    if robot_model not in ('skeleton', 'mesh'):
        raise ValueError(f'Unknown robot model: {robot_model}')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / '.gitignore').write_text('*\n')
    commit, asset = snapshot(revision, output)
    source = asset / 'HW/urdf/urdf/HEXAPEDAL_URDF.xacro'
    robot = ET.parse(source).getroot()
    if lidar_tf_source == 'measured':
        sensor_transform = measured_transform()
        add_sensor_frames(robot)
        active_lidar_frame = SENSOR_FRAME
        extrinsic_source = 'User dimensions relative to chassis underside, 2026-09-06'
    elif lidar_tf_source == 'urdf':
        sensor_transform = fixed_transform(robot, lidar_frame)
        active_lidar_frame = lidar_frame
        extrinsic_source = 'URDF CAD frame; physical calibration unverified'
    else:
        raise ValueError(f'Unknown LiDAR TF source: {lidar_tf_source}')
    for element in list(robot):
        if element.tag.startswith('{') or element.tag == 'gazebo':
            robot.remove(element)
    for link in robot.findall('link'):
        for collision in list(link.findall('collision')):
            link.remove(collision)
        inertia = link.find('inertial')
        if inertia is not None and not _has_positive_inertia(inertia):
            link.remove(inertia)
    material = ET.SubElement(robot, 'material', name='silver')
    ET.SubElement(material, 'color', rgba='0.65 0.69 0.73 1')
    for mesh in robot.iter('mesh'):
        name = mesh.get('filename')
        prefix = 'package://HEXAPEDAL_URDF_description/'
        if not name.startswith(prefix):
            raise ValueError(f'Unsupported mesh URI: {name}')
        mesh.set('filename', str(asset / 'HW/urdf' / name[len(prefix):]))
    settings = robot.find('mujoco')
    if settings is None:
        settings = ET.SubElement(robot, 'mujoco')
    compiler = settings.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(settings, 'compiler')
    compiler.set('discardvisual', 'false')
    compiler.set('strippath', 'false')
    # Keep the URDF base and its inertial until the floating parent exists.
    # Fusing into world here would discard the fixed chassis mass/inertia.
    compiler.set('fusestatic', 'false')
    urdf = output / 'robot.urdf'
    ET.ElementTree(robot).write(urdf, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(urdf))
    scene_path = output / 'scene.xml'
    mujoco.mj_saveLastXML(str(scene_path), model)
    tree = ET.parse(scene_path)
    root = tree.getroot()
    root.set('model', f'Hexapod foothold preview {commit[:7]} - RL zero')
    _add_scene_elements(root)
    root.find('compiler').set('fusestatic', 'true')
    root.find('option').set('gravity', '0 0 0')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'headlight', ambient='0.4 0.4 0.4', diffuse='0.7 0.7 0.7')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='900')
    body = root.find('./worldbody/body[@name="hexapod"]')
    for geom in body.iter('geom'):
        # CAD remains visible and occludes rays, but this preview has no dynamics.
        geom.set('group', '1')
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
        mesh_name = geom.get('mesh', '').lower()
        if 'livox' in mesh_name or 'mid-360' in mesh_name:
            # Exclude the emitter housing and CAD FOV helper from its own rays.
            geom.set('group', '5')
        if 'fov' in mesh_name:
            geom.set('rgba', '0 0 0 0')
    sensor_quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(sensor_quaternion, sensor_transform[:3, :3].reshape(9))
    ET.SubElement(body, 'site', name='lidar_origin', pos=numbers(sensor_transform[:3, 3]),
                  quat=numbers(sensor_quaternion), size='0.008', rgba='0 0.8 1 1')
    world = root.find('worldbody')
    obstacles = add_obstacle_course(world, terrain)
    tree.write(scene_path, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if robot_model == 'skeleton':
        # Reuse compiled kinematics with the former MJX training primitives.
        replace_with_training_links(root, model)
        tree.write(scene_path, encoding='utf-8', xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    data.qpos[2] = STAND_HEIGHT
    data.qpos[3:7] = ROOT_QUATERNION
    for leg in LEG_PREFIXES:
        for joint in (1, 2, 3):
            data.qpos[model.joint(f'{leg}_{joint}').qposadr[0]] = _standing_angle(leg, joint)
    mujoco.mj_forward(model, data)
    tips = []
    for leg in LEG_PREFIXES:
        if robot_model == 'skeleton':
            tip = data.site_xpos[model.site(f'{leg}_sole').id].copy()
            tip[2] -= FOOT_RADIUS
            tips.append(tip)
            continue
        distal = model.body(f'{leg}_motor_horn_3_1').id
        points = []
        for geom_id in range(model.ngeom):
            mesh_id = int(model.geom_dataid[geom_id])
            if (int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)
                    or model.geom_bodyid[geom_id] != distal):
                continue
            if not model.mesh(mesh_id).name.startswith(f'{leg}_foot_'):
                continue
            start = model.mesh_vertadr[mesh_id]
            end = start + model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start:end]
            points.append(vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id])
        if not points:
            raise ValueError(f'No foot CAD vertices found for {leg}')
        vertices = np.concatenate(points)
        tip = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.0005].mean(axis=0)
        local_tip = data.xmat[distal].reshape(3, 3).T @ (tip - data.xpos[distal])
        target_body = root.find(f'.//body[@name="{leg}_motor_horn_3_1"]')
        ET.SubElement(target_body, 'site', name=f'{leg}_sole', pos=numbers(local_tip),
                      size='0.004', rgba='1 0.65 0.1 1')
        tips.append(tip)
    # Place the lowest home foot on the reference plane, without changing CAD axes.
    data.qpos[2] -= min(tip[2] for tip in tips)
    keyframes = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframes, 'key', name='home', qpos=numbers(data.qpos))
    ET.indent(tree, space='  ')
    tree.write(scene_path, encoding='utf-8', xml_declaration=True)
    metadata = {
        'commit': commit, 'urdf': str(source), 'terrain': terrain,
        'robot_model': robot_model,
        'geometry_source': ('mjx/prepare_rl_scene.py@3a817c4 training primitives'
                            if robot_model == 'skeleton' else 'URDF CAD meshes'),
        'lidar_occlusion': 'active robot model geometry',
        'foot_contact': ('sphere centre minus 0.032 m along world Z'
                         if robot_model == 'skeleton' else 'CAD sole site'),
        'lidar_frame': active_lidar_frame, 'T_base_lidar': sensor_transform.tolist(),
        'lidar_tf_source': lidar_tf_source,
        'sensor_extrinsic_source': extrinsic_source,
        'lidar_measurement': measurement_metadata() if lidar_tf_source == 'measured' else None,
        'state_source': 'MuJoCo pose stub, no LIO estimator',
        'mode': 'keyboard continuous tripod kinematic preview; no physics stepping',
        'arena_size_m': [12, 12], 'obstacles': obstacles,
        'rl_residual': 0, 'map_frame': 'rolling local grid with world-aligned odom history',
        'lidar_pattern': 'angular proxy with denser sampling near lower FOV edge, not Livox pattern',
        'mid360_fov': mid360_metadata(),
    }
    (output / 'scene_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return scene_path, metadata
