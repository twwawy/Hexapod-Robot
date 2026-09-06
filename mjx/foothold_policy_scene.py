"""Use the explorer's obstacle course in the archived PPO dynamics environment."""
from dataclasses import replace
import xml.etree.ElementTree as ET

import jax.numpy as jp
import mujoco
import numpy as np

from foothold_preview_scene import add_obstacle_course, numbers, fixed_transform
from lidar_extrinsics import measured_transform


def course_heightfield(terrain, spacing=0.02):
    world = ET.Element('worldbody')
    ET.SubElement(world, 'geom', name='floor')
    add_obstacle_course(world, terrain)
    axis = np.linspace(-6, 6, int(round(12/spacing))+1)
    x, y = np.meshgrid(axis, axis)
    heights = np.zeros_like(x)
    for geom in list(world)[1:]:
        center = np.fromstring(geom.get('pos'), sep=' ')
        size = np.fromstring(geom.get('size'), sep=' ')
        if geom.get('type') == 'ellipsoid':
            radial = ((x-center[0])/size[0])**2 + ((y-center[1])/size[1])**2
            top = np.where(radial <= 1, center[2]+size[2]*np.sqrt(np.maximum(0, 1-radial)), 0)
        else:
            # Course boxes are axis aligned except for the ramp's Y rotation.
            pitch = np.fromstring(geom.get('euler', '0 0 0'), sep=' ')[1]
            c, s = np.cos(pitch), np.sin(pitch)
            origin = (c*(x-center[0])+s*center[2], y-center[1],
                      s*(x-center[0])-c*center[2])
            direction = (-s, 0.0, c)
            low, high = np.full_like(x, -np.inf), np.full_like(x, np.inf)
            valid = np.ones_like(x, dtype=bool)
            for component, slope, half in zip(origin, direction, size):
                if abs(slope) < 1e-10:
                    valid &= np.abs(component) <= half
                else:
                    a, b = (-half-component)/slope, (half-component)/slope
                    low, high = np.maximum(low, np.minimum(a, b)), np.minimum(high, np.maximum(a, b))
            top = np.where(valid & (low <= high), high, 0)
        heights = np.maximum(heights, top)
    return heights


def make_explorer_environment(module, config, level, request, source):
    """Called after archived robot imports, inside the isolated policy process."""
    import terrain_curriculum as curriculum
    field = course_heightfield(request['terrain'])
    # One heightfield keeps the MJX contact graph bounded across the wide arena.
    # Its 2 cm raster is used for both the physical surface and LiDAR raycasts.
    height_scale = max(float(field.max()), 0.01)
    sensor = measured_transform()
    if request['lidar_tf_source'] == 'urdf':
        robot = ET.parse(source/'HW/urdf/urdf/HEXAPEDAL_URDF.xacro').getroot()
        sensor = fixed_transform(robot, request['lidar_frame'])
    original_prepare = module.prepare_rl_scene

    def prepare_scene(path):
        original_prepare(path)
        tree = ET.parse(path)
        root = tree.getroot()
        hfield = root.find('./asset/hfield[@name="rough_hfield"]')
        hfield.set('nrow', str(field.shape[0]))
        hfield.set('ncol', str(field.shape[1]))
        hfield.set('size', f'6 6 {height_scale} 0.05')
        hfield.set('elevation', numbers((field/height_scale).ravel()))
        root.find('./worldbody/geom[@name="floor"]').set('size', '6 6 0.1')
        robot = root.find('./worldbody/body[@name="hexapod"]')
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, sensor[:3, :3].reshape(9))
        ET.SubElement(robot, 'site', name='lidar_origin', pos=numbers(sensor[:3, 3]),
                      quat=numbers(quat), size='0.008', rgba='0 0.8 1 1')
        for body in root.iter('body'):
            for site in list(body.findall('site')):
                if site.get('name', '').endswith('_foot_site'):
                    ET.SubElement(body, 'site', name=site.get('name').replace('_foot_site', '_sole'),
                                  pos=site.get('pos'), size='0.004', rgba='0 0 0 0')
        tree.write(path, encoding='utf-8', xml_declaration=True)
        return path

    module.prepare_rl_scene = prepare_scene
    curriculum.TERRAIN_START_X = module.TERRAIN_START_X = 1.24
    curriculum.STAIR_DEPTH = module.STAIR_DEPTH = 0.32
    if request['terrain'] == 'steps':
        specs = list(curriculum.TERRAIN_LEVELS)
        specs[level] = replace(specs[level], stair_count=6, stair_riser=0.04,
                               name='explorer_course_6x4cm')
        curriculum.TERRAIN_LEVELS = tuple(specs)
    config.dr_enabled = False
    config.command_min_speed = config.command_max_speed = config.command_max_yaw_rate = 0.0
    config.command_delay = 0.0

    class ExplorerEnvironment(module.HexapodRoughTerrainEnv):
        def _configure_terrain_geometry(self):
            for name in ['rough_hfield_geom', 'terrain_ramp', 'terrain_plateau'] + [
                    f'stair_{i+1}' for i in range(curriculum.MAX_STAIR_COUNT)]:
                self._set_geom_active(name, False, (0, 0, 0, 0))
                self._mj_model.geom_group[self._mj_model.geom(name).id] = 5
            surface = self._set_geom_active('rough_hfield_geom', True, (0.43, 0.46, 0.40, 1))
            self._mj_model.geom_pos[surface.id] = (0, 0, 0)
            self._mj_model.geom_group[surface.id] = 0
            # Hide the coincident floor in the raster extent to avoid z fighting.
            floor = self._mj_model.geom('floor').id
            self._mj_model.geom_rgba[floor, 3] = 0
            self._mj_model.geom_contype[floor] = self._mj_model.geom_conaffinity[floor] = 0
            for i in range(self._mj_model.ngeom):
                if self._mj_model.geom_bodyid[i] != 0:
                    self._mj_model.geom_group[i] = 1
            self._course_grid = jp.asarray(field)

        def _terrain_height(self, xy):
            # Match the course grid instead of the archived straight-stair map.
            col = (xy[..., 0]+6)*(field.shape[1]-1)/12
            row = (xy[..., 1]+6)*(field.shape[0]-1)/12
            c = jp.clip(jp.floor(col).astype(jp.int32), 0, field.shape[1]-2)
            r = jp.clip(jp.floor(row).astype(jp.int32), 0, field.shape[0]-2)
            u, v = jp.clip(col-c, 0, 1), jp.clip(row-r, 0, 1)
            h = self._course_grid
            z = ((1-v)*((1-u)*h[r, c]+u*h[r, c+1])
                 + v*((1-u)*h[r+1, c]+u*h[r+1, c+1]))
            return jp.where((jp.abs(xy) <= 6).all(axis=-1), z, 0)

    env = ExplorerEnvironment(config=config, terrain_level=level)
    env._terrain_goal_x = 5.8  # Keep exploring after the central staircase.
    return env, sensor
