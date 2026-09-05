"""CPU perception worker and drawing helpers for the interactive explorer."""
from __future__ import annotations

import copy
import time

import mujoco
import numpy as np

from foothold_planner import ElevationMap, MapConfig, PlannerConfig, plan_with_nominal_fallback
from tripod_controller import LEG_PREFIXES
from mid360_profile import HORIZONTAL_FOV_DEG, VERTICAL_FOV_DEG, MIN_RANGE_M, LOW_DENSITY_SPLIT_DEG


class SiteIK:
    def __init__(self, model, home):
        self.model, self.home = model, home.copy()
        self.data = mujoco.MjData(model)
        self.jacobian = np.zeros((3, model.nv))
        self.site_ids = [model.site(f'{leg}_sole').id for leg in LEG_PREFIXES]
        self.joints = [[model.joint(f'{leg}_{j}').id for j in (1, 2, 3)] for leg in LEG_PREFIXES]

    def solve(self, leg, target, seed=None):
        ids = self.joints[leg]
        qids, vids = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        limits = self.model.jnt_range[ids]
        self.data.qpos[:] = self.home
        if seed is not None:
            self.data.qpos[qids] = seed
        for _ in range(40):
            # Only position kinematics / Jacobian preparation are required here.
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            error = target - self.data.site_xpos[self.site_ids[leg]]
            if np.linalg.norm(error) < 0.001:
                return self.data.qpos[qids].copy()
            mujoco.mj_jacSite(self.model, self.data, self.jacobian, None, self.site_ids[leg])
            jac = self.jacobian[:, vids]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(3), error)
            self.data.qpos[qids] = np.clip(self.data.qpos[qids] + np.clip(delta, -0.15, 0.15),
                                          limits[:, 0] + 0.01, limits[:, 1] - 0.01)
        return None


class LidarProxy:
    """Upward LiDAR with unchanged FOV; extra samples near its lower FOV edge.

    This remains an angular proxy, not a measured Livox scan pattern. Occluding
    robot geoms block rays; filtering a self return never exposes the ground behind it.
    """
    def __init__(self, model, azimuth_samples, elevation_samples, max_range):
        self.model = model
        self.site = model.site('lidar_origin').id
        self.azimuth_samples, self.elevation_samples = azimuth_samples, elevation_samples
        self.max_range = max_range
        self.sequence = 0
        self.ray_count = azimuth_samples * elevation_samples
        self.distances = np.empty(self.ray_count)
        self.geom_ids = np.empty(self.ray_count, dtype=np.int32)
        self.groups = np.array((1, 1, 0, 0, 0, 0), dtype=np.uint8)

    def scan(self, data):
        fraction = (self.sequence * 0.61803398875) % 1
        azimuth = (np.arange(self.azimuth_samples) + fraction) * np.deg2rad(HORIZONTAL_FOV_DEG) / self.azimuth_samples
        low_count = max(1, min(self.elevation_samples - 1, int(self.elevation_samples * 0.75)))
        upper_count = self.elevation_samples - low_count
        lower, upper = VERTICAL_FOV_DEG
        split = LOW_DENSITY_SPLIT_DEG
        elevation = np.deg2rad(np.r_[
            lower + (np.arange(low_count) + fraction) * (split - lower) / low_count,
            split + (np.arange(upper_count) + fraction) * (upper - split) / upper_count])
        azimuth, elevation = np.meshgrid(azimuth, elevation, indexing='ij')
        directions = np.stack((np.cos(elevation) * np.cos(azimuth),
                                np.cos(elevation) * np.sin(azimuth), np.sin(elevation)), axis=-1).reshape(-1, 3)
        origin = data.site_xpos[self.site].copy()
        rays = np.ascontiguousarray(directions @ data.site_xmat[self.site].reshape(3, 3).T)
        mujoco.mj_multiRay(self.model, data, origin, rays.reshape(-1), self.groups,
                           True, -1, self.geom_ids, self.distances, None, self.ray_count, self.max_range)
        hit = (self.distances >= MIN_RANGE_M) & (self.distances <= self.max_range) & (self.geom_ids >= 0)
        robot = hit & (self.model.geom_group[np.maximum(self.geom_ids, 0)] != 0)
        ground = hit & ~robot
        self.sequence += 1
        points = origin + rays[ground] * self.distances[ground, None]
        stats = dict(ground_hits=int(ground.sum()), body_hits=int(robot.sum()),
                     downward_rays=int(np.sum(rays[:, 2] < 0)), rays=self.ray_count)
        return points, stats


class PerceptionWorker:
    """One worker owns its map, raycast MjData and IK MjData; UI never mutates them."""
    def __init__(self, model, args, home):
        self.model, self.args = model, args
        self.data = mujoco.MjData(model)
        self.ik = SiteIK(model, home)
        self.reference_data = mujoco.MjData(model)
        self.neutral_qpos = home.copy()
        self.lidar = LidarProxy(model, args.azimuth_samples, args.elevation_samples, args.lidar_range)
        self.map_cfg = MapConfig(args.map_half_extent, args.map_resolution, args.map_max_age)
        self.grid = ElevationMap(self.map_cfg)
        self.cfg = PlannerConfig()
        self.generation = -1
        self.points = np.empty((0, 3))
        self.stats = dict(ground_hits=0, body_hits=0, downward_rays=0, rays=self.lidar.ray_count)

    def compute(self, qpos, command, generation, scanning):
        begin = time.monotonic()
        if generation != self.generation:
            self.grid = ElevationMap(self.map_cfg)
            self.points = np.empty((0, 3))
            self.generation = generation
        self.grid.recenter(qpos[:2])
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.ik.home = qpos.copy()
        starts = self.data.site_xpos[self.ik.site_ids].copy()
        self.reference_data.qpos[:] = self.neutral_qpos
        self.reference_data.qpos[:7] = qpos[:7]
        mujoco.mj_kinematics(self.model, self.reference_data)
        neutral_feet = self.reference_data.site_xpos[self.ik.site_ids].copy()
        root_rotation = self.data.xmat[self.model.body('hexapod').id].reshape(3, 3)
        forward, left = -root_rotation[:, 1], root_rotation[:, 0]
        vx, vy, wz = command
        moving = np.max(np.abs(command)) > 0.001
        displacement = (vx * forward + vy * left) * 0.6 if moving else self.args.step_length * forward
        yaw_delta = wz * 0.6
        c, s = np.cos(yaw_delta), np.sin(yaw_delta)
        yaw_rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        nominal = (neutral_feet - qpos[:3]) @ yaw_rotation.T + qpos[:3] + displacement
        scan_begin = time.monotonic()
        if scanning:
            self.points, self.stats = self.lidar.scan(self.data)
            self.grid.update(self.points, time.monotonic())
        scan_ms = (time.monotonic() - scan_begin) * 1000
        plan_begin = time.monotonic()
        plans = []
        for leg in range(6):
            cache = {}

            def reachable(point):
                key = tuple(np.round(point, 6))
                if key not in cache:
                    cache[key] = self.ik.solve(leg, point) is not None
                return cache[key]

            plans.append(plan_with_nominal_fallback(
                self.grid, starts[leg], nominal[leg], plan_begin, reachable, self.cfg,
                allow_unknown=not self.args.require_observed))
        # Publish independent snapshots; only the worker writes its live grid.
        grid = copy.copy(self.grid)
        for name in ('height', 'vertical_range', 'timestamp', 'center'):
            setattr(grid, name, getattr(self.grid, name).copy())
        valid = grid.valid(time.monotonic())
        local = np.linalg.norm(grid.centers() - qpos[:2], axis=-1) < 0.65
        return dict(grid=grid, points=self.points.copy(), plans=plans, starts=starts,
                    nominal=nominal, pose=qpos.copy(), generation=generation,
                    stamp=begin, completed=time.monotonic(), scan_ms=scan_ms,
                    plan_ms=(time.monotonic() - plan_begin) * 1000, stats=self.stats.copy(),
                    near_valid_fraction=float(valid[local].mean()))


def add_sphere(scene, point, radius, color, label=''):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
                        np.asarray(point, dtype=float), np.eye(3).reshape(-1), np.asarray(color, np.float32))
    if label:
        geom.label = label
    scene.ngeom += 1


def add_line(scene, start, end, color, width=0.002):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).reshape(-1), np.asarray(color, np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(start, dtype=float), np.asarray(end, dtype=float))
    scene.ngeom += 1


def draw_robot_skeleton(scene, data, joint_ids, sole_ids):
    """Connect actual world-space joint anchors and sole sites, with no CAD rendering.

    Overlay geometry is purely visual and cannot create phantom LiDAR returns.
    Joint order and foot locations are shared with IK, including CAD-derived soles.
    """
    hips = np.array([data.xanchor[ids[0]] for ids in joint_ids])
    body_color = (0.65, 0.73, 0.82, 1)
    # RF -> RM -> RB -> LB -> LM -> LF closes the chassis outline.
    perimeter = (0, 1, 2, 5, 4, 3, 0)
    for a, b in zip(perimeter[:-1], perimeter[1:]):
        add_line(scene, hips[a], hips[b], body_color, 0.005)
    for leg, ids in enumerate(joint_ids):
        chain = np.vstack((data.xanchor[ids], data.site_xpos[sole_ids[leg]]))
        for start, end in zip(chain[:-1], chain[1:]):
            if np.linalg.norm(end - start) > 1e-8:
                add_line(scene, start, end, (0.9, 0.93, 0.97, 1), 0.006)
        for anchor in chain[:-1]:
            add_sphere(scene, anchor, 0.010, body_color)
        add_sphere(scene, chain[-1], 0.008, (1, 1, 1, 1))


def add_height_cell(scene, point, resolution, color):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.array((resolution * 0.43, resolution * 0.43, 0.0015)),
                        np.asarray(point, dtype=float) + (0, 0, 0.005), np.eye(3).reshape(-1),
                        np.asarray(color, np.float32))
    scene.ngeom += 1


def draw_mid360_fov(scene, origin, rotation, radius):
    """Draw angular boundaries in the same world transform as the ray emitter.

    Wires depict the unobstructed angular envelope, not measured returns. Radius
    is for readability only and is independent of the simulated detection range.
    """
    azimuths = np.linspace(0, np.deg2rad(HORIZONTAL_FOV_DEG), 73)
    lower, upper = np.deg2rad(VERTICAL_FOV_DEG)
    colors = ((1.0, 0.6, 0.1, 0.65), (0.25, 0.65, 1.0, 0.65))
    for elevation, color in zip((lower, upper), colors):
        ring_sensor = np.column_stack((np.cos(elevation) * np.cos(azimuths),
                                       np.cos(elevation) * np.sin(azimuths),
                                       np.full_like(azimuths, np.sin(elevation))))
        ring_world = origin + radius * (ring_sensor @ rotation.T)
        for start, end in zip(ring_world[:-1], ring_world[1:]):
            add_line(scene, start, end, color, 0.0015)
        for endpoint in ring_world[:-1:6]:
            add_line(scene, origin, endpoint, color, 0.001)
    elevations = np.linspace(lower, upper, 13)
    for azimuth in azimuths[:-1:6]:
        arc_sensor = np.column_stack((np.cos(elevations) * np.cos(azimuth),
                                      np.cos(elevations) * np.sin(azimuth), np.sin(elevations)))
        arc_world = origin + radius * (arc_sensor @ rotation.T)
        for start, end in zip(arc_world[:-1], arc_world[1:]):
            add_line(scene, start, end, (0.3, 0.8, 0.95, 0.3), 0.001)
