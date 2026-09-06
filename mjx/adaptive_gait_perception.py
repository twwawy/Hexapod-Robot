"""Batched JAX MID-360 angular ray proxy and timestamped rolling height map.

Only true ray returns update the map. The sensor model has first-hit robot and
terrain occlusion, sparse angular sampling, dropout and range noise. It is not
Livox's measured non-repeating pattern. Hfields are rejected by the environment
because the installed MJX ray primitive does not intersect them.
"""
from typing import NamedTuple

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

from lidar_extrinsics import measured_transform

GRID_N = 64
RESOLUTION = .05
MAX_AGE = 60.  # retain front scans long enough to reach the rear legs at slow speed
SENSOR_PERIOD = 5  # 100 ms, simulated time


class MapState(NamedTuple):
    height: jax.Array
    timestamp: jax.Array
    spread: jax.Array
    center: jax.Array
    hits: jax.Array


def initial_map(xy):
    return MapState(jp.zeros((GRID_N, GRID_N)), jp.full((GRID_N, GRID_N), -1e6),
                    jp.zeros((GRID_N, GRID_N)), jp.round(xy/RESOLUTION)*RESOLUTION, jp.asarray(0))


def sample(grid, xy, now):
    ij = jp.floor((xy-grid.center)/RESOLUTION + GRID_N/2).astype(jp.int32)
    inside = ((ij >= 0) & (ij < GRID_N)).all(axis=-1)
    ij = jp.clip(ij, 0, GRID_N-1)
    age = now-grid.timestamp[ij[..., 0], ij[..., 1]]
    valid = inside & (age >= 0.) & (age <= MAX_AGE)
    return grid.height[ij[..., 0], ij[..., 1]], valid, age, grid.spread[ij[..., 0], ij[..., 1]]


class AngularLidar:
    def __init__(self, model, root_id, azimuths=90, elevations=8, dropout=.05, noise=.005):
        if azimuths < 1 or elevations < 2 or not 0. <= dropout <= 1. or noise < 0.:
            raise ValueError('LiDAR requires positive azimuths, >=2 elevations, dropout in [0,1], noise >=0')
        self.model, self.root_id = model, root_id
        self.azimuths, self.elevations = azimuths, elevations
        self.dropout, self.noise = dropout, noise
        self.tf = jp.asarray(measured_transform())
        self.geom_is_terrain = jp.asarray(np.asarray(model.geom_bodyid) == 0)

    def update(self, data, grid, key, sequence):
        key, noise_key, dropout_key = jax.random.split(key, 3)
        fraction = jp.mod(sequence*.61803398875, 1.)
        az = (jp.arange(self.azimuths)+fraction)*2*jp.pi/self.azimuths
        # Focus most rays at the lower FOV edge, matching the upward sensor.
        low = max(1, int(self.elevations*.75))
        el = jp.deg2rad(jp.concatenate((
            -7.+(jp.arange(low)+fraction)*17./low,
            10.+(jp.arange(self.elevations-low)+fraction)*42./max(1, self.elevations-low))))
        az, el = jp.meshgrid(az, el, indexing='ij')
        local = jp.stack((jp.cos(el)*jp.cos(az), jp.cos(el)*jp.sin(az), jp.sin(el)), axis=-1).reshape(-1, 3)
        rotation = data.xmat[self.root_id]
        origin = data.xpos[self.root_id] + rotation @ self.tf[:3, 3]
        directions = local @ (rotation @ self.tf[:3, :3]).T
        distances, ids = jax.vmap(lambda ray: mjx.ray(
            self.model, data, origin, ray, geomgroup=(1, 1, 0, 0, 0, 0)))(directions)
        valid = (distances >= .1) & (distances <= 8.) & (ids >= 0)
        valid &= self.geom_is_terrain[jp.maximum(ids, 0)]
        valid &= jax.random.uniform(dropout_key, distances.shape) >= self.dropout
        distances = distances + self.noise*jax.random.normal(noise_key, distances.shape)
        points = origin + directions*distances[:, None]
        center = jp.round(data.qpos[:2]/RESOLUTION)*RESOLUTION
        shift = jp.rint((center-grid.center)/RESOLUTION).astype(jp.int32)
        i, j = jp.meshgrid(jp.arange(GRID_N), jp.arange(GRID_N), indexing='ij')
        oi, oj = i+shift[0], j+shift[1]
        overlap = (oi >= 0) & (oi < GRID_N) & (oj >= 0) & (oj < GRID_N)
        oi, oj = jp.clip(oi, 0, GRID_N-1), jp.clip(oj, 0, GRID_N-1)
        height = jp.where(overlap, grid.height[oi, oj], 0.)
        stamp = jp.where(overlap, grid.timestamp[oi, oj], -1e6)
        spread = jp.where(overlap, grid.spread[oi, oj], 0.)
        ij = jp.floor((points[:, :2]-center)/RESOLUTION + GRID_N/2).astype(jp.int32)
        valid &= ((ij >= 0) & (ij < GRID_N)).all(axis=-1)
        ij = jp.clip(ij, 0, GRID_N-1)
        flat = ij[:, 0]*GRID_N+ij[:, 1]
        maximum = jp.full(GRID_N*GRID_N, -jp.inf).at[flat].max(jp.where(valid, points[:, 2], -jp.inf))
        minimum = jp.full(GRID_N*GRID_N, jp.inf).at[flat].min(jp.where(valid, points[:, 2], jp.inf))
        seen = jp.isfinite(maximum).reshape(GRID_N, GRID_N)
        new_high = maximum.reshape(GRID_N, GRID_N)
        new_low = minimum.reshape(GRID_N, GRID_N)
        fresh = data.time-stamp <= MAX_AGE
        new_high = jp.where(fresh, jp.maximum(new_high, height), new_high)
        new_low = jp.where(fresh, jp.minimum(new_low, height-spread), new_low)
        updated = jp.where(seen, new_high, height)
        variation = jp.where(seen, new_high-new_low, spread)
        return MapState(updated, jp.where(seen, data.time, stamp), variation, center, jp.sum(valid)), key
