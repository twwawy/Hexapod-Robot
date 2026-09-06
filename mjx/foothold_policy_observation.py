"""LiDAR-only policy samples, plus a separate simulator-reference diagnostic."""
import time

import numpy as np


class LidarPolicyObservation:
    def __init__(self, env):
        self.body_id = env.mj_model.body('hexapod').id
        self.height_samples = np.asarray(env._height_samples)
        self.grid = None

    def sample(self, host, support_height):
        rotation = host.xmat[self.body_id].reshape(3, 3)
        forward = -rotation[:, 1]
        yaw = np.arctan2(forward[1], forward[0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy = self.height_samples @ np.array(((c, s), (-s, c))) + host.qpos[:2]
        now = time.monotonic()
        z, valid, age = np.full(15, np.nan), np.zeros(15, dtype=bool), np.full(15, np.inf)
        if self.grid is not None:
            z, valid = self.grid.sample(xy, now)
            indices, _ = self.grid.indices(xy)
            age = np.where(valid, now-self.grid.timestamp[indices[:, 0], indices[:, 1]], np.inf)
        features = np.where(valid, z-support_height, 0.).astype(np.float32)
        return dict(features=features, valid=valid, age_s=age, xy_odom=xy,
                    lidar_height_odom=np.where(valid, z, np.nan),
                    support_height=np.asarray(support_height), pose=host.qpos[:7].copy(),
                    simulation_time=np.asarray(host.time), monotonic_time=np.asarray(now))


def compare_with_reference(sample, field):
    """Logging only: compare identical XY samples to the course height raster.

    This matches the environment's bilinear reference, not exact MuJoCo triangle
    interpolation at edges. Neither these heights nor errors go to the actor.
    """
    xy = sample['xy_odom']
    col = (xy[:, 0]+6)*(field.shape[1]-1)/12
    row = (xy[:, 1]+6)*(field.shape[0]-1)/12
    c = np.clip(np.floor(col).astype(int), 0, field.shape[1]-2)
    r = np.clip(np.floor(row).astype(int), 0, field.shape[0]-2)
    u, v = np.clip(col-c, 0, 1), np.clip(row-r, 0, 1)
    gt = ((1-v)*((1-u)*field[r, c]+u*field[r, c+1])
          + v*((1-u)*field[r+1, c]+u*field[r+1, c+1]))
    inside = (np.abs(xy) <= 6).all(axis=1)
    gt = np.where(inside, gt, np.nan)
    mask = sample['valid'] & inside
    error = sample['lidar_height_odom'][mask] - gt[mask]
    summary = dict(observed_count=int(sample['valid'].sum()), compared_count=int(mask.sum()),
                   sample_count=len(mask), observation_fraction=float(sample['valid'].mean()),
                   rmse_m=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                   bias_m=float(np.mean(error)) if len(error) else None,
                   sample_simulation_time=float(sample['simulation_time']),
                   reference='2cm course raster, bilinear; diagnostic only')
    return summary, dict(**sample, gt_height_odom=gt, comparison_valid=mask)
