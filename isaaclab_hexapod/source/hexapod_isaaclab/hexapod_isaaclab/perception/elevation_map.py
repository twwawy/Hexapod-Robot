"""Gravity-aligned LiDAR/depth fusion into a local elevation map."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ElevationMapCfg:
    x_min: float = -0.4
    x_max: float = 1.2
    y_min: float = -0.6
    y_max: float = 0.6
    resolution: float = 0.05
    minimum_points: int = 1

    @property
    def rows(self) -> int:
        return round((self.x_max - self.x_min) / self.resolution)

    @property
    def cols(self) -> int:
        return round((self.y_max - self.y_min) / self.resolution)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.rows, self.cols, 3


DEFAULT_ELEVATION_MAP_CFG = ElevationMapCfg()


def quat_apply_wxyz(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by a wxyz quaternion with broadcast-compatible leading dims."""
    q_xyz = quaternion[..., 1:]
    q_w = quaternion[..., :1]
    while q_xyz.ndim < vectors.ndim:
        q_xyz = q_xyz.unsqueeze(-2)
        q_w = q_w.unsqueeze(-2)
    cross = torch.linalg.cross(q_xyz.expand_as(vectors), vectors, dim=-1)
    return vectors + 2.0 * (q_w * cross + torch.linalg.cross(q_xyz.expand_as(vectors), cross, dim=-1))


def gravity_align_heading(
    points_body: torch.Tensor, body_quaternion_wxyz: torch.Tensor
) -> torch.Tensor:
    """Remove body roll/pitch while retaining robot heading as local +X."""
    points_world_aligned = quat_apply_wxyz(body_quaternion_wxyz, points_body)
    w, x, y, z = body_quaternion_wxyz.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    px, py, pz = points_world_aligned.unbind(dim=-1)
    while cosine.ndim < px.ndim:
        cosine = cosine.unsqueeze(-1)
        sine = sine.unsqueeze(-1)
    model_x = cosine * px + sine * py
    model_y = -sine * px + cosine * py
    # The CAD/MJX body frame uses -Y as forward and +X as lateral.  Publish
    # the map in the controller convention: +X forward, +Y left, +Z up.
    return torch.stack((-model_y, model_x, pz), dim=-1)


def fuse_point_clouds(*clouds: torch.Tensor | None) -> torch.Tensor:
    available = [cloud for cloud in clouds if cloud is not None]
    if not available:
        raise ValueError("at least one point cloud is required")
    batch = available[0].shape[0]
    if any(cloud.ndim != 3 or cloud.shape[0] != batch or cloud.shape[-1] != 3 for cloud in available):
        raise ValueError("point clouds must all have shape [N, P, 3]")
    return torch.cat(available, dim=1)


def rasterize_elevation_map(
    points_heading: torch.Tensor,
    cfg: ElevationMapCfg = DEFAULT_ELEVATION_MAP_CFG,
) -> torch.Tensor:
    """Return height, confidence, and vertical roughness as ``[N,32,24,3]``."""
    if points_heading.ndim != 3 or points_heading.shape[-1] != 3:
        raise ValueError("points must have shape [N, P, 3]")
    num_envs, num_points, _ = points_heading.shape
    x, y, z = points_heading.unbind(dim=-1)
    row = torch.floor((x - cfg.x_min) / cfg.resolution).to(torch.long)
    col = torch.floor((y - cfg.y_min) / cfg.resolution).to(torch.long)
    valid = (
        torch.isfinite(points_heading).all(dim=-1)
        & (row >= 0)
        & (row < cfg.rows)
        & (col >= 0)
        & (col < cfg.cols)
    )
    cell_count = cfg.rows * cfg.cols
    local_index = row.clamp(0, cfg.rows - 1) * cfg.cols + col.clamp(0, cfg.cols - 1)
    env_offset = torch.arange(num_envs, device=points_heading.device)[:, None] * cell_count
    flat_index = (local_index + env_offset).reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_z = z.reshape(-1)
    total_cells = num_envs * cell_count

    z_max = torch.full((total_cells,), -torch.inf, dtype=z.dtype, device=z.device)
    z_min = torch.full((total_cells,), torch.inf, dtype=z.dtype, device=z.device)
    counts = torch.zeros((total_cells,), dtype=torch.int32, device=z.device)
    z_max.scatter_reduce_(0, flat_index[flat_valid], flat_z[flat_valid], reduce="amax", include_self=True)
    z_min.scatter_reduce_(0, flat_index[flat_valid], flat_z[flat_valid], reduce="amin", include_self=True)
    counts.scatter_add_(0, flat_index[flat_valid], torch.ones_like(flat_index[flat_valid], dtype=torch.int32))

    observed = counts >= cfg.minimum_points
    height = torch.where(observed, z_max, torch.zeros_like(z_max))
    roughness = torch.where(observed, z_max - z_min, torch.zeros_like(z_max))
    confidence = observed.to(dtype=z.dtype)
    grid = torch.stack((height, confidence, roughness), dim=-1)
    return grid.reshape(num_envs, cfg.rows, cfg.cols, 3)


def build_elevation_map(
    *,
    lidar_points_body: torch.Tensor,
    depth_points_body: torch.Tensor | None,
    body_quaternion_wxyz: torch.Tensor,
    cfg: ElevationMapCfg = DEFAULT_ELEVATION_MAP_CFG,
) -> torch.Tensor:
    fused = fuse_point_clouds(lidar_points_body, depth_points_body)
    aligned = gravity_align_heading(fused, body_quaternion_wxyz)
    return rasterize_elevation_map(aligned, cfg)
