"""Sensor-only geometric foothold planning, independent of MuJoCo and RL.

Points and targets use a gravity-aligned odom frame, in metres. The local grid
follows translation on cell boundaries while keeping world-aligned history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MapConfig:
    half_extent: float = 4.0
    resolution: float = 0.04
    max_age: float = 60.0

    def __post_init__(self):
        if min(self.half_extent, self.resolution, self.max_age) <= 0:
            raise ValueError('Map extent, resolution and age must be positive')


class ElevationMap:
    """Keep observed cell height/range and timestamps; never fill unknowns."""

    def __init__(self, cfg: MapConfig = MapConfig()):
        self.cfg = cfg
        self.n = int(np.ceil(2 * cfg.half_extent / cfg.resolution))
        shape = (self.n, self.n)
        self.height = np.full(shape, np.nan)
        self.vertical_range = np.full(shape, np.nan)
        self.timestamp = np.full(shape, -np.inf)
        self.version = 0
        self.center = np.zeros(2)

    def recenter(self, xy):
        """Roll by whole cells, retain overlapping odom cells, clear new strips."""
        shift = np.rint((np.asarray(xy) - self.center) / self.cfg.resolution).astype(int)
        if not shift.any():
            return
        self.center += shift * self.cfg.resolution
        for name, fill in (('height', np.nan), ('vertical_range', np.nan), ('timestamp', -np.inf)):
            values = getattr(self, name)
            if np.any(np.abs(shift) >= self.n):
                values.fill(fill)
            else:
                values = np.roll(values, tuple(-shift), axis=(0, 1))
                for axis, amount in enumerate(shift):
                    if amount:
                        region = [slice(None), slice(None)]
                        region[axis] = slice(-amount, None) if amount > 0 else slice(None, -amount)
                        values[tuple(region)] = fill
                setattr(self, name, values)
        self.version += 1

    def indices(self, xy):
        xy = np.asarray(xy, dtype=float)
        finite = np.isfinite(xy).all(axis=-1)
        index = np.floor((np.where(np.isfinite(xy), xy, 0) - self.center + self.cfg.half_extent)
                         / self.cfg.resolution).astype(int)
        valid = finite & (index >= 0).all(axis=-1) & (index < self.n).all(axis=-1)
        return np.clip(index, 0, self.n - 1), valid

    def centers(self):
        axis = (np.arange(self.n) + 0.5) * self.cfg.resolution - self.cfg.half_extent
        return np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1) + self.center

    def valid(self, now):
        age = now - self.timestamp
        return np.isfinite(self.height) & (age >= 0) & (age <= self.cfg.max_age)

    def update(self, points_odom, timestamp):
        points = np.asarray(points_odom, dtype=float).reshape(-1, 3)
        index, inside = self.indices(points[:, :2])
        inside &= np.isfinite(points).all(axis=1)
        index, points = index[inside], points[inside]
        if not len(points):
            return
        flat = index[:, 0] * self.n + index[:, 1]
        zmax = np.full(self.n * self.n, -np.inf)
        zmin = np.full(self.n * self.n, np.inf)
        np.maximum.at(zmax, flat, points[:, 2])
        np.minimum.at(zmin, flat, points[:, 2])
        observed = np.isfinite(zmax)
        # Overwrite with the latest scan, rather than retaining old obstacles forever.
        self.height.flat[observed] = zmax[observed]
        self.vertical_range.flat[observed] = (zmax - zmin)[observed]
        self.timestamp.flat[observed] = timestamp
        self.version += 1

    def sample(self, xy, now):
        index, inside = self.indices(xy)
        row, col = index[..., 0], index[..., 1]
        return self.height[row, col], inside & self.valid(now)[row, col]


@dataclass(frozen=True)
class PlannerConfig:
    search_radius: float = 0.14
    support_radius: float = 0.032
    max_patch_height_range: float = 0.025
    max_vertical_range: float = 0.015
    max_slope_deg: float = 20.0
    max_touchdown_height_change: float = 0.10
    min_clearance: float = 0.06
    max_clearance: float = 0.20
    max_ik_candidates: int = 6

    def __post_init__(self):
        if min(self.search_radius, self.support_radius, self.min_clearance) <= 0:
            raise ValueError('Planner distances must be positive')
        if self.max_clearance < self.min_clearance or self.max_ik_candidates < 1:
            raise ValueError('Invalid clearance or candidate budget')


@dataclass
class Plan:
    nominal: np.ndarray
    candidates: np.ndarray
    accepted: np.ndarray
    reasons: list[str]
    selected: np.ndarray | None
    path: np.ndarray | None
    status: str
    map_version: int
    mode: str = 'hold'
    geometric_candidate: np.ndarray | None = None


def footprint_offsets(resolution, radius):
    """Cells intersecting the support disk, instead of an oversized square."""
    reach = int(np.ceil(radius / resolution))
    offsets = np.array([(i, j) for i in range(-reach, reach + 1)
                        for j in range(-reach, reach + 1)])
    distance_to_cell = np.maximum(np.abs(offsets) * resolution - resolution / 2, 0)
    return offsets[np.linalg.norm(distance_to_cell, axis=1) <= radius]


def swing_path(start, end, clearance, samples=41):
    """Lift, transfer, lower with zero endpoint velocity and acceleration.

    Clearance is measured above the higher endpoint, independently of touchdown Z.
    This deliberately simple inspection trajectory is not the firmware gait.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    phase = np.linspace(0, 1, samples)
    smooth = lambda x: x**3 * (10 + x * (-15 + 6 * x))
    top = max(start[2], end[2]) + clearance
    path = np.tile(start, (samples, 1))
    xy_progress = smooth(np.clip((phase - 0.25) / 0.5, 0, 1))
    path[:, :2] += xy_progress[:, None] * (end[:2] - start[:2])
    up = smooth(np.clip(phase / 0.25, 0, 1))
    down = smooth(np.clip((phase - 0.75) / 0.25, 0, 1))
    path[:, 2] = start[2] + up * (top - start[2]) + down * (end[2] - top)
    return path


def plan_foothold(grid: ElevationMap, start, nominal, now: float,
                  reachable: Callable[[np.ndarray], bool],
                  cfg: PlannerConfig = PlannerConfig(), allow_unknown_path=False) -> Plan:
    """Rank geometric patches, then validate sampled paths with caller's IK.

    No simulator geometry or ground-truth height is accessible from this module.
    Candidate budget exhaustion returns failure, never an unchecked target.
    """
    start, nominal = np.asarray(start, float), np.asarray(nominal, float)
    xy = grid.centers()
    cells = np.argwhere(np.linalg.norm(xy - nominal[:2], axis=-1) <= cfg.search_radius)
    candidates, accepted, reasons, costs = [], [], [], []
    valid = grid.valid(now)
    offsets = footprint_offsets(grid.cfg.resolution, cfg.support_radius)
    fit_xy = offsets * grid.cfg.resolution
    fit = np.column_stack((fit_xy, np.ones(len(offsets))))
    fit_inverse = np.linalg.pinv(fit)
    for row, col in cells:
        z = grid.height[row, col]
        candidate = np.r_[xy[row, col], z if np.isfinite(z) else nominal[2]]
        patch = offsets + (row, col)
        reason, slope, spread = 'valid_patch', 0.0, 0.0
        if (patch < 0).any() or (patch >= grid.n).any():
            reason = 'map_boundary'
        else:
            r, c = patch.T
            if not valid[r, c].all():
                reason = 'unknown_or_stale'
            else:
                heights = grid.height[r, c]
                spread = float(np.ptp(heights))
                slope = float(np.linalg.norm((fit_inverse @ heights)[:2]))
                if np.max(grid.vertical_range[r, c]) > cfg.max_vertical_range:
                    reason = 'vertical_surface'
                elif spread > cfg.max_patch_height_range:
                    reason = 'edge_or_step'
                elif slope > np.tan(np.deg2rad(cfg.max_slope_deg)):
                    reason = 'slope'
                elif abs(z - start[2]) > cfg.max_touchdown_height_change:
                    reason = 'height_change'
        candidates.append(candidate)
        accepted.append(reason == 'valid_patch')
        reasons.append(reason)
        costs.append(np.linalg.norm(candidate[:2] - nominal[:2]) + 2 * spread + 0.04 * slope)
    candidates = np.asarray(candidates).reshape(-1, 3)
    accepted = np.asarray(accepted, dtype=bool)
    selected, path, status = None, None, 'no_valid_candidate'
    first_candidate = None
    order = np.argsort(costs)
    tried = 0
    for index in order:
        if not accepted[index]:
            continue
        if tried >= cfg.max_ik_candidates:
            status = 'candidate_budget_exhausted'
            break
        tried += 1
        end = candidates[index]
        if not reachable(end):
            accepted[index], reasons[index] = False, 'endpoint_ik'
            continue
        selected = end.copy()
        if first_candidate is None:
            first_candidate = selected.copy()
        trial = swing_path(start, end, cfg.min_clearance)
        # Inspect the full swept foot footprint during horizontal transfer.
        transfer = trial[10:31, :2]
        sweep_xy = transfer[:, None, :] + fit_xy[None, :, :]
        heights, observed = grid.sample(sweep_xy, now)
        if not observed.all() and not allow_unknown_path:
            reasons[index], status = 'path_unknown_or_stale', 'path_unknown_or_stale'
            continue
        obstacle_top = float(np.max(heights[observed])) if observed.any() else max(start[2], end[2])
        clearance = max(cfg.min_clearance,
                        obstacle_top - max(start[2], end[2]) + cfg.min_clearance)
        if clearance > cfg.max_clearance:
            reasons[index], status = 'clearance_limit', 'clearance_limit'
            continue
        trial = swing_path(start, end, clearance)
        if not all(reachable(point) for point in trial):
            reasons[index], status = 'path_ik', 'path_ik'
            continue
        path = trial
        status = 'ready' if observed.all() else 'endpoint_observed_path_partial'
        break
    if path is None:
        selected = first_candidate
    mode = ('geometric' if status == 'ready' else 'geometric_partial') if path is not None else 'hold'
    return Plan(nominal.copy(), candidates, accepted, reasons, selected, path,
                status, grid.version, mode, selected.copy() if selected is not None else None)


def plan_with_nominal_fallback(grid, start, nominal, now, reachable,
                               cfg=PlannerConfig(), allow_unknown=True):
    """Unknown terrain may use nominal motion; observed hazards still veto it.

    This preview supplies a lift/transfer/lower nominal path. A walking adapter
    must supply its existing controller trajectory at the same handoff boundary.
    The returned nominal mode explicitly does NOT mean terrain-verified safe.
    """
    plan = plan_foothold(grid, start, nominal, now, reachable, cfg, allow_unknown_path=allow_unknown)
    if plan.path is not None or not allow_unknown:
        return plan
    nominal_path = swing_path(start, nominal, cfg.min_clearance)
    offsets = footprint_offsets(grid.cfg.resolution, cfg.support_radius) * grid.cfg.resolution
    heights, observed = grid.sample(nominal_path[:, None, :2] + offsets[None], now)
    if observed.all():
        return plan  # Known terrain failure is not a reason to enable blind mode.
    end_heights, end_observed = heights[-1], observed[-1]
    known_collision = np.any(observed & (heights > nominal_path[:, None, 2] + 0.015))
    known_endpoint_mismatch = np.any(end_observed & (np.abs(end_heights - nominal[2]) > 0.015))
    if known_collision or known_endpoint_mismatch:
        plan.selected, plan.path, plan.mode = None, None, 'hold'
        plan.status = 'hold_known_hazard'
        return plan
    if not all(reachable(point) for point in nominal_path):
        plan.selected, plan.path, plan.mode = None, None, 'hold'
        plan.status = 'hold_nominal_ik'
        return plan
    plan.selected = np.asarray(nominal, dtype=float).copy()
    plan.path = nominal_path
    plan.status = 'nominal_unknown_terrain'
    plan.mode = 'nominal'
    return plan
