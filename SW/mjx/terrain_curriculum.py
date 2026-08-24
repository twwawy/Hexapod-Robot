"""Canonical mixed-terrain difficulty schedule.

Each competence level owns one fixed geometry band and one reset-time patch
distribution.  A training stage samples one reproducible geometry from its
level band; vectorized environments then sample terrain lanes on reset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainLevel:
    # Total elevation gain across the complete staircase, not each riser.
    stair_total_rise_range_m: tuple[float, float]
    ramp_rise_range_m: tuple[float, float]
    yaw_limit_rps: float
    # flat, curb, ramp, blocks, stairs, rough
    patch_probabilities: tuple[float, float, float, float, float, float]


TERRAIN_LEVELS = (
    TerrainLevel(
        stair_total_rise_range_m=(0.02, 0.04),
        ramp_rise_range_m=(0.04, 0.08),
        yaw_limit_rps=0.00,
        # Roughness precedes stairs; flat/curb/ramp establish the transition.
        patch_probabilities=(0.40, 0.20, 0.15, 0.10, 0.00, 0.15),
    ),
    TerrainLevel(
        stair_total_rise_range_m=(0.04, 0.08),
        ramp_rise_range_m=(0.08, 0.12),
        yaw_limit_rps=0.05,
        patch_probabilities=(0.25, 0.15, 0.15, 0.20, 0.05, 0.20),
    ),
    TerrainLevel(
        stair_total_rise_range_m=(0.08, 0.12),
        ramp_rise_range_m=(0.12, 0.16),
        yaw_limit_rps=0.10,
        patch_probabilities=(0.20, 0.10, 0.15, 0.20, 0.15, 0.20),
    ),
    TerrainLevel(
        stair_total_rise_range_m=(0.12, 0.16),
        ramp_rise_range_m=(0.16, 0.20),
        yaw_limit_rps=0.20,
        patch_probabilities=(0.15, 0.10, 0.15, 0.20, 0.25, 0.15),
    ),
    TerrainLevel(
        stair_total_rise_range_m=(0.16, 0.20),
        ramp_rise_range_m=(0.20, 0.24),
        yaw_limit_rps=0.35,
        patch_probabilities=(0.10, 0.10, 0.15, 0.20, 0.30, 0.15),
    ),
)


def terrain_level(level: int) -> TerrainLevel:
    if not 0 <= level < len(TERRAIN_LEVELS):
        raise ValueError(f"terrain level must be 0..{len(TERRAIN_LEVELS) - 1}, got {level}")
    return TERRAIN_LEVELS[level]


for _level in TERRAIN_LEVELS:
    if abs(sum(_level.patch_probabilities) - 1.0) > 1e-9:
        raise ValueError("terrain patch probabilities must sum to one")
    if not 0.0 <= _level.stair_total_rise_range_m[0] <= _level.stair_total_rise_range_m[1] <= 0.20:
        raise ValueError("stair curriculum total rise must stay within 0..0.20 m")
