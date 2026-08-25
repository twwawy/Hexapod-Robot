"""Fixed-shape terrain curriculum shared by scene, MJX env and launcher."""

from __future__ import annotations

from dataclasses import dataclass
import math


FLAT_GOAL_X = 1.00
TERRAIN_START_X = 0.55
TERRAIN_HALF_WIDTH = 0.60

ROUGH_COLUMNS = 8
ROUGH_ROWS = 4
ROUGH_TILE_DEPTH = 0.22
ROUGH_TILE_WIDTH = 2.0 * TERRAIN_HALF_WIDTH / ROUGH_ROWS
ROUGH_LENGTH = ROUGH_COLUMNS * ROUGH_TILE_DEPTH
ROUGH_END_X = TERRAIN_START_X + ROUGH_COLUMNS * ROUGH_TILE_DEPTH
# One heightfield replaces 32 individual collision boxes.  The repeated border
# samples keep the original vertical edge while the interior samples retain the
# deterministic roughness pattern.
ROUGH_HFIELD_NCOL = ROUGH_COLUMNS + 2
ROUGH_HFIELD_NROW = ROUGH_ROWS + 2

RAMP_LENGTH = 1.20
PLATEAU_DEPTH = 0.50
RAMP_END_X = TERRAIN_START_X + RAMP_LENGTH

STAIR_DEPTH = 0.25
MAX_STAIR_COUNT = 10


@dataclass(frozen=True)
class TerrainLevel:
    level: int
    name: str
    kind: str
    rough_amplitude: float = 0.0
    slope_degrees: float = 0.0
    stair_count: int = 0
    stair_riser: float = 0.0

    @property
    def final_height(self) -> float:
        if self.kind == "ramp":
            return RAMP_LENGTH * math.tan(math.radians(self.slope_degrees))
        if self.kind == "stairs":
            return self.stair_count * self.stair_riser
        return 0.0

    @property
    def goal_x(self) -> float:
        if self.kind == "flat":
            return FLAT_GOAL_X
        if self.kind == "rough":
            return ROUGH_END_X
        if self.kind == "ramp":
            return RAMP_END_X + 0.5 * PLATEAU_DEPTH
        if self.kind == "stairs":
            stair_end = TERRAIN_START_X + self.stair_count * STAIR_DEPTH
            return stair_end + 0.5 * PLATEAU_DEPTH
        raise ValueError(f"unknown terrain kind: {self.kind}")

    @property
    def requires_final_height(self) -> bool:
        return self.kind in {"ramp", "stairs"}

    @property
    def description(self) -> str:
        if self.kind == "flat":
            return "flat"
        if self.kind == "rough":
            return f"rough amplitude {100.0 * self.rough_amplitude:.1f}cm"
        if self.kind == "ramp":
            return f"{self.slope_degrees:.0f}deg ramp"
        return (
            f"{self.stair_count} stairs x {100.0 * self.stair_riser:.2f}cm "
            f"= {100.0 * self.final_height:.1f}cm"
        )


def _seven_step_total(level: int, total_height: float) -> TerrainLevel:
    return TerrainLevel(
        level,
        f"stairs_total_{int(round(100 * total_height))}cm",
        "stairs",
        stair_count=7,
        stair_riser=total_height / 7.0,
    )


TERRAIN_LEVELS = (
    TerrainLevel(0, "flat", "flat"),
    TerrainLevel(1, "rough", "rough", rough_amplitude=0.025),
    TerrainLevel(2, "rough_hard", "rough", rough_amplitude=0.050),
    TerrainLevel(3, "ramp", "ramp", slope_degrees=8.0),
    TerrainLevel(4, "ramp_steep", "ramp", slope_degrees=15.0),
    _seven_step_total(5, 0.05),
    _seven_step_total(6, 0.10),
    _seven_step_total(7, 0.15),
    _seven_step_total(8, 0.20),
    TerrainLevel(9, "stairs_riser_5cm", "stairs", stair_count=10, stair_riser=0.05),
    TerrainLevel(10, "stairs_riser_10cm", "stairs", stair_count=10, stair_riser=0.10),
    TerrainLevel(11, "stairs_riser_15cm", "stairs", stair_count=10, stair_riser=0.15),
    TerrainLevel(12, "stairs_riser_20cm", "stairs", stair_count=10, stair_riser=0.20),
)
MAX_TERRAIN_LEVEL = len(TERRAIN_LEVELS) - 1


def terrain_level(level: int) -> TerrainLevel:
    if level not in range(len(TERRAIN_LEVELS)):
        raise ValueError(f"terrain level must be 0..{MAX_TERRAIN_LEVEL}")
    spec = TERRAIN_LEVELS[level]
    if spec.level != level:
        raise RuntimeError("terrain level table is not contiguous")
    return spec


def rough_tile_centers() -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            TERRAIN_START_X + (column + 0.5) * ROUGH_TILE_DEPTH,
            -TERRAIN_HALF_WIDTH + (row + 0.5) * ROUGH_TILE_WIDTH,
        )
        for column in range(ROUGH_COLUMNS)
        for row in range(ROUGH_ROWS)
    )


def rough_tile_heights(amplitude: float) -> tuple[float, ...]:
    """Deterministic non-periodic-looking positive bumps over the ground plane."""
    return tuple(
        amplitude * (1.0 + ((column * 5 + row * 3) % 7)) / 7.0
        for column in range(ROUGH_COLUMNS)
        for row in range(ROUGH_ROWS)
    )


def rough_heightfield_grid(amplitude: float) -> tuple[tuple[float, ...], ...]:
    """Return a row-major rough heightfield with repeated boundary samples."""
    heights = rough_tile_heights(amplitude)
    interior = tuple(
        tuple(heights[column * ROUGH_ROWS + row] for column in range(ROUGH_COLUMNS))
        for row in range(ROUGH_ROWS)
    )
    bordered = tuple((row[0], *row, row[-1]) for row in interior)
    return (bordered[0], *bordered, bordered[-1])
