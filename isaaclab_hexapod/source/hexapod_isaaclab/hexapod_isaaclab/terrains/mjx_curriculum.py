"""GPU-efficient Isaac Lab terrain curriculum matching MJX levels 0 through 9."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass


MAX_TRAINING_TERRAIN_LEVEL = 9
TERRAIN_START_DISTANCE_M = 0.55
TERRAIN_GOAL_DISTANCE_M = (
    1.00,
    2.31,
    2.31,
    2.00,
    2.00,
    2.55,
    2.55,
    2.55,
    2.55,
    2.55,
)

_ROUGH_AMPLITUDES = {1: 0.025, 2: 0.050}
_RAMP_ANGLES_DEG = {3: 8.0, 4: 15.0}
_STAIR_RISERS_M = {5: 0.050, 6: 0.065, 7: 0.080, 8: 0.100, 9: 0.150}


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def mjx_curriculum_terrain(
    difficulty: float, cfg: "MjxCurriculumSubTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the discrete MJX terrain selected by the curriculum row."""

    level = min(int(difficulty * (MAX_TRAINING_TERRAIN_LEVEL + 1)), MAX_TRAINING_TERRAIN_LEVEL)
    size_x, size_y = cfg.size
    # The CAD's body -Y axis becomes world +X under HOME_ROOT_QUAT_WXYZ.
    # Build every obstacle in that actual world-forward direction.
    origin = np.array((0.65, 0.5 * size_y, 0.0), dtype=np.float64)
    start_x = origin[0] + TERRAIN_START_DISTANCE_M
    corridor_width = min(1.20, size_y - 0.20)
    meshes = [_box((size_x, size_y, 0.10), (0.5 * size_x, 0.5 * size_y, -0.05))]

    if level in _ROUGH_AMPLITUDES:
        amplitude = _ROUGH_AMPLITUDES[level]
        columns, rows = 8, 4
        tile_depth = 0.22
        tile_width = corridor_width / rows
        for column in range(columns):
            for row in range(rows):
                height = amplitude * (1.0 + ((column * 5 + row * 3) % 7)) / 7.0
                center_x = start_x + (column + 0.5) * tile_depth
                center_y = 0.5 * size_y - 0.5 * corridor_width + (row + 0.5) * tile_width
                meshes.append(_box((tile_depth, tile_width, height), (center_x, center_y, 0.5 * height)))
    elif level in _RAMP_ANGLES_DEG:
        angle = math.radians(_RAMP_ANGLES_DEG[level])
        length = 1.20
        rise = length * math.tan(angle)
        ramp = trimesh.creation.box(extents=(length, corridor_width, 0.10))
        transform = trimesh.transformations.euler_matrix(0.0, -angle, 0.0, axes="sxyz")
        transform[:3, 3] = (start_x + 0.5 * length, 0.5 * size_y, 0.5 * rise - 0.05)
        ramp.apply_transform(transform)
        meshes.append(ramp)
        plateau_length = max(size_x - start_x - length, 0.20)
        meshes.append(
            _box(
                (plateau_length, corridor_width, rise),
                (start_x + length + 0.5 * plateau_length, 0.5 * size_y, 0.5 * rise),
            )
        )
    elif level in _STAIR_RISERS_M:
        riser = _STAIR_RISERS_M[level]
        step_depth = 0.25
        stair_count = 7
        for step in range(stair_count):
            height = (step + 1) * riser
            center_x = start_x + (step + 0.5) * step_depth
            meshes.append(
                _box((step_depth, corridor_width, height), (center_x, 0.5 * size_y, 0.5 * height))
            )
        plateau_start_x = start_x + stair_count * step_depth
        plateau_length = max(size_x - plateau_start_x, 0.20)
        final_height = stair_count * riser
        meshes.append(
            _box(
                (plateau_length, corridor_width, final_height),
                (plateau_start_x + 0.5 * plateau_length, 0.5 * size_y, 0.5 * final_height),
            )
        )

    return meshes, origin


@configclass
class MjxCurriculumSubTerrainCfg(SubTerrainBaseCfg):
    function: Callable = mjx_curriculum_terrain


def mjx_curriculum_terrain_cfg() -> TerrainImporterCfg:
    generator = TerrainGeneratorCfg(
        seed=0,
        curriculum=True,
        size=(4.0, 3.0),
        border_width=1.0,
        num_rows=MAX_TRAINING_TERRAIN_LEVEL + 1,
        num_cols=1,
        color_scheme="height",
        difficulty_range=(0.0, 1.0),
        use_cache=True,
        sub_terrains={"mjx_levels_0_to_9": MjxCurriculumSubTerrainCfg(proportion=1.0)},
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )
