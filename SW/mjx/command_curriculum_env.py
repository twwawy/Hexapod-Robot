"""Flat walking-and-turning curriculum on the shared residual controller."""

from __future__ import annotations

from typing import Any, Optional

from ml_collections import config_dict

from rough_terrain_env import HexapodRoughTerrainEnv, default_config


class HexapodCommandCurriculumEnv(HexapodRoughTerrainEnv):
    """Train walk and yaw tracking as one randomized flat-ground curriculum.

    Difficulty expands from straight walking to limited and then full yaw,
    but an exact command is sampled every 1.5--4.0 seconds inside the active
    bounds.  There is no deterministic forward/turn command script for the
    policy to memorize.  The policy sees `[forward_speed, yaw_rate]` and
    always controls the same 22-D Cartesian/gait residual action.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[dict[str, Any]] = None,
        *,
        fixed_curriculum_stage: Optional[int] = None,
        scripted_commands: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            config_overrides=config_overrides,
            terrain="flat",
            command_curriculum=True,
            fixed_curriculum_stage=fixed_curriculum_stage,
            scripted_commands=scripted_commands,
        )
