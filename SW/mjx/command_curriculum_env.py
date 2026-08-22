"""Flat walking-and-turning curriculum on the shared residual controller."""

from __future__ import annotations

from typing import Any, Optional

from ml_collections import config_dict

from rough_terrain_env import HexapodRoughTerrainEnv, default_config


class HexapodCommandCurriculumEnv(HexapodRoughTerrainEnv):
    """Train walk and yaw tracking as one ordered flat-ground curriculum.

    A 1,000-step episode progresses through straight walking, limited yaw,
    then the full walking-and-turning command range.  The policy still sees
    the current `[forward_speed, yaw_rate]` command and always controls the
    same 22-D Cartesian/gait residual action.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            config=config,
            config_overrides=config_overrides,
            terrain="flat",
            command_curriculum=True,
        )
