"""Exact legacy 15-point height scanner pattern."""

from __future__ import annotations

from collections.abc import Callable

import torch

from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


FORWARD_SAMPLES = (-0.10, 0.15, 0.40, 0.65, 0.90)
LATERAL_SAMPLES = (-0.24, 0.00, 0.24)


def hexapod_15_point_pattern(
    cfg: "Hexapod15PointPatternCfg", device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    forward = torch.tensor(FORWARD_SAMPLES, dtype=torch.float32, device=device).repeat(3)
    lateral = torch.tensor(LATERAL_SAMPLES, dtype=torch.float32, device=device).repeat_interleave(5)
    starts = torch.stack(
        (lateral, -forward, torch.full_like(forward, cfg.start_height)), dim=-1
    )
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    return starts, directions


@configclass
class Hexapod15PointPatternCfg(PatternBaseCfg):
    func: Callable = hexapod_15_point_pattern
    start_height: float = 1.0
