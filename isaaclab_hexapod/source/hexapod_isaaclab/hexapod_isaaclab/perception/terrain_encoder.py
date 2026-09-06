"""Small deployment-oriented CNN for the 32 x 24 x 3 terrain map."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def deterministic_terrain_features(elevation_map: torch.Tensor) -> torch.Tensor:
    """Compress the fused map to 64 policy features without frozen random weights.

    The PPO actor can learn from these stable inputs directly: 48 local-height
    samples plus eight confidence and eight roughness samples.
    """
    if elevation_map.ndim != 4 or elevation_map.shape[1:] != (32, 24, 3):
        raise ValueError("elevation map must have shape [N, 32, 24, 3]")
    channels = elevation_map.permute(0, 3, 1, 2).contiguous()
    height = F.adaptive_avg_pool2d(channels[:, 0:1], (8, 6)).flatten(1)
    confidence = F.adaptive_avg_pool2d(channels[:, 1:2], (4, 2)).flatten(1)
    roughness = F.adaptive_max_pool2d(channels[:, 2:3], (4, 2)).flatten(1)
    return torch.cat((height, confidence, roughness), dim=-1)


class TerrainEncoder(nn.Module):
    def __init__(self, latent_size: int = 64) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(nn.Linear(32 * 4 * 3, latent_size), nn.ELU())

    def forward(self, elevation_map: torch.Tensor) -> torch.Tensor:
        if elevation_map.ndim != 4 or elevation_map.shape[1:] != (32, 24, 3):
            raise ValueError("elevation map must have shape [N, 32, 24, 3]")
        channels_first = elevation_map.permute(0, 3, 1, 2).contiguous()
        return self.projection(self.features(channels_first))
