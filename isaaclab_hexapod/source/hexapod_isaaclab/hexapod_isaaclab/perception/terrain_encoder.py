"""Small deployment-oriented CNN for the 32 x 24 x 3 terrain map."""

from __future__ import annotations

import torch
from torch import nn


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
