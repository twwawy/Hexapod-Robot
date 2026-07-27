# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight MuJoCo bootstrap for standalone Gym environment registration."""

from importlib import import_module

__all__ = ["bootstrap"]


def bootstrap() -> None:
    """Import the MuJoCo direct environment package to trigger Gym registration."""
    import_module(".hexapedal_direct", __name__)


bootstrap()
