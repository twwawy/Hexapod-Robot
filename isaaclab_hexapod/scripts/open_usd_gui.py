#!/usr/bin/env python3
"""Open one USD in Isaac Sim and frame the Hexapod in the viewport."""

from __future__ import annotations

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("usd", type=Path)
args = parser.parse_args()
usd_path = args.usd.expanduser().resolve()
if not usd_path.is_file():
    raise FileNotFoundError(usd_path)

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1440,
        "height": 900,
        "sync_loads": True,
    }
)

import omni.usd

from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.core.utils.viewports import set_camera_view


try:
    # The GUI creates its empty startup stage after SimulationApp processes the
    # open_usd config.  Open explicitly only after the full app is ready.
    if not open_stage(str(usd_path)):
        raise RuntimeError(f"Isaac Sim failed to open USD: {usd_path}")
    while is_stage_loading():
        simulation_app.update()
    for _ in range(10):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    root_prims = list(stage.GetPseudoRoot().GetChildren())
    if not root_prims:
        raise RuntimeError(f"USD opened as an empty stage: {usd_path}")

    # The CAD is centered around the origin and is less than one metre wide.
    set_camera_view(
        eye=(1.15, -1.15, 0.75),
        target=(0.0, 0.0, 0.05),
        camera_prim_path="/OmniverseKit_Persp",
    )
    print(
        f"OPENED_USD={usd_path} ROOT_PRIMS="
        f"{','.join(str(prim.GetPath()) for prim in root_prims)}",
        flush=True,
    )
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
