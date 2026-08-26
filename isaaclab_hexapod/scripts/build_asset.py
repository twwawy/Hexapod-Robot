#!/usr/bin/env python3
"""Convert the parity MJCF to USD and keep one floating articulation root."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

from pxr import Usd, UsdPhysics

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


def main() -> None:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(input_path),
            usd_dir=str(output_path.parent),
            usd_file_name=output_path.name,
            fix_base=False,
            import_sites=True,
            force_usd_conversion=True,
            make_instanceable=False,
        )
    )

    stage = Usd.Stage.Open(converter.usd_path)
    roots = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    robot_roots = [prim for prim in roots if prim.GetName() == "hexapod"]
    if len(robot_roots) != 1:
        paths = [str(prim.GetPath()) for prim in roots]
        raise RuntimeError(f"expected one hexapod articulation, found {paths}")
    for prim in roots:
        if prim != robot_roots[0]:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    stage.GetRootLayer().Save()

    verified = Usd.Stage.Open(converter.usd_path)
    final_roots = [
        str(prim.GetPath())
        for prim in verified.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(final_roots) != 1:
        raise RuntimeError(f"USD articulation normalization failed: {final_roots}")
    print(f"Generated {converter.usd_path}")
    print(f"Articulation root: {final_roots[0]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
