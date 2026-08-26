#!/usr/bin/env python3
"""Create a robot-only MJCF while preserving final RL mass and primitives."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_rl_scene import RL_SCENE_OUTPUT, prepare_rl_scene


OUTPUT = Path(__file__).resolve().parent / "generated/hexapod_isaac_asset.xml"


def main() -> None:
    source = prepare_rl_scene(RL_SCENE_OUTPUT)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "hexapod_mjx_parity_asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no worldbody")
    robot = worldbody.find("body[@name='hexapod']")
    if robot is None:
        raise RuntimeError("MJCF has no hexapod root body")
    for child in list(worldbody):
        if child is not robot:
            worldbody.remove(child)
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "hfield":
                asset.remove(child)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
