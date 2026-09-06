#!/usr/bin/env python3
"""Create the robot-only Isaac MJCF with full CAD visuals and RL collisions.

The MJX training scene intentionally strips the CAD mesh geoms for fast
batched collision.  Isaac Sim should still show the real robot, so this export
starts from the unstripped CAD scene, makes every mesh visual-only, and adds
the same primitive colliders used by MJX.  This keeps rendering fidelity from
changing the learned contact/dynamics contract.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from prepare_rl_scene import (
    TARGET_ROBOT_MASS_KG,
    _add_robot_colliders,
    _replace_inertials,
    _robot_body_ids,
)
from prepare_scene import SCENE_OUTPUT, prepare_scene


OUTPUT = (
    Path(__file__).resolve().parent
    / "generated/hexapod_isaac_full_mesh_asset.xml"
)


def _make_cad_visual_only(root: ET.Element) -> int:
    count = 0
    for geom in root.iter("geom"):
        if geom.get("type") != "mesh":
            continue
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")
        count += 1
    return count


def main() -> None:
    source = prepare_scene(SCENE_OUTPUT)
    source_model = mujoco.MjModel.from_xml_path(str(source))
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "hexapod_mjx_full_cad_parity_asset")
    option = root.find("option")
    if option is not None:
        option.set("timestep", "0.0025")

    _replace_inertials(root, source_model)
    mesh_geom_count = _make_cad_visual_only(root)
    if mesh_geom_count == 0:
        raise RuntimeError("full-mesh Isaac export found no CAD mesh geoms")
    _add_robot_colliders(root, source_model)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no worldbody")
    robot = worldbody.find("body[@name='hexapod']")
    if robot is None:
        raise RuntimeError("MJCF has no hexapod root body")
    for child in list(worldbody):
        if child is not robot:
            worldbody.remove(child)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    robot_mass = sum(
        float(model.body_mass[body_id]) for body_id in _robot_body_ids(model)
    )
    if not np.isclose(robot_mass, TARGET_ROBOT_MASS_KG, atol=1.0e-6):
        raise RuntimeError(
            f"Isaac asset mass is {robot_mass:.9g} kg, expected "
            f"{TARGET_ROBOT_MASS_KG:.9g} kg"
        )
    active_collision_count = sum(
        int(model.geom_contype[index] != 0 or model.geom_conaffinity[index] != 0)
        for index in range(model.ngeom)
    )
    print(OUTPUT.resolve())
    print(
        f"CAD_VISUAL_MESH_GEOMS={mesh_geom_count} "
        f"ACTIVE_PRIMITIVE_COLLIDERS={active_collision_count} "
        f"ROBOT_MASS_KG={robot_mass:.6f}"
    )


if __name__ == "__main__":
    main()
