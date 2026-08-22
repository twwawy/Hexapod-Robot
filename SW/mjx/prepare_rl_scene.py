#!/usr/bin/env python3
"""Build mesh-free primitive-collision scenes used for MJX training."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from prepare_scene import SCENE_OUTPUT, prepare_scene
from tripod_controller import LEG_PREFIXES


RL_SCENE_OUTPUT = Path(__file__).resolve().parent / "generated/hexapod_rl.xml"
FLAT_RL_SCENE_OUTPUT = Path(__file__).resolve().parent / "generated/hexapod_flat_rl.xml"
STEP_START_X = 0.55
STEP_DEPTH = 0.25
STEP_HEIGHT = 0.05
STEP_COUNT = 7


def _numbers(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _replace_inertials(root: ET.Element, model) -> None:
    import mujoco

    for body in root.iter("body"):
        name = body.get("name")
        if not name:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        old = body.find("inertial")
        if old is not None:
            body.remove(old)
        inertial = ET.Element(
            "inertial",
            {
                "pos": _numbers(model.body_ipos[body_id]),
                "quat": _numbers(model.body_iquat[body_id]),
                "mass": f"{float(model.body_mass[body_id]):.9g}",
                "diaginertia": _numbers(model.body_inertia[body_id]),
            },
        )
        body.insert(0, inertial)


def _strip_cad_meshes(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("type") == "mesh":
                body.remove(geom)


def _add_robot_colliders(root: ET.Element, *, foot_friction: float = 1.2) -> None:
    robot = root.find("./worldbody/body[@name='hexapod']")
    if robot is None:
        raise ValueError("Missing hexapod body")
    robot.append(
        ET.Element(
            "geom",
            {
                "name": "torso_collision",
                "type": "box",
                "size": "0.17 0.15 0.045",
                "rgba": "0.22 0.30 0.38 1",
                "friction": "0.8 0.005 0.0001",
            },
        )
    )
    robot.append(
        ET.Element(
            "site",
            {"name": "imu", "pos": "0 0 0", "size": "0.01"},
        )
    )

    for prefix in LEG_PREFIXES:
        body1 = root.find(f".//body[@name='{prefix}_motor_horn_1_1']")
        body2 = root.find(f".//body[@name='{prefix}_DS51150_270_2_1']")
        body3 = root.find(f".//body[@name='{prefix}_motor_horn_3_1']")
        if body1 is None or body2 is None or body3 is None:
            raise ValueError(f"Missing articulated bodies for {prefix}")

        segment1 = np.fromstring(body2.get("pos", ""), sep=" ")
        segment2 = np.fromstring(body3.get("pos", ""), sep=" ")
        origin = np.fromstring(body1.get("pos", ""), sep=" ")
        outward = origin.copy()
        outward[2] = 0.0
        outward /= np.linalg.norm(outward)
        segment3 = outward * 0.230

        body1.append(
            ET.Element(
                "geom",
                {
                    "name": f"{prefix}_coxa_collision",
                    "type": "capsule",
                    "fromto": f"0 0 0 {_numbers(segment1)}",
                    "size": "0.028",
                    "rgba": "0.30 0.38 0.45 1",
                },
            )
        )
        body2.append(
            ET.Element(
                "geom",
                {
                    "name": f"{prefix}_femur_collision",
                    "type": "capsule",
                    "fromto": f"0 0 0 {_numbers(segment2)}",
                    "size": "0.026",
                    "rgba": "0.34 0.43 0.52 1",
                },
            )
        )
        body3.append(
            ET.Element(
                "geom",
                {
                    "name": f"{prefix}_tibia_collision",
                    "type": "capsule",
                    "fromto": f"0 0 0 {_numbers(segment3)}",
                    "size": "0.023",
                    "rgba": "0.40 0.50 0.60 1",
                },
            )
        )
        body3.append(
            ET.Element(
                "geom",
                {
                    "name": f"{prefix}_foot_collision",
                    "type": "sphere",
                    "pos": _numbers(segment3),
                    "size": "0.032",
                    "friction": f"{foot_friction:.6g} 0.01 0.001",
                    "rgba": "0.85 0.45 0.12 1",
                },
            )
        )
        body3.append(
            ET.Element(
                "site",
                {
                    "name": f"{prefix}_foot_site",
                    "pos": _numbers(segment3),
                    "size": "0.012",
                    "rgba": "1 0.2 0.1 1",
                },
            )
        )


def _add_stairs(
    root: ET.Element,
    *,
    step_start_x: float = STEP_START_X,
    step_depth: float = STEP_DEPTH,
    step_height: float = STEP_HEIGHT,
    step_count: int = STEP_COUNT,
    friction: float = 1.1,
) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Missing worldbody")
    for index in range(step_count):
        height = step_height * (index + 1)
        center_x = step_start_x + step_depth * index
        worldbody.insert(
            2 + index,
            ET.Element(
                "geom",
                {
                    "name": f"stair_{index + 1}",
                    "type": "box",
                    "pos": f"{center_x:.6g} 0 {height / 2:.6g}",
                    "size": f"{step_depth / 2:.6g} 1.0 {height / 2:.6g}",
                    "friction": f"{friction:.6g} 0.01 0.001",
                    "rgba": "0.34 0.42 0.50 1",
                },
            ),
        )


def prepare_rl_scene(
    output: Path = RL_SCENE_OUTPUT,
    *,
    terrain: str = "stairs",
    step_start_x: float = STEP_START_X,
    step_depth: float = STEP_DEPTH,
    step_height: float = STEP_HEIGHT,
    step_count: int = STEP_COUNT,
    friction: float = 1.0,
) -> Path:
    """Create a mesh-free RL scene for either ``flat`` or ``stairs`` terrain.

    Both variants use exactly the same robot collision geometry and position
    actuators.  Keeping that contract makes the flat command curriculum and
    terrain policy directly comparable while avoiding CAD mesh contact in MJX.
    """
    import mujoco

    if terrain not in {"flat", "stairs"}:
        raise ValueError(f"terrain must be 'flat' or 'stairs', got {terrain!r}")

    prepare_scene(SCENE_OUTPUT)
    source_model = mujoco.MjModel.from_xml_path(str(SCENE_OUTPUT))
    tree = ET.parse(SCENE_OUTPUT)
    root = tree.getroot()
    root.set("model", f"hexapod_{terrain}_rl")
    option = root.find("option")
    if option is not None:
        option.set("timestep", "0.0025")

    if step_depth <= 0.0 or step_height < 0.0 or step_count < 0:
        raise ValueError("terrain dimensions must satisfy depth > 0, height >= 0, count >= 0")
    if friction <= 0.0:
        raise ValueError("terrain friction must be positive")

    _replace_inertials(root, source_model)
    _strip_cad_meshes(root)
    _add_robot_colliders(root, foot_friction=1.2 * friction)
    for geom in root.findall("./worldbody/geom"):
        if geom.get("type") == "plane":
            geom.set("friction", f"{friction:.6g} 0.01 0.001")
    if terrain == "stairs":
        _add_stairs(
            root,
            step_start_x=step_start_x,
            step_depth=step_depth,
            step_height=step_height,
            step_count=step_count,
            friction=1.1 * friction,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    if model.ngeom >= 60:
        raise ValueError(f"RL collision model is unexpectedly large: {model.ngeom}")
    return output


def prepare_flat_rl_scene(
    output: Path = FLAT_RL_SCENE_OUTPUT, *, friction: float = 1.0
) -> Path:
    """Create the mesh-free plane scene for walking-and-turning curriculum."""
    return prepare_rl_scene(output, terrain="flat", friction=friction)


def main() -> None:
    path = prepare_rl_scene()
    print(path.resolve())


if __name__ == "__main__":
    main()
