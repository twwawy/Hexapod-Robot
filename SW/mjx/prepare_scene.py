#!/usr/bin/env python3
"""Build a MuJoCo scene with a floor and the documented standing pose."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_urdf import DEFAULT_OUTPUT as URDF_OUTPUT
from prepare_urdf import SOURCE_URDF, prepare_urdf
from urdf_kinematics import STAND_ROOT_HEIGHT


SCENE_OUTPUT = Path(__file__).resolve().parent / "generated/hexapod_scene.xml"
LEG_PREFIXES = ("RB", "RM", "RF", "LB", "LM", "LF")
RIGHT_LEGS = frozenset(("RB", "RM", "RF"))
STAND_HEIGHT = STAND_ROOT_HEIGHT
ROOT_QUATERNION = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
HIP_ANGLE = math.radians(0.0)
KNEE_ANGLE = math.radians(30.0)
ANKLE_ANGLE = math.radians(50.0)


def _standing_angle(prefix: str, joint_number: int) -> float:
    """Map documented servo angles to the mirrored raw URDF joint axes."""
    side_sign = -1.0 if prefix in RIGHT_LEGS else 1.0
    return {
        1: HIP_ANGLE,
        2: side_sign * KNEE_ANGLE,
        3: -side_sign * ANKLE_ANGLE,
    }[joint_number]


def _add_scene_elements(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("autolimits", "true")

    option = ET.Element(
        "option",
        {
            "timestep": "0.002",
            "integrator": "implicitfast",
            "gravity": "0 0 -9.81",
        },
    )
    root.insert(1, option)

    asset = root.find("asset")
    if asset is None:
        raise ValueError("Converted model has no asset section")
    asset.insert(
        0,
        ET.Element(
            "texture",
            {
                "name": "ground_texture",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.18 0.22 0.26",
                "rgb2": "0.08 0.10 0.12",
                "width": "512",
                "height": "512",
            },
        ),
    )
    asset.insert(
        1,
        ET.Element(
            "material",
            {
                "name": "ground_material",
                "texture": "ground_texture",
                "texrepeat": "4 4",
                "reflectance": "0.15",
            },
        ),
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Converted model has no worldbody")

    robot_parts = list(worldbody)
    for part in robot_parts:
        worldbody.remove(part)

    worldbody.append(
        ET.Element(
            "light",
            {
                "name": "sun",
                "pos": "0 0 3",
                "dir": "0 0 -1",
                "directional": "true",
                "castshadow": "true",
            },
        )
    )
    worldbody.append(
        ET.Element(
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "3 3 0.1",
                "material": "ground_material",
                "friction": "1.0 0.005 0.0001",
                "condim": "3",
            },
        )
    )

    robot = ET.Element("body", {"name": "hexapod"})
    robot.append(ET.Element("freejoint", {"name": "root"}))
    for part in robot_parts:
        robot.append(part)
    worldbody.append(robot)

    actuator = ET.SubElement(root, "actuator")
    for prefix in LEG_PREFIXES:
        for joint_number in (1, 2, 3):
            joint_name = f"{prefix}_{joint_number}"
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": f"{joint_name}_position",
                    "joint": joint_name,
                    "kp": "120",
                    "kv": "3",
                    "ctrlrange": "-2.356194 2.356194",
                    "forcerange": "-8 8",
                },
            )


def _format_qpos(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def prepare_scene(output: Path = SCENE_OUTPUT) -> Path:
    """Convert the URDF to MJCF and add the floor, free joint and home pose."""
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is not installed. Activate .venv or install mjx/requirements.txt"
        ) from exc

    prepare_urdf(SOURCE_URDF, URDF_OUTPUT)
    urdf_model = mujoco.MjModel.from_xml_path(str(URDF_OUTPUT))

    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output), urdf_model)

    tree = ET.parse(output)
    root = tree.getroot()
    root.set("model", "hexapod_scene")
    _add_scene_elements(root)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    # Compile once to obtain the exact qpos ordering, then create a named reset
    # pose. The first seven entries belong to the floating base.
    model = mujoco.MjModel.from_xml_path(str(output))
    qpos = [0.0] * model.nq
    qpos[2] = STAND_HEIGHT
    qpos[3:7] = ROOT_QUATERNION
    controls: list[float] = []
    for prefix in LEG_PREFIXES:
        # Controller documentation uses the same positive servo angles on all
        # legs. Raw URDF axes are mirrored, so qpos signs differ by side.
        for joint_number in (1, 2, 3):
            angle = _standing_angle(prefix, joint_number)
            joint_name = f"{prefix}_{joint_number}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Missing joint: {joint_name}")
            qpos[int(model.jnt_qposadr[joint_id])] = angle
            controls.append(angle)

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": _format_qpos(qpos),
            "ctrl": _format_qpos(controls),
        },
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    # Final compilation catches invalid keyframe dimensions and scene errors.
    mujoco.MjModel.from_xml_path(str(output))
    return output


def main() -> None:
    print(prepare_scene().resolve())


if __name__ == "__main__":
    main()
