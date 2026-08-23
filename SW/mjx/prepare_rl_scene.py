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
MIXED_RL_SCENE_OUTPUT = Path(__file__).resolve().parent / "generated/hexapod_mixed_rl.xml"
STEP_START_X = 0.55
STEP_DEPTH = 0.25
STEP_HEIGHT = 0.05
STEP_COUNT = 7
DRY_ASPHALT_FRICTION = 0.8
MIXED_PATCH_NAMES = ("flat", "curb", "ramp", "blocks", "stairs", "rough")
MIXED_PATCH_Y = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
MIXED_LANE_HALF_WIDTH = 0.9
MIXED_CURB = (0.55, 0.55, 0.04)  # start, length, height
MIXED_RAMP = (0.45, 1.20, 0.10)  # start, length, rise
MIXED_BLOCKS = (
    (0.50, -0.28, 0.22, 0.22, 0.035),
    (0.82, 0.20, 0.20, 0.28, 0.060),
    (1.14, -0.12, 0.24, 0.22, 0.045),
    (1.47, 0.30, 0.20, 0.20, 0.075),
    (1.78, -0.26, 0.26, 0.24, 0.050),
)
MIXED_ROUGH = (
    (0.42, -0.42, 0.24, 0.32, 0.018),
    (0.42, 0.02, 0.24, 0.32, 0.032),
    (0.42, 0.46, 0.24, 0.32, 0.012),
    (0.78, -0.40, 0.24, 0.30, 0.040),
    (0.78, 0.00, 0.24, 0.30, 0.020),
    (0.78, 0.40, 0.24, 0.30, 0.052),
    (1.14, -0.30, 0.26, 0.42, 0.028),
    (1.14, 0.28, 0.26, 0.42, 0.044),
)


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
    center_y: float = 0.0,
    name_prefix: str = "stair",
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
                    "name": f"{name_prefix}_{index + 1}",
                    "type": "box",
                    "pos": f"{center_x:.6g} {center_y:.6g} {height / 2:.6g}",
                    "size": f"{step_depth / 2:.6g} {MIXED_LANE_HALF_WIDTH:.6g} {height / 2:.6g}",
                    "friction": f"{friction:.6g} 0.01 0.001",
                    "rgba": "0.34 0.42 0.50 1",
                },
            ),
        )


def _add_patch_box(
    worldbody: ET.Element,
    *,
    name: str,
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    height: float,
    friction: float,
    rgba: str,
) -> None:
    worldbody.append(
        ET.Element(
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": f"{center_x:.6g} {center_y:.6g} {height / 2:.6g}",
                "size": f"{length / 2:.6g} {width / 2:.6g} {height / 2:.6g}",
                "friction": f"{friction:.6g} 0.01 0.001",
                "rgba": rgba,
            },
        )
    )


def _add_mixed_patches(root: ET.Element, *, friction: float) -> None:
    """Add several terrain families as parallel lanes in one MJX model."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Missing worldbody")

    curb_start, curb_length, curb_height = MIXED_CURB
    _add_patch_box(
        worldbody,
        name="mixed_curb",
        center_x=curb_start + curb_length / 2,
        center_y=MIXED_PATCH_Y[1],
        length=curb_length,
        width=2 * MIXED_LANE_HALF_WIDTH,
        height=curb_height,
        friction=friction,
        rgba="0.35 0.48 0.58 1",
    )

    ramp_start, ramp_length, ramp_rise = MIXED_RAMP
    angle = float(np.arctan2(ramp_rise, ramp_length))
    thickness = 0.04
    worldbody.append(
        ET.Element(
            "geom",
            {
                "name": "mixed_ramp",
                "type": "box",
                "pos": f"{ramp_start + ramp_length / 2:.6g} {MIXED_PATCH_Y[2]:.6g} {ramp_rise / 2 - thickness / 2:.6g}",
                "size": f"{np.hypot(ramp_length, ramp_rise) / 2:.6g} {MIXED_LANE_HALF_WIDTH:.6g} {thickness / 2:.6g}",
                "euler": f"0 {-angle:.9g} 0",
                "friction": f"{friction:.6g} 0.01 0.001",
                "rgba": "0.38 0.52 0.38 1",
            },
        )
    )
    _add_patch_box(
        worldbody,
        name="mixed_ramp_plateau",
        center_x=ramp_start + ramp_length + 0.45,
        center_y=MIXED_PATCH_Y[2],
        length=0.9,
        width=2 * MIXED_LANE_HALF_WIDTH,
        height=ramp_rise,
        friction=friction,
        rgba="0.38 0.52 0.38 1",
    )

    for index, (x, y_offset, length, width, height) in enumerate(MIXED_BLOCKS):
        _add_patch_box(
            worldbody,
            name=f"mixed_block_{index}",
            center_x=x,
            center_y=MIXED_PATCH_Y[3] + y_offset,
            length=length,
            width=width,
            height=height,
            friction=friction,
            rgba="0.52 0.42 0.30 1",
        )

    _add_stairs(
        root,
        step_start_x=0.50,
        step_depth=0.28,
        step_height=0.035,
        step_count=6,
        friction=friction,
        center_y=MIXED_PATCH_Y[4],
        name_prefix="mixed_stair",
    )

    for index, (x, y_offset, length, width, height) in enumerate(MIXED_ROUGH):
        _add_patch_box(
            worldbody,
            name=f"mixed_rough_{index}",
            center_x=x,
            center_y=MIXED_PATCH_Y[5] + y_offset,
            length=length,
            width=width,
            height=height,
            friction=friction,
            rgba="0.42 0.36 0.30 1",
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
    """Create a mesh-free RL scene for ``flat``, ``stairs``, or ``mixed`` terrain.

    Both variants use exactly the same robot collision geometry and position
    actuators.  Keeping that contract makes the flat command curriculum and
    terrain policy directly comparable while avoiding CAD mesh contact in MJX.
    """
    import mujoco

    if terrain not in {"flat", "stairs", "mixed"}:
        raise ValueError(f"terrain must be 'flat', 'stairs', or 'mixed', got {terrain!r}")

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
    # Equal-priority MuJoCo contacts take the larger geom friction.  Matching
    # the plane and foot values makes ``friction`` the actual flat contact
    # coefficient; rough terrain keeps the conservative legacy multipliers.
    foot_friction = friction if terrain == "flat" else 1.2 * friction
    _add_robot_colliders(root, foot_friction=foot_friction)
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
    elif terrain == "mixed":
        _add_mixed_patches(root, friction=1.1 * friction)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    if model.ngeom >= 80:
        raise ValueError(f"RL collision model is unexpectedly large: {model.ngeom}")
    return output


def prepare_flat_rl_scene(
    output: Path = FLAT_RL_SCENE_OUTPUT, *, friction: float = DRY_ASPHALT_FRICTION
) -> Path:
    """Create the mesh-free plane scene with nominal dry-asphalt friction."""
    return prepare_rl_scene(output, terrain="flat", friction=friction)


def main() -> None:
    path = prepare_rl_scene()
    print(path.resolve())


if __name__ == "__main__":
    main()
