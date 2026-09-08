#!/usr/bin/env python3
"""Build the fixed-shape primitive terrain scene used for MJX training."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from prepare_scene import SCENE_OUTPUT, prepare_scene
from terrain_curriculum import (
    MAX_STAIR_COUNT,
    PLATEAU_DEPTH,
    RAMP_LENGTH,
    ROUGH_HFIELD_NCOL,
    ROUGH_HFIELD_NROW,
    ROUGH_LENGTH,
    TERRAIN_HALF_WIDTH,
    TERRAIN_LEVELS,
    TERRAIN_START_X,
    rough_heightfield_grid,
)
from tripod_controller import LEG_PREFIXES


RL_SCENE_OUTPUT = Path(__file__).resolve().parent / "generated/hexapod_rl.xml"
TARGET_ROBOT_MASS_KG = 10.0
# Backward-compatible names used by the firmware-only runner.  Its default
# stairs terrain is the final curriculum: ten consecutive 20 cm risers.
STEP_START_X = TERRAIN_START_X
STEP_DEPTH = 0.25
STEP_COUNT = MAX_STAIR_COUNT
STEP_HEIGHT = TERRAIN_LEVELS[-1].stair_riser
STAIR_TOTAL_RISE = TERRAIN_LEVELS[-1].final_height
FOOT_COLLISION_RADIUS_M = 0.032


def _numbers(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _robot_body_ids(model) -> set[int]:
    import mujoco

    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hexapod")
    if robot_id < 0:
        raise ValueError("Missing hexapod body")

    body_ids: set[int] = set()
    for body_id in range(1, model.nbody):
        ancestor_id = body_id
        while ancestor_id > 0 and ancestor_id != robot_id:
            ancestor_id = int(model.body_parentid[ancestor_id])
        if ancestor_id == robot_id:
            body_ids.add(body_id)
    return body_ids


def _replace_inertials(root: ET.Element, model) -> None:
    import mujoco

    robot_body_ids = _robot_body_ids(model)
    source_mass = sum(float(model.body_mass[body_id]) for body_id in robot_body_ids)
    if source_mass <= 0.0:
        raise ValueError("Hexapod mass must be positive")
    mass_scale = TARGET_ROBOT_MASS_KG / source_mass

    for body in root.iter("body"):
        name = body.get("name")
        if not name:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id not in robot_body_ids:
            continue
        old = body.find("inertial")
        if old is not None:
            body.remove(old)
        inertial = ET.Element(
            "inertial",
            {
                "pos": _numbers(model.body_ipos[body_id]),
                "quat": _numbers(model.body_iquat[body_id]),
                "mass": f"{float(model.body_mass[body_id]) * mass_scale:.12g}",
                "diaginertia": _numbers(
                    model.body_inertia[body_id] * mass_scale
                ),
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


def _quat_rotate_vectors(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate row vectors by one WXYZ quaternion."""

    xyz = quaternion[1:]
    return vectors + 2.0 * (
        quaternion[0] * np.cross(xyz, vectors)
        + np.cross(xyz, np.cross(xyz, vectors))
    )


def _mesh_foot_tip_local(model, prefix: str, outward: np.ndarray) -> np.ndarray:
    """Return the CAD foot assembly's outer support point in distal-body axes."""

    import mujoco

    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_motor_horn_3_1"
    )
    if body_id < 0:
        raise ValueError(f"Missing distal body for {prefix}")

    vertices: list[np.ndarray] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = model.mesh(mesh_id).name or ""
        if not mesh_name.startswith(f"{prefix}_foot_"):
            continue
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        mesh_vertices = np.asarray(
            model.mesh_vert[vertex_start : vertex_start + vertex_count]
        )
        vertices.append(
            _quat_rotate_vectors(model.geom_quat[geom_id], mesh_vertices)
            + model.geom_pos[geom_id]
        )
    if not vertices:
        raise ValueError(f"Missing merged CAD foot meshes for {prefix}")

    foot_vertices = np.concatenate(vertices, axis=0)
    support = foot_vertices @ outward
    support_max = float(np.max(support))
    # The support set is normally one CAD vertex.  Averaging tied vertices
    # gives a stable center if a future foot terminates in a flat face.
    tip = np.mean(
        foot_vertices[np.isclose(support, support_max, rtol=0.0, atol=1.0e-6)],
        axis=0,
    )
    # Preserve the exact maximum projection after averaging a flat support set.
    return tip + (support_max - float(tip @ outward)) * outward


def _add_robot_colliders(root: ET.Element, source_model) -> None:
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
        foot_tip = _mesh_foot_tip_local(source_model, prefix, outward)
        foot_center = foot_tip - FOOT_COLLISION_RADIUS_M * outward

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
                    "fromto": f"0 0 0 {_numbers(foot_center)}",
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
                    "pos": _numbers(foot_center),
                    "size": f"{FOOT_COLLISION_RADIUS_M:.9g}",
                    "friction": "1.2 0.01 0.001",
                    "rgba": "0.85 0.45 0.12 1",
                },
            )
        )
        body3.append(
            ET.Element(
                "site",
                {
                    "name": f"{prefix}_foot_site",
                    "pos": _numbers(foot_tip),
                    "size": "0.012",
                    "rgba": "1 0.2 0.1 1",
                },
            )
        )


def _hidden_attributes() -> dict[str, str]:
    return {"contype": "0", "conaffinity": "0", "rgba": "0 0 0 0"}


def _add_terrain_pool(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Missing worldbody")
    asset = root.find("asset")
    if asset is None:
        raise ValueError("Missing asset")

    insertion = 2
    placeholder = rough_heightfield_grid(1.0)
    asset.append(
        ET.Element(
            "hfield",
            {
                "name": "rough_hfield",
                "nrow": str(ROUGH_HFIELD_NROW),
                "ncol": str(ROUGH_HFIELD_NCOL),
                "size": f"{ROUGH_LENGTH / 2:.6g} {TERRAIN_HALF_WIDTH:.6g} 1 0.001",
                "elevation": " ".join(
                    f"{height:.9g}" for row in placeholder for height in row
                ),
            },
        )
    )
    worldbody.insert(
        insertion,
        ET.Element(
            "geom",
            {
                "name": "rough_hfield_geom",
                "type": "hfield",
                "hfield": "rough_hfield",
                "pos": f"{TERRAIN_START_X + ROUGH_LENGTH / 2:.6g} 0 0",
                "friction": "1.1 0.01 0.001",
                **_hidden_attributes(),
            },
        ),
    )
    insertion += 1

    ramp_attributes = {
        "name": "terrain_ramp",
        "type": "box",
        "pos": f"{TERRAIN_START_X + RAMP_LENGTH / 2:.6g} 0 0",
        "size": f"{RAMP_LENGTH / 2:.6g} {TERRAIN_HALF_WIDTH:.6g} 0.025",
        "friction": "1.1 0.01 0.001",
        **_hidden_attributes(),
    }
    worldbody.insert(insertion, ET.Element("geom", ramp_attributes))
    insertion += 1

    for index in range(STEP_COUNT):
        height = STEP_HEIGHT * (index + 1)
        center_x = STEP_START_X + STEP_DEPTH * (index + 0.5)
        worldbody.insert(
            insertion,
            ET.Element(
                "geom",
                {
                    "name": f"stair_{index + 1}",
                    "type": "box",
                    "pos": f"{center_x:.6g} 0 {height / 2:.6g}",
                    "size": (
                        f"{STEP_DEPTH / 2:.6g} {TERRAIN_HALF_WIDTH:.6g} "
                        f"{height / 2:.6g}"
                    ),
                    "friction": "1.1 0.01 0.001",
                    "rgba": "0.34 0.42 0.50 1",
                },
            ),
        )
        insertion += 1

    final_stair_end = STEP_START_X + STEP_COUNT * STEP_DEPTH
    worldbody.insert(
        insertion,
        ET.Element(
            "geom",
            {
                "name": "terrain_plateau",
                "type": "box",
                "pos": (
                    f"{final_stair_end + PLATEAU_DEPTH / 2:.6g} 0 "
                    f"{STAIR_TOTAL_RISE / 2:.6g}"
                ),
                "size": (
                    f"{PLATEAU_DEPTH / 2:.6g} {TERRAIN_HALF_WIDTH:.6g} "
                    f"{STAIR_TOTAL_RISE / 2:.6g}"
                ),
                "friction": "1.1 0.01 0.001",
                "rgba": "0.30 0.39 0.48 1",
            },
        ),
    )


def prepare_rl_scene(output: Path = RL_SCENE_OUTPUT, *, rough_boxes: bool = False) -> Path:
    import mujoco

    prepare_scene(SCENE_OUTPUT)
    source_model = mujoco.MjModel.from_xml_path(str(SCENE_OUTPUT))
    tree = ET.parse(SCENE_OUTPUT)
    root = tree.getroot()
    root.set("model", "hexapod_terrain_curriculum")
    option = root.find("option")
    if option is not None:
        option.set("timestep", "0.0025")

    _replace_inertials(root, source_model)
    _strip_cad_meshes(root)
    _add_robot_colliders(root, source_model)
    _add_terrain_pool(root)
    if rough_boxes:
        from terrain_curriculum import rough_tile_centers, ROUGH_TILE_DEPTH, ROUGH_TILE_WIDTH
        world = root.find('worldbody')
        # Remove the unsupported type entirely: invisible hfields can still be
        # encountered by a ray implementation's static dispatch.
        old = world.find("geom[@name='rough_hfield_geom']")
        old.attrib.pop('hfield')
        old.set('type', 'box')
        old.set('size', '.001 .001 .001')
        old.set('group', '5')
        asset = root.find('asset')
        asset.remove(asset.find("hfield[@name='rough_hfield']"))
        for index, (x, y) in enumerate(rough_tile_centers()):
            ET.SubElement(world, 'geom', dict(name=f'adaptive_rough_{index}', type='box',
                pos=_numbers((x, y, .0005)), size=_numbers((ROUGH_TILE_DEPTH/2, ROUGH_TILE_WIDTH/2, .0005)),
                friction='1.1 0.01 0.001', **_hidden_attributes()))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    if model.ngeom >= (152 if rough_boxes else 120):
        raise ValueError(f"RL collision model is unexpectedly large: {model.ngeom}")
    robot_mass = sum(
        float(model.body_mass[body_id]) for body_id in _robot_body_ids(model)
    )
    if not np.isclose(robot_mass, TARGET_ROBOT_MASS_KG, atol=1.0e-6):
        raise ValueError(
            f"RL hexapod mass is {robot_mass:.9g} kg, expected "
            f"{TARGET_ROBOT_MASS_KG:.9g} kg"
        )
    return output


def main() -> None:
    path = prepare_rl_scene()
    print(path.resolve())


if __name__ == "__main__":
    main()
