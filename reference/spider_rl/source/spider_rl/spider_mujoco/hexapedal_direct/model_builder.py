from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import math
import xml.etree.ElementTree as ET


LEG_ORDER = ("LF", "LM", "LB", "RF", "RM", "RB")
STAGE_ORDER = ("motor_horn_1_1", "DS51150_270_2_1", "motor_horn_3_1")
RETAINED_LINK_NAMES = (
    ("base_link",)
    + tuple(f"{leg}_motor_horn_1_1" for leg in LEG_ORDER)
    + tuple(f"{leg}_DS51150_270_2_1" for leg in LEG_ORDER)
    + tuple(f"{leg}_motor_horn_3_1" for leg in LEG_ORDER)
)
JOINT_NAME_ORDER = tuple(
    f"{leg}_motor_horn_{stage}_joint"
    for leg in LEG_ORDER
    for stage in (1, 2, 3)
)
DESIRED_CONTACT_SITE_NAMES = tuple(f"{leg}_motor_horn_3_1_contact_site" for leg in LEG_ORDER)
UNDESIRED_CONTACT_BODY_PATTERNS = (
    "base_link",
    "*_motor_horn_1_1",
    "*_motor_horn_2_1",
)
DEFAULT_URDF_PATH = Path("/home/huro/spider_ws/HEXAPEDAL_URDF_description/urdf/HEXAPEDAL_URDF_fixed.urdf")
URDF_ENV_VAR = "SPIDER_HEXAPEDAL_URDF_PATH"
MESH_DIR_ENV_VAR = "SPIDER_HEXAPEDAL_MESH_DIR"
PACKAGE_ROOT = Path(__file__).resolve().parent
ASSET_DIR = PACKAGE_ROOT / "assets"
XML_PATH = ASSET_DIR / "hexapedal.xml"
SOURCE_MAP_PATH = ASSET_DIR / "source_map.yaml"


def _discover_urdf_candidates() -> tuple[Path, ...]:
    explicit = os.environ.get(URDF_ENV_VAR)
    if explicit:
        return (Path(explicit).expanduser(),)
    candidates = [DEFAULT_URDF_PATH]
    relative = Path("HEXAPEDAL_URDF_description") / "urdf" / "HEXAPEDAL_URDF_fixed.urdf"
    for parent in PACKAGE_ROOT.parents:
        candidate = parent / relative
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def resolve_source_asset_paths() -> tuple[Path, Path]:
    explicit_mesh_dir = os.environ.get(MESH_DIR_ENV_VAR)
    for urdf_candidate in _discover_urdf_candidates():
        urdf_path = urdf_candidate.expanduser()
        if not urdf_path.is_file():
            continue
        mesh_dir = Path(explicit_mesh_dir).expanduser() if explicit_mesh_dir else urdf_path.parent.parent / "meshes"
        if not mesh_dir.is_dir():
            raise FileNotFoundError(
                f"Resolved URDF at {urdf_path}, but mesh directory {mesh_dir} is missing. "
                f"Set {MESH_DIR_ENV_VAR} to the meshes directory if it lives elsewhere."
            )
        return urdf_path.resolve(), mesh_dir.resolve()
    searched = ", ".join(str(path) for path in _discover_urdf_candidates())
    raise FileNotFoundError(
        "Could not locate HEXAPEDAL URDF source assets. "
        f"Set {URDF_ENV_VAR} (and optionally {MESH_DIR_ENV_VAR}) or place the asset repo beside this checkout. "
        f"Searched: {searched}"
    )


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    source_link: str
    kind: str
    geom_type: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    mesh_name: str | None
    size: tuple[float, ...]
    fromto: tuple[float, ...] | None


@dataclass(frozen=True)
class InertialSpec:
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class BodySpec:
    name: str
    parent_name: str | None
    urdf_link_name: str
    joint_name: str | None
    urdf_joint_name: str | None
    joint_axis: tuple[float, float, float] | None
    joint_range: tuple[float, float] | None
    body_pos: tuple[float, float, float]
    body_quat: tuple[float, float, float, float]
    merged_links: tuple[str, ...]
    contact_sources: tuple[str, ...]
    inertial: InertialSpec
    visual_geoms: tuple[GeometrySpec, ...]
    collision_geoms: tuple[GeometrySpec, ...]


@dataclass(frozen=True)
class HexapedalModel:
    xml_path: Path
    source_map_path: Path
    urdf_path: Path
    joint_names: tuple[str, ...]
    desired_contact_site_names: tuple[str, ...]
    undesired_contact_body_patterns: tuple[str, ...]
    retained_bodies: dict[str, BodySpec]
    source_map: dict[str, object]


@dataclass(frozen=True)
class LinkData:
    name: str
    inertial: ET.Element | None
    visuals: tuple[ET.Element, ...]
    collisions: tuple[ET.Element, ...]


@dataclass(frozen=True)
class JointData:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    limit: tuple[float, float] | None


@dataclass(frozen=True)
class ParsedUrdf:
    links: dict[str, LinkData]
    joints: dict[str, JointData]
    link_order: tuple[str, ...]
    child_joint_by_link: dict[str, JointData]
    child_links_by_parent: dict[str, tuple[str, ...]]
    world_from_link: dict[str, tuple[tuple[float, ...], ...]]


def _fmt_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.12g}"
    return "0" if text == "-0" else text


def _fmt_vec(values: tuple[float, ...]) -> str:
    return " ".join(_fmt_number(value) for value in values)


def _identity_matrix() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix_multiply(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    rows = []
    for row_index in range(4):
        row = []
        for col_index in range(4):
            row.append(sum(left[row_index][k] * right[k][col_index] for k in range(4)))
        rows.append(tuple(row))
    return tuple(rows)


def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, ...], ...]:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matrix_from_xyz_rpy(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> tuple[tuple[float, ...], ...]:
    rotation = _rotation_matrix_from_rpy(*rpy)
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], xyz[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], xyz[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], xyz[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix_inverse(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    rotation = tuple(tuple(matrix[row][col] for col in range(3)) for row in range(3))
    translation = (matrix[0][3], matrix[1][3], matrix[2][3])
    rotation_t = tuple(tuple(rotation[col][row] for col in range(3)) for row in range(3))
    inv_translation = tuple(-sum(rotation_t[row][col] * translation[col] for col in range(3)) for row in range(3))
    return (
        (rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inv_translation[0]),
        (rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inv_translation[1]),
        (rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inv_translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _transform_point(matrix: tuple[tuple[float, ...], ...], point: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][col] * point[col] for col in range(3)) + matrix[row][3]
        for row in range(3)
    )


def _rotation_from_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(matrix[row][col] for col in range(3)) for row in range(3))


def _quaternion_from_rotation(rotation: tuple[tuple[float, ...], ...]) -> tuple[float, float, float, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2][1] - rotation[1][2]) / scale
        y = (rotation[0][2] - rotation[2][0]) / scale
        z = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        w = (rotation[2][1] - rotation[1][2]) / scale
        x = 0.25 * scale
        y = (rotation[0][1] + rotation[1][0]) / scale
        z = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        w = (rotation[0][2] - rotation[2][0]) / scale
        x = (rotation[0][1] + rotation[1][0]) / scale
        y = 0.25 * scale
        z = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        w = (rotation[1][0] - rotation[0][1]) / scale
        x = (rotation[0][2] + rotation[2][0]) / scale
        y = (rotation[1][2] + rotation[2][1]) / scale
        z = 0.25 * scale
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    quaternion = (w / norm, x / norm, y / norm, z / norm)
    if quaternion[0] < 0.0:
        return tuple(-value for value in quaternion)
    return quaternion


def _quat_from_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, float, float, float]:
    return _quaternion_from_rotation(_rotation_from_matrix(matrix))


def _rotate_vector(rotation: tuple[tuple[float, ...], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(rotation[row][col] * vector[col] for col in range(3)) for row in range(3))


def _vector_add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(left[index] + right[index] for index in range(3))


def _vector_subtract(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))


def _outer(vector: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(vector[row] * vector[col] for col in range(3)) for row in range(3))


def _matrix3_add(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(left[row][col] + right[row][col] for col in range(3)) for row in range(3))


def _matrix3_scale(matrix: tuple[tuple[float, ...], ...], scale: float) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(matrix[row][col] * scale for col in range(3)) for row in range(3))


def _matrix3_multiply(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _matrix3_transpose(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(matrix[col][row] for col in range(3)) for row in range(3))


def _parse_xyz(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) for value in text.split())


def _parse_origin(element: ET.Element | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return _parse_xyz(element.attrib.get("xyz")), _parse_xyz(element.attrib.get("rpy"))


def _parse_geometry_size(geometry_element: ET.Element) -> tuple[str, str | None, tuple[float, ...], tuple[float, ...] | None]:
    mesh = geometry_element.find("mesh")
    if mesh is not None:
        scale = tuple(float(value) for value in mesh.attrib.get("scale", "1 1 1").split())
        return "mesh", mesh.attrib["filename"], scale, None
    box = geometry_element.find("box")
    if box is not None:
        return "box", None, tuple(float(value) for value in box.attrib["size"].split()), None
    cylinder = geometry_element.find("cylinder")
    if cylinder is not None:
        return (
            "cylinder",
            None,
            (float(cylinder.attrib["radius"]), float(cylinder.attrib["length"])),
            None,
        )
    sphere = geometry_element.find("sphere")
    if sphere is not None:
        return "sphere", None, (float(sphere.attrib["radius"]),), None
    raise ValueError(f"Unsupported geometry in {ET.tostring(geometry_element, encoding='unicode')}")


def _mesh_name_from_filename(filename: str) -> str:
    return Path(filename).stem


def _parse_urdf(urdf_path: Path) -> ParsedUrdf:
    root = ET.parse(urdf_path).getroot()
    links: dict[str, LinkData] = {}
    link_order = []
    for link_element in root.findall("link"):
        name = link_element.attrib["name"]
        link_order.append(name)
        links[name] = LinkData(
            name=name,
            inertial=link_element.find("inertial"),
            visuals=tuple(link_element.findall("visual")),
            collisions=tuple(link_element.findall("collision")),
        )
    joints: dict[str, JointData] = {}
    child_joint_by_link: dict[str, JointData] = {}
    child_links_by_parent: dict[str, list[str]] = {}
    for joint_element in root.findall("joint"):
        origin_element = joint_element.find("origin")
        origin_xyz, origin_rpy = _parse_origin(origin_element)
        axis_element = joint_element.find("axis")
        axis = _parse_xyz(axis_element.attrib.get("xyz") if axis_element is not None else None)
        limit_element = joint_element.find("limit")
        limit = None
        if limit_element is not None and "lower" in limit_element.attrib and "upper" in limit_element.attrib:
            limit = (float(limit_element.attrib["lower"]), float(limit_element.attrib["upper"]))
        joint = JointData(
            name=joint_element.attrib["name"],
            joint_type=joint_element.attrib["type"],
            parent=joint_element.find("parent").attrib["link"],
            child=joint_element.find("child").attrib["link"],
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
            limit=limit,
        )
        joints[joint.name] = joint
        child_joint_by_link[joint.child] = joint
        child_links_by_parent.setdefault(joint.parent, []).append(joint.child)
    world_from_link = {"base_link": _identity_matrix()}
    stack = ["base_link"]
    while stack:
        parent = stack.pop()
        parent_world = world_from_link[parent]
        for child in child_links_by_parent.get(parent, []):
            joint = child_joint_by_link[child]
            local = _matrix_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
            world_from_link[child] = _matrix_multiply(parent_world, local)
            stack.append(child)
    return ParsedUrdf(
        links=links,
        joints=joints,
        link_order=tuple(link_order),
        child_joint_by_link=child_joint_by_link,
        child_links_by_parent={key: tuple(value) for key, value in child_links_by_parent.items()},
        world_from_link=world_from_link,
    )


def _retained_links() -> set[str]:
    return set(RETAINED_LINK_NAMES)


def _retained_owner(link_name: str, parsed: ParsedUrdf) -> str:
    retained_links = _retained_links()
    current = link_name
    while current not in retained_links:
        joint = parsed.child_joint_by_link[current]
        if joint.joint_type != "fixed":
            raise ValueError(f"{link_name} is not on a fixed chain below a retained link")
        current = joint.parent
    return current


def _exported_joint_name_for_child(link_name: str) -> str:
    leg, part, _, suffix = link_name.split("_", 3)
    if part != "motor" and part != "DS51150":
        raise ValueError(f"Unexpected retained child link {link_name}")
    if link_name.endswith("motor_horn_1_1"):
        return f"{leg}_motor_horn_1_joint"
    if link_name.endswith("DS51150_270_2_1"):
        return f"{leg}_motor_horn_2_joint"
    if link_name.endswith("motor_horn_3_1"):
        return f"{leg}_motor_horn_3_joint"
    raise ValueError(f"Unexpected retained child link {link_name}")


def _merge_inertials(link_names: tuple[str, ...], parsed: ParsedUrdf, body_name: str) -> InertialSpec:
    body_world = parsed.world_from_link[body_name]
    body_from_world = _matrix_inverse(body_world)
    mass_entries: list[tuple[float, tuple[float, float, float], tuple[tuple[float, ...], ...]]] = []
    total_mass = 0.0
    weighted_com = (0.0, 0.0, 0.0)
    for link_name in link_names:
        inertial = parsed.links[link_name].inertial
        if inertial is None:
            continue
        mass = float(inertial.find("mass").attrib["value"])
        origin_xyz, origin_rpy = _parse_origin(inertial.find("origin"))
        inertial_from_link = _matrix_from_xyz_rpy(origin_xyz, origin_rpy)
        inertial_world = _matrix_multiply(parsed.world_from_link[link_name], inertial_from_link)
        inertial_body = _matrix_multiply(body_from_world, inertial_world)
        com = (inertial_body[0][3], inertial_body[1][3], inertial_body[2][3])
        inertia_element = inertial.find("inertia")
        inertia = (
            (float(inertia_element.attrib["ixx"]), float(inertia_element.attrib["ixy"]), float(inertia_element.attrib["ixz"])),
            (float(inertia_element.attrib["ixy"]), float(inertia_element.attrib["iyy"]), float(inertia_element.attrib["iyz"])),
            (float(inertia_element.attrib["ixz"]), float(inertia_element.attrib["iyz"]), float(inertia_element.attrib["izz"])),
        )
        rotation = _rotation_from_matrix(inertial_body)
        rotated_inertia = _matrix3_multiply(_matrix3_multiply(rotation, inertia), _matrix3_transpose(rotation))
        mass_entries.append((mass, com, rotated_inertia))
        total_mass += mass
        weighted_com = tuple(weighted_com[index] + mass * com[index] for index in range(3))
    if total_mass == 0.0:
        return InertialSpec(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    combined_com = tuple(weighted_com[index] / total_mass for index in range(3))
    combined_inertia = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    for mass, com, inertia in mass_entries:
        offset = _vector_subtract(com, combined_com)
        distance_sq = sum(value * value for value in offset)
        shift = _matrix3_add(
            _matrix3_scale(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), distance_sq),
            _matrix3_scale(_outer(offset), -1.0),
        )
        combined_inertia = _matrix3_add(combined_inertia, _matrix3_add(inertia, _matrix3_scale(shift, mass)))
    return InertialSpec(
        mass=total_mass,
        com=combined_com,
        inertia=(
            combined_inertia[0][0],
            combined_inertia[1][1],
            combined_inertia[2][2],
            combined_inertia[0][1],
            combined_inertia[0][2],
            combined_inertia[1][2],
        ),
    )


def _build_geom_specs(
    body_name: str,
    link_names: tuple[str, ...],
    parsed: ParsedUrdf,
) -> tuple[tuple[GeometrySpec, ...], tuple[GeometrySpec, ...], dict[str, tuple[str, tuple[float, float, float]]]]:
    body_world = parsed.world_from_link[body_name]
    body_from_world = _matrix_inverse(body_world)
    visual_geoms: list[GeometrySpec] = []
    collision_geoms: list[GeometrySpec] = []
    mesh_assets: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for link_name in link_names:
        link_world = parsed.world_from_link[link_name]
        for kind, elements, target in (
            ("visual", parsed.links[link_name].visuals, visual_geoms),
            ("collision", parsed.links[link_name].collisions, collision_geoms),
        ):
            for index, element in enumerate(elements):
                local_xyz, local_rpy = _parse_origin(element.find("origin"))
                geom_local = _matrix_from_xyz_rpy(local_xyz, local_rpy)
                geom_world = _matrix_multiply(link_world, geom_local)
                geom_body = _matrix_multiply(body_from_world, geom_world)
                geometry_type, mesh_filename, size, fromto = _parse_geometry_size(element.find("geometry"))
                mesh_name = None
                if mesh_filename is not None:
                    mesh_name = _mesh_name_from_filename(mesh_filename)
                    mesh_assets[mesh_name] = (Path(mesh_filename).name, size)
                target.append(
                    GeometrySpec(
                        name=f"{link_name}_{kind}_{index}",
                        source_link=link_name,
                        kind=kind,
                        geom_type=geometry_type,
                        pos=(geom_body[0][3], geom_body[1][3], geom_body[2][3]),
                        quat=_quat_from_matrix(geom_body),
                        mesh_name=mesh_name,
                        size=size,
                        fromto=fromto,
                    )
                )
    return tuple(visual_geoms), tuple(collision_geoms), mesh_assets


def _build_body_specs(parsed: ParsedUrdf) -> tuple[dict[str, BodySpec], dict[str, tuple[str, tuple[float, float, float]]], dict[str, str]]:
    retained_links = _retained_links()
    merge_map: dict[str, list[str]] = {name: [] for name in RETAINED_LINK_NAMES}
    for link_name in parsed.link_order:
        if link_name in retained_links:
            continue
        merge_map[_retained_owner(link_name, parsed)].append(link_name)
    body_specs: dict[str, BodySpec] = {}
    mesh_assets: dict[str, tuple[str, tuple[float, float, float]]] = {}
    provenance_tags: dict[str, str] = {}
    for body_name in RETAINED_LINK_NAMES:
        merged_links = tuple([body_name] + merge_map[body_name])
        if body_name == "base_link":
            parent_name = None
            urdf_joint_name = None
            exported_joint_name = None
            joint_axis = None
            joint_range = None
            body_pose = _identity_matrix()
        else:
            child_joint = parsed.child_joint_by_link[body_name]
            urdf_joint_name = child_joint.name
            exported_joint_name = _exported_joint_name_for_child(body_name)
            parent_name = _retained_owner(child_joint.parent, parsed)
            body_pose = _matrix_multiply(_matrix_inverse(parsed.world_from_link[parent_name]), parsed.world_from_link[body_name])
            joint_axis = child_joint.axis
            joint_range = child_joint.limit
        visual_geoms, collision_geoms, body_mesh_assets = _build_geom_specs(body_name, merged_links, parsed)
        mesh_assets.update(body_mesh_assets)
        body_specs[body_name] = BodySpec(
            name=body_name,
            parent_name=parent_name,
            urdf_link_name=body_name,
            joint_name=exported_joint_name,
            urdf_joint_name=urdf_joint_name,
            joint_axis=joint_axis,
            joint_range=joint_range,
            body_pos=(body_pose[0][3], body_pose[1][3], body_pose[2][3]),
            body_quat=_quat_from_matrix(body_pose),
            merged_links=merged_links,
            contact_sources=tuple(
                link_name
                for link_name in merged_links
                if link_name == "base_link" or link_name.endswith("motor_horn_1_1") or link_name.endswith("motor_horn_2_1")
            ),
            inertial=_merge_inertials(merged_links, parsed, body_name),
            visual_geoms=visual_geoms,
            collision_geoms=collision_geoms,
        )
        for link_name in merged_links:
            provenance_tags[link_name] = body_name
    return body_specs, mesh_assets, provenance_tags


def _mjcf_geom_xml(geom: GeometrySpec) -> str:
    attrs = [f'name="{geom.name}"', f'pos="{_fmt_vec(geom.pos)}"', f'quat="{_fmt_vec(geom.quat)}"']
    if geom.kind == "visual":
        attrs.extend(['class="visual"', 'contype="0"', 'conaffinity="0"'])
    else:
        attrs.extend(['class="collision"'])
    attrs.append(f'user="0 0 0 0"')
    if geom.geom_type == "mesh":
        attrs.append('type="mesh"')
        attrs.append(f'mesh="{geom.mesh_name}"')
    elif geom.geom_type == "box":
        attrs.append('type="box"')
        half_size = tuple(value / 2.0 for value in geom.size)
        attrs.append(f'size="{_fmt_vec(half_size)}"')
    elif geom.geom_type == "cylinder":
        attrs.append('type="cylinder"')
        attrs.append(f'size="{_fmt_vec((geom.size[0], geom.size[1] / 2.0))}"')
    elif geom.geom_type == "sphere":
        attrs.append('type="sphere"')
        attrs.append(f'size="{_fmt_vec(geom.size)}"')
    else:
        raise ValueError(f"Unsupported MuJoCo geom type {geom.geom_type}")
    return f"<geom {' '.join(attrs)}/>"


def _render_body_xml(body_name: str, children: dict[str, list[str]], bodies: dict[str, BodySpec], indent: str) -> list[str]:
    body = bodies[body_name]
    lines = []
    body_attrs = [f'name="{body.name}"']
    if body.parent_name is not None:
        body_attrs.append(f'pos="{_fmt_vec(body.body_pos)}"')
        body_attrs.append(f'quat="{_fmt_vec(body.body_quat)}"')
    lines.append(f"{indent}<body {' '.join(body_attrs)}>")
    if body.parent_name is None:
        lines.append(f"{indent}  <freejoint name=\"base_freejoint\"/>")
    else:
        lines.append(
            f"{indent}  <joint name=\"{body.joint_name}\" type=\"hinge\" pos=\"0 0 0\" axis=\"{_fmt_vec(body.joint_axis)}\" range=\"{_fmt_vec(body.joint_range)}\" damping=\"4\"/>")
    inertial = body.inertial
    lines.append(
        f"{indent}  <inertial pos=\"{_fmt_vec(inertial.com)}\" mass=\"{_fmt_number(inertial.mass)}\" fullinertia=\"{_fmt_vec(inertial.inertia)}\"/>")
    for geom in body.visual_geoms:
        lines.append(f"{indent}  {_mjcf_geom_xml(geom)}")
    for geom in body.collision_geoms:
        lines.append(f"{indent}  {_mjcf_geom_xml(geom)}")
    if body.name.endswith("motor_horn_3_1"):
        lines.append(
            f"{indent}  <site name=\"{body.name}_contact_site\" pos=\"0 0 0\" quat=\"1 0 0 0\" type=\"sphere\" size=\"0.003\" rgba=\"0 1 0 1\"/>")
        lines.append(
            f"{indent}  <geom name=\"{body.name}_stance_pad\" pos=\"0 0 -0.09\" quat=\"1 0 0 0\" class=\"collision\" user=\"0 0 0 0\" type=\"box\" size=\"0.02 0.02 0.004\"/>"
        )
    for child_name in children.get(body_name, []):
        lines.extend(_render_body_xml(child_name, children, bodies, indent + "  "))
    lines.append(f"{indent}</body>")
    return lines


def _build_children_graph(bodies: dict[str, BodySpec]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {name: [] for name in bodies}
    for leg in LEG_ORDER:
        graph["base_link"].append(f"{leg}_motor_horn_1_1")
        graph[f"{leg}_motor_horn_1_1"].append(f"{leg}_DS51150_270_2_1")
        graph[f"{leg}_DS51150_270_2_1"].append(f"{leg}_motor_horn_3_1")
    return graph


def _render_xml(
    parsed: ParsedUrdf,
    bodies: dict[str, BodySpec],
    mesh_assets: dict[str, tuple[str, tuple[float, float, float]]],
    mesh_dir: Path,
) -> str:
    del parsed
    meshdir_text = os.path.relpath(mesh_dir, XML_PATH.parent).replace(os.sep, "/")
    lines = [
        '<mujoco model="hexapedal_direct">',
        f'  <compiler angle="radian" autolimits="true" meshdir="{meshdir_text}"/>',
        '  <option timestep="0.008333333333333333" gravity="0 0 -9.81"/>',
        '  <default>',
        '    <default class="visual">',
        '      <geom group="1" rgba="0.7 0.7 0.7 1" contype="0" conaffinity="0"/>',
        '    </default>',
        '    <default class="collision">',
        '      <geom group="0" contype="1" conaffinity="0" friction="1 0.02 0.002" solref="0.005 1" solimp="0.9 0.95 0.001"/>',
        '    </default>',
        '  </default>',
        '  <asset>',
    ]
    for mesh_name in sorted(mesh_assets):
        filename, scale = mesh_assets[mesh_name]
        lines.append(f'    <mesh name="{mesh_name}" file="{filename}" scale="{_fmt_vec(scale)}"/>')
    lines.extend([
        '  </asset>',
        '  <worldbody>',
        '    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.2 0.25 0.3 1" contype="0" conaffinity="1" friction="1 0.02 0.002"/>',
    ])
    children = _build_children_graph(bodies)
    lines.extend(_render_body_xml("base_link", children, bodies, "    "))
    lines.extend([
        '  </worldbody>',
        '  <actuator>',
    ])
    actuators = {body.joint_name: body for body in bodies.values() if body.joint_name is not None}
    for joint_name in JOINT_NAME_ORDER:
        body = actuators[joint_name]
        lines.append(
            f'    <motor name="{joint_name}" joint="{joint_name}" gear="1" ctrllimited="true" ctrlrange="-1 1"/>'
        )
    lines.extend([
        '  </actuator>',
        '</mujoco>',
        '',
    ])
    return "\n".join(lines)


def _source_map_dict(bodies: dict[str, BodySpec], urdf_path: Path) -> dict[str, object]:
    retained_bodies = {}
    merged_links = {}
    for body_name, body in bodies.items():
        retained_bodies[body_name] = {
            "urdfLink": body.urdf_link_name,
            "parentBody": body.parent_name,
            "jointName": body.joint_name,
            "urdfJointName": body.urdf_joint_name,
            "mergedLinks": list(body.merged_links),
            "contactSources": list(body.contact_sources),
        }
        for link_name in body.merged_links:
            merged_links[link_name] = {
                "retainedBody": body_name,
                "contactClass": (
                    "desired" if body_name.endswith("motor_horn_3_1") and link_name.endswith("motor_horn_3_1") else
                    "undesired" if link_name == "base_link" or link_name.endswith("motor_horn_1_1") or link_name.endswith("motor_horn_2_1") else
                    "neutral"
                ),
            }
    return {
        "version": 1,
        "urdfPath": str(urdf_path),
        "xmlPath": str(XML_PATH),
        "jointNames": list(JOINT_NAME_ORDER),
        "desiredContactSiteNames": list(DESIRED_CONTACT_SITE_NAMES),
        "undesiredContactBodyPatterns": list(UNDESIRED_CONTACT_BODY_PATTERNS),
        "retainedBodies": retained_bodies,
        "mergedLinks": merged_links,
        "contractSites": {
            site_name: {
                "body": site_name.removesuffix("_contact_site"),
                "rule": "zero_transform",
                "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
            for site_name in DESIRED_CONTACT_SITE_NAMES
        },
    }


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _fmt_number(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        safe = value.replace("'", "''")
        return f"'{safe}'"
    raise TypeError(f"Unsupported scalar type: {type(value)!r}")


def _yaml_dump(value: object, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def _render_source_map(source_map: dict[str, object]) -> str:
    return "\n".join(_yaml_dump(source_map) + [""])


def build_hexapedal_assets() -> tuple[str, str, HexapedalModel]:
    urdf_path, mesh_dir = resolve_source_asset_paths()
    parsed = _parse_urdf(urdf_path)
    bodies, mesh_assets, _provenance = _build_body_specs(parsed)
    source_map = _source_map_dict(bodies, urdf_path)
    xml_text = _render_xml(parsed, bodies, mesh_assets, mesh_dir)
    source_map_text = _render_source_map(source_map)
    metadata = HexapedalModel(
        xml_path=XML_PATH,
        source_map_path=SOURCE_MAP_PATH,
        urdf_path=urdf_path,
        joint_names=JOINT_NAME_ORDER,
        desired_contact_site_names=DESIRED_CONTACT_SITE_NAMES,
        undesired_contact_body_patterns=UNDESIRED_CONTACT_BODY_PATTERNS,
        retained_bodies=bodies,
        source_map=source_map,
    )
    return xml_text, source_map_text, metadata


def write_hexapedal_assets() -> HexapedalModel:
    xml_text, source_map_text, metadata = build_hexapedal_assets()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    XML_PATH.write_text(xml_text, encoding="utf-8")
    SOURCE_MAP_PATH.write_text(source_map_text, encoding="utf-8")
    return metadata


def load_hexapedal_model(validate_generated: bool = True) -> HexapedalModel:
    xml_text, source_map_text, metadata = build_hexapedal_assets()
    if validate_generated:
        if not XML_PATH.exists() or not SOURCE_MAP_PATH.exists():
            raise FileNotFoundError("Generated MuJoCo assets are missing; run write_hexapedal_assets().")
        if XML_PATH.read_text(encoding="utf-8") != xml_text:
            raise RuntimeError(f"{XML_PATH} is stale; regenerate it with write_hexapedal_assets().")
        if SOURCE_MAP_PATH.read_text(encoding="utf-8") != source_map_text:
            raise RuntimeError(f"{SOURCE_MAP_PATH} is stale; regenerate it with write_hexapedal_assets().")
    return metadata


if __name__ == "__main__":
    write_hexapedal_assets()
