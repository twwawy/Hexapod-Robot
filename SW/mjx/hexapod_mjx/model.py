from __future__ import annotations

"""Model-loading utilities for the Hexapod MJX experiments.

The original repository stores a detailed URDF meant for CAD-faithful robot
inspection. That is useful for design work, but much heavier than what we want
for the first MJX locomotion experiment. This module therefore prepares a
training-oriented MuJoCo model with three explicit goals:

1. keep the kinematic structure and joint ordering faithful,
2. make contact geometry cheap and stable enough for batched simulation,
3. package every indexing detail once so the training code can stay readable.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

# Alternating tripod gait means three legs move together while the other three
# provide support. Using leg names directly keeps this convention easy to audit
# against the physical robot instead of hiding it behind integer IDs.
TRIPOD_A = {"LF", "RM", "LB"}
TRIPOD_B = {"RF", "LM", "RB"}
LEG_NAMES = ("LF", "LM", "LB", "RF", "RM", "RB")

# The original URDF contains many mesh collisions. Those are expensive and can
# be numerically noisy for locomotion, so training uses a hybrid model:
# - keep the original mesh geoms as visual-only geometry,
# - add one coarse collision box for the body,
# - add explicit foot-contact spheres for ground interaction.
#
# This keeps the learning physics simple while making preview/training renders
# look like the physical robot instead of a box-and-stick proxy.
BASE_COLLISION_HALF_SIZE = (0.19, 0.11, 0.03)
BASE_COLLISION_POS = (0.0, 0.0, 0.025)
FOOT_COLLISION_RADIUS = 0.018
LEG_FRAME_RADIUS = 0.0075
LEFT_FRAME_RGBA = "0.25 0.70 0.95 1"
RIGHT_FRAME_RGBA = "0.95 0.65 0.25 1"

DEFAULT_STAND_ROOT_HEIGHT = 0.06
FLOOR_VISUAL_HALF_SIZE = (1.15, 1.15, 0.015)
FLOOR_VISUAL_POS = (0.0, 0.0, -0.015)

# Standing pose used both at reset and as the sinusoid center point.
# Suffix ``_1/_2/_3`` corresponds to the three actuated joints in each leg.
#
# ``*_1`` is the body-nearest yaw/sweep joint, so a photo-like neutral stance
# needs front/mid/rear legs to use different values instead of one shared sweep.
# ``*_2`` and ``*_3`` are the two pitch joints that fold the leg downward.
#
# The reference hardware photo shows three consistent characteristics we want to
# preserve in the default pose:
# 1. the body sits above the legs,
# 2. the front and rear legs splay fore/aft while the middle legs stay lateral,
# 3. the pitch joints fold below the body instead of arching up over it.
STAND_POSE = {
    "LF_1": -0.7,
    "LF_2": 0.1,
    "LF_3": -1.0,
    "RF_1": 0.7,
    "RF_2": -0.1,
    "RF_3": 1.0,
    "LM_1": -0.4,
    "LM_2": 0.1,
    "LM_3": -1.0,
    "RM_1": 0.4,
    "RM_2": -0.1,
    "RM_3": 1.0,
    "LB_1": 0.7,
    "LB_2": 0.1,
    "LB_3": -1.0,
    "RB_1": -0.7,
    "RB_2": -0.1,
    "RB_3": 1.0
}



@dataclass(frozen=True)
class HexapodModelBundle:
    """Everything downstream code needs after one-time model preparation.

    The bundle deliberately stores index arrays such as ``joint_qpos_adr`` and
    ``joint_dof_adr``. MuJoCo models are compact and fast, but the indexing is
    not self-explanatory. Precomputing those lookups once removes repetitive,
    easy-to-get-wrong boilerplate from rollout and visualization code.
    """

    repo_root: Path
    generated_mjcf_path: Path
    model: mujoco.MjModel
    mjx_model: mjx.Model
    joint_names: tuple[str, ...]
    joint_qpos_adr: np.ndarray
    joint_dof_adr: np.ndarray
    default_joint_pose: jnp.ndarray
    joint_group_index: jnp.ndarray
    tripod_phase_offset: jnp.ndarray


def repo_root_from(path: str | Path) -> Path:
    """Resolve any path inside the repo back to the repository root."""
    path = Path(path).resolve()
    if path.is_dir() and (path / "HW").exists() and (path / "SW").exists():
        return path
    for parent in [path, *path.parents]:
        if (parent / "HW").exists() and (parent / "SW").exists():
            return parent
    raise FileNotFoundError(f"Could not locate Hexapod-Robot repo root from: {path}")
def _stable_root_height_from_foot_body_z(foot_z: np.ndarray) -> float:
    """Place the root high enough that the lowest modeled foot touches the floor.

    The caller should provide foot/sample Z values measured with the floating base
    still at world Z=0. The required lift is then simply `-min(z)`, clamped at
    zero so already-above-ground poses stay unchanged.
    """
    return float(max(0.0, -float(np.min(foot_z))))



def estimate_standing_root_height(bundle: HexapodModelBundle) -> float:
    """Infer a stable reset height and never let it drop below the 0.06 floor."""
    data = mujoco.MjData(bundle.model)
    data.qpos[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[bundle.joint_qpos_adr] = np.asarray(bundle.default_joint_pose, dtype=np.float64)
    mujoco.mj_forward(bundle.model, data)

    foot_world_z = []
    for leg in LEG_NAMES:
        foot_geom_id = mujoco.mj_name2id(bundle.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_motor_horn_3_1_contact")
        foot_world_z.append(float(data.geom_xpos[foot_geom_id][2]))
    auto_height = _stable_root_height_from_foot_body_z(np.asarray(foot_world_z, dtype=np.float64))
    return float(max(DEFAULT_STAND_ROOT_HEIGHT, auto_height))



def _clean_urdf_text(repo_root: Path) -> str:
    """Load the URDF and strip tags/path styles that MuJoCo does not need.

    Two cleanups happen here:
    - ROS/Gazebo-only tags are removed because they do not help MuJoCo parsing.
    - ``package://...`` mesh URIs are rewritten to absolute filesystem paths.
    """
    urdf_path = repo_root / "HW" / "urdf" / "urdf" / "HEXAPEDAL_URDF.urdf"
    mesh_root = (repo_root / "HW" / "urdf" / "meshes").resolve()
    text = urdf_path.read_text(encoding="utf-8")
    text = re.sub(r"\s*<transmission\b.*?</transmission>\s*", "\n", text, flags=re.S)
    text = re.sub(r"\s*<gazebo\b.*?</gazebo>\s*", "\n", text, flags=re.S)
    text = text.replace("package://HEXAPEDAL_URDF_description/meshes/", mesh_root.as_posix() + "/")
    return text


def _iter_parent_child(element: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """Return every ``(parent, child)`` pair in a subtree.

    ``xml.etree`` elements do not store back-pointers to their parents. We need
    parent references when removing or replacing nested geometry elements.
    """
    pairs: list[tuple[ET.Element, ET.Element]] = []
    for child in list(element):
        pairs.append((element, child))
        pairs.extend(_iter_parent_child(child))
    return pairs


def _find_body(root: ET.Element, body_name: str) -> ET.Element:
    """Find a MuJoCo body by name and fail loudly if the expected name is gone."""
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    raise KeyError(f"Could not find body '{body_name}' while simplifying MJCF.")


def _leg_prefix(name: str) -> str | None:
    for leg in (*sorted(TRIPOD_A), *sorted(TRIPOD_B)):
        if name.startswith(f"{leg}_"):
            return leg
    return None



def _frame_rgba_for_leg(leg: str) -> str:
    return LEFT_FRAME_RGBA if leg.startswith("L") else RIGHT_FRAME_RGBA



def _add_visual_leg_frames(robot_body: ET.Element, foot_contacts: dict[str, str]) -> None:
    """Add non-colliding capsule geoms so the simplified robot is still readable.

    The training model intentionally strips mesh detail, but a pure body box plus
    foot spheres makes it hard to see whether a leg is actually swinging. These
    frame capsules are visual-only (`contype=conaffinity=0`, `mass=0`) so they
    do not change contacts or dynamics.
    """
    parents = [robot_body, *[body for body in robot_body.iter("body") if body is not robot_body]]
    for parent in parents:
        parent_leg = _leg_prefix(parent.get("name", "")) if parent is not robot_body else None
        for child in parent.findall("body"):
            child_name = child.get("name", "")
            child_leg = _leg_prefix(child_name)
            if child_leg is None:
                continue
            if parent_leg is not None and child_leg != parent_leg:
                continue
            child_pos = child.get("pos", "0 0 0")
            ET.SubElement(
                parent,
                "geom",
                {
                    "name": f"{child_name}_frame",
                    "type": "capsule",
                    "fromto": f"0 0 0 {child_pos}",
                    "size": f"{LEG_FRAME_RADIUS:.4f}",
                    "rgba": _frame_rgba_for_leg(child_leg),
                    "contype": "0",
                    "conaffinity": "0",
                    "mass": "0",
                },
            )

    for body_name, pos in foot_contacts.items():
        leg = _leg_prefix(body_name)
        if leg is None:
            continue
        ET.SubElement(
            _find_body(robot_body, body_name),
            "geom",
            {
                "name": f"{body_name}_foot_frame",
                "type": "capsule",
                "fromto": f"0 0 0 {pos}",
                "size": f"{LEG_FRAME_RADIUS:.4f}",
                "rgba": _frame_rgba_for_leg(leg),
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
            },
        )


def _format_pos(vec: np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in vec)


def _geom_rotation_matrix(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(model.geom_quat[geom_id], dtype=np.float64))
    return mat.reshape(3, 3)


def _infer_foot_contacts_from_model(model: mujoco.MjModel) -> dict[str, str]:
    """Infer contact points from the actual foot mesh geometry, not mesh origins."""
    foot_contacts: dict[str, str] = {}
    for leg in LEG_NAMES:
        body_name: str | None = None
        body_vertices: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name is None or not mesh_name.startswith(f"{leg}_foot_"):
                continue

            geom_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
            if geom_body_name is None:
                continue
            body_name = geom_body_name

            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64)
            rot = _geom_rotation_matrix(model, geom_id)
            pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            body_vertices.append(mesh_vertices @ rot.T + pos)

        if body_name is None or not body_vertices:
            continue

        vertices = np.concatenate(body_vertices, axis=0)
        min_z = float(np.min(vertices[:, 2]))
        lowest_band = vertices[vertices[:, 2] <= min_z + 0.002]
        contact = np.array([
            float(np.mean(lowest_band[:, 0])),
            float(np.mean(lowest_band[:, 1])),
            min_z,
        ], dtype=np.float64)
        foot_contacts[body_name] = _format_pos(contact)

    if len(foot_contacts) != 6:
        raise RuntimeError(f"Expected 6 inferred foot contact points, found {len(foot_contacts)}.")
    return foot_contacts



def _add_foot_contact_geoms(robot_body: ET.Element, foot_contacts: dict[str, str], *, collidable: bool) -> None:
    for body_name, pos in foot_contacts.items():
        attrs = {
            "name": f"{body_name}_contact",
            "type": "sphere",
            "pos": pos,
            "size": f"{FOOT_COLLISION_RADIUS:.4f}",
            "rgba": "0.98 0.35 0.18 0.55",
        }
        if collidable:
            attrs.update({
                "friction": "1.8 0.02 0.01",
                "condim": "3",
            })
        else:
            attrs.update({
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
            })
        ET.SubElement(_find_body(robot_body, body_name), "geom", attrs)


def _simplify_training_geoms(robot_body: ET.Element, foot_contacts: dict[str, str]) -> None:
    """Keep original mesh visuals but replace their collisions with simple proxies."""

    # Leave URDF-expanded geoms in place for visualization, but make them
    # visual-only so training contacts come only from the coarse proxies below.
    for body in robot_body.iter("body"):
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("density", "0")

    ET.SubElement(
        robot_body,
        "geom",
        {
            "name": "base_collision",
            "type": "box",
            "pos": " ".join(f"{value:.4f}" for value in BASE_COLLISION_POS),
            "size": " ".join(f"{value:.4f}" for value in BASE_COLLISION_HALF_SIZE),
            "rgba": "0.58 0.64 0.72 0.18",
            "friction": "0.9 0.05 0.02",
            "condim": "3",
        },
    )

    _add_foot_contact_geoms(robot_body, foot_contacts, collidable=True)



def build_floating_base_mjcf(repo_root: str | Path, *, base_height: float = 0.22, simplified: bool = True) -> Path:
    """Generate a free-floating MJCF for either training or mesh-faithful preview."""
    repo_root = repo_root_from(repo_root)
    generated_dir = repo_root / "SW" / "mjx" / "artifacts"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_name = "hexapod_floating_base.xml" if simplified else "hexapod_floating_base_visual.xml"
    generated_path = generated_dir / generated_name

    clean_urdf = _clean_urdf_text(repo_root)
    with tempfile.TemporaryDirectory(prefix="hexapod_mjx_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        urdf_path = tmpdir_path / "hexapod_clean.urdf"
        saved_xml_path = tmpdir_path / "hexapod_saved.xml"
        urdf_path.write_text(clean_urdf, encoding="utf-8")

        # Let MuJoCo perform the URDF -> MJCF expansion once, then edit the
        # resulting XML tree. That is much simpler than reimplementing the full
        # conversion logic ourselves.
        urdf_model = mujoco.MjModel.from_xml_path(str(urdf_path))
        foot_contacts = _infer_foot_contacts_from_model(urdf_model)
        mujoco.mj_saveLastXML(str(saved_xml_path), urdf_model)

        tree = ET.parse(saved_xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("MuJoCo-exported XML is missing <worldbody>.")

        # Wrap the robot inside a new free-floating root body so the first gait
        # experiment can focus on whole-body balance without pinning the base.
        robot_body = ET.Element("body", {"name": "hexapod_root", "pos": f"0 0 {base_height:.4f}"})
        ET.SubElement(robot_body, "freejoint", {"name": "root_free"})

        for child in list(worldbody):
            worldbody.remove(child)
            robot_body.append(child)

        if simplified:
            _simplify_training_geoms(robot_body, foot_contacts)
        else:
            _add_foot_contact_geoms(robot_body, foot_contacts, collidable=False)

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "4 4 0.1",
                "rgba": "0.78 0.80 0.82 1",
                "friction": "1.2 0.05 0.02",
                "condim": "3",
            },
        )
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor_visual",
                "type": "box",
                "pos": " ".join(f"{value:.4f}" for value in FLOOR_VISUAL_POS),
                "size": " ".join(f"{value:.4f}" for value in FLOOR_VISUAL_HALF_SIZE),
                "rgba": "0.86 0.88 0.90 1",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
            },
        )
        worldbody.append(robot_body)

        tree.write(generated_path, encoding="unicode")

    return generated_path



def _load_hexapod_model(repo_root: str | Path, *, simplified: bool) -> HexapodModelBundle:
    """Load a generated MJCF and package all indexing metadata."""
    repo_root = repo_root_from(repo_root)
    mjcf_path = build_floating_base_mjcf(repo_root, simplified=simplified)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mjx_model = mjx.put_model(model)

    joint_names: list[str] = []
    joint_qpos_adr: list[int] = []
    joint_dof_adr: list[int] = []
    joint_group_index: list[int] = []
    tripod_phase_offset: list[float] = []
    default_joint_pose: list[float] = []

    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name == "root_free":
            continue
        if name is None:
            continue

        joint_names.append(name)
        joint_qpos_adr.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_adr.append(int(model.jnt_dofadr[joint_id]))

        suffix = int(name.split("_")[-1])
        joint_group_index.append(suffix - 1)
        leg = name.split("_")[0]
        if leg not in TRIPOD_A and leg not in TRIPOD_B:
            raise KeyError(f"Unexpected leg prefix in joint name: {name}")
        tripod_phase_offset.append(0.0 if leg in TRIPOD_A else np.pi)
        default_joint_pose.append(STAND_POSE[name])

    return HexapodModelBundle(
        repo_root=repo_root,
        generated_mjcf_path=mjcf_path,
        model=model,
        mjx_model=mjx_model,
        joint_names=tuple(joint_names),
        joint_qpos_adr=np.asarray(joint_qpos_adr, dtype=np.int32),
        joint_dof_adr=np.asarray(joint_dof_adr, dtype=np.int32),
        default_joint_pose=jnp.asarray(default_joint_pose, dtype=jnp.float32),
        joint_group_index=jnp.asarray(joint_group_index, dtype=jnp.int32),
        tripod_phase_offset=jnp.asarray(tripod_phase_offset, dtype=jnp.float32),
    )


def load_hexapod_model(repo_root: str | Path) -> HexapodModelBundle:
    """Load the simplified training/visualization model."""
    return _load_hexapod_model(repo_root, simplified=True)


def load_hexapod_visual_model(repo_root: str | Path) -> HexapodModelBundle:
    """Load a mesh-faithful preview model that preserves the original visuals."""
    return _load_hexapod_model(repo_root, simplified=False)