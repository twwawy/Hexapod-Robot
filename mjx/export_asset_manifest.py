#!/usr/bin/env python3
"""Export the final MJX model as the versioned Isaac asset source of truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import mujoco
import numpy as np

from prepare_rl_scene import RL_SCENE_OUTPUT, _robot_body_ids, prepare_rl_scene


SCHEMA = "hexapod_mjx_asset_v1"
CANONICAL_LEGS = ("RB", "RM", "RF", "LB", "LM", "LF")
JOINT_ORDER = tuple(
    f"{leg}_{joint}" for leg in CANONICAL_LEGS for joint in (1, 2, 3)
)


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or ""


def _array(value: Any) -> list[Any]:
    result = np.asarray(value)
    if np.issubdtype(result.dtype, np.integer):
        return result.astype(np.int64).tolist()
    return result.astype(np.float64).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str], root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def build_manifest(repo_root: Path) -> dict[str, Any]:
    xml_path = prepare_rl_scene(RL_SCENE_OUTPUT).resolve()
    urdf_path = (xml_path.parent / "hexapod.urdf").resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    robot_ids = sorted(_robot_body_ids(model))

    bodies = []
    for body_id in robot_ids:
        parent_id = int(model.body_parentid[body_id])
        bodies.append(
            {
                "id": body_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "parent": _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
                "local_position": _array(model.body_pos[body_id]),
                "local_quaternion_wxyz": _array(model.body_quat[body_id]),
                "mass_kg": float(model.body_mass[body_id]),
                "inertial_com_position": _array(model.body_ipos[body_id]),
                "inertial_quaternion_wxyz": _array(model.body_iquat[body_id]),
                "diagonal_inertia_kg_m2": _array(model.body_inertia[body_id]),
            }
        )

    joints = []
    for joint_name in JOINT_ORDER:
        joint_id = model.joint(joint_name).id
        child_id = int(model.jnt_bodyid[joint_id])
        parent_id = int(model.body_parentid[child_id])
        joints.append(
            {
                "id": int(joint_id),
                "name": joint_name,
                "parent_body": _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id),
                "child_body": _name(model, mujoco.mjtObj.mjOBJ_BODY, child_id),
                "axis": _array(model.jnt_axis[joint_id]),
                "position": _array(model.jnt_pos[joint_id]),
                "limited": bool(model.jnt_limited[joint_id]),
                "range_rad": _array(model.jnt_range[joint_id]),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
            }
        )

    actuators = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_name = _name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        actuators.append(
            {
                "id": actuator_id,
                "name": actuator_name,
                "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "gain_parameters": _array(model.actuator_gainprm[actuator_id]),
                "bias_parameters": _array(model.actuator_biasprm[actuator_id]),
                "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
                "control_range": _array(model.actuator_ctrlrange[actuator_id]),
                "force_limited": bool(model.actuator_forcelimited[actuator_id]),
                "force_range_nm": _array(model.actuator_forcerange[actuator_id]),
                "gear": _array(model.actuator_gear[actuator_id]),
            }
        )

    collisions = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id not in robot_ids:
            continue
        collisions.append(
            {
                "id": geom_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "type": mujoco.mjtGeom(int(model.geom_type[geom_id])).name,
                "local_position": _array(model.geom_pos[geom_id]),
                "local_quaternion_wxyz": _array(model.geom_quat[geom_id]),
                "size": _array(model.geom_size[geom_id]),
                "friction": _array(model.geom_friction[geom_id]),
                "collision_type_mask": int(model.geom_contype[geom_id]),
                "collision_affinity_mask": int(model.geom_conaffinity[geom_id]),
            }
        )

    sites = []
    for site_id in range(model.nsite):
        body_id = int(model.site_bodyid[site_id])
        if body_id not in robot_ids:
            continue
        sites.append(
            {
                "id": site_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_SITE, site_id),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "local_position": _array(model.site_pos[site_id]),
                "local_quaternion_wxyz": _array(model.site_quat[site_id]),
            }
        )

    home_id = model.key("home").id
    home_qpos = np.asarray(model.key_qpos[home_id], dtype=np.float64)
    home_ctrl = np.asarray(model.key_ctrl[home_id], dtype=np.float64)
    joint_qpos = [int(model.joint(name).qposadr[0]) for name in JOINT_ORDER]
    actuator_ids = [model.actuator(f"{name}_position").id for name in JOINT_ORDER]
    status = _git(["status", "--short"], repo_root)

    return {
        "schema": SCHEMA,
        "source": {
            "git_commit": _git(["rev-parse", "HEAD"], repo_root),
            "git_dirty": bool(status),
            "git_status": status.splitlines(),
            "mujoco_version": mujoco.__version__,
            "xml_path": str(xml_path.relative_to(repo_root)),
            "xml_sha256": _sha256(xml_path),
            "urdf_path": str(urdf_path.relative_to(repo_root)),
            "urdf_sha256": _sha256(urdf_path),
        },
        "contract": {
            "canonical_leg_order": list(CANONICAL_LEGS),
            "canonical_joint_order": list(JOINT_ORDER),
            "root_home_position": _array(home_qpos[:3]),
            "root_home_quaternion_wxyz": _array(home_qpos[3:7]),
            "joint_home_position_rad": _array(home_qpos[joint_qpos]),
            "actuator_home_target_rad": _array(home_ctrl[actuator_ids]),
            "total_robot_mass_kg": float(
                sum(model.body_mass[body_id] for body_id in robot_ids)
            ),
        },
        "bodies": bodies,
        "joints": joints,
        "actuators": actuators,
        "collisions": collisions,
        "sites": sites,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "golden/asset_manifest_v1.json",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = build_manifest(repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
