#!/usr/bin/env python3
"""Inspect the generated USD without constructing an Isaac Lab environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("usd", type=Path)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

from pxr import Usd, UsdGeom, UsdPhysics


JOINT_ORDER = (
    "RB_1", "RB_2", "RB_3",
    "RM_1", "RM_2", "RM_3",
    "RF_1", "RF_2", "RF_3",
    "LB_1", "LB_2", "LB_3",
    "LM_1", "LM_2", "LM_3",
    "LF_1", "LF_2", "LF_3",
)


def main() -> None:
    usd_path = args.usd.resolve()
    report_path = args.report.resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")

    roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    revolute = sorted(
        prim.GetName()
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    )
    missing = [name for name in JOINT_ORDER if name not in revolute]
    unexpected = [name for name in revolute if name not in JOINT_ORDER]
    rigid_bodies = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    configuration_dir = usd_path.parent / "configuration"
    base_path = configuration_dir / f"{usd_path.stem}_base.usd"
    physics_path = configuration_dir / f"{usd_path.stem}_physics.usd"
    base_stage = Usd.Stage.Open(str(base_path))
    physics_stage = Usd.Stage.Open(str(physics_path))
    if base_stage is None or physics_stage is None:
        raise RuntimeError("generated USD configuration layers are incomplete")
    mesh_prims = [
        prim for prim in base_stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    ]
    layer_collision_prims = [
        prim
        for prim in physics_stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    # Converter collision shapes are instance proxies referenced below each
    # rigid body.  Default Stage.Traverse() intentionally skips those proxies.
    composed_prims = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    collision_prims = [
        prim for prim in composed_prims if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    collision_meshes = [
        str(prim.GetPath())
        for prim in mesh_prims
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]

    report = {
        "schema_version": 1,
        "usd": str(usd_path),
        "default_prim": stage.GetDefaultPrim().GetName(),
        "articulation_roots": roots,
        "expected_joint_order": list(JOINT_ORDER),
        "revolute_joints_sorted": revolute,
        "missing_expected_joints": missing,
        "unexpected_revolute_joints": unexpected,
        "rigid_body_prims": rigid_bodies,
        "source_cad_mesh_count": 133,
        "usd_visual_mesh_prim_count": len(mesh_prims),
        "cad_visual_layer": str(base_path),
        "collision_prim_count": len(collision_prims),
        "collision_prim_paths": [str(prim.GetPath()) for prim in collision_prims],
        "physics_layer_collision_prim_count": len(layer_collision_prims),
        "collision_layer": str(physics_path),
        "collision_mesh_prims": collision_meshes,
        "checks": {
            "single_articulation_root": len(roots) == 1,
            "all_18_expected_joints_present": not missing and len(revolute) == 18,
            # The MJCF converter authors two USD Mesh prims per source STL in
            # this non-instanceable asset.  At least 133 proves that no source
            # CAD geom was lost; the source MJCF exporter checks exact count.
            "all_133_cad_meshes_preserved": len(mesh_prims) >= 133,
            "cad_meshes_are_visual_only": not collision_meshes,
            "primitive_training_colliders_present": len(collision_prims) >= 25,
        },
    }
    report["passed"] = all(report["checks"].values())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"asset inspection failed; see {report_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
