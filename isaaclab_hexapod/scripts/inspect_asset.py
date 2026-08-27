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

from pxr import Usd, UsdPhysics


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
        "checks": {
            "single_articulation_root": len(roots) == 1,
            "all_18_expected_joints_present": not missing and len(revolute) == 18,
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
