#!/usr/bin/env python3
"""Prepare the ROS-exported hexapod URDF for standalone MuJoCo loading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URDF = REPO_ROOT / "HW/urdf/urdf/HEXAPEDAL_URDF.xacro"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated/hexapod.urdf"
PACKAGE_PREFIX = "package://HEXAPEDAL_URDF_description/"
XACRO_INCLUDE = "{http://www.ros.org/wiki/xacro}include"


def _has_positive_inertia(inertial: ET.Element) -> bool:
    mass = inertial.find("mass")
    tensor = inertial.find("inertia")
    if mass is None or tensor is None or float(mass.get("value", "0")) <= 0:
        return False

    ixx = float(tensor.get("ixx", "0"))
    iyy = float(tensor.get("iyy", "0"))
    izz = float(tensor.get("izz", "0"))
    ixy = float(tensor.get("ixy", "0"))
    ixz = float(tensor.get("ixz", "0"))
    iyz = float(tensor.get("iyz", "0"))
    minor_2 = ixx * iyy - ixy * ixy
    determinant = (
        ixx * iyy * izz
        + 2 * ixy * ixz * iyz
        - ixx * iyz * iyz
        - iyy * ixz * ixz
        - izz * ixy * ixy
    )
    return ixx > 0 and minor_2 > 0 and determinant > 0


def prepare_urdf(source: Path, output: Path) -> Path:
    """Create a plain URDF with mesh paths relative to the generated file."""
    tree = ET.parse(source)
    root = tree.getroot()

    # The source uses xacro only for top-level ROS/Gazebo includes. MuJoCo does
    # not need those blocks, so the geometry can be made standalone without ROS.
    for include in list(root.findall(XACRO_INCLUDE)):
        root.remove(include)

    material = ET.Element("material", {"name": "silver"})
    ET.SubElement(material, "color", {"rgba": "0.70 0.70 0.72 1.0"})
    root.insert(0, material)

    mesh_root = REPO_ROOT / "HW/urdf"
    relative_mesh_root = Path(os.path.relpath(mesh_root, start=output.parent))
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(PACKAGE_PREFIX):
            relative_name = filename.removeprefix(PACKAGE_PREFIX)
            mesh.set("filename", (relative_mesh_root / relative_name).as_posix())

    # The CAD exporter emitted zero/singular inertia tensors for decorative
    # fixed links. MuJoCo rejects those tensors; removing them lets it infer a
    # valid inertia from the attached mesh when needed.
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None and not _has_positive_inertia(inertial):
            link.remove(inertial)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = prepare_urdf(args.source.resolve(), args.output.resolve())
    print(output)


if __name__ == "__main__":
    main()
