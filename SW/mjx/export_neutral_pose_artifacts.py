from __future__ import annotations

"""Export the current neutral standing pose as a PNG + JSON artifact.

This is meant for fresh residual-RL runs after pose changes. It captures the
exact default pose and inferred reset height that the run starts from so later
reward comparisons have an auditable baseline image and metadata record.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from hexapod_mjx.model import STAND_POSE, estimate_standing_root_height, load_hexapod_model, load_hexapod_visual_model, repo_root_from



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Hexapod MJX neutral pose as PNG + JSON.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--visual-style", choices=("mesh", "simplified"), default="mesh")
    parser.add_argument("--output-image-path", type=str, required=True)
    parser.add_argument("--output-metadata-path", type=str, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-distance", type=float, default=1.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.08)
    return parser.parse_args()



def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def main() -> None:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)
    output_image_path = _resolve_repo_path(repo_root, args.output_image_path)
    output_metadata_path = _resolve_repo_path(repo_root, args.output_metadata_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_hexapod_visual_model(repo_root) if args.visual_style == "mesh" else load_hexapod_model(repo_root)


    reset_root_height = estimate_standing_root_height(bundle)

    data = mujoco.MjData(bundle.model)
    data.qpos[0:3] = np.array([0.0, 0.0, reset_root_height], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[bundle.joint_qpos_adr] = np.asarray(bundle.default_joint_pose, dtype=np.float64)
    mujoco.mj_forward(bundle.model, data)

    renderer = mujoco.Renderer(bundle.model, args.width, args.height)
    try:
        camera = mujoco.MjvCamera()
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        camera.lookat[:] = [args.lookat_x, args.lookat_y, args.lookat_z]
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        Image.fromarray(pixels).save(output_image_path)
    finally:
        renderer.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "visual_style": args.visual_style,
        "generated_mjcf": str(bundle.generated_mjcf_path),
        "reset_root_height": float(reset_root_height),
        "stand_pose": {name: float(value) for name, value in STAND_POSE.items()},
        "joint_order": list(bundle.joint_names),
        "default_joint_pose": [float(value) for value in np.asarray(bundle.default_joint_pose)],
        "image_path": str(output_image_path),
        "camera": {
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
            "lookat": [args.lookat_x, args.lookat_y, args.lookat_z],
            "width": args.width,
            "height": args.height,
        },
    }
    output_metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"neutral_pose_image: {output_image_path}")
    print(f"neutral_pose_metadata: {output_metadata_path}")
    print(f"reset_root_height: {reset_root_height:.6f}")


if __name__ == "__main__":
    main()
