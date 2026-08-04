from __future__ import annotations

"""Preview candidate neutral standing poses without touching training code."""

import argparse
import json
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from hexapod_mjx.model import LEG_NAMES, STAND_POSE, _leg_support_surface_world_z, _stable_root_height_from_foot_body_z, estimate_standing_root_height, load_hexapod_model, load_hexapod_visual_model, repo_root_from






GROUP_TO_LEGS = {
    "front": ("LF", "RF"),
    "mid": ("LM", "RM"),
    "rear": ("LB", "RB"),
}
GROUP_TO_LEFT_LEG = {
    "front": "LF",
    "mid": "LM",
    "rear": "LB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a candidate MJX stand pose as PNG or in the MuJoCo viewer.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--visual-style", choices=("mesh", "simplified"), default="mesh", help="Use the original mesh visuals or the simplified training model.")
    parser.add_argument("--q1", type=float, default=None, help="Base value for all left-side *_1 joints.")
    parser.add_argument("--q2", type=float, default=None, help="Base value for all left-side *_2 joints.")
    parser.add_argument("--q3", type=float, default=None, help="Base value for all left-side *_3 joints.")
    for group in ("front", "mid", "rear"):
        parser.add_argument(f"--{group}-q1", type=float, default=None)
        parser.add_argument(f"--{group}-q2", type=float, default=None)
        parser.add_argument(f"--{group}-q3", type=float, default=None)
    parser.add_argument("--viewer", action="store_true", help="Open a live passive viewer instead of saving a PNG.")
    parser.add_argument("--save-to-model", action="store_true", help="Persist the resolved STAND_POSE into SW/mjx/hexapod_mjx/model.py.")
    parser.add_argument("--root-height", type=float, default=None, help="Override the resolved floating-base height directly.")
    parser.add_argument("--root-height-offset", type=float, default=0.0, help="Add an offset on top of the auto-resolved floating-base height.")
    parser.add_argument("--output-image-path", type=str, default="/tmp/stand_pose_preview.png")
    parser.add_argument("--output-json-path", type=str, default="/tmp/stand_pose_preview.json")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=1.45)
    parser.add_argument("--camera-azimuth", type=float, default=115.0)
    parser.add_argument("--camera-elevation", type=float, default=-10.0)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.12)
    return parser.parse_args()




def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _clamp_render_size(model: mujoco.MjModel, width: int, height: int) -> tuple[int, int]:
    return min(width, int(model.vis.global_.offwidth)), min(height, int(model.vis.global_.offheight))


def _current_left_group_values() -> dict[str, tuple[float, float, float]]:
    values: dict[str, tuple[float, float, float]] = {}
    for group, leg in GROUP_TO_LEFT_LEG.items():
        values[group] = (
            float(STAND_POSE[f"{leg}_1"]),
            float(STAND_POSE[f"{leg}_2"]),
            float(STAND_POSE[f"{leg}_3"]),
        )
    return values


def _resolve_left_group_values(args: argparse.Namespace) -> dict[str, tuple[float, float, float]]:
    values = _current_left_group_values()
    for group in ("front", "mid", "rear"):
        q1, q2, q3 = values[group]
        q1 = args.q1 if args.q1 is not None else q1
        q2 = args.q2 if args.q2 is not None else q2
        q3 = args.q3 if args.q3 is not None else q3
        group_q1 = getattr(args, f"{group}_q1")
        group_q2 = getattr(args, f"{group}_q2")
        group_q3 = getattr(args, f"{group}_q3")
        if group_q1 is not None:
            q1 = group_q1
        if group_q2 is not None:
            q2 = group_q2
        if group_q3 is not None:
            q3 = group_q3
        values[group] = (float(q1), float(q2), float(q3))
    return values


def _build_pose(values: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    pose: dict[str, float] = {}
    for group, (left_leg, right_leg) in GROUP_TO_LEGS.items():
        left_q1, left_q2, left_q3 = values[group]
        pose[f"{left_leg}_1"] = left_q1
        pose[f"{left_leg}_2"] = left_q2
        pose[f"{left_leg}_3"] = left_q3
        pose[f"{right_leg}_1"] = -left_q1
        pose[f"{right_leg}_2"] = -left_q2
        pose[f"{right_leg}_3"] = -left_q3
    return pose


def _configure_camera(camera: mujoco.MjvCamera, args: argparse.Namespace) -> None:
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.lookat[:] = [args.lookat_x, args.lookat_y, args.lookat_z]


def _foot_samples(bundle, data: mujoco.MjData) -> tuple[list[float], list[float]]:
    root_body_id = mujoco.mj_name2id(bundle.model, mujoco.mjtObj.mjOBJ_BODY, "hexapod_root")
    root_pos = data.xpos[root_body_id].copy()
    root_rot = data.xmat[root_body_id].reshape(3, 3).copy()
    foot_body_z: list[float] = []
    foot_world_z: list[float] = []
    for leg in LEG_NAMES:
        foot_geom_id = mujoco.mj_name2id(bundle.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_motor_horn_3_1_contact")
        foot_world = data.geom_xpos[foot_geom_id]
        foot_body = root_rot.T @ (foot_world - root_pos)
        foot_body_z.append(float(foot_body[2]))
        foot_world_z.append(_leg_support_surface_world_z(bundle.model, data, leg))
    return foot_body_z, foot_world_z



def _build_pose_data(bundle, pose: dict[str, float], args: argparse.Namespace) -> tuple[mujoco.MjData, float, float, list[float]]:
    data = mujoco.MjData(bundle.model)
    data.qpos[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for idx, joint_name in enumerate(bundle.joint_names):
        data.qpos[bundle.joint_qpos_adr[idx]] = pose[joint_name]
    mujoco.mj_forward(bundle.model, data)
    foot_body_z, foot_world_z = _foot_samples(bundle, data)
    auto_root_height = _stable_root_height_from_foot_body_z(np.asarray(foot_world_z, dtype=np.float64))
    base_root_height = float(estimate_standing_root_height(bundle))
    resolved_root_height = float(args.root_height) if args.root_height is not None else (base_root_height + float(args.root_height_offset))
    data.qpos[2] = resolved_root_height
    mujoco.mj_forward(bundle.model, data)
    return data, auto_root_height, resolved_root_height, foot_body_z




def _payload(repo_root: Path, pose: dict[str, float], auto_root_height: float, resolved_root_height: float, foot_body_z: list[float], args: argparse.Namespace) -> dict[str, object]:
    return {
        "repo_root": str(repo_root),
        "visual_style": args.visual_style,
        "auto_root_height": auto_root_height,
        "root_height_offset": float(args.root_height_offset),
        "root_height_override": args.root_height,
        "reset_root_height": resolved_root_height,
        "foot_body_z": foot_body_z,
        "stand_pose": pose,
        "camera": {
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
            "lookat": [args.lookat_x, args.lookat_y, args.lookat_z],
            "width": args.width,
            "height": args.height,
        },
        "python_block": "STAND_POSE = " + json.dumps(pose, indent=4, ensure_ascii=False),
    }


def _save_pose_to_model(repo_root: Path, pose: dict[str, float]) -> Path:
    model_path = repo_root / "SW" / "mjx" / "hexapod_mjx" / "model.py"
    text = model_path.read_text(encoding="utf-8")
    replacement = "STAND_POSE = " + json.dumps(pose, indent=4, ensure_ascii=False)
    updated, count = re.subn(r"STAND_POSE = \{.*?\n\}", replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Could not find STAND_POSE block in {model_path}")
    model_path.write_text(updated, encoding="utf-8")
    return model_path




def main() -> None:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)
    bundle = load_hexapod_visual_model(repo_root) if args.visual_style == "mesh" else load_hexapod_model(repo_root)
    values = _resolve_left_group_values(args)
    pose = _build_pose(values)
    data, auto_root_height, resolved_root_height, foot_body_z = _build_pose_data(bundle, pose, args)
    payload = _payload(repo_root, pose, auto_root_height, resolved_root_height, foot_body_z, args)
    if args.save_to_model:
        model_path = _save_pose_to_model(repo_root, pose)
        payload["saved_model_path"] = str(model_path)

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    output_json_path = _resolve_path(repo_root, args.output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.viewer:
        with mujoco.viewer.launch_passive(bundle.model, data) as viewer:
            _configure_camera(viewer.cam, args)
            viewer.sync()
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 60.0)
        return

    output_image_path = _resolve_path(repo_root, args.output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = _clamp_render_size(bundle.model, args.width, args.height)
    renderer = mujoco.Renderer(bundle.model, height=height, width=width)
    try:
        camera = mujoco.MjvCamera()
        _configure_camera(camera, args)
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        Image.fromarray(pixels).save(output_image_path)
    finally:
        renderer.close()

    print(f"render_size: {args.width}x{args.height} -> {width}x{height}")
    print(f"preview_image: {output_image_path}")
    print(f"preview_json: {output_json_path}")


if __name__ == "__main__":
    main()
