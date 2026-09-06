#!/usr/bin/env python3
"""Open the full-CAD Hexapod on the latest MJX training terrain."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--terrain-level",
    type=int,
    default=None,
    help="Override the latest MJX terrain level recorded in the handoff.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Exit after N physics steps; 0 keeps the GUI open.",
)
parser.add_argument(
    "--report",
    type=Path,
    default=None,
    help="Optionally write a JSON scene/smoke-test report when the viewer exits.",
)
parser.add_argument(
    "--rtx-lidar",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Attach and visualize the Livox MID-360 RTX proxy (default: enabled).",
)
parser.add_argument(
    "--lidar-fov-vis",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw MID-360 direction/FOV guides (default: enabled).",
)
parser.add_argument(
    "--lidar-fov-vis-range",
    type=float,
    default=3.0,
    help="Radius of the visible FOV envelope in metres (default: 3).",
)
parser.add_argument(
    "--lidar-pointcloud-vis",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Draw rotating RTX returns (default: disabled; FOV envelope only).",
)
parser.add_argument(
    "--frame-markers",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also draw the LiDAR origin and six CAD foot support points.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from hexapod_isaaclab.assets import HEXAPOD_CFG
from hexapod_isaaclab.assets.joint_contract import (
    FOOT_BODY_ORDER,
    FOOT_SITE_LOCAL_POS,
)
from hexapod_isaaclab.contracts import load_training_handoff
from hexapod_isaaclab.sensors.mid360 import (
    MID360_MAX_RANGE_M,
    MID360_MIN_RANGE_M,
    MID360_POSITION_BODY,
    MID360_ROTATION_BODY_WXYZ,
    MID360_VERTICAL_FOV_DEG,
    create_rtx_mid360,
    initialize_rtx_mid360,
)


def _terrain_spec(handoff: dict, level: int) -> dict:
    for spec in handoff["contracts"]["terrain_levels"]:
        if spec["level"] == level:
            return spec
    raise ValueError(f"terrain level {level} is not in the MJX handoff")


def _spawn_static_box(
    path: str,
    size: tuple[float, float, float],
    position,
    orientation=(1.0, 0.0, 0.0, 0.0),
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.30, 0.39, 0.48)
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.1,
            dynamic_friction=1.1,
            restitution=0.0,
        ),
    )
    cfg.func(path, cfg, translation=position, orientation=orientation)


def _spawn_terrain(handoff: dict, spec: dict) -> None:
    # Do not use GroundPlaneCfg here: Isaac Lab 2.3 resolves its visual USD
    # from NVIDIA S3, which makes this local viewer depend on network/cache
    # availability.  A local static slab is deterministic and self-contained.
    _spawn_static_box(
        "/World/Ground",
        (20.0, 20.0, 0.10),
        (0.0, 0.0, -0.05),
    )
    geometry = handoff["contracts"]["terrain_geometry"]
    width = 2.0 * geometry["half_width_m"]
    if spec["kind"] == "stairs":
        depth = geometry["stair_depth_m"]
        start_x = geometry["start_x_m"]
        riser = spec["stair_riser_m"]
        for index in range(spec["stair_count"]):
            height = riser * (index + 1)
            _spawn_static_box(
                f"/World/Terrain/Stair_{index + 1}",
                (depth, width, height),
                (start_x + depth * (index + 0.5), 0.0, height / 2.0),
            )
        stair_end = start_x + spec["stair_count"] * depth
        plateau_depth = geometry["plateau_depth_m"]
        _spawn_static_box(
            "/World/Terrain/Plateau",
            (plateau_depth, width, spec["final_height_m"]),
            (
                stair_end + plateau_depth / 2.0,
                0.0,
                spec["final_height_m"] / 2.0,
            ),
        )
    elif spec["kind"] == "ramp":
        length = geometry["ramp_length_m"]
        angle = math.radians(spec["slope_degrees"])
        thickness = 0.05
        _spawn_static_box(
            "/World/Terrain/Ramp",
            (length, width, thickness),
            (
                geometry["start_x_m"] + length / 2.0,
                0.0,
                spec["final_height_m"] / 2.0,
            ),
            (math.cos(angle / 2.0), 0.0, -math.sin(angle / 2.0), 0.0),
        )
    elif spec["kind"] == "rough":
        print(
            "[WARN] Rough heightfield visualization is not implemented in this "
            "viewer; showing the full-mesh robot on the base plane."
        )


def _print_lidar_health(lidar) -> None:
    """Print numerical proof that RTX rays are returning geometry hits."""

    name = "IsaacExtractRTXSensorPointCloudNoAccumulator"
    payload = lidar.get_current_frame().get(name)
    if not isinstance(payload, dict) or "data" not in payload:
        print("[MID360] warming up: no point-cloud frame yet", flush=True)
        return
    points = np.asarray(payload["data"])
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        print("[MID360] FAIL: returns=0 (no geometry hits)", flush=True)
        return
    ranges = np.linalg.norm(points, axis=1)
    ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]
    if ranges.size == 0:
        print("[MID360] FAIL: all returned points are invalid", flush=True)
        return
    print(
        "[MID360] OK "
        f"returns={ranges.size} "
        f"range_m[min/median/max]="
        f"{ranges.min():.3f}/{np.median(ranges):.3f}/{ranges.max():.3f}",
        flush=True,
    )


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hamilton product for WXYZ quaternions."""

    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a unit WXYZ quaternion."""

    xyz = quat[1:]
    return vector + 2.0 * (
        quat[0] * np.cross(xyz, vector)
        + np.cross(xyz, np.cross(xyz, vector))
    )


def _draw_mid360_fov(
    draw,
    root_state_w: torch.Tensor,
    display_range: float,
    foot_body_pos_w: torch.Tensor | None = None,
    foot_body_quat_w: torch.Tensor | None = None,
) -> None:
    """Draw only the MID-360 FOV envelope plus optional frame markers."""

    state = root_state_w[0].detach().cpu().numpy()
    root_position = state[:3].astype(np.float64)
    root_rotation = state[3:7].astype(np.float64)
    sensor_rotation = _quat_multiply(
        root_rotation, np.asarray(MID360_ROTATION_BODY_WXYZ)
    )
    sensor_origin = root_position + _quat_rotate(
        root_rotation, np.asarray(MID360_POSITION_BODY)
    )

    line_starts: list[tuple[float, float, float]] = []
    line_ends: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    sizes: list[float] = []

    envelope = (0.0, 0.9, 1.0, 0.85)
    lidar_origin_color = (1.0, 0.0, 1.0, 1.0)
    foot_site_color = (1.0, 0.25, 0.05, 1.0)

    def point(radius: float, azimuth_deg: float, elevation_deg: float):
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        local = radius * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        return sensor_origin + _quat_rotate(sensor_rotation, local)

    def add_line(start, end, color, size=1.5):
        line_starts.append(tuple(float(value) for value in start))
        line_ends.append(tuple(float(value) for value in end))
        colors.append(color)
        sizes.append(size)

    # FOV boundary only: lower/upper limits and a sparse outer-range cage.
    for elevation in MID360_VERTICAL_FOV_DEG:
        for azimuth in range(0, 360, 5):
            add_line(
                point(display_range, azimuth, elevation),
                point(display_range, azimuth + 5, elevation),
                envelope,
                1.0,
            )
    elevation_min, elevation_max = MID360_VERTICAL_FOV_DEG
    for azimuth in range(0, 360, 30):
        for elevation in np.arange(elevation_min, elevation_max, 2.0):
            add_line(
                point(display_range, azimuth, float(elevation)),
                point(
                    display_range,
                    azimuth,
                    min(float(elevation + 2.0), elevation_max),
                ),
                envelope,
                1.0,
            )
        for elevation in MID360_VERTICAL_FOV_DEG:
            add_line(
                point(MID360_MIN_RANGE_M, azimuth, elevation),
                point(display_range, azimuth, elevation),
                envelope,
                1.0,
            )

    def add_cross(position, color, half_size=0.025, size=3.0):
        for axis in np.eye(3):
            add_line(
                position - half_size * axis,
                position + half_size * axis,
                color,
                size,
            )

    if args.frame_markers:
        add_cross(sensor_origin, lidar_origin_color, half_size=0.035, size=3.0)
        if foot_body_pos_w is not None and foot_body_quat_w is not None:
            positions = foot_body_pos_w.detach().cpu().numpy()
            rotations = foot_body_quat_w.detach().cpu().numpy()
            for position, rotation, local_site in zip(
                positions, rotations, FOOT_SITE_LOCAL_POS, strict=True
            ):
                site_position = position + _quat_rotate(
                    rotation, np.asarray(local_site)
                )
                add_cross(site_position, foot_site_color)
    draw.clear_lines()
    draw.draw_lines(line_starts, line_ends, colors, sizes)


def main() -> None:
    handoff = load_training_handoff()
    level = (
        handoff["latest_attempt"]["terrain_level"]
        if args.terrain_level is None
        else args.terrain_level
    )
    spec = _terrain_spec(handoff, level)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.0025, device=args.device)
    )
    print("[INFO] Simulation context created", flush=True)
    sim.set_camera_view([3.2, -3.0, 2.2], [1.1, 0.0, 0.5])
    _spawn_terrain(handoff, spec)
    light = sim_utils.DomeLightCfg(
        intensity=2500.0, color=(0.78, 0.78, 0.78)
    )
    light.func("/World/Light", light)
    print("[INFO] Terrain and light created", flush=True)

    robot_cfg = HEXAPOD_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(robot_cfg)
    print("[INFO] Full-CAD robot USD spawned", flush=True)
    lidar = None
    lidar_prim_path = None
    if args.rtx_lidar:
        lidar_prim_path = (
            "/World/Robot/hexapod/hexapod/Sensors/LivoxMID360"
        )
        lidar = create_rtx_mid360(lidar_prim_path)
        print("[INFO] MID-360 RTX prim created", flush=True)
    sim.reset()
    print("[INFO] Simulation reset complete", flush=True)
    if lidar is not None:
        initialize_rtx_mid360(
            lidar,
            debug_vis=args.lidar_pointcloud_vis and not args.headless,
        )
        print("[INFO] MID-360 point-cloud output initialized", flush=True)

    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    robot.reset()
    foot_body_ids, foot_body_names = robot.find_bodies(
        list(FOOT_BODY_ORDER), preserve_order=True
    )
    if tuple(foot_body_names) != FOOT_BODY_ORDER:
        raise RuntimeError(
            f"foot marker contract mismatch: {foot_body_names}"
        )

    print(
        "[INFO] Full-CAD Hexapod scene ready | "
        f"terrain={level}:{spec['name']} | "
        f"latest_run={handoff['latest_attempt']['run_id']} | "
        "MJX_WEIGHT_AUTOLOAD=disabled"
        f" | RTX_MID360={'enabled' if lidar is not None else 'disabled'}"
    )
    count = 0
    next_lidar_report = time.monotonic() + 1.0
    next_fov_update = 0.0
    fov_draw = None
    if lidar is not None and args.lidar_fov_vis and not args.headless:
        if not (
            MID360_MIN_RANGE_M
            <= args.lidar_fov_vis_range
            <= MID360_MAX_RANGE_M
        ):
            raise ValueError(
                "--lidar-fov-vis-range must be within the configured "
                f"MID-360 range [{MID360_MIN_RANGE_M}, {MID360_MAX_RANGE_M}] m"
            )
        from isaacsim.util.debug_draw import _debug_draw

        fov_draw = _debug_draw.acquire_debug_draw_interface()
        marker_legend = (
            ", magenta=optical origin, orange=CAD foot support points"
            if args.frame_markers
            else ""
        )
        print(
            f"[MID360 FOV] cyan=coverage envelope{marker_legend} | "
            f"shown={args.lidar_fov_vis_range:.1f}m, "
            f"sensor_max={MID360_MAX_RANGE_M:.1f}m, "
            f"rotating_returns={'on' if args.lidar_pointcloud_vis else 'off'}",
            flush=True,
        )
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running() and (args.steps == 0 or count < args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        count += 1
        if lidar is not None and time.monotonic() >= next_lidar_report:
            _print_lidar_health(lidar)
            next_lidar_report = time.monotonic() + 1.0
        if fov_draw is not None and time.monotonic() >= next_fov_update:
            _draw_mid360_fov(
                fov_draw,
                robot.data.root_state_w,
                args.lidar_fov_vis_range,
                robot.data.body_pos_w[0, foot_body_ids],
                robot.data.body_quat_w[0, foot_body_ids],
            )
            next_fov_update = time.monotonic() + 0.1
    if args.report is not None:
        report = {
            "schema_version": 1,
            "passed": True,
            "terrain_level": level,
            "terrain_name": spec["name"],
            "latest_run": handoff["latest_attempt"]["run_id"],
            "full_mesh_usd": str(robot_cfg.spawn.usd_path),
            "completed_physics_steps": count,
            "device": args.device,
            "mjx_weight_autoload": False,
            "rtx_mid360_enabled": lidar is not None,
            "rtx_mid360_prim": lidar_prim_path,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
