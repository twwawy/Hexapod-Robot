#!/usr/bin/env python3
"""Run the documented tripod gait controller against the MuJoCo model."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from tripod_controller import GaitConfig, TripodGaitController
from view_robot import ensure_model


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.06,
        help="signed forward speed [m/s] (negative is reverse)",
    )
    parser.add_argument("--phase-time", type=float, default=0.5, help="tripod phase [s]")
    parser.add_argument("--swing-height", type=float, default=0.06, help="foot lift [m]")
    parser.add_argument("--radial-offset", type=float, default=0.01, help="swing outward offset [m]")
    parser.add_argument("--duration", type=float, default=10.0, help="run time [s]")
    parser.add_argument("--headless", action="store_true", help="run without the viewer")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "walk.png",
        help="final frame for --headless",
    )
    return parser.parse_args()


def _make_simulation(args: argparse.Namespace):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ensure_model()))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    controller = TripodGaitController(
        model,
        GaitConfig(
            speed=args.speed,
            phase_time=args.phase_time,
            swing_height=args.swing_height,
            radial_offset=args.radial_offset,
        ),
    )
    return mujoco, model, data, controller


def _step_controller(mujoco, model, data, controller, next_control: float) -> float:
    while data.time + 1e-12 >= next_control:
        data.ctrl[:] = controller.targets(float(data.time))
        next_control += controller.config.control_dt
    mujoco.mj_step(model, data)
    return next_control


def _print_result(data, start_position: np.ndarray) -> None:
    displacement = data.qpos[:3] - start_position
    print(
        f"Finished at t={data.time:.2f}s | "
        f"world displacement=({displacement[0]:+.3f}, "
        f"{displacement[1]:+.3f}, {displacement[2]:+.3f}) m"
    )


def _run_headless(args: argparse.Namespace) -> None:
    mujoco, model, data, controller = _make_simulation(args)
    start_position = data.qpos[:3].copy()
    next_control = 0.0
    while data.time < args.duration:
        next_control = _step_controller(
            mujoco, model, data, controller, next_control
        )

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.qpos[:3]
    camera.distance = 1.45
    camera.azimuth = 135
    camera.elevation = -25
    renderer.update_scene(data, camera=camera)

    from PIL import Image

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.render()).save(args.output)
    _print_result(data, start_position)
    print(f"Saved final frame to {args.output.resolve()}")


def _run_viewer(args: argparse.Namespace) -> None:
    import mujoco.viewer

    mujoco, model, data, controller = _make_simulation(args)
    start_position = data.qpos[:3].copy()
    next_control = 0.0
    wall_start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.12)
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while viewer.is_running() and data.time < args.duration:
            step_start = time.monotonic()
            next_control = _step_controller(
                mujoco, model, data, controller, next_control
            )
            viewer.cam.lookat[:2] = data.qpos[:2]
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    _print_result(data, start_position)
    print(f"Wall time: {time.monotonic() - wall_start:.2f}s")


def main() -> None:
    args = _arguments()
    if args.headless:
        _run_headless(args)
    else:
        _run_viewer(args)


if __name__ == "__main__":
    main()
