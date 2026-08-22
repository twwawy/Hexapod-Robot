#!/usr/bin/env python3
"""Preview the staircase training scene with the zero-residual base gait."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from prepare_rl_scene import STEP_START_X, prepare_rl_scene
from tripod_controller import GaitConfig, TripodGaitController


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.10, help="forward speed [m/s]")
    parser.add_argument("--duration", type=float, default=12.0, help="preview time [s]")
    parser.add_argument("--phase-time", type=float, default=0.5, help="tripod phase [s]")
    parser.add_argument("--swing-height", type=float, default=0.07, help="foot lift [m]")
    parser.add_argument(
        "--radial-offset", type=float, default=0.01, help="swing outward offset [m]"
    )
    parser.add_argument(
        "--headless", action="store_true", help="save an animated GIF instead of opening a window"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "rl_scene_preview.gif",
        help="animated GIF path used with --headless",
    )
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    return parser.parse_args()


def _make_simulation(args: argparse.Namespace):
    import mujoco

    scene = prepare_rl_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    controller = TripodGaitController(
        model,
        GaitConfig(
            speed=args.speed,
            phase_time=args.phase_time,
            swing_height=args.swing_height,
            radial_offset=args.radial_offset,
        ),
    )
    print(
        f"Loaded {scene.name}: {model.nbody} bodies, {model.ngeom} geoms, "
        f"{model.nu} actuators"
    )
    return mujoco, model, data, controller


def _step(mujoco, model, data, controller, next_control: float) -> float:
    while data.time + 1e-12 >= next_control:
        data.ctrl[:] = controller.targets(float(data.time))
        next_control += controller.config.control_dt
    mujoco.mj_step(model, data)
    return next_control


def _camera(camera, data) -> None:
    camera.lookat[:] = (
        max(float(data.qpos[0]) + 0.35, STEP_START_X),
        float(data.qpos[1]),
        max(float(data.qpos[2]) - 0.18, 0.12),
    )
    camera.distance = 1.75
    camera.azimuth = 135
    camera.elevation = -24


def _result(data, start: np.ndarray) -> None:
    displacement = data.qpos[:3] - start
    print(
        f"Finished at t={data.time:.2f}s | world displacement="
        f"({displacement[0]:+.3f}, {displacement[1]:+.3f}, "
        f"{displacement[2]:+.3f}) m"
    )
    print(
        "This is the zero-residual baseline. Its progress on the stairs is the "
        "pre-training comparison, not a learned-policy result."
    )


def _run_viewer(args: argparse.Namespace) -> None:
    import mujoco.viewer

    mujoco, model, data, controller = _make_simulation(args)
    start = data.qpos[:3].copy()
    next_control = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _camera(viewer.cam, data)
        while viewer.is_running() and data.time < args.duration:
            step_start = time.monotonic()
            next_control = _step(mujoco, model, data, controller, next_control)
            _camera(viewer.cam, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)
    _result(data, start)


def _run_headless(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero")

    from PIL import Image

    mujoco, model, data, controller = _make_simulation(args)
    start = data.qpos[:3].copy()
    renderer = mujoco.Renderer(model, height=360, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    frames: list[Image.Image] = []
    next_control = 0.0
    next_frame = 0.0
    frame_dt = 1.0 / args.fps

    while data.time < args.duration:
        next_control = _step(mujoco, model, data, controller, next_control)
        if data.time + 1e-12 >= next_frame:
            _camera(camera, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            frames.append(frame.convert("P", palette=Image.ADAPTIVE))
            next_frame += frame_dt

    if not frames:
        raise RuntimeError("No preview frames were rendered")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=False,
    )
    renderer.close()
    _result(data, start)
    print(f"Saved animated preview to {args.output.resolve()}")


def main() -> None:
    args = _arguments()
    if args.headless:
        _run_headless(args)
    else:
        _run_viewer(args)


if __name__ == "__main__":
    main()
