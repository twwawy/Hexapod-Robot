from __future__ import annotations

"""Replay a saved MJX tripod gait in a MuJoCo viewer or exported frames.

This script intentionally uses plain NumPy + MuJoCo for playback instead of MJX.
The training loop already proved that the batched JAX/MJX rollout works; for
visual inspection we want the opposite trade-off:

- easy to single-step and inspect,
- easy to attach a live viewer,
- easy to export a few frames in headless mode.

The input contract is the JSON produced by ``train_tripod_cem.py``. The script
reads ``best_params`` from that JSON and replays the same open-loop tripod gait
with the same PD controller logic that the CEM search evaluated.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from hexapod_mjx.cem import PARAMETER_NAMES, PD_KD, PD_KP, TORQUE_LIMIT
from hexapod_mjx.model import HexapodModelBundle, estimate_standing_root_height, load_hexapod_model, repo_root_from


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for live viewing or headless frame export."""
    parser = argparse.ArgumentParser(
        description="Visualize a saved Hexapod MJX tripod-gait result.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path inside or at the Hexapod-Robot repo. Defaults to this script location.",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default="SW/mjx/artifacts/hexapod_cem_gpu_baseline.json",
        help="Repo-relative or absolute JSON result path produced by train_tripod_cem.py.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=8.0,
        help="How long to replay the gait for.",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=None,
        help="Override the action-repeat from the saved JSON config.",
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default=None,
        help="If set, save .ppm frames here instead of opening a live viewer.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Save one frame every N control updates in render-dir mode.",
    )
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=540, help="Render height.")
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sleep to roughly match simulated time while using the live viewer.",
    )
    parser.add_argument("--camera-distance", type=float, default=2.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.08)
    return parser.parse_args()


def _resolve_result_path(repo_root: Path, result_path: str) -> Path:
    """Allow the caller to pass either a repo-relative or absolute JSON path."""
    path = Path(result_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_result_payload(result_path: Path) -> dict:
    """Load and validate the saved search result JSON."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    best_params = payload.get("best_params", {})
    missing = [name for name in PARAMETER_NAMES if name not in best_params]
    if missing:
        raise KeyError(f"Result JSON is missing best_params entries: {missing}")
    return payload


def _best_param_vector(payload: dict) -> np.ndarray:
    """Convert the named JSON params into the exact vector order used in training."""
    best_params = payload["best_params"]
    return np.asarray([float(best_params[name]) for name in PARAMETER_NAMES], dtype=np.float32)


def _make_initial_data(bundle: HexapodModelBundle, *, base_height: float) -> mujoco.MjData:
    """Reset the free-floating robot to the same standing pose used in training.

    The base is free-jointed, so we must set both:
    - root position/quaternion for the floating base, and
    - per-joint standing pose for all 18 leg joints.
    """
    data = mujoco.MjData(bundle.model)
    data.qpos[0:3] = np.array([0.0, 0.0, base_height], dtype=np.float32)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    data.qpos[bundle.joint_qpos_adr] = np.asarray(bundle.default_joint_pose, dtype=np.float32)
    mujoco.mj_forward(bundle.model, data)
    return data


def _joint_targets(bundle: HexapodModelBundle, params: np.ndarray, time_s: float) -> np.ndarray:
    """Rebuild the open-loop tripod gait target for one control instant.

    The parameter vector is the same one optimized by CEM:
    - frequency_hz controls gait speed,
    - amplitudes control sinusoid swing size,
    - bias deltas shift the nominal hip/knee bend,
    - knee_phase_offset lets the knee lead or lag the upper leg.

    ``joint_group_index`` maps joint suffixes to a compact index:
    - 0 -> first joint in each leg (hip yaw / swing),
    - 1 -> second joint in each leg,
    - 2 -> third joint in each leg.
    """
    phase = 2.0 * np.pi * float(params[0]) * time_s + np.asarray(bundle.tripod_phase_offset, dtype=np.float32)
    knee_phase = phase + float(params[6])
    group = np.asarray(bundle.joint_group_index, dtype=np.int32)
    default_pose = np.asarray(bundle.default_joint_pose, dtype=np.float32)

    hip1 = float(params[1]) * np.sin(phase)
    hip2 = float(params[4]) + float(params[2]) * np.sin(phase)
    knee = float(params[5]) + float(params[3]) * np.sin(knee_phase)

    return (
        default_pose
        + np.where(group == 0, hip1, 0.0)
        + np.where(group == 1, hip2, 0.0)
        + np.where(group == 2, knee, 0.0)
    ).astype(np.float32)


def _pd_torque(bundle: HexapodModelBundle, data: mujoco.MjData, desired_qpos: np.ndarray) -> np.ndarray:
    """Apply the same simple per-joint PD controller that training scored.

    CEM does not output torques directly. It outputs trajectory parameters, then
    a hand-written PD controller tracks those joint-angle targets. Keeping the
    exact same controller here matters; otherwise the visualization could look
    better or worse than what the optimizer actually evaluated.
    """
    group = np.asarray(bundle.joint_group_index, dtype=np.int32)
    qj = data.qpos[bundle.joint_qpos_adr]
    qv = data.qvel[bundle.joint_dof_adr]
    kp = np.asarray(PD_KP, dtype=np.float32)[group]
    kd = np.asarray(PD_KD, dtype=np.float32)[group]
    tau_limit = np.asarray(TORQUE_LIMIT, dtype=np.float32)[group]
    tau = kp * (desired_qpos - qj) - kd * qv
    return np.clip(tau, -tau_limit, tau_limit)


def _step_controller(
    bundle: HexapodModelBundle,
    data: mujoco.MjData,
    params: np.ndarray,
    *,
    control_time_s: float,
    action_repeat: int,
) -> None:
    """Advance physics by one control update.

    Training held one torque command for ``action_repeat`` physics steps. We do
    the same here so the replay matches the optimization environment instead of
    silently using a higher control frequency.
    """
    desired_qpos = _joint_targets(bundle, params, control_time_s)
    tau = _pd_torque(bundle, data, desired_qpos)
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[bundle.joint_dof_adr] = tau
    for _ in range(action_repeat):
        mujoco.mj_step(bundle.model, data)


def _configure_camera(camera: mujoco.MjvCamera, args: argparse.Namespace) -> None:
    """Set a stable free-camera pose that shows the whole robot and floor."""
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.lookat[0] = args.lookat_x
    camera.lookat[1] = args.lookat_y
    camera.lookat[2] = args.lookat_z


def _write_ppm(path: Path, pixels: np.ndarray) -> None:
    """Write an RGB frame without extra dependencies.

    PPM is intentionally boring: it is a trivial binary image format that lets
    us export frames even when Pillow / imageio / ffmpeg are not installed.
    """
    rgb = np.ascontiguousarray(pixels[::-1])
    header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + rgb.tobytes())


def _clamp_render_size(model: mujoco.MjModel, width: int, height: int) -> tuple[int, int]:
    """Clamp headless render size to MuJoCo's configured offscreen framebuffer.

    MuJoCo's software/EGL renderer cannot exceed ``vis.global.offwidth`` and
    ``offheight`` unless the model XML explicitly requests a larger framebuffer.
    Clamping here keeps the default headless path usable out of the box.
    """
    offwidth = int(model.vis.global_.offwidth)
    offheight = int(model.vis.global_.offheight)
    return min(width, offwidth), min(height, offheight)


def _render_frames(
    bundle: HexapodModelBundle,
    data: mujoco.MjData,
    params: np.ndarray,
    *,
    action_repeat: int,
    duration_sec: float,
    render_dir: Path,
    frame_stride: int,
    width: int,
    height: int,
) -> int:
    """Export a sparse frame sequence for headless inspection."""
    render_dir.mkdir(parents=True, exist_ok=True)
    total_control_steps = max(1, int(duration_sec / (bundle.model.opt.timestep * action_repeat)))
    saved_frames = 0
    width, height = _clamp_render_size(bundle.model, width, height)
    renderer = mujoco.Renderer(bundle.model, height=height, width=width)
    try:
        for control_step in range(total_control_steps):
            control_time_s = control_step * bundle.model.opt.timestep * action_repeat
            _step_controller(
                bundle,
                data,
                params,
                control_time_s=control_time_s,
                action_repeat=action_repeat,
            )
            if control_step % max(1, frame_stride) != 0:
                continue
            renderer.update_scene(data)
            pixels = renderer.render()
            frame_path = render_dir / f"frame_{saved_frames:04d}.ppm"
            _write_ppm(frame_path, pixels)
            saved_frames += 1
    finally:
        renderer.close()
    return saved_frames


def _run_live_viewer(
    bundle: HexapodModelBundle,
    data: mujoco.MjData,
    params: np.ndarray,
    *,
    action_repeat: int,
    duration_sec: float,
    realtime: bool,
    camera_args: argparse.Namespace,
) -> None:
    """Open an interactive MuJoCo viewer and replay the gait."""
    control_dt = bundle.model.opt.timestep * action_repeat
    total_control_steps = max(1, int(duration_sec / control_dt))
    with mujoco.viewer.launch_passive(bundle.model, data) as viewer:
        _configure_camera(viewer.cam, camera_args)
        viewer.sync()
        wall_start = time.perf_counter()
        for control_step in range(total_control_steps):
            if not viewer.is_running():
                break
            control_time_s = control_step * control_dt
            _step_controller(
                bundle,
                data,
                params,
                control_time_s=control_time_s,
                action_repeat=action_repeat,
            )
            viewer.sync()
            if realtime:
                target_wall = wall_start + (control_step + 1) * control_dt
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)


def main() -> None:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)
    bundle = load_hexapod_model(repo_root)

    result_path = _resolve_result_path(repo_root, args.result_path)
    payload = _load_result_payload(result_path)
    params = _best_param_vector(payload)
    saved_base_height = payload.get("config", {}).get("base_height")
    base_height = float(saved_base_height) if saved_base_height is not None else estimate_standing_root_height(bundle)
    action_repeat = int(args.action_repeat or payload.get("config", {}).get("action_repeat", 2))
    data = _make_initial_data(bundle, base_height=base_height)

    print(f"repo_root: {repo_root}")
    print(f"result_path: {result_path}")
    print(f"generated_mjcf: {bundle.generated_mjcf_path}")
    print(f"action_repeat: {action_repeat}")
    print("best_params:")
    for name, value in zip(PARAMETER_NAMES, params):
        print(f"  - {name}: {float(value):.6f}")

    if args.render_dir:
        render_dir = _resolve_result_path(repo_root, args.render_dir)
        saved_frames = _render_frames(
            bundle,
            data,
            params,
            action_repeat=action_repeat,
            duration_sec=args.duration_sec,
            render_dir=render_dir,
            frame_stride=args.frame_stride,
            width=args.width,
            height=args.height,
        )
        print(f"render_dir: {render_dir}")
        print(f"render_size: {args.width}x{args.height} -> {min(args.width, int(bundle.model.vis.global_.offwidth))}x{min(args.height, int(bundle.model.vis.global_.offheight))}")
        print(f"saved_frames: {saved_frames}")
    else:
        _run_live_viewer(
            bundle,
            data,
            params,
            action_repeat=action_repeat,
            duration_sec=args.duration_sec,
            realtime=args.realtime,
            camera_args=args,
        )


if __name__ == "__main__":
    main()
