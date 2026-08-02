from __future__ import annotations

"""Replay a trained residual-RL policy in the MuJoCo viewer or headless frames."""

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from hexapod_mjx.model import load_hexapod_model, repo_root_from
from hexapod_mjx.residual_controller import (
    ResidualControllerConfig,
    body_velocity_components,
    build_residual_controller,
    controller_step,
    policy_dt,
    quat_roll_pitch_yaw,
    reset_controller_state,
)
from hexapod_mjx.residual_rl import load_checkpoint, policy_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained residual-RL hexapod policy.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default="SW/mjx/artifacts/residual_rl_policy.pkl")
    parser.add_argument("--forward-cmd", type=float, default=0.18)
    parser.add_argument("--lateral-cmd", type=float, default=0.0)
    parser.add_argument("--yaw-cmd", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--hold-sec", type=float, default=1.5, help="Hold the neutral pose for this many seconds before policy rollout.")
    parser.add_argument("--render-dir", type=str, default=None)
    parser.add_argument("--output-video", type=str, default=None)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=0.0, help="Override output-video FPS. Default derives from policy_dt and frame_stride.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=1.45)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-16.0)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.18)
    return parser.parse_args()



def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()



def _write_ppm(path: Path, pixels: np.ndarray) -> None:
    rgb = np.ascontiguousarray(pixels[::-1])
    header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + rgb.tobytes())



def _clamp_render_size(model: mujoco.MjModel, width: int, height: int) -> tuple[int, int]:
    return min(width, int(model.vis.global_.offwidth)), min(height, int(model.vis.global_.offheight))


def _video_fps(step_dt: float, frame_stride: int, override_fps: float) -> float:
    if override_fps > 0.0:
        return override_fps
    return max(1.0, 1.0 / (step_dt * max(1, frame_stride)))



def _configure_camera(camera: mujoco.MjvCamera, args: argparse.Namespace) -> None:
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.lookat[0] = args.lookat_x
    camera.lookat[1] = args.lookat_y
    camera.lookat[2] = args.lookat_z



def main() -> None:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)
    bundle = load_hexapod_model(repo_root)
    controller_config = ResidualControllerConfig()
    controller_bundle = build_residual_controller(bundle, controller_config)
    train_state, metadata = load_checkpoint(_resolve_path(repo_root, args.policy_path))

    model = bundle.model
    data = mujoco.MjData(model)
    data.qpos[0:3] = np.array([0.0, 0.0, float(controller_bundle.reset_root_height)], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[bundle.joint_qpos_adr] = np.asarray(controller_bundle.neutral_joint_pose)
    mujoco.mj_forward(model, data)

    controller_state = reset_controller_state(1)
    command = jnp.asarray([[args.forward_cmd, args.lateral_cmd, args.yaw_cmd]], dtype=jnp.float32)
    body_velocity_world = jnp.zeros((1, 3), dtype=jnp.float32)
    yaw_rate = jnp.zeros((1,), dtype=jnp.float32)
    prev_root_pos = jnp.asarray(data.qpos[0:3][None, :], dtype=jnp.float32)
    _, _, prev_yaw = quat_roll_pitch_yaw(jnp.asarray(data.qpos[3:7][None, :], dtype=jnp.float32))
    prev_foot_world = jnp.asarray(data.geom_xpos[controller_bundle.foot_geom_ids][None, :, :], dtype=jnp.float32)

    qpos_adr = np.asarray(bundle.joint_qpos_adr, dtype=np.int32)
    dof_adr = np.asarray(bundle.joint_dof_adr, dtype=np.int32)
    group_index = np.asarray([int(name.split('_')[-1]) - 1 for name in bundle.joint_names], dtype=np.int32)
    kp = np.asarray(controller_config.pd_kp, dtype=np.float32)[group_index]
    kd = np.asarray(controller_config.pd_kd, dtype=np.float32)[group_index]
    tau_limit = np.asarray(controller_config.torque_limit, dtype=np.float32)[group_index]

    def make_obs() -> jnp.ndarray:
        quat = jnp.asarray(data.qpos[3:7][None, :], dtype=jnp.float32)
        roll, pitch, _ = quat_roll_pitch_yaw(quat)
        forward_velocity, lateral_velocity = body_velocity_components(quat, body_velocity_world)
        body_height_error = jnp.asarray([[data.qpos[2] - float(controller_bundle.reset_root_height)]], dtype=jnp.float32)
        contacts = jnp.asarray((data.geom_xpos[controller_bundle.foot_geom_ids, 2] < controller_config.foot_contact_height).astype(np.float32)[None, :])
        phase_angle = controller_state.phase * (2.0 * jnp.pi)
        return jnp.concatenate(
            [
                command,
                jnp.stack([forward_velocity, lateral_velocity, yaw_rate], axis=-1),
                jnp.concatenate([roll[:, None], pitch[:, None], body_height_error], axis=-1),
                jnp.asarray(data.qpos[qpos_adr][None, :], dtype=jnp.float32),
                jnp.asarray(data.qvel[dof_adr][None, :], dtype=jnp.float32),
                contacts,
                jnp.stack([jnp.sin(phase_angle), jnp.cos(phase_angle)], axis=-1),
                controller_state.prev_action,
            ],
            axis=-1,
        )

    def step_policy() -> None:
        nonlocal controller_state, body_velocity_world, yaw_rate, prev_root_pos, prev_yaw, prev_foot_world
        obs = make_obs()
        action = np.asarray(policy_mean(train_state.params, obs))[0]
        for _ in range(controller_config.policy_controls_per_action):
            controller_state, joint_targets, _ = controller_step(
                bundle,
                controller_bundle,
                controller_config,
                controller_state,
                command,
                jnp.asarray(action[None, :], dtype=jnp.float32),
            )
            joint_targets_np = np.asarray(joint_targets)[0]
            qj = data.qpos[qpos_adr]
            qv = data.qvel[dof_adr]
            tau = kp * (joint_targets_np - qj) - kd * qv
            tau = np.clip(tau, -tau_limit, tau_limit)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_adr] = tau
            for _ in range(controller_config.physics_steps_per_control):
                mujoco.mj_step(model, data)

        root_pos = jnp.asarray(data.qpos[0:3][None, :], dtype=jnp.float32)
        body_velocity_world = (root_pos - prev_root_pos) / policy_dt(bundle, controller_config)
        _, _, yaw = quat_roll_pitch_yaw(jnp.asarray(data.qpos[3:7][None, :], dtype=jnp.float32))
        yaw_delta = jnp.arctan2(jnp.sin(yaw - prev_yaw), jnp.cos(yaw - prev_yaw))
        yaw_rate = yaw_delta / policy_dt(bundle, controller_config)
        prev_root_pos = root_pos
        prev_yaw = yaw
        prev_foot_world = jnp.asarray(data.geom_xpos[controller_bundle.foot_geom_ids][None, :, :], dtype=jnp.float32)

    step_dt = policy_dt(bundle, controller_config)
    total_steps = max(1, int(args.duration_sec / step_dt))
    print(f"repo_root: {repo_root}")
    print(f"policy_metadata: {metadata}")

    if args.render_dir or args.output_video:
        render_dir = _resolve_path(repo_root, args.render_dir) if args.render_dir else None
        output_video = _resolve_path(repo_root, args.output_video) if args.output_video else None
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
        if output_video is not None:
            output_video.parent.mkdir(parents=True, exist_ok=True)
        width, height = _clamp_render_size(model, args.width, args.height)
        video_fps = _video_fps(step_dt, args.frame_stride, args.fps)
        hold_frames = max(0, int(round(args.hold_sec * video_fps)))
        renderer = mujoco.Renderer(model, height=height, width=width)
        writer = imageio.get_writer(str(output_video), fps=video_fps, codec="libx264", pixelformat="yuv420p") if output_video is not None else None

        def save_frame(frame_idx: int) -> None:
            renderer.update_scene(data)
            pixels = renderer.render()
            if render_dir is not None:
                _write_ppm(render_dir / f"frame_{frame_idx:04d}.ppm", pixels)
            if writer is not None:
                writer.append_data(np.ascontiguousarray(pixels[::-1]))

        try:
            saved_frames = 0
            for _hold_idx in range(hold_frames):
                save_frame(saved_frames)
                saved_frames += 1
            for step_idx in range(total_steps):
                step_policy()
                if step_idx % max(1, args.frame_stride) != 0:
                    continue
                save_frame(saved_frames)
                saved_frames += 1
        finally:
            if writer is not None:
                writer.close()
            renderer.close()
        if render_dir is not None:
            print(f"render_dir: {render_dir}")
        if output_video is not None:
            print(f"output_video: {output_video}")
            print(f"video_fps: {video_fps:.2f}")
        print(f"hold_sec: {args.hold_sec:.2f}")
        print(f"render_size: {args.width}x{args.height} -> {width}x{height}")
        print(f"saved_frames: {saved_frames}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_camera(viewer.cam, args)
        viewer.sync()
        hold_steps = max(0, int(args.hold_sec / step_dt))
        wall_start = time.perf_counter()
        for hold_idx in range(hold_steps):
            if not viewer.is_running():
                break
            viewer.sync()
            target_time = wall_start + (hold_idx + 1) * step_dt
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        for step_idx in range(total_steps):
            if not viewer.is_running():
                break
            step_policy()
            viewer.sync()
            target_time = wall_start + (hold_steps + step_idx + 1) * step_dt
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
