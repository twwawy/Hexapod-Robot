#!/usr/bin/env python3
"""Run the pulled STM32 controller in MuJoCo using ground-truth sensing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from firmware_controller import FirmwareController, FirmwareControllerState
from prepare_rl_scene import STEP_START_X, prepare_rl_scene
from prepare_rl_scene import STAIR_TOTAL_RISE, STEP_COUNT, STEP_DEPTH, STEP_HEIGHT
from terrain_curriculum import PLATEAU_DEPTH
from tripod_controller import LEG_PREFIXES, RIGHT_LEGS


CONTROL_DT = 0.005
JOINT_LIMIT = np.deg2rad(135.0)
JOINT_LIMIT_MARGIN = np.deg2rad(1.0)
MAX_TILT = np.deg2rad(45.0)
MIN_CLEARANCE = 0.14
MAX_ROOT_LINEAR_SPEED = 1.5
MAX_ROOT_ANGULAR_SPEED = 6.0
MAX_JOINT_SPEED = 20.0
MODEL_SIGNS = np.asarray(
    [
        (1.0, -1.0, 1.0) if prefix in RIGHT_LEGS else (1.0, 1.0, -1.0)
        for prefix in LEG_PREFIXES
    ]
)


@dataclass(frozen=True)
class GroundTruth:
    body_position: np.ndarray
    attitude: np.ndarray
    joint_angles: np.ndarray
    foot_contacts: np.ndarray


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.08, help="target forward speed [m/s]")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="target yaw rate [rad/s]")
    parser.add_argument("--duration", type=float, default=12.0, help="simulation time [s]")
    parser.add_argument(
        "--command-delay", type=float, default=1.0, help="standing time before motion [s]"
    )
    parser.add_argument(
        "--terrain",
        choices=("flat", "stairs"),
        default="flat",
        help="base validation terrain",
    )
    parser.add_argument("--headless", action="store_true", help="save a GIF without a window")
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="continue after the RL controller-failure termination condition",
    )
    parser.add_argument("--fps", type=int, default=20, help="headless GIF frame rate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "firmware_base.gif",
        help="headless GIF path",
    )
    return parser.parse_args()


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.asarray((1.0, -1.0, -1.0, -1.0))


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quat_to_euler(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw))


class MujocoGroundTruthAdapter:
    """Read perfect pose, joint position and foot contact from MuJoCo data."""

    def __init__(self, model, initial_quaternion: np.ndarray) -> None:
        self._model = model
        self._initial_quaternion = initial_quaternion.copy()
        self._joint_qpos_ids = np.asarray(
            [
                model.joint(f"{prefix}_{joint}").qposadr[0]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._joint_qvel_ids = np.asarray(
            [
                model.joint(f"{prefix}_{joint}").dofadr[0]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._foot_geom_to_leg = {
            model.geom(f"{prefix}_foot_collision").id: leg
            for leg, prefix in enumerate(LEG_PREFIXES)
        }

    def read(self, data) -> GroundTruth:
        contacts = np.zeros(6, dtype=bool)
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.dist > 0.002:
                continue
            for geom_id in (int(contact.geom1), int(contact.geom2)):
                leg = self._foot_geom_to_leg.get(geom_id)
                if leg is not None:
                    contacts[leg] = True

        # The generated scene rotates the CAD frame into the controller world
        # frame. q * inv(q_home) is therefore the controller-frame attitude.
        relative_quaternion = _quat_multiply(
            np.asarray(data.qpos[3:7]), _quat_conjugate(self._initial_quaternion)
        )
        raw_joints = np.asarray(data.qpos[self._joint_qpos_ids]).reshape(6, 3)
        return GroundTruth(
            body_position=np.asarray(data.qpos[:3]).copy(),
            attitude=_quat_to_euler(relative_quaternion),
            joint_angles=(raw_joints * MODEL_SIGNS).reshape(18),
            foot_contacts=contacts,
        )


class Simulation:
    def __init__(self, args: argparse.Namespace) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(prepare_rl_scene()))
        if args.terrain == "flat":
            for index in range(1, 8):
                geom = self.model.geom(f"stair_{index}")
                self.model.geom_contype[geom.id] = 0
                self.model.geom_conaffinity[geom.id] = 0
                self.model.geom_rgba[geom.id, 3] = 0.0

        self.data = mujoco.MjData(self.model)
        home = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, home)
        mujoco.mj_forward(self.model, self.data)
        self.start_position = np.asarray(self.data.qpos[:3]).copy()
        self.gt = MujocoGroundTruthAdapter(self.model, np.asarray(self.data.qpos[3:7]))
        self.controller = FirmwareController()
        self.terrain = args.terrain
        self.actuator_ids = np.asarray(
            [
                self.model.actuator(f"{prefix}_{joint}_position").id
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self.next_control = 0.0
        self.last_state: FirmwareControllerState | None = None
        self.unsafe_reason: str | None = None
        print(
            f"Loaded firmware base | terrain={args.terrain} | "
            f"bodies={self.model.nbody} geoms={self.model.ngeom}",
            flush=True,
        )

    def step(self, args: argparse.Namespace) -> None:
        while self.data.time + 1.0e-12 >= self.next_control:
            ground_truth = self.gt.read(self.data)
            command_active = self.data.time >= args.command_delay
            self.last_state = self.controller.step(
                target_vx=args.speed if command_active else 0.0,
                target_wz=args.yaw_rate if command_active else 0.0,
                body_position=ground_truth.body_position,
                attitude=ground_truth.attitude,
                contacts=ground_truth.foot_contacts,
            )
            model_targets = self.last_state.joint_angles.reshape(6, 3) * MODEL_SIGNS
            self.data.ctrl[self.actuator_ids] = model_targets.reshape(18)
            self.next_control += CONTROL_DT
        self.mujoco.mj_step(self.model, self.data)
        if not args.allow_unsafe and self.unsafe_reason is None and self.data.time > 0.25:
            self.unsafe_reason = self._unsafe_reason()

    def _terrain_height(self) -> float:
        if self.terrain == "flat":
            return 0.0
        x = float(self.data.qpos[0])
        heights = [
            STEP_HEIGHT * (index + 1)
            for index in range(STEP_COUNT)
            if abs(x - (STEP_START_X + STEP_DEPTH * (index + 0.5)))
            <= STEP_DEPTH / 2.0
        ]
        stair_end = STEP_START_X + STEP_COUNT * STEP_DEPTH
        if stair_end < x <= stair_end + PLATEAU_DEPTH:
            heights.append(STAIR_TOTAL_RISE)
        return max(heights, default=0.0)

    def _unsafe_reason(self) -> str | None:
        """Mirror the catastrophic termination used by residual training."""
        if self.last_state is None:
            return None
        ground_truth = self.gt.read(self.data)
        if not (
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.last_state.joint_angles))
        ):
            return "non-finite simulation/controller state"
        invalid = np.flatnonzero(~self.last_state.ik_valid)
        if invalid.size:
            return f"firmware IK invalid on leg {int(invalid[0]) + 1}"
        joint_margin = JOINT_LIMIT - float(np.max(np.abs(self.last_state.joint_angles)))
        if joint_margin <= JOINT_LIMIT_MARGIN:
            return f"firmware joint target reached limit margin ({np.rad2deg(joint_margin):.2f} deg)"
        if np.max(np.abs(ground_truth.attitude[:2])) > MAX_TILT:
            return "body roll/pitch exceeded 45 deg"
        clearance = float(self.data.qpos[2]) - self._terrain_height()
        if clearance < MIN_CLEARANCE:
            return f"body clearance fell below {MIN_CLEARANCE:.2f} m"
        if np.linalg.norm(self.data.qvel[:3]) > MAX_ROOT_LINEAR_SPEED:
            return "root linear velocity diverged"
        if np.linalg.norm(self.data.qvel[3:6]) > MAX_ROOT_ANGULAR_SPEED:
            return "root angular velocity diverged"
        joint_velocity = self.data.qvel[self.gt._joint_qvel_ids]
        if np.max(np.abs(joint_velocity)) > MAX_JOINT_SPEED:
            return "joint velocity diverged"
        return None

    def close(self) -> None:
        self.controller.close()


def _camera(camera, simulation: Simulation, terrain: str) -> None:
    data = simulation.data
    lookahead = 0.4 if terrain == "stairs" else 0.15
    camera.lookat[:] = (
        max(float(data.qpos[0]) + lookahead, STEP_START_X if terrain == "stairs" else 0.0),
        float(data.qpos[1]),
        max(float(data.qpos[2]) - 0.18, 0.12),
    )
    camera.distance = 1.65
    camera.azimuth = 135
    camera.elevation = -24


def _print_result(simulation: Simulation) -> None:
    displacement = np.asarray(simulation.data.qpos[:3]) - simulation.start_position
    state = simulation.last_state
    ground_truth = simulation.gt.read(simulation.data)
    print(
        f"Finished t={simulation.data.time:.2f}s | displacement="
        f"({displacement[0]:+.3f}, {displacement[1]:+.3f}, {displacement[2]:+.3f}) m"
    )
    if simulation.unsafe_reason is not None:
        print(f"SAFETY TERMINATION: {simulation.unsafe_reason}")
    if state is not None:
        print(
            f"gait_enabled={state.gait_enabled} contacts="
            f"{ground_truth.foot_contacts.astype(int).tolist()} "
            f"ik_valid={state.ik_valid.astype(int).tolist()}\n"
            f"GT rpy_deg={np.rad2deg(ground_truth.attitude).round(2).tolist()} "
            f"applied_twist={state.applied_twist.round(3).tolist()}"
        )


def _run_viewer(args: argparse.Namespace) -> None:
    import mujoco.viewer

    simulation = Simulation(args)
    try:
        with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
            _camera(viewer.cam, simulation, args.terrain)
            while viewer.is_running() and simulation.data.time < args.duration:
                started = time.monotonic()
                simulation.step(args)
                if simulation.unsafe_reason is not None:
                    break
                _camera(viewer.cam, simulation, args.terrain)
                viewer.sync()
                remaining = simulation.model.opt.timestep - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
        _print_result(simulation)
    finally:
        simulation.close()


def _run_headless(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero")

    from PIL import Image

    simulation = Simulation(args)
    renderer = simulation.mujoco.Renderer(simulation.model, height=360, width=640)
    camera = simulation.mujoco.MjvCamera()
    simulation.mujoco.mjv_defaultCamera(camera)
    frames: list[Image.Image] = []
    next_frame = 0.0
    frame_dt = 1.0 / args.fps
    try:
        while simulation.data.time < args.duration:
            simulation.step(args)
            if simulation.data.time + 1.0e-12 >= next_frame:
                _camera(camera, simulation, args.terrain)
                renderer.update_scene(simulation.data, camera=camera)
                frame = Image.fromarray(renderer.render())
                frames.append(frame.convert("P", palette=Image.ADAPTIVE))
                next_frame += frame_dt
            if simulation.unsafe_reason is not None:
                break

        args.output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            args.output,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / args.fps),
            loop=0,
            optimize=False,
        )
        _print_result(simulation)
        print(f"Saved preview to {args.output.resolve()}")
    finally:
        renderer.close()
        simulation.close()


def main() -> None:
    args = _arguments()
    if args.headless:
        _run_headless(args)
    else:
        _run_viewer(args)


if __name__ == "__main__":
    main()
