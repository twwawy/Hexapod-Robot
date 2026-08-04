from __future__ import annotations

"""Classical locomotion controller with a small residual-RL action interface.

The controller follows the design split requested in the brief:
- command filtering, gait timing, safety projection, and inverse-kinematics-like
  mapping stay explicit and deterministic,
- RL only nudges foothold/gait/body parameters through a tiny residual action.

To keep MJX training cheap, the IK layer is a *linearized* per-leg Jacobian
inverse around the standing pose instead of a heavy nonlinear solve every tick.
That still preserves the intended architecture: the policy reasons in foot/body
space while the controller owns the joint-space conversion and safety clamps.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .model import FOOT_CONTACT_POINT_COUNT, HexapodModelBundle, TRIPOD_A, estimate_standing_root_height






LEGS = ("LF", "LM", "LB", "RF", "RM", "RB")
ACTION_DIM = 7


class ResidualControllerBundle(NamedTuple):
    neutral_joint_pose: jnp.ndarray
    neutral_foot_body_pos: jnp.ndarray
    foot_jacobian_pinv: jnp.ndarray
    leg_joint_indices: jnp.ndarray
    leg_joint_pose: jnp.ndarray
    joint_lower: jnp.ndarray
    joint_upper: jnp.ndarray
    hip_body_pos: jnp.ndarray
    side_sign: jnp.ndarray
    front_sign: jnp.ndarray
    tripod_is_a: jnp.ndarray
    foot_geom_ids: jnp.ndarray
    foot_support_geom_ids: jnp.ndarray
    reset_root_height: jnp.ndarray



@dataclass(frozen=True)
class ResidualControllerConfig:
    physics_steps_per_control: int = 5
    policy_controls_per_action: int = 2
    base_step_period: float = 0.42
    command_deadzone: float = 0.03
    command_rate_limit: tuple[float, float, float] = (0.9, 0.6, 1.2)
    command_limits: tuple[float, float, float] = (0.35, 0.18, 0.9)
    translation_step_gain: tuple[float, float] = (0.65, 0.55)
    turn_step_gain: float = 0.35
    nominal_swing_height: float = 0.055
    residual_step_x: float = 0.055
    residual_step_y: float = 0.040
    residual_swing_height: float = 0.040
    residual_step_period: float = 0.12
    residual_body_height: float = 0.025
    residual_roll_trim: float = 0.030
    residual_pitch_trim: float = 0.030
    workspace_delta_min: tuple[float, float, float] = (-0.14, -0.10, -0.10)
    workspace_delta_max: tuple[float, float, float] = (0.14, 0.10, 0.08)
    pd_kp: tuple[float, float, float] = (16.0, 42.0, 42.0)
    pd_kd: tuple[float, float, float] = (0.9, 2.1, 2.1)
    torque_limit: tuple[float, float, float] = (10.0, 30.0, 30.0)
    foot_contact_height: float = 0.03
    startup_duration_sec: float = 0.30
    joint_limit_margin: tuple[float, float, float] = (0.0, 0.0, 0.0)


def controller_config_from_metadata(metadata: Mapping[str, Any] | None) -> "ResidualControllerConfig":
    ppo_config = metadata.get("ppo_config") if isinstance(metadata, Mapping) else None
    if not isinstance(ppo_config, Mapping):
        return ResidualControllerConfig()
    return ResidualControllerConfig(
        joint_limit_margin=(
            float(ppo_config.get("joint_limit_margin_1", 0.0)),
            float(ppo_config.get("joint_limit_margin_2", 0.0)),
            float(ppo_config.get("joint_limit_margin_3", 0.0)),
        )
    )



class ResidualControllerState(NamedTuple):
    phase: jnp.ndarray
    filtered_command: jnp.ndarray
    prev_action: jnp.ndarray
    elapsed_sec: jnp.ndarray


class ResidualActionTerms(NamedTuple):
    step_x: jnp.ndarray
    step_y: jnp.ndarray
    swing_height: jnp.ndarray
    step_period: jnp.ndarray
    body_height: jnp.ndarray
    roll_trim: jnp.ndarray
    pitch_trim: jnp.ndarray


def build_residual_controller(
    bundle: HexapodModelBundle,
    config: ResidualControllerConfig,
) -> ResidualControllerBundle:
    """Build neutral foot geometry and a cheap per-leg linearized IK mapping."""
    model = bundle.model
    host_data = mujoco.MjData(model)
    host_data.qpos[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    host_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    host_data.qpos[bundle.joint_qpos_adr] = np.asarray(bundle.default_joint_pose, dtype=np.float64)
    mujoco.mj_forward(model, host_data)

    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hexapod_root")
    root_pos = host_data.xpos[root_body_id].copy()
    root_rot = host_data.xmat[root_body_id].reshape(3, 3).copy()

    leg_joint_indices: list[list[int]] = []
    leg_joint_pose: list[list[float]] = []
    joint_lower: list[list[float]] = []
    joint_upper: list[list[float]] = []
    neutral_foot_body_pos: list[np.ndarray] = []
    hip_body_pos: list[np.ndarray] = []
    side_sign: list[float] = []
    front_sign: list[float] = []
    tripod_is_a: list[bool] = []
    foot_geom_ids: list[int] = []
    foot_support_geom_ids: list[list[int]] = []
    jacobian_pinv: list[np.ndarray] = []


    name_to_index = {name: idx for idx, name in enumerate(bundle.joint_names)}
    name_to_joint_id = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in bundle.joint_names
    }
    joint_limit_margin = np.asarray(config.joint_limit_margin, dtype=np.float64)
    if np.any(joint_limit_margin < 0.0):
        raise ValueError(f"joint_limit_margin must be non-negative, got {config.joint_limit_margin}")

    neutral_foot_world_z = []

    for leg in LEGS:
        leg_indices = [name_to_index[f"{leg}_{suffix}"] for suffix in (1, 2, 3)]
        leg_joint_indices.append(leg_indices)
        leg_joint_pose.append([float(bundle.default_joint_pose[idx]) for idx in leg_indices])
        leg_joint_ids = [name_to_joint_id[f"{leg}_{suffix}"] for suffix in (1, 2, 3)]
        raw_joint_lower = np.asarray([float(model.jnt_range[joint_id, 0]) for joint_id in leg_joint_ids], dtype=np.float64)
        raw_joint_upper = np.asarray([float(model.jnt_range[joint_id, 1]) for joint_id in leg_joint_ids], dtype=np.float64)
        tightened_joint_lower = raw_joint_lower + joint_limit_margin
        tightened_joint_upper = raw_joint_upper - joint_limit_margin
        if np.any(tightened_joint_lower >= tightened_joint_upper):
            raise ValueError(
                f"joint_limit_margin {config.joint_limit_margin} collapses the valid range for {leg}: "
                f"raw_lower={raw_joint_lower.tolist()} raw_upper={raw_joint_upper.tolist()}"
            )
        joint_lower.append(tightened_joint_lower.tolist())
        joint_upper.append(tightened_joint_upper.tolist())

        hip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_motor_horn_1_1")
        hip_world = host_data.xpos[hip_body_id]
        hip_body = root_rot.T @ (hip_world - root_pos)
        hip_body_pos.append(hip_body.astype(np.float32))
        side_sign.append(float(np.sign(hip_body[0])))
        front_sign.append(float(np.sign(-hip_body[1])))
        tripod_is_a.append(leg in TRIPOD_A)

        foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_motor_horn_3_1_contact")
        foot_geom_ids.append(foot_geom_id)
        support_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_motor_horn_3_1_contact_{idx}")
            for idx in range(FOOT_CONTACT_POINT_COUNT)
        ]
        foot_support_geom_ids.append([support_id if support_id >= 0 else foot_geom_id for support_id in support_ids])
        foot_world = host_data.geom_xpos[foot_geom_id]
        neutral_foot_world_z.append(float(foot_world[2]))
        foot_body = root_rot.T @ (foot_world - root_pos)
        neutral_foot_body_pos.append(foot_body.astype(np.float32))


        jacobian = np.zeros((3, 3), dtype=np.float64)
        eps = 1e-4
        for axis_idx, joint_index in enumerate(leg_indices):
            perturbed = mujoco.MjData(model)
            perturbed.qpos[:] = host_data.qpos
            perturbed.qvel[:] = 0.0
            perturbed.qpos[bundle.joint_qpos_adr[joint_index]] += eps
            mujoco.mj_forward(model, perturbed)
            foot_world_eps = perturbed.geom_xpos[foot_geom_id]
            foot_body_eps = root_rot.T @ (foot_world_eps - root_pos)
            jacobian[:, axis_idx] = (foot_body_eps - foot_body) / eps
        jacobian_pinv.append(np.linalg.pinv(jacobian).astype(np.float32))

    neutral_foot_body = np.asarray(neutral_foot_body_pos, dtype=np.float32)
    _ = neutral_foot_world_z
    reset_root_height = np.asarray(float(estimate_standing_root_height(bundle)), dtype=np.float32)

    return ResidualControllerBundle(
        neutral_joint_pose=jnp.asarray(np.asarray(bundle.default_joint_pose, dtype=np.float32)),
        neutral_foot_body_pos=jnp.asarray(neutral_foot_body),
        foot_jacobian_pinv=jnp.asarray(np.asarray(jacobian_pinv, dtype=np.float32)),
        leg_joint_indices=jnp.asarray(np.asarray(leg_joint_indices, dtype=np.int32)),
        leg_joint_pose=jnp.asarray(np.asarray(leg_joint_pose, dtype=np.float32)),
        joint_lower=jnp.asarray(np.asarray(joint_lower, dtype=np.float32)),
        joint_upper=jnp.asarray(np.asarray(joint_upper, dtype=np.float32)),
        hip_body_pos=jnp.asarray(np.asarray(hip_body_pos, dtype=np.float32)),
        side_sign=jnp.asarray(np.asarray(side_sign, dtype=np.float32)),
        front_sign=jnp.asarray(np.asarray(front_sign, dtype=np.float32)),
        tripod_is_a=jnp.asarray(np.asarray(tripod_is_a, dtype=bool)),
        foot_geom_ids=jnp.asarray(np.asarray(foot_geom_ids, dtype=np.int32)),
        foot_support_geom_ids=jnp.asarray(np.asarray(foot_support_geom_ids, dtype=np.int32)),
        reset_root_height=jnp.asarray(reset_root_height),
    )


def control_dt(bundle: HexapodModelBundle, config: ResidualControllerConfig) -> float:
    return float(bundle.model.opt.timestep) * float(config.physics_steps_per_control)


def policy_dt(bundle: HexapodModelBundle, config: ResidualControllerConfig) -> float:
    return control_dt(bundle, config) * float(config.policy_controls_per_action)


def reset_controller_state(batch_size: int) -> ResidualControllerState:
    return ResidualControllerState(
        phase=jnp.zeros((batch_size,), dtype=jnp.float32),
        filtered_command=jnp.zeros((batch_size, 3), dtype=jnp.float32),
        prev_action=jnp.zeros((batch_size, ACTION_DIM), dtype=jnp.float32),
        elapsed_sec=jnp.zeros((batch_size,), dtype=jnp.float32),
    )



def quat_to_rotmat(quat: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            jnp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            jnp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def quat_up_z(quat: jnp.ndarray) -> jnp.ndarray:
    return 1.0 - 2.0 * (quat[..., 1] * quat[..., 1] + quat[..., 2] * quat[..., 2])


def quat_roll_pitch_yaw(quat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = jnp.arcsin(jnp.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def world_to_body(root_pos: jnp.ndarray, root_quat: jnp.ndarray, world_points: jnp.ndarray) -> jnp.ndarray:
    rot = quat_to_rotmat(root_quat)
    relative = world_points - root_pos[:, None, :]
    return jnp.einsum("bij,bkj->bki", jnp.swapaxes(rot, -1, -2), relative)


def body_to_world(root_pos: jnp.ndarray, root_quat: jnp.ndarray, body_points: jnp.ndarray) -> jnp.ndarray:
    rot = quat_to_rotmat(root_quat)
    rotated = jnp.einsum("bij,bkj->bki", rot, body_points)
    return rotated + root_pos[:, None, :]


def body_velocity_components(root_quat: jnp.ndarray, world_velocity: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    rot_t = jnp.swapaxes(quat_to_rotmat(root_quat), -1, -2)
    body_velocity = jnp.einsum("bij,bj->bi", rot_t, world_velocity)
    lateral_velocity = body_velocity[:, 0]
    forward_velocity = -body_velocity[:, 1]
    return forward_velocity, lateral_velocity


def _slew_limit(current: jnp.ndarray, target: jnp.ndarray, rate_limit: jnp.ndarray, dt: float) -> jnp.ndarray:
    delta = jnp.clip(target - current, -rate_limit[None, :] * dt, rate_limit[None, :] * dt)
    return current + delta


def _apply_deadzone(command: jnp.ndarray, deadzone: float) -> jnp.ndarray:
    return jnp.where(jnp.abs(command) < deadzone, 0.0, command)


def _scale_action(action: jnp.ndarray, config: ResidualControllerConfig) -> ResidualActionTerms:
    bounded = jnp.tanh(action)
    return ResidualActionTerms(
        step_x=bounded[:, 0] * config.residual_step_x,
        step_y=bounded[:, 1] * config.residual_step_y,
        swing_height=bounded[:, 2] * config.residual_swing_height,
        step_period=bounded[:, 3] * config.residual_step_period,
        body_height=bounded[:, 4] * config.residual_body_height,
        roll_trim=bounded[:, 5] * config.residual_roll_trim,
        pitch_trim=bounded[:, 6] * config.residual_pitch_trim,
    )


def _bezier_cubic(start: jnp.ndarray, end: jnp.ndarray, height: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    height = jnp.asarray(height, dtype=start.dtype)
    while height.ndim > start.ndim - 1:
        height = jnp.squeeze(height, axis=-1)
    height = jnp.broadcast_to(height, start[..., 2].shape)
    lift = jnp.zeros_like(start)
    lift = lift.at[..., 2].set(height)
    p1 = start + lift
    p2 = end + lift
    one_minus = 1.0 - s
    return (
        (one_minus**3) * start
        + 3.0 * (one_minus**2) * s * p1
        + 3.0 * one_minus * (s**2) * p2
        + (s**3) * end
    )


def controller_step(
    model_bundle: HexapodModelBundle,
    controller_bundle: ResidualControllerBundle,
    config: ResidualControllerConfig,
    state: ResidualControllerState,
    command_target: jnp.ndarray,
    action: jnp.ndarray,
) -> tuple[ResidualControllerState, jnp.ndarray, jnp.ndarray]:
    """Advance the deterministic gait controller by one control tick."""
    dt = control_dt(model_bundle, config)
    command_limit = jnp.asarray(config.command_limits, dtype=jnp.float32)
    command_rate_limit = jnp.asarray(config.command_rate_limit, dtype=jnp.float32)
    command_target = _apply_deadzone(jnp.clip(command_target, -command_limit, command_limit), config.command_deadzone)
    filtered_command = _slew_limit(state.filtered_command, command_target, command_rate_limit, dt)

    action_terms = _scale_action(action, config)
    step_period = jnp.clip(
        config.base_step_period + action_terms.step_period,
        0.22,
        0.72,
    )
    phase = jnp.mod(state.phase + dt / step_period, 1.0)
    elapsed_sec = state.elapsed_sec + dt
    startup_ramp = jnp.clip(elapsed_sec / config.startup_duration_sec, 0.0, 1.0)[:, None]

    first_half = phase[:, None] < 0.5
    local_phase = jnp.where(first_half, phase[:, None] * 2.0, (phase[:, None] - 0.5) * 2.0)
    swing_mask = jnp.where(controller_bundle.tripod_is_a[None, :], first_half, ~first_half)

    command_lateral = filtered_command[:, 1]
    command_forward = filtered_command[:, 0]
    command_yaw = filtered_command[:, 2]
    translation_xy = jnp.stack(
        [
            config.translation_step_gain[1] * command_lateral + action_terms.step_x,
            -config.translation_step_gain[0] * command_forward + action_terms.step_y,
        ],
        axis=-1,
    )

    hip_xy = controller_bundle.hip_body_pos[:, 0:2]
    yaw_offset = config.turn_step_gain * jnp.stack(
        [
            -command_yaw[:, None] * hip_xy[None, :, 1],
            command_yaw[:, None] * hip_xy[None, :, 0],
        ],
        axis=-1,
    )
    step_offset_xy = (translation_xy[:, None, :] * step_period[:, None, None] + yaw_offset * step_period[:, None, None]) * startup_ramp[:, :, None]

    trim_z = (
        action_terms.body_height[:, None]
        + controller_bundle.side_sign[None, :] * action_terms.roll_trim[:, None]
        + controller_bundle.front_sign[None, :] * action_terms.pitch_trim[:, None]
    ) * startup_ramp

    neutral = controller_bundle.neutral_foot_body_pos[None, :, :]
    forward_target = neutral + jnp.concatenate([step_offset_xy, trim_z[:, :, None]], axis=-1)
    backward_target = neutral + jnp.concatenate([-step_offset_xy, trim_z[:, :, None]], axis=-1)

    swing_height = (config.nominal_swing_height + action_terms.swing_height[:, None, None]) * startup_ramp[:, :, None]
    swing_target = _bezier_cubic(backward_target, forward_target, swing_height, local_phase[:, :, None])
    stance_target = (1.0 - local_phase[:, :, None]) * forward_target + local_phase[:, :, None] * backward_target
    desired_foot_body = jnp.where(swing_mask[:, :, None], swing_target, stance_target)


    workspace_min = controller_bundle.neutral_foot_body_pos[None, :, :] + jnp.asarray(config.workspace_delta_min, dtype=jnp.float32)
    workspace_max = controller_bundle.neutral_foot_body_pos[None, :, :] + jnp.asarray(config.workspace_delta_max, dtype=jnp.float32)
    desired_foot_body = jnp.clip(desired_foot_body, workspace_min, workspace_max)

    foot_delta = desired_foot_body - controller_bundle.neutral_foot_body_pos[None, :, :]
    joint_delta = jnp.einsum("lij,blj->bli", controller_bundle.foot_jacobian_pinv, foot_delta)
    leg_joint_targets = jnp.clip(
        controller_bundle.leg_joint_pose[None, :, :] + joint_delta,
        controller_bundle.joint_lower[None, :, :],
        controller_bundle.joint_upper[None, :, :],
    )

    batch_size = action.shape[0]
    joint_targets = jnp.broadcast_to(controller_bundle.neutral_joint_pose[None, :], (batch_size, controller_bundle.neutral_joint_pose.shape[0]))
    flat_indices = controller_bundle.leg_joint_indices.reshape(-1)
    joint_targets = joint_targets.at[:, flat_indices].set(leg_joint_targets.reshape(batch_size, -1))

    next_state = ResidualControllerState(
        phase=phase,
        filtered_command=filtered_command,
        prev_action=jnp.tanh(action),
        elapsed_sec=elapsed_sec,
    )
    return next_state, joint_targets, desired_foot_body
