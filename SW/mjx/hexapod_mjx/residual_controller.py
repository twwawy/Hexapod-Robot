from __future__ import annotations

"""Tripod nominal controller with a Cartesian foot-height residual interface.

The controller owns the complete nominal locomotion path:

``command -> position/heading PI -> tripod gait -> nominal feet -> RL residual
   -> contact safety -> posture PI -> IK -> joint limits``.

The first residual-RL curriculum deliberately exposes only one scalar per leg:
the swing-foot vertical correction ``Δz``.  It therefore cannot change the
nominal step length, gait timing, landing XY location, or body attitude.  The
policy action is masked to zero for stance legs, and early swing contact holds
the current foot target instead of allowing RL to pull it through the ground.

The nominal path is ported from ``~/Downloads/mjx/tripod_controller.py``:
quintic stance/swing timing, radial swing clearance, the documented analytical
three-link IK, and its mirrored left/right servo mapping.  The policy still
acts in Cartesian foot space; the controller alone owns conversion to joint
targets and all clamps.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .model import FOOT_CONTACT_POINT_COUNT, HexapodModelBundle, estimate_standing_root_height






LEGS = ("LF", "LM", "LB", "RF", "RM", "RB")
# One swing-foot vertical residual per leg: [LF, LM, LB, RF, RM, RB].
ACTION_DIM = len(LEGS)
RESIDUAL_INTERFACE = "downloads_tripod_ik_swing_delta_z_v3"
# Matches the original controller exactly.  The old project convention used
# the complementary group as phase zero; the two are dynamically equivalent,
# but preserving this order makes direct comparison with Downloads/mjx simple.
REFERENCE_TRIPOD_A = frozenset(("RF", "RB", "LM"))
RIGHT_LEGS = frozenset(("RF", "RM", "RB"))
LINK_1 = 0.074
LINK_2 = 0.121
LINK_3 = 0.230
NOMINAL_LOCAL_FOOT = (0.218728, 0.0, -0.287006)
MODEL_FORWARD = (0.0, -1.0, 0.0)


def validate_residual_interface(metadata: Mapping[str, Any] | None) -> None:
    """Reject policies trained with a nominal controller of different meaning."""
    saved_interface = metadata.get("residual_interface") if isinstance(metadata, Mapping) else None
    if saved_interface != RESIDUAL_INTERFACE:
        raise ValueError(
            "Checkpoint controller interface is incompatible: "
            f"checkpoint has {saved_interface!r}, current controller requires {RESIDUAL_INTERFACE}. "
            "Run a fresh source-tripod residual training before evaluation or visualization."
        )


class ResidualControllerBundle(NamedTuple):
    neutral_joint_pose: jnp.ndarray
    leg_joint_indices: jnp.ndarray
    joint_lower: jnp.ndarray
    joint_upper: jnp.ndarray
    hip_body_pos: jnp.ndarray
    leg_outward_body: jnp.ndarray
    leg_tangent_body: jnp.ndarray
    reference_foot_body_pos: jnp.ndarray
    right_leg_mask: jnp.ndarray
    tripod_is_a: jnp.ndarray
    foot_geom_ids: jnp.ndarray
    foot_support_geom_ids: jnp.ndarray
    reset_root_height: jnp.ndarray



@dataclass(frozen=True)
class ResidualControllerConfig:
    # Downloads/mjx advances a 2 ms model and refreshes targets every 5 ms.
    # Three 2 ms physics steps is the closest JAX-scan representation.
    physics_steps_per_control: int = 3
    policy_controls_per_action: int = 2
    command_deadzone: float = 0.03
    command_rate_limit: tuple[float, float, float] = (0.9, 0.6, 1.2)
    # A source gait speed is a physical m/s target, not an arbitrary neural
    # action amplitude.  Sampling commands above this would make the reward
    # ask for speeds the copied nominal controller can never produce.
    command_limits: tuple[float, float, float] = (0.06, 0.04, 0.25)
    # Classical body-twist controller.  RC commands define a moving desired
    # body position/heading; these PI loops turn its tracking error back into
    # the twist used by the nominal tripod gait.
    body_twist_limits: tuple[float, float, float] = (0.06, 0.04, 0.25)
    # The imported controller is open-loop at its nominal layer.  Keep these
    # optional PI hooks at zero until the direct tripod baseline is validated.
    translation_pi_kp: tuple[float, float] = (0.0, 0.0)
    translation_pi_ki: tuple[float, float] = (0.0, 0.0)
    translation_integral_limit: tuple[float, float] = (0.25, 0.20)
    heading_pi_kp: float = 0.0
    heading_pi_ki: float = 0.0
    heading_integral_limit: float = 0.60
    # IMU-style roll/pitch/height PI.  Its output remains classical and is
    # applied as a bounded Cartesian foot-height overlay; RL cannot command it.
    posture_pi_kp: tuple[float, float, float] = (0.0, 0.0, 0.0)
    posture_pi_ki: tuple[float, float, float] = (0.0, 0.0, 0.0)
    posture_integral_limit: tuple[float, float, float] = (0.50, 0.50, 0.06)
    posture_foot_z_limit: float = 0.020
    # Direct values from Downloads/mjx/GaitConfig.
    reference_phase_time: float = 0.50
    reference_ramp_time: float = 1.0
    nominal_swing_height: float = 0.060
    nominal_radial_offset: float = 0.010
    # Initial residual curriculum: RL may only lift/lower a swing foot by ±3 cm.
    # Larger Cartesian residuals and XY foothold changes are deliberately out of
    # scope until this 6-D policy is stable.
    residual_swing_z: float = 0.030
    workspace_delta_min: tuple[float, float, float] = (-0.14, -0.10, -0.10)
    workspace_delta_max: tuple[float, float, float] = (0.14, 0.10, 0.08)
    pd_kp: tuple[float, float, float] = (120.0, 120.0, 120.0)
    pd_kd: tuple[float, float, float] = (3.0, 3.0, 3.0)
    torque_limit: tuple[float, float, float] = (8.0, 8.0, 8.0)
    foot_contact_height: float = 0.03
    # Ignore the normal contact at swing take-off; only later contact is an
    # early landing that should override the residual and hold the foot.
    early_contact_phase_min: float = 0.25
    startup_duration_sec: float = 1.0
    joint_limit_margin: tuple[float, float, float] = (0.0, 0.0, 0.0)


def controller_config_from_metadata(metadata: Mapping[str, Any] | None) -> "ResidualControllerConfig":
    ppo_config = metadata.get("ppo_config") if isinstance(metadata, Mapping) else None
    if not isinstance(ppo_config, Mapping):
        return ResidualControllerConfig()

    def config_tuple(key: str, default: tuple[float, ...]) -> tuple[float, ...]:
        value = ppo_config.get(key, default)
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            return default
        return tuple(float(item) for item in value)

    return ResidualControllerConfig(
        joint_limit_margin=(
            float(ppo_config.get("joint_limit_margin_1", 0.0)),
            float(ppo_config.get("joint_limit_margin_2", 0.0)),
            float(ppo_config.get("joint_limit_margin_3", 0.0)),
        ),
        residual_swing_z=float(ppo_config.get("residual_swing_z", ResidualControllerConfig.residual_swing_z)),
        translation_pi_kp=config_tuple("translation_pi_kp", ResidualControllerConfig.translation_pi_kp),
        translation_pi_ki=config_tuple("translation_pi_ki", ResidualControllerConfig.translation_pi_ki),
        heading_pi_kp=float(ppo_config.get("heading_pi_kp", ResidualControllerConfig.heading_pi_kp)),
        heading_pi_ki=float(ppo_config.get("heading_pi_ki", ResidualControllerConfig.heading_pi_ki)),
        posture_pi_kp=config_tuple("posture_pi_kp", ResidualControllerConfig.posture_pi_kp),
        posture_pi_ki=config_tuple("posture_pi_ki", ResidualControllerConfig.posture_pi_ki),
        posture_foot_z_limit=float(ppo_config.get("posture_foot_z_limit", ResidualControllerConfig.posture_foot_z_limit)),
    )



class ResidualControllerState(NamedTuple):
    phase: jnp.ndarray
    filtered_command: jnp.ndarray
    prev_action: jnp.ndarray
    elapsed_sec: jnp.ndarray
    desired_position_world: jnp.ndarray
    desired_heading: jnp.ndarray
    translation_integral: jnp.ndarray
    heading_integral: jnp.ndarray
    posture_integral: jnp.ndarray


class ResidualActionTerms(NamedTuple):
    """Bounded Cartesian residual in metres, one vertical term per leg."""

    swing_z: jnp.ndarray


def build_residual_controller(
    bundle: HexapodModelBundle,
    config: ResidualControllerConfig,
) -> ResidualControllerBundle:
    """Build leg frames for the imported documented analytical IK."""
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
    joint_lower: list[list[float]] = []
    joint_upper: list[list[float]] = []
    hip_body_pos: list[np.ndarray] = []
    tripod_is_a: list[bool] = []
    foot_geom_ids: list[int] = []
    foot_support_geom_ids: list[list[int]] = []
    leg_outward_body: list[np.ndarray] = []
    leg_tangent_body: list[np.ndarray] = []
    reference_foot_body_pos: list[np.ndarray] = []
    right_leg_mask: list[bool] = []


    name_to_index = {name: idx for idx, name in enumerate(bundle.joint_names)}
    name_to_joint_id = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in bundle.joint_names
    }
    joint_limit_margin = np.asarray(config.joint_limit_margin, dtype=np.float64)
    if np.any(joint_limit_margin < 0.0):
        raise ValueError(f"joint_limit_margin must be non-negative, got {config.joint_limit_margin}")

    for leg in LEGS:
        leg_indices = [name_to_index[f"{leg}_{suffix}"] for suffix in (1, 2, 3)]
        leg_joint_indices.append(leg_indices)
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
        outward = hip_body.copy()
        outward[2] = 0.0
        outward /= np.linalg.norm(outward)
        tangent = np.asarray((-outward[1], outward[0], 0.0), dtype=np.float64)
        reference_foot = hip_body + outward * NOMINAL_LOCAL_FOOT[0]
        reference_foot[2] = hip_body[2] + NOMINAL_LOCAL_FOOT[2]
        leg_outward_body.append(outward.astype(np.float32))
        leg_tangent_body.append(tangent.astype(np.float32))
        reference_foot_body_pos.append(reference_foot.astype(np.float32))
        right_leg_mask.append(leg in RIGHT_LEGS)
        tripod_is_a.append(leg in REFERENCE_TRIPOD_A)

        foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_motor_horn_3_1_contact")
        foot_geom_ids.append(foot_geom_id)
        support_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_motor_horn_3_1_contact_{idx}")
            for idx in range(FOOT_CONTACT_POINT_COUNT)
        ]
        foot_support_geom_ids.append([support_id if support_id >= 0 else foot_geom_id for support_id in support_ids])
    reset_root_height = np.asarray(float(estimate_standing_root_height(bundle)), dtype=np.float32)

    return ResidualControllerBundle(
        neutral_joint_pose=jnp.asarray(np.asarray(bundle.default_joint_pose, dtype=np.float32)),
        leg_joint_indices=jnp.asarray(np.asarray(leg_joint_indices, dtype=np.int32)),
        joint_lower=jnp.asarray(np.asarray(joint_lower, dtype=np.float32)),
        joint_upper=jnp.asarray(np.asarray(joint_upper, dtype=np.float32)),
        hip_body_pos=jnp.asarray(np.asarray(hip_body_pos, dtype=np.float32)),
        leg_outward_body=jnp.asarray(np.asarray(leg_outward_body, dtype=np.float32)),
        leg_tangent_body=jnp.asarray(np.asarray(leg_tangent_body, dtype=np.float32)),
        reference_foot_body_pos=jnp.asarray(np.asarray(reference_foot_body_pos, dtype=np.float32)),
        right_leg_mask=jnp.asarray(np.asarray(right_leg_mask, dtype=bool)),
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
        desired_position_world=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        desired_heading=jnp.zeros((batch_size,), dtype=jnp.float32),
        translation_integral=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        heading_integral=jnp.zeros((batch_size,), dtype=jnp.float32),
        posture_integral=jnp.zeros((batch_size, 3), dtype=jnp.float32),
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


def _wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _world_xy_from_body_command(forward: jnp.ndarray, lateral: jnp.ndarray, heading: jnp.ndarray) -> jnp.ndarray:
    """Map this robot's body convention (forward=-Y, lateral=+X) to world XY."""
    cos_heading = jnp.cos(heading)
    sin_heading = jnp.sin(heading)
    return jnp.stack(
        [
            cos_heading * lateral + sin_heading * forward,
            sin_heading * lateral - cos_heading * forward,
        ],
        axis=-1,
    )


def _body_command_error_from_world_xy(world_error: jnp.ndarray, heading: jnp.ndarray) -> jnp.ndarray:
    """Return position error ordered as (forward, lateral) in the body frame."""
    cos_heading = jnp.cos(heading)
    sin_heading = jnp.sin(heading)
    lateral_error = cos_heading * world_error[:, 0] + sin_heading * world_error[:, 1]
    body_y_error = -sin_heading * world_error[:, 0] + cos_heading * world_error[:, 1]
    return jnp.stack([-body_y_error, lateral_error], axis=-1)


def _classical_body_twist(
    config: ResidualControllerConfig,
    state: ResidualControllerState,
    filtered_command: jnp.ndarray,
    body_position_world: jnp.ndarray,
    body_quat: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate RC references and close the nominal position/heading PI loops."""
    _, _, measured_heading = quat_roll_pitch_yaw(body_quat)
    desired_heading = _wrap_angle(state.desired_heading + filtered_command[:, 2] * dt)
    desired_position_world = state.desired_position_world + _world_xy_from_body_command(
        filtered_command[:, 0],
        filtered_command[:, 1],
        desired_heading,
    ) * dt

    position_error = _body_command_error_from_world_xy(
        desired_position_world - body_position_world[:, 0:2],
        measured_heading,
    )
    translation_integral_limit = jnp.asarray(config.translation_integral_limit, dtype=jnp.float32)
    translation_integral = jnp.clip(
        state.translation_integral + position_error * dt,
        -translation_integral_limit,
        translation_integral_limit,
    )
    translation_twist = (
        filtered_command[:, 0:2]
        + position_error * jnp.asarray(config.translation_pi_kp, dtype=jnp.float32)
        + translation_integral * jnp.asarray(config.translation_pi_ki, dtype=jnp.float32)
    )

    heading_error = _wrap_angle(desired_heading - measured_heading)
    heading_integral = jnp.clip(
        state.heading_integral + heading_error * dt,
        -config.heading_integral_limit,
        config.heading_integral_limit,
    )
    yaw_twist = (
        filtered_command[:, 2]
        + config.heading_pi_kp * heading_error
        + config.heading_pi_ki * heading_integral
    )
    body_twist = jnp.concatenate([translation_twist, yaw_twist[:, None]], axis=-1)
    body_twist_limit = jnp.asarray(config.body_twist_limits, dtype=jnp.float32)
    body_twist = jnp.clip(body_twist, -body_twist_limit, body_twist_limit)
    return body_twist, desired_position_world, desired_heading, translation_integral, heading_integral


def _scale_action(action: jnp.ndarray, config: ResidualControllerConfig) -> ResidualActionTerms:
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"Residual action must have {ACTION_DIM} terms, got {action.shape[-1]}.")
    bounded = jnp.clip(action, -1.0, 1.0)
    return ResidualActionTerms(swing_z=bounded * config.residual_swing_z)


def residual_action_metres(action: jnp.ndarray, config: ResidualControllerConfig) -> jnp.ndarray:
    """Return the bounded physical residual (metres) used by the controller."""
    return _scale_action(action, config).swing_z


def _apply_contact_adaptation(
    nominal_foot_body: jnp.ndarray,
    swing_z_residual: jnp.ndarray,
    swing_mask: jnp.ndarray,
    foot_contacts: jnp.ndarray | None,
    current_foot_body: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RL only in swing, then let contact safety override it.

    ``foot_contacts`` and ``current_foot_body`` are measured at the start of
    the control tick.  A contact detected during nominal swing is an early
    landing.  In that case the foot is held at its measured position, which
    prevents the residual from penetrating or dragging the contact point.
    """
    residual_z = jnp.where(swing_mask, swing_z_residual, 0.0)
    corrected = nominal_foot_body.at[:, :, 2].add(residual_z)

    if foot_contacts is None or current_foot_body is None:
        return corrected, residual_z

    early_landing = swing_mask & foot_contacts.astype(bool)
    corrected = jnp.where(early_landing[:, :, None], current_foot_body, corrected)
    applied_residual_z = jnp.where(early_landing, 0.0, residual_z)
    return corrected, applied_residual_z


def _apply_nominal_posture_overlay(
    foot_body: jnp.ndarray,
    body_quat: jnp.ndarray,
    body_height: jnp.ndarray,
    controller_bundle: ResidualControllerBundle,
    config: ResidualControllerConfig,
    posture_integral: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Close roll/pitch/height PI and add its bounded classical foot overlay.

    The body convention is lateral ``+X``, forward ``-Y``, and up ``+Z``.
    A virtual corrective roll/pitch rotates the nominal foot plane while the
    height loop moves all targets down when the body is below its stand height.
    The learned residual is not part of this calculation.
    """
    roll, pitch, _ = quat_roll_pitch_yaw(body_quat)
    posture_error = jnp.stack(
        [
            -roll,
            -pitch,
            controller_bundle.reset_root_height - body_height,
        ],
        axis=-1,
    )
    integral_limit = jnp.asarray(config.posture_integral_limit, dtype=jnp.float32)
    next_posture_integral = jnp.clip(
        posture_integral + posture_error * dt,
        -integral_limit,
        integral_limit,
    )
    posture_control = (
        posture_error * jnp.asarray(config.posture_pi_kp, dtype=jnp.float32)
        + next_posture_integral * jnp.asarray(config.posture_pi_ki, dtype=jnp.float32)
    )
    virtual_roll = posture_control[:, 0, None]
    virtual_pitch = posture_control[:, 1, None]
    height_control = posture_control[:, 2, None]
    # For a small virtual rotation dtheta, (dtheta x p)_z = droll*y - dpitch*x.
    posture_z = virtual_roll * foot_body[:, :, 1] - virtual_pitch * foot_body[:, :, 0] - height_control
    posture_z = jnp.clip(posture_z, -config.posture_foot_z_limit, config.posture_foot_z_limit)
    return foot_body.at[:, :, 2].add(posture_z), next_posture_integral


def _quintic(tau: jnp.ndarray) -> jnp.ndarray:
    """The zero-velocity/zero-acceleration timing curve from Downloads/mjx."""
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def _documented_inverse_kinematics(
    foot_body: jnp.ndarray,
    controller_bundle: ResidualControllerBundle,
) -> jnp.ndarray:
    """Port of ``TripodGaitController._inverse_kinematics`` in batch form.

    ``foot_body`` is first expressed in each leg's outward/tangent frame.  The
    branch and left/right raw-axis conversion are deliberately identical to
    the standalone controller, replacing the old standing-pose Jacobian
    approximation.
    """
    relative = foot_body - controller_bundle.hip_body_pos[None, :, :]
    x = jnp.sum(relative * controller_bundle.leg_outward_body[None, :, :], axis=-1)
    y = jnp.sum(relative * controller_bundle.leg_tangent_body[None, :, :], axis=-1)
    z = relative[:, :, 2]
    radius = jnp.hypot(x, y)
    rho = radius - LINK_1
    cosine3 = (rho * rho + z * z - LINK_2**2 - LINK_3**2) / (2.0 * LINK_2 * LINK_3)
    cosine3 = jnp.clip(cosine3, -1.0, 1.0)
    theta3 = jnp.arctan2(-jnp.sqrt(jnp.maximum(0.0, 1.0 - cosine3 * cosine3)), cosine3)
    theta2 = jnp.arctan2(z, rho) - jnp.arctan2(
        LINK_3 * jnp.sin(theta3), LINK_2 + LINK_3 * jnp.cos(theta3)
    )
    servo = jnp.stack([jnp.arctan2(y, x), -theta2, -theta3], axis=-1)
    right = controller_bundle.right_leg_mask[None, :]
    raw_joint_2 = jnp.where(right, -servo[:, :, 1], servo[:, :, 1])
    raw_joint_3 = jnp.where(right, servo[:, :, 2], -servo[:, :, 2])
    return jnp.stack([servo[:, :, 0], raw_joint_2, raw_joint_3], axis=-1)




def controller_step(
    model_bundle: HexapodModelBundle,
    controller_bundle: ResidualControllerBundle,
    config: ResidualControllerConfig,
    state: ResidualControllerState,
    command_target: jnp.ndarray,
    action: jnp.ndarray,
    *,
    foot_contacts: jnp.ndarray | None = None,
    current_foot_body: jnp.ndarray | None = None,
    body_position_world: jnp.ndarray | None = None,
    body_quat: jnp.ndarray | None = None,
) -> tuple[ResidualControllerState, jnp.ndarray, jnp.ndarray]:
    """Advance one tick of the nominal gait plus its bounded foot residual.

    Args:
        foot_contacts: Measured ``(batch, 6)`` contacts before this tick.
        current_foot_body: Measured ``(batch, 6, 3)`` foot positions in the
            body frame.  Together with ``foot_contacts`` this enables the
            early-landing safety override after the RL residual.
        body_position_world: Measured floating-base position for the classical
            position and height PI loops.
        body_quat: Measured floating-base quaternion for heading and posture
            PI.  RL never receives authority over these classical corrections.
    """
    dt = control_dt(model_bundle, config)
    command_limit = jnp.asarray(config.command_limits, dtype=jnp.float32)
    command_rate_limit = jnp.asarray(config.command_rate_limit, dtype=jnp.float32)
    command_target = _apply_deadzone(jnp.clip(command_target, -command_limit, command_limit), config.command_deadzone)
    filtered_command = _slew_limit(state.filtered_command, command_target, command_rate_limit, dt)

    batch_size = action.shape[0]
    if body_position_world is None:
        body_position_world = jnp.zeros((batch_size, 3), dtype=jnp.float32)
        body_position_world = body_position_world.at[:, 2].set(controller_bundle.reset_root_height)
    if body_quat is None:
        body_quat = jnp.zeros((batch_size, 4), dtype=jnp.float32).at[:, 0].set(1.0)
    body_twist, desired_position_world, desired_heading, translation_integral, heading_integral = _classical_body_twist(
        config,
        state,
        filtered_command,
        body_position_world,
        body_quat,
        dt,
    )

    action_terms = _scale_action(action, config)
    elapsed_sec = state.elapsed_sec + dt
    gait_elapsed_sec = jnp.maximum(elapsed_sec - config.startup_duration_sec, 0.0)
    full_cycle_time = 2.0 * config.reference_phase_time
    phase = jnp.mod(gait_elapsed_sec / full_cycle_time, 1.0)
    gait_active = elapsed_sec >= config.startup_duration_sec
    first_half = phase[:, None] < 0.5
    local_phase = jnp.where(first_half, phase[:, None] * 2.0, (phase[:, None] - 0.5) * 2.0)
    source_swing_mask = jnp.where(controller_bundle.tripod_is_a[None, :], first_half, ~first_half)
    swing_mask = source_swing_mask & gait_active[:, None]

    # Direct port of TripodGaitController._foot_target.  Its forward stride is
    # model-local -Y, its swing uses a quintic phase, and only swing feet get
    # the radial clearance/lift terms.  Lateral/yaw are conservative additive
    # extensions; at the initial forward-only curriculum they are identically
    # zero, so the motion is exactly the imported reference gait.
    smooth = _quintic(local_phase)
    stride_phase = jnp.where(source_swing_mask, smooth - 0.5, 0.5 - local_phase)
    ramp = jnp.clip(gait_elapsed_sec / config.reference_ramp_time, 0.0, 1.0)
    forward_step = body_twist[:, 0] * config.reference_phase_time * ramp
    lateral_step = body_twist[:, 1] * config.reference_phase_time * ramp
    yaw_step = body_twist[:, 2] * config.reference_phase_time * ramp
    forward_offset = jnp.asarray(MODEL_FORWARD, dtype=jnp.float32)[None, None, :] * (
        stride_phase[:, :, None] * forward_step[:, None, None]
    )
    lateral_offset = jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float32)[None, None, :] * (
        stride_phase[:, :, None] * lateral_step[:, None, None]
    )
    hip_xy = controller_bundle.hip_body_pos[None, :, 0:2]
    yaw_xy = jnp.stack(
        [-yaw_step[:, None] * hip_xy[:, :, 1], yaw_step[:, None] * hip_xy[:, :, 0]],
        axis=-1,
    ) * stride_phase[:, :, None]
    yaw_offset = jnp.concatenate([yaw_xy, jnp.zeros_like(yaw_xy[:, :, :1])], axis=-1)
    swing_arc = 4.0 * smooth * (1.0 - smooth) * source_swing_mask.astype(jnp.float32)
    radial_offset = controller_bundle.leg_outward_body[None, :, :] * (
        config.nominal_radial_offset * swing_arc[:, :, None]
    )
    lift = jnp.zeros_like(forward_offset).at[:, :, 2].set(config.nominal_swing_height * swing_arc)
    moving_nominal = (
        controller_bundle.reference_foot_body_pos[None, :, :]
        + forward_offset
        + lateral_offset
        + yaw_offset
        + radial_offset
        + lift
    )
    nominal_foot_body = jnp.where(
        gait_active[:, None, None], moving_nominal, controller_bundle.reference_foot_body_pos[None, :, :]
    )
    early_landing = (
        swing_mask
        & foot_contacts.astype(bool)
        & (local_phase >= config.early_contact_phase_min)
        if foot_contacts is not None
        else None
    )
    corrected_foot_body, _ = _apply_contact_adaptation(
        nominal_foot_body,
        action_terms.swing_z,
        swing_mask,
        early_landing,
        current_foot_body,
    )
    # The posture layer follows contact adaptation and remains fully classical.
    # Re-applying the hold below preserves the higher-priority early-contact
    # safety decision even when posture PI is active.
    desired_foot_body, posture_integral = _apply_nominal_posture_overlay(
        corrected_foot_body,
        body_quat,
        body_position_world[:, 2],
        controller_bundle,
        config,
        state.posture_integral,
        dt,
    )
    if early_landing is not None and current_foot_body is not None:
        desired_foot_body = jnp.where(early_landing[:, :, None], current_foot_body, desired_foot_body)

    workspace_min = controller_bundle.reference_foot_body_pos[None, :, :] + jnp.asarray(config.workspace_delta_min, dtype=jnp.float32)
    workspace_max = controller_bundle.reference_foot_body_pos[None, :, :] + jnp.asarray(config.workspace_delta_max, dtype=jnp.float32)
    desired_foot_body = jnp.clip(desired_foot_body, workspace_min, workspace_max)

    leg_joint_targets = jnp.clip(
        _documented_inverse_kinematics(desired_foot_body, controller_bundle),
        controller_bundle.joint_lower[None, :, :],
        controller_bundle.joint_upper[None, :, :],
    )

    joint_targets = jnp.broadcast_to(controller_bundle.neutral_joint_pose[None, :], (batch_size, controller_bundle.neutral_joint_pose.shape[0]))
    flat_indices = controller_bundle.leg_joint_indices.reshape(-1)
    joint_targets = joint_targets.at[:, flat_indices].set(leg_joint_targets.reshape(batch_size, -1))

    next_state = ResidualControllerState(
        phase=phase,
        filtered_command=filtered_command,
        prev_action=jnp.clip(action, -1.0, 1.0),
        elapsed_sec=elapsed_sec,
        desired_position_world=desired_position_world,
        desired_heading=desired_heading,
        translation_integral=translation_integral,
        heading_integral=heading_integral,
        posture_integral=posture_integral,
    )
    return next_state, joint_targets, desired_foot_body
