"""JAX port of the STM32 manual-walking controller used by MJX training.

The deployable implementation lives under
``SW/STM32/workspace/Hexapod/Core/Src/high_control``.  This module keeps the
same 5 ms controller state, gait/contact transitions, trajectories, posture
feedback, IK limits and joint rate limit in JAX so that thousands of MJX
environments can be vmapped on a GPU.  The policy is inserted only as a
bounded Cartesian foot residual immediately before the final workspace/IK
safety layer.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


FIRMWARE_CONTROL_DT = 0.005
GAIT_PHASE_TIME = 0.5
MAX_LINEAR_SPEED = 0.28
MAX_YAW_RATE = jp.deg2rad(45.0)
SWING_HEIGHT = 0.06
SWING_HEIGHT_MIN = 0.04
SWING_HEIGHT_MAX = 0.25
SWING_RADIAL_OFFSET = 0.07
EARLY_LANDING_PROGRESS = 0.50
LATE_LANDING_SPEED = 0.20
LATE_INWARD_SPEED = 0.16
JOINT_LIMIT = jp.deg2rad(135.0)
JOINT_RATE = jp.deg2rad(315.8)

LINK_1 = 0.074
LINK_2 = 0.121
LINK_3 = 0.230
ROOT_DISTANCE = 0.1845
BASE_FOOT_RADIUS = 0.218728
BASE_FOOT_Z = -0.287006
# Keep commanded feet one millimetre inside the analytic 2-link workspace.
# This mirrors ROBOT_WORKSPACE_MARGIN_M in the deployable STM32 controller.
WORKSPACE_MARGIN = 0.001

LEG_STANCE = 0
LEG_SWING = 1
LEG_LATE_LANDING = 2

LEG_ANGLES = jp.deg2rad(jp.array((-45.0, -90.0, -135.0, 45.0, 90.0, 135.0)))
_DIAGONAL = ROOT_DISTANCE / jp.sqrt(2.0)
LEG_ROOTS = jp.array(
    (
        (_DIAGONAL, -_DIAGONAL, 0.0),
        (0.0, -ROOT_DISTANCE, 0.0),
        (-_DIAGONAL, -_DIAGONAL, 0.0),
        (_DIAGONAL, _DIAGONAL, 0.0),
        (0.0, ROOT_DISTANCE, 0.0),
        (-_DIAGONAL, _DIAGONAL, 0.0),
    )
)
BASE_FEET = LEG_ROOTS + jp.stack(
    (
        BASE_FOOT_RADIUS * jp.cos(LEG_ANGLES),
        BASE_FOOT_RADIUS * jp.sin(LEG_ANGLES),
        jp.full(6, BASE_FOOT_Z),
    ),
    axis=-1,
)
MODEL_SIGNS = jp.array(
    (
        (1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0),
    )
)
# Local Cartesian residual limits.  Swing X/Y remain Cartesian offsets, while
# Z has phase-dependent semantics: swing Z selects a 4--25 cm arc height,
# stance Z is limited to 2 cm and late-landing Z is controller-owned.
RESIDUAL_SCALE = jp.array((0.04, 0.02, 0.02))


class FirmwareState(NamedTuple):
    first_step: jax.Array
    throttle_filter: jax.Array
    yaw_filter: jax.Array
    yaw_reference: jax.Array
    position_reference: jax.Array
    previous_twist: jax.Array
    gait_applied: jax.Array
    phase_index: jax.Array
    phase_time: jax.Array
    airborne_seen: jax.Array
    landed: jax.Array
    gait_initialized: jax.Array
    gait_running: jax.Array
    stop_pending: jax.Array
    foot_memory: jax.Array
    swing_start: jax.Array
    previous_leg_state: jax.Array
    adapted_stance: jax.Array
    custom_swing: jax.Array
    posture_command: jax.Array
    last_ik: jax.Array
    previous_joint: jax.Array
    residual_filter: jax.Array


class FirmwareOutput(NamedTuple):
    model_joint_targets: jax.Array
    servo_joint_targets: jax.Array
    foot_targets_body: jax.Array
    applied_twist: jax.Array
    gait_progress: jax.Array
    gait_state: jax.Array
    swing_height_command: jax.Array
    ik_valid: jax.Array
    policy_valid: jax.Array
    foot_limited: jax.Array
    gait_enabled: jax.Array
    gait_accepted: jax.Array
    posture_accepted: jax.Array


def _body_to_leg(feet_body: jax.Array) -> jax.Array:
    delta = feet_body - LEG_ROOTS
    cosine = jp.cos(LEG_ANGLES)
    sine = jp.sin(LEG_ANGLES)
    return jp.stack(
        (
            delta[..., 0] * cosine + delta[..., 1] * sine,
            -delta[..., 0] * sine + delta[..., 1] * cosine,
            delta[..., 2],
        ),
        axis=-1,
    )


def _leg_to_body(feet_local: jax.Array) -> jax.Array:
    cosine = jp.cos(LEG_ANGLES)
    sine = jp.sin(LEG_ANGLES)
    return LEG_ROOTS + jp.stack(
        (
            feet_local[..., 0] * cosine - feet_local[..., 1] * sine,
            feet_local[..., 0] * sine + feet_local[..., 1] * cosine,
            feet_local[..., 2],
        ),
        axis=-1,
    )


def _solve_ik(feet_body: jax.Array) -> tuple[jax.Array, jax.Array]:
    local = _body_to_leg(feet_body)
    radial = jp.linalg.norm(local[..., :2], axis=-1)
    planar = radial - LINK_1
    cosine_knee_raw = (
        planar * planar + local[..., 2] * local[..., 2] - LINK_2**2 - LINK_3**2
    ) / (2.0 * LINK_2 * LINK_3)
    cosine_knee = jp.clip(cosine_knee_raw, -1.0, 1.0)
    sine_knee = jp.sqrt(jp.maximum(0.0, 1.0 - cosine_knee * cosine_knee))
    angles = jp.stack(
        (
            jp.arctan2(local[..., 1], local[..., 0]),
            jp.arctan2(-local[..., 2], planar)
            - jp.arctan2(LINK_3 * sine_knee, LINK_2 + LINK_3 * cosine_knee),
            jp.arctan2(sine_knee, cosine_knee),
        ),
        axis=-1,
    )
    finite = jp.all(jp.isfinite(feet_body), axis=-1) & jp.all(
        jp.isfinite(angles), axis=-1
    )
    reachable = (cosine_knee_raw >= -1.000001) & (cosine_knee_raw <= 1.000001)
    within_joint_limits = jp.all(jp.abs(angles) <= JOINT_LIMIT, axis=-1)
    return angles, finite & reachable & within_joint_limits


def _limit_foot_reach(feet_body: jax.Array) -> tuple[jax.Array, jax.Array]:
    local = _body_to_leg(feet_body)
    radial = jp.linalg.norm(local[..., :2], axis=-1)
    planar = radial - LINK_1
    reach = jp.sqrt(planar * planar + local[..., 2] * local[..., 2])
    limited_reach = jp.clip(
        reach,
        abs(LINK_2 - LINK_3) + WORKSPACE_MARGIN,
        LINK_2 + LINK_3 - WORKSPACE_MARGIN,
    )
    was_limited = jp.abs(limited_reach - reach) > 1.0e-9
    scale = limited_reach / jp.maximum(reach, 1.0e-9)
    planar_limited = jp.where(was_limited, planar * scale, planar)
    z_limited = jp.where(was_limited, local[..., 2] * scale, local[..., 2])
    xy_scale = (LINK_1 + planar_limited) / jp.maximum(radial, 1.0e-9)
    local_limited = local.at[..., 0].set(
        jp.where(was_limited, local[..., 0] * xy_scale, local[..., 0])
    )
    local_limited = local_limited.at[..., 1].set(
        jp.where(was_limited, local[..., 1] * xy_scale, local[..., 1])
    )
    local_limited = local_limited.at[..., 2].set(z_limited)
    return _leg_to_body(local_limited), was_limited


def _rotation_matrix(rpy: jax.Array) -> jax.Array:
    roll, pitch, yaw = rpy
    cr, sr = jp.cos(roll), jp.sin(roll)
    cp, sp = jp.cos(pitch), jp.sin(pitch)
    cy, sy = jp.cos(yaw), jp.sin(yaw)
    return jp.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def _rotate_inverse(vectors: jax.Array, rpy: jax.Array) -> jax.Array:
    return vectors @ _rotation_matrix(rpy)


def _all_feet_valid(feet_body: jax.Array, posture: jax.Array) -> jax.Array:
    _, valid = _solve_ik(_rotate_inverse(feet_body, posture))
    return jp.all(valid)


def _quintic(progress: jax.Array) -> jax.Array:
    progress = jp.clip(progress, 0.0, 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _bridge_normalized_command(
    value: jax.Array, maximum: jax.Array, deadband: float
) -> jax.Array:
    """Mirror Bridge_CommandToRaw followed by DroneController_Normalize."""
    normalized = jp.clip(value / maximum, -1.0, 1.0)
    raw_magnitude = deadband + jp.abs(normalized) * (1000.0 - deadband)
    # C lroundf rounds halves away from zero.
    raw = jp.sign(normalized) * jp.floor(raw_magnitude + 0.5)
    return jp.where(
        jp.abs(value) < 1.0e-6,
        0.0,
        jp.sign(raw) * (jp.abs(raw) - deadband) / (1000.0 - deadband),
    )


def _swing(
    progress: jax.Array,
    start: jax.Array,
    end: jax.Array,
    height: float = SWING_HEIGHT,
    radial_offset: float = SWING_RADIAL_OFFSET,
) -> jax.Array:
    scaled = _quintic(progress)
    # The firmware uses a cubic Bezier with duplicated start/end controls for
    # XY, after applying quintic time scaling to the Bezier parameter.
    blend = (scaled * scaled * (3.0 - 2.0 * scaled))[..., None]
    result = start + blend * (end - start)
    lift = 4.0 * height * scaled * (1.0 - scaled)
    bulge = 4.0 * radial_offset * scaled * (1.0 - scaled)
    return result + jp.stack(
        (
            bulge * jp.cos(LEG_ANGLES),
            bulge * jp.sin(LEG_ANGLES),
            lift * jp.ones(6),
        ),
        axis=-1,
    )


def _adaptive_swing_height(action_z: jax.Array) -> jax.Array:
    """Map zero action to the 6 cm nominal and use the full signed range."""
    action_z = jp.clip(action_z, -1.0, 1.0)
    return jp.where(
        action_z >= 0.0,
        SWING_HEIGHT + action_z * (SWING_HEIGHT_MAX - SWING_HEIGHT),
        SWING_HEIGHT + action_z * (SWING_HEIGHT - SWING_HEIGHT_MIN),
    )


def _phase_gated_policy_residual(
    filtered_action: jax.Array,
    gait_state: jax.Array,
    gait_progress: jax.Array,
    swing_boost: jax.Array = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Return local residuals without moving swing takeoff or touchdown Z."""
    action = filtered_action.reshape(6, 3)
    swing = gait_state == LEG_SWING
    stance = gait_state == LEG_STANCE
    swing_height = _adaptive_swing_height(action[:, 2])
    scaled_progress = _quintic(gait_progress)
    swing_envelope = 4.0 * scaled_progress * (1.0 - scaled_progress)

    xy = action[:, :2] * RESIDUAL_SCALE[:2]
    xy = jp.where(swing[:, None], xy, 0.0)
    bounded_boost = jp.clip(swing_boost, 0.0, 0.06)
    swing_z = jp.minimum(
        (swing_height - SWING_HEIGHT + bounded_boost) * swing_envelope,
        SWING_HEIGHT_MAX - SWING_HEIGHT,
    )
    stance_z = action[:, 2] * RESIDUAL_SCALE[2]
    z = jp.where(swing, swing_z, jp.where(stance, stance_z, 0.0))
    return jp.concatenate((xy, z[:, None]), axis=-1), swing_height


def _preview_gait(candidate: jax.Array, posture: jax.Array) -> jax.Array:
    displacement = jp.stack(
        (
            GAIT_PHASE_TIME * (-candidate[0] + candidate[3] * BASE_FEET[:, 1]),
            GAIT_PHASE_TIME * (-candidate[1] - candidate[3] * BASE_FEET[:, 0]),
            jp.full(6, GAIT_PHASE_TIME * -candidate[2]),
        ),
        axis=-1,
    )
    front = BASE_FEET - 0.5 * displacement
    rear = BASE_FEET + 0.5 * displacement
    progress = jp.linspace(0.0, 1.0, 21)[:, None]
    stance_samples = front[None, ...] + progress[..., None] * (
        rear - front
    )[None, ...]
    swing_samples = jax.vmap(_swing, in_axes=(0, None, None))(
        progress[:, 0], rear, front
    )
    _, base_valid = _solve_ik(_rotate_inverse(BASE_FEET, posture))
    _, stance_valid = _solve_ik(_rotate_inverse(stance_samples, posture))
    _, swing_valid = _solve_ik(_rotate_inverse(swing_samples, posture))
    return jp.all(base_valid) & jp.all(stance_valid) & jp.all(swing_valid)


def initial_state() -> FirmwareState:
    base_angles, base_valid = _solve_ik(BASE_FEET)
    base_angles = jp.where(base_valid[:, None], base_angles, jp.zeros_like(base_angles))
    return FirmwareState(
        first_step=jp.ones((), dtype=jp.bool_),
        throttle_filter=jp.zeros(()),
        yaw_filter=jp.zeros(()),
        yaw_reference=jp.zeros(()),
        position_reference=jp.zeros(2),
        previous_twist=jp.zeros(4),
        gait_applied=jp.zeros(4),
        phase_index=jp.zeros((), dtype=jp.int32),
        phase_time=jp.zeros(()),
        airborne_seen=jp.zeros(6, dtype=jp.bool_),
        landed=jp.zeros(6, dtype=jp.bool_),
        gait_initialized=jp.zeros((), dtype=jp.bool_),
        gait_running=jp.zeros((), dtype=jp.bool_),
        stop_pending=jp.zeros((), dtype=jp.bool_),
        foot_memory=BASE_FEET,
        swing_start=BASE_FEET,
        previous_leg_state=jp.zeros(6, dtype=jp.int32),
        adapted_stance=jp.zeros(6, dtype=jp.bool_),
        custom_swing=jp.zeros(6, dtype=jp.bool_),
        posture_command=jp.zeros(3),
        last_ik=base_angles,
        previous_joint=base_angles,
        residual_filter=jp.zeros(18),
    )


def initial_output() -> FirmwareOutput:
    state = initial_state()
    return FirmwareOutput(
        model_joint_targets=state.previous_joint * MODEL_SIGNS,
        servo_joint_targets=state.previous_joint,
        foot_targets_body=BASE_FEET,
        applied_twist=jp.zeros(4),
        gait_progress=jp.zeros(6),
        gait_state=jp.zeros(6, dtype=jp.int32),
        swing_height_command=jp.full(6, SWING_HEIGHT),
        ik_valid=jp.ones(6, dtype=jp.bool_),
        policy_valid=jp.ones(6, dtype=jp.bool_),
        foot_limited=jp.zeros(6, dtype=jp.bool_),
        gait_enabled=jp.zeros((), dtype=jp.bool_),
        gait_accepted=jp.ones((), dtype=jp.bool_),
        posture_accepted=jp.ones((), dtype=jp.bool_),
    )


def _update_gait(
    state: FirmwareState, tripod_enable: jax.Array, contacts: jax.Array
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    new_run = tripod_enable & (~state.gait_running)
    running = state.gait_running | tripod_enable
    stop_pending = jp.where(tripod_enable, False, state.stop_pending | running)
    initialized = jp.where(new_run, False, state.gait_initialized)
    phase_index = jp.where(new_run, 0, state.phase_index)
    phase_time = jp.where(new_run, 0.0, state.phase_time)
    airborne = jp.where(new_run, jp.zeros_like(state.airborne_seen), state.airborne_seen)
    landed = jp.where(new_run, jp.zeros_like(state.landed), state.landed)

    phase_time = jp.where(
        running & initialized,
        phase_time + FIRMWARE_CONTROL_DT,
        phase_time,
    )
    initialized = initialized | running
    progress = jp.clip(phase_time / GAIT_PHASE_TIME, 0.0, 1.0)
    leg_group_135 = (jp.arange(6) % 2) == 0
    swing_group = leg_group_135 == ((phase_index % 2) == 0)
    airborne = airborne | (running & swing_group & (~contacts))
    landed = landed | (
        running
        & swing_group
        & (airborne | stop_pending)
        & contacts
        & (progress >= EARLY_LANDING_PROGRESS)
    )
    all_swing_landed = jp.all((~swing_group) | landed)
    completed = running & (progress >= 1.0) & all_swing_landed
    stop_completed = completed & stop_pending
    running = running & (~stop_completed)
    initialized = initialized & (~stop_completed)
    stop_pending = stop_pending & (~stop_completed)
    phase_index = jp.where(completed & (~stop_completed), phase_index + 1, phase_index)
    phase_time = jp.where(completed, 0.0, phase_time)
    airborne = jp.where(completed, jp.zeros_like(airborne), airborne)
    landed = jp.where(completed, jp.zeros_like(landed), landed)
    progress = jp.where(completed, 0.0, progress)
    swing_group = leg_group_135 == ((phase_index % 2) == 0)
    gait_state = jp.where(
        running & swing_group & (~landed),
        jp.where(progress >= 1.0, LEG_LATE_LANDING, LEG_SWING),
        LEG_STANCE,
    ).astype(jp.int32)
    gait_progress = jp.where(running, jp.full(6, progress), jp.zeros(6))
    updates = {
        "phase_index": phase_index,
        "phase_time": phase_time,
        "airborne_seen": airborne,
        "landed": landed,
        "gait_initialized": initialized,
        "gait_running": running,
        "stop_pending": stop_pending,
    }
    output = {
        "state": gait_state,
        "progress": gait_progress,
        "startup": running & (phase_index == 0),
        "enabled": running,
    }
    return updates, output


def _foot_trajectory(
    state: FirmwareState,
    gait: dict[str, jax.Array],
    applied_twist: jax.Array,
    tripod_enable: jax.Array,
) -> tuple[dict[str, jax.Array], jax.Array]:
    displacement = jp.stack(
        (
            GAIT_PHASE_TIME * (-applied_twist[0] + applied_twist[3] * BASE_FEET[:, 1]),
            GAIT_PHASE_TIME * (-applied_twist[1] - applied_twist[3] * BASE_FEET[:, 0]),
            jp.full(6, GAIT_PHASE_TIME * -applied_twist[2]),
        ),
        axis=-1,
    )
    front = BASE_FEET - 0.5 * displacement
    rear = BASE_FEET + 0.5 * displacement
    default_start = jp.where(gait["startup"], BASE_FEET, rear)
    current = gait["state"]
    previous = state.previous_leg_state
    progress = gait["progress"]

    entering_swing = (current == LEG_SWING) & (previous != LEG_SWING)
    swing_start = jp.where(
        entering_swing[:, None],
        jp.where(state.adapted_stance[:, None], state.foot_memory, default_start),
        state.swing_start,
    )
    custom_swing = jp.where(entering_swing, state.adapted_stance, state.custom_swing)
    adapted = jp.where(entering_swing, False, state.adapted_stance)
    actual_swing_start = jp.where(custom_swing[:, None], swing_start, default_start)
    swing_target = _swing(progress, actual_swing_start, front)

    late_repeat = (current == LEG_LATE_LANDING) & (previous == LEG_LATE_LANDING)
    inward = jp.stack(
        (
            jp.cos(LEG_ANGLES),
            jp.sin(LEG_ANGLES),
            jp.zeros(6),
        ),
        axis=-1,
    )
    late_target = state.foot_memory - late_repeat[:, None] * (
        FIRMWARE_CONTROL_DT
        * (LATE_INWARD_SPEED * inward + jp.array((0.0, 0.0, LATE_LANDING_SPEED)))
    )
    adapted = adapted | (current == LEG_LATE_LANDING)

    previous_late = (current == LEG_STANCE) & (previous == LEG_LATE_LANDING)
    early_landing = (
        (current == LEG_STANCE)
        & (previous == LEG_SWING)
        & (progress >= EARLY_LANDING_PROGRESS)
    )
    adapted = adapted | previous_late | early_landing
    early_target = _swing(progress, actual_swing_start, front)
    stance_integrated = state.foot_memory + FIRMWARE_CONTROL_DT * jp.stack(
        (
            -applied_twist[0] + applied_twist[3] * state.foot_memory[:, 1],
            -applied_twist[1] - applied_twist[3] * state.foot_memory[:, 0],
            jp.full(6, -applied_twist[2]),
        ),
        axis=-1,
    )
    normal_stance = jp.where(
        gait["startup"],
        BASE_FEET + progress[:, None] * (rear - BASE_FEET),
        front + progress[:, None] * (rear - front),
    )
    all_stance_mode = ~tripod_enable
    stance_target = jp.where(
        all_stance_mode,
        stance_integrated,
        jp.where(
            previous_late[:, None],
            state.foot_memory,
            jp.where(
                early_landing[:, None],
                early_target,
                jp.where(adapted[:, None], stance_integrated, normal_stance),
            ),
        ),
    )
    target = jp.where(
        (current == LEG_SWING)[:, None],
        swing_target,
        jp.where(
            (current == LEG_LATE_LANDING)[:, None], late_target, stance_target
        ),
    )
    return {
        "foot_memory": target,
        "swing_start": swing_start,
        "previous_leg_state": current,
        "adapted_stance": adapted,
        "custom_swing": custom_swing,
    }, target


def step(
    state: FirmwareState,
    *,
    target_velocity: jax.Array,
    body_position_world: jax.Array,
    attitude_rpy: jax.Array,
    contacts: jax.Array,
    policy_action: jax.Array,
    pitch_target: jax.Array = 0.0,
    swing_boost: jax.Array = 0.0,
) -> tuple[FirmwareState, FirmwareOutput]:
    """Advance the source-equivalent controller by one 5 ms firmware tick."""
    target_velocity = jp.clip(
        target_velocity, jp.array((-MAX_LINEAR_SPEED, -MAX_YAW_RATE)),
        jp.array((MAX_LINEAR_SPEED, MAX_YAW_RATE))
    )
    alpha = jp.exp(-2.0 * jp.pi * 5.0 * FIRMWARE_CONTROL_DT)
    throttle_target = _bridge_normalized_command(
        target_velocity[0], MAX_LINEAR_SPEED, 20.0
    )
    yaw_target = _bridge_normalized_command(target_velocity[1], MAX_YAW_RATE, 50.0)
    throttle_filter = alpha * state.throttle_filter + (1.0 - alpha) * throttle_target
    yaw_filter = alpha * state.yaw_filter + (1.0 - alpha) * yaw_target
    user_vx = MAX_LINEAR_SPEED * throttle_filter
    user_wz = MAX_YAW_RATE * yaw_filter

    yaw_reference = jp.where(
        state.first_step,
        attitude_rpy[2],
        (state.yaw_reference + user_wz * FIRMWARE_CONTROL_DT + jp.pi)
        % (2.0 * jp.pi)
        - jp.pi,
    )
    position_reference = jp.where(
        state.first_step,
        body_position_world[:2],
        state.position_reference,
    )
    reference_velocity_world = jp.array(
        (jp.cos(yaw_reference) * user_vx, jp.sin(yaw_reference) * user_vx)
    )
    position_reference = position_reference + reference_velocity_world * FIRMWARE_CONTROL_DT
    position_feedback_world = jp.clip(
        position_reference - body_position_world[:2], -0.05, 0.05
    )
    cosine, sine = jp.cos(attitude_rpy[2]), jp.sin(attitude_rpy[2])
    feedback_body = jp.array(
        (
            cosine * position_feedback_world[0] + sine * position_feedback_world[1],
            -sine * position_feedback_world[0] + cosine * position_feedback_world[1],
        )
    )
    yaw_error = (yaw_reference - attitude_rpy[2] + jp.pi) % (2.0 * jp.pi) - jp.pi
    candidate_twist = jp.array(
        (
            jp.clip(user_vx + feedback_body[0], -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED),
            jp.clip(feedback_body[1], -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED),
            0.0,
            jp.clip(user_wz + 2.0 * yaw_error, -MAX_YAW_RATE, MAX_YAW_RATE),
        )
    )
    twist_step = jp.array((0.5, 0.5, 0.5, jp.deg2rad(90.0))) * FIRMWARE_CONTROL_DT
    candidate_twist = state.previous_twist + jp.clip(
        candidate_twist - state.previous_twist, -twist_step, twist_step
    )

    gait_preview_valid = _preview_gait(candidate_twist, state.posture_command)
    gait_accepted = state.first_step | gait_preview_valid
    gait_applied = jp.where(
        state.first_step,
        jp.zeros(4),
        jp.where(gait_preview_valid, candidate_twist, state.gait_applied),
    )
    tripod_enable = (jp.abs(user_vx) >= 0.005) | (jp.abs(user_wz) >= jp.deg2rad(1.0))
    gait_updates, gait = _update_gait(state, tripod_enable, contacts)
    foot_updates, nominal_feet = _foot_trajectory(
        state, gait, gait_applied, tripod_enable
    )

    posture_error = jp.stack((-attitude_rpy[0], pitch_target - attitude_rpy[1]))
    posture_rate = jp.clip(2.0 * posture_error, -jp.deg2rad(15.0), jp.deg2rad(15.0))
    posture_candidate = state.posture_command.at[:2].set(
        jp.clip(
            state.posture_command[:2] + posture_rate * FIRMWARE_CONTROL_DT,
            -jp.deg2rad(45.0),
            jp.deg2rad(45.0),
        )
    )
    posture_candidate = posture_candidate.at[2].set(
        state.posture_command[2]
        + jp.clip(-state.posture_command[2], -jp.deg2rad(15.0), jp.deg2rad(15.0))
        * FIRMWARE_CONTROL_DT
    )
    posture_accepted = _all_feet_valid(nominal_feet, posture_candidate)
    posture_command = jp.where(
        state.first_step,
        jp.zeros(3),
        jp.where(posture_accepted, posture_candidate, state.posture_command),
    )
    nominal_posture_feet = _rotate_inverse(nominal_feet, posture_command)

    residual_alpha = jp.exp(-FIRMWARE_CONTROL_DT / 0.10)
    residual_filter = residual_alpha * state.residual_filter + (
        1.0 - residual_alpha
    ) * jp.clip(policy_action, -1.0, 1.0)
    residual_local, swing_height_command = _phase_gated_policy_residual(
        residual_filter,
        gait["state"],
        gait["progress"],
        swing_boost,
    )
    candidate_local = _body_to_leg(nominal_feet) + residual_local
    residual_feet = _rotate_inverse(_leg_to_body(candidate_local), posture_command)
    _, policy_valid = _solve_ik(residual_feet)
    safe_feet = jp.where(policy_valid[:, None], residual_feet, nominal_posture_feet)

    limited_feet, foot_limited = _limit_foot_reach(safe_feet)
    ik_candidate, ik_valid = _solve_ik(limited_feet)
    last_ik = jp.where(ik_valid[:, None], ik_candidate, state.last_ik)
    desired_joint = last_ik
    joint_step = JOINT_RATE * FIRMWARE_CONTROL_DT
    previous_joint = jp.clip(
        state.previous_joint
        + jp.clip(desired_joint - state.previous_joint, -joint_step, joint_step),
        -JOINT_LIMIT,
        JOINT_LIMIT,
    )

    next_state = state._replace(
        first_step=jp.zeros((), dtype=jp.bool_),
        throttle_filter=throttle_filter,
        yaw_filter=yaw_filter,
        yaw_reference=yaw_reference,
        position_reference=position_reference,
        previous_twist=candidate_twist,
        gait_applied=gait_applied,
        posture_command=posture_command,
        last_ik=last_ik,
        previous_joint=previous_joint,
        residual_filter=residual_filter,
        **gait_updates,
        **foot_updates,
    )
    output = FirmwareOutput(
        model_joint_targets=previous_joint * MODEL_SIGNS,
        servo_joint_targets=previous_joint,
        foot_targets_body=limited_feet,
        applied_twist=gait_applied,
        gait_progress=gait["progress"],
        gait_state=gait["state"],
        swing_height_command=swing_height_command,
        ik_valid=ik_valid,
        policy_valid=policy_valid,
        foot_limited=foot_limited,
        gait_enabled=gait["enabled"],
        gait_accepted=gait_accepted,
        posture_accepted=posture_accepted,
    )
    return next_state, output
