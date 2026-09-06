"""23-D parameter policy on the firmware's contact-aware tripod controller.

The adaptive environment calls this controller explicitly. Legacy 18-D
environments keep their own controller. Foot memory is updated before posture/IK.
"""
from collections import namedtuple

import jax
import jax.numpy as jp
import firmware_mjx_controller as fw

ACTION_SIZE = 23
ACTION_CONTRACT = 'adaptive_tripod_reference_residual_23_v2'
LEG_ORDER = ('RF', 'RM', 'RB', 'LF', 'LM', 'LB')
AdaptiveState = namedtuple('AdaptiveState', fw.FirmwareState._fields + (
    'request', 'proposal_end', 'proposal_known', 'proposal_safe', 'proposal_clearance',
    'proposal_posture', 'proposal_stride', 'proposal_period',
    'phase_duration', 'stride_scale', 'swing_end', 'swing_clearance', 'active_known',
    'adapt_posture', 'accepted_action', 'plan_rejected',
    'root_rotation', 'root_position', 'proposal_world', 'goal_world',
    'posture_target', 'height_applied',
    'proposal_index', 'active_index',
))


def initial_state():
    return AdaptiveState(*fw.initial_state(), jp.zeros(23), fw.BASE_FEET,
                         jp.zeros(6, dtype=jp.bool_), jp.ones(6, dtype=jp.bool_), jp.full(6, .06),
                         jp.zeros(3), jp.asarray(1.), jp.asarray(.5),
                         jp.asarray(.5), jp.asarray(1.), fw.BASE_FEET, jp.full(6, .06),
                         jp.zeros(6, dtype=jp.bool_), jp.zeros(3), jp.zeros(23), jp.asarray(False),
                         jp.eye(3), jp.zeros(3), jp.zeros((6, 3)), jp.zeros((6, 3)),
                         jp.zeros(3), jp.asarray(0.),
                         jp.full(6, -1, dtype=jp.int32), jp.full(6, -1, dtype=jp.int32))


def decode(action):
    a = jp.clip(action, -1., 1.)
    stride = 1. + jp.where(a[21] < 0., .5*a[21], .3*a[21])
    period = .5 * (1. + .4*a[22])
    # policy order pitch/roll/height; controller order roll/pitch/height
    posture = jp.array((a[19]*jp.deg2rad(5.), a[18]*jp.deg2rad(10.), a[20]*.03))
    return a[:12].reshape(6, 2)*.04, a[12:18]*.04, posture, stride, period


def planned_swing(progress, start, end, clearance):
    """Controller-owned lift/transfer/lower; clearance is above the higher end."""
    phase = jp.broadcast_to(progress, start.shape[:-1])
    xy = fw._quintic(jp.clip((phase-.25)/.5, 0., 1.))
    up = fw._quintic(jp.clip(phase/.25, 0., 1.))
    down = fw._quintic(jp.clip((phase-.75)/.25, 0., 1.))
    top = jp.maximum(start[..., 2], end[..., 2]) + clearance
    points = start + xy[..., None]*(end-start)
    return points.at[..., 2].set(start[..., 2] + up*(top-start[..., 2]) + down*(end[..., 2]-top))


from firmware_mjx_controller import (
    BASE_FEET,
    EARLY_LANDING_PROGRESS,
    EFFECTIVE_PITCH_MAX_RAD,
    FIRMWARE_CONTROL_DT,
    FirmwareOutput,
    JOINT_LIMIT,
    JOINT_RATE,
    LATE_INWARD_SPEED,
    LATE_LANDING_SPEED,
    LEG_ANGLES,
    LEG_LATE_LANDING,
    LEG_STANCE,
    LEG_SWING,
    MAX_LINEAR_SPEED,
    MAX_YAW_RATE,
    SWING_HEIGHT_MIN,
    SWING_HEIGHT,
    _all_feet_valid,
    _apply_height_offset,
    _bridge_normalized_command,
    _limit_foot_reach,
    _rotate_inverse,
    _solve_ik,
    _swing,
    apply_joint_sign_pattern
)


def _preview_gait(candidate: jax.Array, posture: jax.Array, duration: jax.Array) -> jax.Array:
    displacement = jp.stack(
        (
            duration * (-candidate[0] + candidate[3] * BASE_FEET[:, 1]),
            duration * (-candidate[1] - candidate[3] * BASE_FEET[:, 0]),
            jp.full(6, duration * -candidate[2]),
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


def _update_gait(
    state: AdaptiveState, tripod_enable: jax.Array, contacts: jax.Array
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
    progress = jp.clip(phase_time / state.phase_duration, 0.0, 1.0)
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
    state: AdaptiveState,
    gait: dict[str, jax.Array],
    applied_twist: jax.Array,
    tripod_enable: jax.Array,
) -> tuple[dict[str, jax.Array], jax.Array]:
    displacement = jp.stack(
        (
            state.phase_duration * (-applied_twist[0] + applied_twist[3] * BASE_FEET[:, 1]),
            state.phase_duration * (-applied_twist[1] - applied_twist[3] * BASE_FEET[:, 0]),
            jp.full(6, state.phase_duration * -applied_twist[2]),
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
    swing_end = jp.where(entering_swing[:, None], state.proposal_end, state.swing_end)
    swing_clearance = jp.where(entering_swing, state.proposal_clearance, state.swing_clearance)
    active_known = jp.where(entering_swing, state.proposal_known, state.active_known)
    # Mapped swings start from the same pre-posture memory as stance, never an
    # externally transformed world-space path. Nominal blind swing is unchanged.
    actual_swing_start = jp.where((active_known & entering_swing)[:, None], state.foot_memory, actual_swing_start)
    swing_start = jp.where((active_known & entering_swing)[:, None], state.foot_memory, swing_start)
    custom_swing = custom_swing | active_known
    actual_swing_start = jp.where(custom_swing[:, None], swing_start, actual_swing_start)
    swing_target = jp.where(active_known[:, None],
        planned_swing(progress, actual_swing_start, swing_end, swing_clearance),
        _swing(progress, actual_swing_start, front))

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
    early_target = jp.where(active_known[:, None], state.foot_memory,
                            _swing(progress, actual_swing_start, front))
    adapted = adapted | (active_known & (current != LEG_SWING))
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
        "swing_end": swing_end,
        "swing_clearance": swing_clearance,
        "active_known": active_known,
        "foot_memory": target,
        "swing_start": swing_start,
        "previous_leg_state": current,
        "adapted_stance": adapted,
        "custom_swing": custom_swing,
    }, target


def step(
    state: AdaptiveState,
    *,
    target_velocity: jax.Array,
    body_position_world: jax.Array,
    attitude_rpy: jax.Array,
    contacts: jax.Array,
    policy_action: jax.Array,
    pitch_ff: jax.Array = 0.0,
    roll_cmd: jax.Array = 0.0,
    pitch_cmd: jax.Array = 0.0,
    height_offset: jax.Array = 0.0,
    swing_boost: jax.Array = 0.0,
    swing_height_floor: jax.Array = SWING_HEIGHT_MIN,
) -> tuple[AdaptiveState, FirmwareOutput]:
    """Advance one firmware tick using the environment's prepared parameters."""
    # Ratio coupling: shortening stride and duration together makes shorter,
    # faster steps without simply changing nominal mean speed.
    speed_ratio = state.stride_scale / (state.phase_duration / .5)
    target_velocity = target_velocity.at[0].multiply(speed_ratio)
    target_pose = state.posture_target
    rate = jp.array((jp.deg2rad(15.), jp.deg2rad(15.), .04)) * FIRMWARE_CONTROL_DT
    adapt_posture = state.adapt_posture + jp.clip(target_pose-state.adapt_posture, -rate, rate)
    roll_cmd = roll_cmd + adapt_posture[0]
    pitch_cmd = pitch_cmd + adapt_posture[1]
    height_offset = jp.clip(height_offset + adapt_posture[2], -.05, .10)
    pitch_ff = jp.asarray(0.)
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

    gait_preview_valid = _preview_gait(candidate_twist, state.posture_command, state.phase_duration)
    gait_accepted = state.first_step | gait_preview_valid
    gait_applied = jp.where(
        state.first_step,
        jp.zeros(4),
        jp.where(gait_preview_valid, candidate_twist, state.gait_applied),
    )
    tripod_enable = (jp.abs(user_vx) >= 0.005) | (jp.abs(user_wz) >= jp.deg2rad(1.0))
    gait_updates, gait = _update_gait(state, tripod_enable, contacts)
    entering = (gait["state"] == LEG_SWING) & (state.previous_leg_state != LEG_SWING)
    # Check the requested stance/swing geometry with the active posture before
    # starting a tripod. A known bad target is never relabeled as blind terrain.
    samples = jax.vmap(planned_swing, in_axes=(0, None, None, None))(
        jp.linspace(0., 1., 21), state.foot_memory, state.proposal_end, state.proposal_clearance)
    shifted_samples = samples.at[..., 2].add(-jp.clip(height_offset, -.10, .10))
    _, path_ik = _solve_ik(_rotate_inverse(shifted_samples, state.posture_command))
    feasible = state.proposal_safe & jp.all(path_ik, axis=0)
    blocked = jp.any(entering & state.proposal_known & ~feasible)
    blocked = blocked | jp.any(entering & ~state.proposal_safe)
    gait_updates = {key: jp.where(blocked, getattr(state, key), value) for key, value in gait_updates.items()}
    gait["state"] = jp.where(blocked, state.previous_leg_state, gait["state"])
    gait["progress"] = jp.where(blocked, jp.clip(state.phase_time/state.phase_duration, 0., 1.), gait["progress"])
    gait["enabled"] = jp.where(blocked, state.gait_running, gait["enabled"])
    gait["startup"] = jp.where(blocked, state.gait_running & (state.phase_index == 0), gait["startup"])
    gait_applied = jp.where(blocked, jp.zeros_like(gait_applied), gait_applied)
    gait_accepted = gait_accepted & ~blocked
    boundary = jp.any(entering) & ~blocked
    phase_duration = jp.where(boundary, state.proposal_period, state.phase_duration)
    stride_scale = jp.where(boundary, state.proposal_stride, state.stride_scale)
    posture_target = jp.where(boundary, state.proposal_posture, state.posture_target)
    accepted_action = state.accepted_action.at[:12].set(jp.where(
        (entering & ~blocked)[:, None], state.request[:12].reshape(6, 2),
        state.accepted_action[:12].reshape(6, 2)).reshape(-1))
    accepted_action = accepted_action.at[12:18].set(jp.where(
        entering & ~blocked, state.request[12:18], state.accepted_action[12:18]))
    accepted_action = accepted_action.at[18:].set(jp.where(boundary, state.request[18:], state.accepted_action[18:]))
    goal_world = jp.where((entering & ~blocked)[:, None], state.proposal_world, state.goal_world)
    # A landing patch stays fixed in world coordinates, but only this controller
    # converts it to its own pre-posture trajectory frame and updates memory.
    model_goal = (goal_world + jp.array((0., 0., .032)) - state.root_position) @ state.root_rotation
    body_goal = jp.stack((-model_goal[:, 1], model_goal[:, 0], model_goal[:, 2]), axis=-1)
    pre_goal = body_goal @ fw._rotation_matrix(state.posture_command).T
    pre_goal = pre_goal.at[:, 2].add(jp.clip(height_offset, -.10, .10))
    trajectory_state = state._replace(phase_duration=phase_duration,
        proposal_end=pre_goal, swing_end=jp.where(state.active_known[:, None], pre_goal, state.swing_end))
    foot_updates, nominal_feet = _foot_trajectory(
        trajectory_state, gait, gait_applied, tripod_enable
    )
    foot_updates = {key: jp.where(blocked, getattr(state, key), value)
                    for key, value in foot_updates.items()}
    nominal_feet = jp.where(blocked, state.foot_memory, nominal_feet)

    effective_pitch = jp.clip(
        pitch_cmd + pitch_ff,
        -EFFECTIVE_PITCH_MAX_RAD,
        EFFECTIVE_PITCH_MAX_RAD,
    )
    posture_error = jp.stack(
        (roll_cmd - attitude_rpy[0], effective_pitch - attitude_rpy[1])
    )
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
    shifted_nominal_feet = _apply_height_offset(nominal_feet, height_offset)
    posture_accepted = _all_feet_valid(shifted_nominal_feet, posture_candidate)
    posture_command = jp.where(
        state.first_step,
        jp.zeros(3),
        jp.where(posture_accepted, posture_candidate, state.posture_command),
    )
    # Reject the combined height/posture request as one workspace-gated unit.
    # This preserves the unshifted last-valid path when a command is unreachable.
    height_applied = jp.where(posture_accepted, height_offset, state.height_applied)
    accepted_nominal_feet = _apply_height_offset(nominal_feet, height_applied)
    nominal_posture_feet = _rotate_inverse(
        accepted_nominal_feet, posture_command
    )

    # The 23-D policy changes trajectory parameters, not a second 18-D foot
    # residual. There is exactly one target/IK path and one foot-memory update.
    residual_filter = jp.zeros(18)
    _, policy_valid = _solve_ik(nominal_posture_feet)
    safe_feet = nominal_posture_feet
    swing_height_command = jp.where(foot_updates['active_known'], foot_updates['swing_clearance'], SWING_HEIGHT)

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
        phase_duration=phase_duration, stride_scale=stride_scale,
        goal_world=goal_world,
        active_index=jp.where(entering & ~blocked, state.proposal_index, state.active_index),
        posture_target=posture_target, height_applied=height_applied,
        adapt_posture=adapt_posture, plan_rejected=blocked,
        accepted_action=accepted_action,
        first_step=jp.zeros((), dtype=jp.bool_),
        throttle_filter=throttle_filter,
        yaw_filter=yaw_filter,
        yaw_reference=yaw_reference,
        position_reference=jp.where(blocked, body_position_world[:2], position_reference),
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
        model_joint_targets=apply_joint_sign_pattern(previous_joint),
        servo_joint_targets=apply_joint_sign_pattern(previous_joint),
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
