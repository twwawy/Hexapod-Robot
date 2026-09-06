"""24-D residual policy on a contact-aware Tripod/Wave controller.

The adaptive environment calls this controller explicitly. Legacy 18-D
environments keep their own controller. Foot memory is updated before posture/IK.
"""
from collections import namedtuple

import jax
import jax.numpy as jp
import firmware_mjx_controller as fw
import wave_gait_scheduler as scheduler

from adaptive_contract import ACTION_SIZE, ACTION_CONTRACT, XY_LIMIT_M, BODY_HEIGHT_LIMIT_M
from adaptive_contract import MAX_LINEAR_SPEED_MPS, MAX_YAW_RATE_RADPS
LEG_ORDER = ('RF', 'RM', 'RB', 'LF', 'LM', 'LB')
AdaptiveState = namedtuple('AdaptiveState', fw.FirmwareState._fields + (
    'request', 'proposal_end', 'proposal_known', 'proposal_safe', 'proposal_clearance',
    'proposal_posture', 'proposal_stride', 'proposal_period',
    'phase_duration', 'stride_scale', 'swing_end', 'swing_clearance', 'active_known',
    'adapt_posture', 'accepted_action', 'plan_rejected',
    'root_rotation', 'root_position', 'proposal_world', 'goal_world',
    'posture_target', 'height_applied',
    'proposal_index', 'active_index',
    'scheduler', 'proposal_mode', 'proposal_permit', 'proposal_epoch',
    'proposal_apex_phase', 'proposal_transfer', 'apex_phase', 'transfer',
    'proposal_speed_scale', 'speed_scale', 'raw_contacts', 'confirmed_contacts',
))


def initial_state():
    return AdaptiveState(*fw.initial_state(), jp.zeros(ACTION_SIZE), fw.BASE_FEET,
                         jp.zeros(6, dtype=jp.bool_), jp.ones(6, dtype=jp.bool_), jp.full(6, .06),
                         jp.zeros(3), jp.asarray(1.), jp.asarray(1.),
                         jp.asarray(1.), jp.asarray(1.), fw.BASE_FEET, jp.full(6, .06),
                         jp.zeros(6, dtype=jp.bool_), jp.zeros(3), jp.zeros(ACTION_SIZE), jp.asarray(False),
                         jp.eye(3), jp.zeros(3), jp.zeros((6, 3)), jp.zeros((6, 3)),
                         jp.zeros(3), jp.asarray(0.),
                         jp.full(6, -1, dtype=jp.int32), jp.full(6, -1, dtype=jp.int32),
                         scheduler.initial_scheduler(), jp.asarray(0), jp.asarray(False), jp.asarray(-1),
                         jp.full(6, .5), jp.full(6, .5), jp.full(6, .5), jp.full(6, .5),
                         jp.asarray(1.), jp.asarray(1.), jp.zeros(6, dtype=jp.bool_), jp.zeros(6, dtype=jp.bool_))


def decode(action):
    a = jp.clip(action, -1., 1.)
    stride = 1. + jp.where(a[21] < 0., .5*a[21], .3*a[21])
    posture = jp.array((a[18]*jp.deg2rad(5.), a[19]*jp.deg2rad(10.), a[20]*BODY_HEIGHT_LIMIT_M))
    return a[:12].reshape(6, 2)*jp.asarray(XY_LIMIT_M), a[12:18]*.04, posture, stride, a[22]*.15, a[23]*.15


def planned_swing(progress, start, end, clearance, apex_phase=.5, transfer=.5):
    """Controller-owned lift/transfer/lower; clearance is above the higher end."""
    phase = jp.broadcast_to(progress, start.shape[:-1])
    apex_phase = jp.clip(apex_phase, .3, .7)
    transfer = jp.clip(transfer, .35, .65)
    xy = fw._quintic(jp.clip((phase-(transfer-.25))/.5, 0., 1.))
    # Preserve the neutral lift/transfer/lower plateau; apex_phase moves the
    # first maximum earlier/later with smooth zero-velocity joins.
    peak_start = apex_phase-.2
    peak_end = apex_phase+.2
    up = fw._quintic(jp.clip(phase/peak_start, 0., 1.))
    down = fw._quintic(jp.clip((phase-peak_end)/(1.-peak_end), 0., 1.))
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


MAX_LINEAR_SPEED = MAX_LINEAR_SPEED_MPS
MAX_YAW_RATE = MAX_YAW_RATE_RADPS


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


def _update_gait(state, tripod_enable, contacts):
    sched, gait = scheduler.advance(state.scheduler,
        requested_mode=state.proposal_mode, permit=state.proposal_permit,
        proposal_epoch=state.proposal_epoch, command_active=tripod_enable,
        contacts=state.confirmed_contacts, raw_contacts=state.raw_contacts,
        duration=state.phase_duration, dt=FIRMWARE_CONTROL_DT)
    return dict(scheduler=sched, phase_index=sched.phase, phase_time=sched.elapsed,
                airborne_seen=sched.airborne, landed=sched.landed,
                gait_initialized=sched.running, gait_running=sched.running,
                stop_pending=~tripod_enable & sched.running), gait


def _foot_trajectory(state, gait, applied_twist, tripod_enable):
    entering = gait['entering']
    start = jp.where(entering[:, None], state.foot_memory, state.swing_start)
    end = jp.where(entering[:, None], state.proposal_end, state.swing_end)
    clearance = jp.where(entering, state.proposal_clearance, state.swing_clearance)
    known = jp.where(entering, state.proposal_known, state.active_known)
    apex = jp.where(entering, state.proposal_apex_phase, state.apex_phase)
    transfer = jp.where(entering, state.proposal_transfer, state.transfer)
    current, previous = gait['state'], state.previous_leg_state
    swing = planned_swing(gait['progress'], start, end, clearance, apex, transfer)
    late = state.foot_memory - FIRMWARE_CONTROL_DT*jp.stack((
        scheduler.LATE_INWARD_SPEED*jp.cos(LEG_ANGLES),
        scheduler.LATE_INWARD_SPEED*jp.sin(LEG_ANGLES),
        jp.full(6, scheduler.LATE_SPEED)), axis=-1)
    # The first Wave revolution moves stance at half rate, as on main.
    startup_scale = jp.where((state.scheduler.mode == scheduler.WAVE) & gait['startup'], .5, 1.)
    velocity = jp.stack((-applied_twist[0]+applied_twist[3]*state.foot_memory[:, 1],
                         -applied_twist[1]-applied_twist[3]*state.foot_memory[:, 0],
                         jp.full(6, -applied_twist[2])), axis=-1)
    stance = state.foot_memory + startup_scale*FIRMWARE_CONTROL_DT*velocity
    landing = (current == scheduler.STANCE) & (previous != scheduler.STANCE)
    stance = jp.where(landing[:, None], state.foot_memory, stance)
    target = jp.where((current == scheduler.SWING)[:, None], swing,
                       jp.where((current == scheduler.LATE)[:, None], late, stance))
    hold = (current == scheduler.HOLD) | (current == scheduler.TOUCHDOWN)
    # During delayed contact, do not continue driving support legs backwards.
    hold |= (current == scheduler.STANCE) & jp.any(current == scheduler.LATE)
    target = jp.where(hold[:, None], state.foot_memory, target)
    return dict(swing_start=start, swing_end=end, swing_clearance=clearance,
                active_known=known, apex_phase=apex, transfer=transfer,
                foot_memory=target, previous_leg_state=current,
                adapted_stance=jp.ones(6, dtype=jp.bool_), custom_swing=jp.ones(6, dtype=jp.bool_)), target


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
    command_requested = (jp.abs(target_velocity[0]) >= .001) | (jp.abs(target_velocity[1]) >= .001)
    # Ratio coupling: shortening stride and duration together makes shorter,
    # faster steps without simply changing nominal mean speed.
    target_velocity = target_velocity * state.speed_scale
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

    gait_preview_valid = jp.asarray(True)  # geometry bank + actual planned_swing preflight below
    gait_accepted = state.first_step | gait_preview_valid
    gait_applied = jp.where(
        state.first_step,
        jp.zeros(4),
        jp.where(gait_preview_valid, candidate_twist, state.gait_applied),
    )
    # Determine intent before Wave's .2 multiplier; a small arrow-key command
    # must not become a false stop after the supervisor reduces speed.
    tripod_enable = command_requested
    gait_updates, gait = _update_gait(state, tripod_enable, contacts)
    entering = gait['entering']
    # Check the requested stance/swing geometry with the active posture before
    # starting a tripod. A known bad target is never relabeled as blind terrain.
    samples = jax.vmap(planned_swing, in_axes=(0, None, None, None, None, None))(
        jp.linspace(0., 1., 21), state.foot_memory, state.proposal_end, state.proposal_clearance,
        state.proposal_apex_phase, state.proposal_transfer)
    shifted_samples = samples.at[..., 2].add(-jp.clip(height_offset, -.10, .10))
    _, path_ik = _solve_ik(_rotate_inverse(shifted_samples, state.posture_command))
    feasible = state.proposal_safe & jp.all(path_ik, axis=0)
    blocked = jp.any(entering & state.proposal_known & ~feasible)
    blocked = blocked | jp.any(entering & ~state.proposal_safe)
    gait_updates = {key: jax.tree_util.tree_map(lambda old, new: jp.where(blocked, old, new),
                    getattr(state, key), value) for key, value in gait_updates.items()}
    gait['entering'] &= ~blocked
    gait["state"] = jp.where(blocked, state.previous_leg_state, gait["state"])
    gait["progress"] = jp.where(blocked, jp.clip(state.phase_time/state.phase_duration, 0., 1.), gait["progress"])
    gait["enabled"] = jp.where(blocked, state.gait_running, gait["enabled"])
    gait["startup"] = jp.where(blocked, state.gait_running & (state.phase_index == 0), gait["startup"])
    gait_applied = jp.where(blocked, jp.zeros_like(gait_applied), gait_applied)
    gait_applied = jp.where(gait['frozen'] | jp.any(gait['state'] == scheduler.LATE), 0., gait_applied)
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
    # Execution endpoints live in the phase-entry pre-posture controller frame.
    # Freeze this exact endpoint like STM32; map/world metadata is diagnostic.
    trajectory_state = state._replace(phase_duration=phase_duration,
                                     scheduler=gait_updates['scheduler'])
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

    # The 24-D policy changes trajectory parameters, not a second 18-D foot
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
        speed_scale=jp.where(boundary, state.proposal_speed_scale, state.speed_scale),
        goal_world=goal_world,
        active_index=jp.where(entering & ~blocked, state.proposal_index, state.active_index),
        posture_target=posture_target, height_applied=height_applied,
        adapt_posture=adapt_posture, plan_rejected=blocked,
        accepted_action=accepted_action,
        first_step=jp.zeros((), dtype=jp.bool_),
        throttle_filter=throttle_filter,
        yaw_filter=yaw_filter,
        yaw_reference=yaw_reference,
        position_reference=jp.where(blocked | gait['frozen'], body_position_world[:2], position_reference),
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
