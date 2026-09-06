"""Explorer handoff on top of the archived 0805164 firmware controller.

The step below retains that revision's nominal gait, posture and actuator safety.
Only this isolated viewer installs it; the training controller is not changed.
Unknown targets use the firmware trajectory and exactly zero residual. Observed
paths are latched at takeoff, then residuals and the original IK/rate limits apply.
"""
from collections import namedtuple

import jax
import jax.numpy as jp
import firmware_mjx_controller as firmware
from firmware_mjx_controller import (
    EFFECTIVE_PITCH_MAX_RAD,
    FIRMWARE_CONTROL_DT,
    FirmwareOutput,
    FirmwareState,
    JOINT_LIMIT,
    JOINT_RATE,
    MAX_LINEAR_SPEED,
    MAX_YAW_RATE,
    MODEL_SIGNS,
    _all_feet_valid,
    _apply_height_offset,
    _body_to_leg,
    _bridge_normalized_command,
    _foot_trajectory,
    _leg_to_body,
    _limit_foot_reach,
    _phase_gated_policy_residual,
    _preview_gait,
    _rotate_inverse,
    _solve_ik,
    _update_gait
)


HandoffState = namedtuple('HandoffState', firmware.FirmwareState._fields + (
    'proposal_path', 'proposal_mode', 'active_path', 'active_mode',
    'body_rotation', 'actual_feet_world', 'anchor_world', 'terrain_features',
    'blocked', 'applied_action',
))


def extend_state(base):
    return HandoffState(*base, jp.zeros((6, 41, 3)), jp.zeros(6, dtype=jp.int32),
                        jp.zeros((6, 41, 3)), jp.zeros(6, dtype=jp.int32),
                        jp.eye(3), jp.zeros((6, 3)), jp.zeros((6, 3)),
                        jp.zeros(15), jp.asarray(False), jp.zeros(18))


def world_to_body(points, state, origin):
    model = (points + jp.array((0., 0., 0.032)) - origin) @ state.body_rotation
    return jp.stack((-model[:, 1], model[:, 0], model[:, 2]), axis=-1)


def step(
    state: FirmwareState,
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
) -> tuple[FirmwareState, FirmwareOutput]:
    """Advance nominal -> observed path -> gated residual -> IK by 5 ms."""
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
    entering = (gait["state"] == firmware.LEG_SWING) & (state.previous_leg_state != firmware.LEG_SWING)
    blocked = jp.any(entering & (state.proposal_mode == 2))
    # A known unsafe target vetoes the next takeoff, while physics/PD continue.
    # Missing observations have mode zero and never trigger this veto.
    gait_updates = {name: jp.where(blocked, getattr(state, name), value)
                    for name, value in gait_updates.items()}
    gait["state"] = jp.where(blocked, state.previous_leg_state, gait["state"])
    gait["progress"] = jp.where(blocked, jp.clip(state.phase_time / firmware.GAIT_PHASE_TIME, 0., 1.), gait["progress"])
    gait_applied = jp.where(blocked, jp.zeros_like(gait_applied), gait_applied)
    entering = entering & ~blocked
    active_mode = jp.where(entering, state.proposal_mode, state.active_mode)
    # Move only the takeoff end to the latest measured foot; keep the chosen
    # touchdown fixed in odom. New scans cannot retarget an airborne leg.
    phase = jp.linspace(0., 1., 41)
    start_delta = state.actual_feet_world - state.proposal_path[:, 0]
    proposal = state.proposal_path + (1. - firmware._quintic(phase))[None, :, None] * start_delta[:, None]
    active_path = jp.where(entering[:, None, None], proposal, state.active_path)
    foot_updates, nominal_feet = _foot_trajectory(
        state, gait, gait_applied, tripod_enable
    )

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
    accepted_nominal_feet = jp.where(
        posture_accepted, shifted_nominal_feet, nominal_feet
    )
    nominal_posture_feet = _rotate_inverse(
        accepted_nominal_feet, posture_command
    )

    mapped = active_mode == 1
    swing = gait["state"] == firmware.LEG_SWING
    touchdown = (gait["state"] == firmware.LEG_STANCE) & (state.previous_leg_state != firmware.LEG_STANCE)
    anchor = jp.where(touchdown[:, None], state.actual_feet_world, state.anchor_world)
    sample = jp.clip(gait["progress"] * 40., 0., 40.)
    lo = jp.minimum(jp.floor(sample).astype(jp.int32), 39)
    blend = (sample - lo)[:, None]
    leg = jp.arange(6)
    path_world = (1.-blend) * active_path[leg, lo] + blend * active_path[leg, lo+1]
    mapped_world = jp.where(swing[:, None], path_world, anchor)
    mapped_body = world_to_body(mapped_world, state, body_position_world)
    # Late landing retains firmware contact search, seeded with corrected memory.
    use_path = mapped & (gait["state"] != firmware.LEG_LATE_LANDING)
    nominal_posture_feet = jp.where(use_path[:, None], mapped_body, nominal_posture_feet)
    # Convert back to the pre-posture frame so the original residual/IK ordering
    # below can be retained verbatim.
    accepted_nominal_feet = nominal_posture_feet @ firmware._rotation_matrix(posture_command).T
    mask = jp.repeat(mapped & swing & ~blocked, 3)
    policy_action = jp.where(mask, policy_action, 0.)
    residual_alpha = jp.exp(-FIRMWARE_CONTROL_DT / 0.10)
    residual_filter = residual_alpha * state.residual_filter + (
        1.0 - residual_alpha
    ) * jp.clip(policy_action, -1.0, 1.0)
    residual_filter = jp.where(mask, residual_filter, 0.)
    residual_local, swing_height_command = _phase_gated_policy_residual(
        residual_filter,
        gait["state"],
        gait["progress"],
        swing_boost,
    )
    # Preserve the selected landing patch. Fade XY out at both endpoints and
    # never let PPO lower the planner's obstacle-clearance arc. Stance and unknown
    # legs are controller-owned, including clearing their residual filter.
    envelope = 4. * firmware._quintic(gait["progress"]) * (1. - firmware._quintic(gait["progress"]))
    residual_local = residual_local.at[:, :2].multiply(envelope[:, None])
    residual_local = residual_local.at[:, 2].set(jp.maximum(residual_local[:, 2], 0.))
    residual_local = jp.where((mapped & swing & ~blocked)[:, None], residual_local, 0.)
    candidate_local = _body_to_leg(accepted_nominal_feet) + residual_local
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

    # Seed the next nominal swing/late-contact search from the corrected foot,
    # rather than jumping back to BASE_FEET when coverage disappears.
    corrected_memory = limited_feet @ firmware._rotation_matrix(posture_command).T
    corrected_memory = corrected_memory.at[:, 2].add(
        jp.where(posture_accepted, jp.clip(height_offset, -firmware.HEIGHT_OFFSET_MAX_M,
                                         firmware.HEIGHT_OFFSET_MAX_M), 0.))
    foot_updates["foot_memory"] = jp.where(mapped[:, None], corrected_memory, foot_updates["foot_memory"])
    foot_updates["adapted_stance"] = foot_updates["adapted_stance"] | (mapped & ~swing)

    next_state = state._replace(
        active_path=active_path, active_mode=active_mode, anchor_world=anchor,
        blocked=blocked, applied_action=policy_action,
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
