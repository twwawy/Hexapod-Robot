"""Batched Torch port of the 5 ms STM32-equivalent MJX controller.

The state machine and safety ordering mirror ``mjx/firmware_mjx_controller.py``.
All public tensors are batch-first so the controller can run once for every
Isaac Lab environment without Python loops.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


DT = 0.005
GAIT_PHASE_TIME = 0.5
MAX_LINEAR_SPEED = 0.28
MAX_YAW_RATE = torch.deg2rad(torch.tensor(45.0)).item()
SWING_HEIGHT = 0.06
SWING_HEIGHT_MIN = 0.04
SWING_HEIGHT_MAX = 0.25
SWING_RADIAL_OFFSET = 0.07
EARLY_LANDING_PROGRESS = 0.50
LATE_LANDING_SPEED = 0.20
LATE_INWARD_SPEED = 0.16
JOINT_LIMIT = torch.deg2rad(torch.tensor(135.0)).item()
JOINT_RATE = torch.deg2rad(torch.tensor(315.8)).item()
LINK_1, LINK_2, LINK_3 = 0.074, 0.121, 0.230
ROOT_DISTANCE = 0.1845
BASE_FOOT_RADIUS = 0.218728
BASE_FOOT_Z = -0.287006
WORKSPACE_MARGIN = 0.001
EFFECTIVE_PITCH_MAX_RAD = 0.5585
HEIGHT_OFFSET_MAX_M = 0.10

LEG_STANCE, LEG_SWING, LEG_LATE_LANDING = 0, 1, 2


class FirmwareState(NamedTuple):
    first_step: torch.Tensor
    throttle_filter: torch.Tensor
    yaw_filter: torch.Tensor
    yaw_reference: torch.Tensor
    position_reference: torch.Tensor
    previous_twist: torch.Tensor
    gait_applied: torch.Tensor
    phase_index: torch.Tensor
    phase_time: torch.Tensor
    airborne_seen: torch.Tensor
    landed: torch.Tensor
    gait_initialized: torch.Tensor
    gait_running: torch.Tensor
    stop_pending: torch.Tensor
    foot_memory: torch.Tensor
    swing_start: torch.Tensor
    previous_leg_state: torch.Tensor
    adapted_stance: torch.Tensor
    custom_swing: torch.Tensor
    posture_command: torch.Tensor
    last_ik: torch.Tensor
    previous_joint: torch.Tensor
    residual_filter: torch.Tensor


class FirmwareOutput(NamedTuple):
    model_joint_targets: torch.Tensor
    servo_joint_targets: torch.Tensor
    foot_targets_body: torch.Tensor
    applied_twist: torch.Tensor
    gait_progress: torch.Tensor
    gait_state: torch.Tensor
    swing_height_command: torch.Tensor
    ik_valid: torch.Tensor
    policy_valid: torch.Tensor
    foot_limited: torch.Tensor
    gait_enabled: torch.Tensor
    gait_accepted: torch.Tensor
    posture_accepted: torch.Tensor


def _geometry(like: torch.Tensor) -> tuple[torch.Tensor, ...]:
    angles = torch.deg2rad(like.new_tensor((-45.0, -90.0, -135.0, 45.0, 90.0, 135.0)))
    diagonal = ROOT_DISTANCE / (2.0**0.5)
    roots = like.new_tensor(
        ((diagonal, -diagonal, 0.0), (0.0, -ROOT_DISTANCE, 0.0),
         (-diagonal, -diagonal, 0.0), (diagonal, diagonal, 0.0),
         (0.0, ROOT_DISTANCE, 0.0), (-diagonal, diagonal, 0.0))
    )
    feet = roots + torch.stack(
        (BASE_FOOT_RADIUS * torch.cos(angles), BASE_FOOT_RADIUS * torch.sin(angles),
         torch.full_like(angles, BASE_FOOT_Z)), dim=-1
    )
    signs = like.new_tensor(
        ((1.0, -1.0, 1.0),) * 3 + ((1.0, 1.0, -1.0),) * 3
    )
    return angles, roots, feet, signs


def _body_to_leg(feet_body: torch.Tensor) -> torch.Tensor:
    angles, roots, _, _ = _geometry(feet_body)
    delta = feet_body - roots
    cosine, sine = torch.cos(angles), torch.sin(angles)
    return torch.stack((delta[..., 0] * cosine + delta[..., 1] * sine,
                        -delta[..., 0] * sine + delta[..., 1] * cosine,
                        delta[..., 2]), dim=-1)


def _leg_to_body(feet_local: torch.Tensor) -> torch.Tensor:
    angles, roots, _, _ = _geometry(feet_local)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    return roots + torch.stack((feet_local[..., 0] * cosine - feet_local[..., 1] * sine,
                                feet_local[..., 0] * sine + feet_local[..., 1] * cosine,
                                feet_local[..., 2]), dim=-1)


def _solve_ik(feet_body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local = _body_to_leg(feet_body)
    radial = torch.linalg.vector_norm(local[..., :2], dim=-1)
    planar = radial - LINK_1
    raw = (planar.square() + local[..., 2].square() - LINK_2**2 - LINK_3**2) / (2 * LINK_2 * LINK_3)
    cosine_knee = raw.clamp(-1.0, 1.0)
    sine_knee = torch.sqrt(torch.clamp(1.0 - cosine_knee.square(), min=0.0))
    angles = torch.stack((torch.atan2(local[..., 1], local[..., 0]),
                          torch.atan2(-local[..., 2], planar) - torch.atan2(LINK_3 * sine_knee, LINK_2 + LINK_3 * cosine_knee),
                          torch.atan2(sine_knee, cosine_knee)), dim=-1)
    finite = torch.isfinite(feet_body).all(-1) & torch.isfinite(angles).all(-1)
    reachable = (raw >= -1.000001) & (raw <= 1.000001)
    return angles, finite & reachable & (angles.abs() <= JOINT_LIMIT).all(-1)


def _limit_foot_reach(feet_body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local = _body_to_leg(feet_body)
    radial = torch.linalg.vector_norm(local[..., :2], dim=-1)
    planar = radial - LINK_1
    reach = torch.sqrt(planar.square() + local[..., 2].square())
    limited = reach.clamp(abs(LINK_2 - LINK_3) + WORKSPACE_MARGIN, LINK_2 + LINK_3 - WORKSPACE_MARGIN)
    changed = (limited - reach).abs() > 1.0e-9
    scale = limited / reach.clamp_min(1.0e-9)
    planar_limited = torch.where(changed, planar * scale, planar)
    xy_scale = (LINK_1 + planar_limited) / radial.clamp_min(1.0e-9)
    local_limited = torch.stack((torch.where(changed, local[..., 0] * xy_scale, local[..., 0]),
                                 torch.where(changed, local[..., 1] * xy_scale, local[..., 1]),
                                 torch.where(changed, local[..., 2] * scale, local[..., 2])), dim=-1)
    return _leg_to_body(local_limited), changed


def _rotation_matrix(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch), torch.cos(yaw), torch.sin(yaw)
    return torch.stack((torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), -1),
                        torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), -1),
                        torch.stack((-sp, cp * sr, cp * cr), -1)), -2)


def _rotate_inverse(vectors: torch.Tensor, rpy: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...li,...ij->...lj", vectors, _rotation_matrix(rpy))


def _all_feet_valid(feet_body: torch.Tensor, posture: torch.Tensor) -> torch.Tensor:
    return _solve_ik(_rotate_inverse(feet_body, posture))[1].all(-1)


def _quintic(progress: torch.Tensor) -> torch.Tensor:
    progress = progress.clamp(0.0, 1.0)
    return 10 * progress**3 - 15 * progress**4 + 6 * progress**5


def _bridge(value: torch.Tensor, maximum: float, deadband: float) -> torch.Tensor:
    normalized = (value / maximum).clamp(-1.0, 1.0)
    raw = torch.sign(normalized) * torch.floor(deadband + normalized.abs() * (1000.0 - deadband) + 0.5)
    return torch.where(value.abs() < 1.0e-6, torch.zeros_like(value), torch.sign(raw) * (raw.abs() - deadband) / (1000.0 - deadband))


def _swing(progress: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    angles, _, _, _ = _geometry(start)
    scaled = _quintic(progress)
    while scaled.ndim < start.ndim - 1:
        scaled = scaled.unsqueeze(-1)
    blend = (scaled.square() * (3.0 - 2.0 * scaled)).unsqueeze(-1)
    lift = 4.0 * SWING_HEIGHT * scaled * (1.0 - scaled)
    bulge = 4.0 * SWING_RADIAL_OFFSET * scaled * (1.0 - scaled)
    radial_x = bulge * torch.cos(angles)
    radial_y = bulge * torch.sin(angles)
    offset = torch.stack((radial_x, radial_y, lift.expand_as(radial_x)), dim=-1)
    return start + blend * (end - start) + offset


def _phase_gated_residual(action_flat: torch.Tensor, gait_state: torch.Tensor,
                          progress: torch.Tensor, swing_boost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    action = action_flat.reshape(-1, 6, 3)
    swing, stance = gait_state == LEG_SWING, gait_state == LEG_STANCE
    az = action[..., 2].clamp(-1.0, 1.0)
    height = torch.where(az >= 0, SWING_HEIGHT + az * (SWING_HEIGHT_MAX - SWING_HEIGHT),
                         SWING_HEIGHT + az * (SWING_HEIGHT - SWING_HEIGHT_MIN))
    envelope = 4.0 * _quintic(progress) * (1.0 - _quintic(progress))
    xy = action[..., :2] * action.new_tensor((0.04, 0.02))
    xy = torch.where(swing[..., None], xy, torch.zeros_like(xy))
    boost = swing_boost.clamp(0.0, 0.06).unsqueeze(-1)
    swing_z = torch.minimum((height - SWING_HEIGHT + boost) * envelope,
                            torch.full_like(height, SWING_HEIGHT_MAX - SWING_HEIGHT))
    z = torch.where(swing, swing_z, torch.where(stance, action[..., 2] * 0.02, torch.zeros_like(height)))
    return torch.cat((xy, z[..., None]), -1), height


def _preview_gait(candidate: torch.Tensor, posture: torch.Tensor) -> torch.Tensor:
    _, _, base, _ = _geometry(candidate)
    base = base.expand(candidate.shape[0], -1, -1)
    displacement = torch.stack((GAIT_PHASE_TIME * (-candidate[:, None, 0] + candidate[:, None, 3] * base[..., 1]),
                                GAIT_PHASE_TIME * (-candidate[:, None, 1] - candidate[:, None, 3] * base[..., 0]),
                                GAIT_PHASE_TIME * -candidate[:, None, 2].expand(-1, 6)), -1)
    front, rear = base - 0.5 * displacement, base + 0.5 * displacement
    p = torch.linspace(0.0, 1.0, 21, device=candidate.device, dtype=candidate.dtype).view(1, 21, 1, 1)
    stance = front[:, None] + p * (rear - front)[:, None]
    swing = _swing(p[..., 0], rear[:, None], front[:, None])
    posture_samples = posture[:, None].expand(-1, 21, -1).reshape(-1, 3)
    base_ok = _solve_ik(_rotate_inverse(base, posture))[1].all(-1)
    stance_ok = _solve_ik(_rotate_inverse(stance.reshape(-1, 6, 3), posture_samples))[1].reshape(-1, 21, 6).all((1, 2))
    swing_ok = _solve_ik(_rotate_inverse(swing.reshape(-1, 6, 3), posture_samples))[1].reshape(-1, 21, 6).all((1, 2))
    return base_ok & stance_ok & swing_ok


def initial_state(num_envs: int, device: str | torch.device, dtype: torch.dtype = torch.float32) -> FirmwareState:
    z = torch.zeros(num_envs, device=device, dtype=dtype)
    _, _, base, _ = _geometry(z)
    base = base.expand(num_envs, -1, -1).clone()
    base_angles, valid = _solve_ik(base)
    base_angles = torch.where(valid[..., None], base_angles, torch.zeros_like(base_angles))
    zb, zl = torch.zeros_like(z, dtype=torch.bool), torch.zeros_like(z, dtype=torch.long)
    return FirmwareState(torch.ones_like(zb), z.clone(), z.clone(), z.clone(), torch.zeros(num_envs, 2, device=device, dtype=dtype),
                         torch.zeros(num_envs, 4, device=device, dtype=dtype), torch.zeros(num_envs, 4, device=device, dtype=dtype), zl.clone(), z.clone(),
                         torch.zeros(num_envs, 6, device=device, dtype=torch.bool), torch.zeros(num_envs, 6, device=device, dtype=torch.bool),
                         zb.clone(), zb.clone(), zb.clone(), base.clone(), base.clone(), torch.zeros(num_envs, 6, device=device, dtype=torch.long),
                         torch.zeros(num_envs, 6, device=device, dtype=torch.bool), torch.zeros(num_envs, 6, device=device, dtype=torch.bool),
                         torch.zeros(num_envs, 3, device=device, dtype=dtype), base_angles.clone(), base_angles.clone(),
                         torch.zeros(num_envs, 18, device=device, dtype=dtype))


def initial_output(state: FirmwareState) -> FirmwareOutput:
    """Build the source controller's valid standing output for a batch."""
    _, _, base_single, signs = _geometry(state.previous_joint)
    base = base_single.expand(state.first_step.shape[0], -1, -1).clone()
    zeros6 = torch.zeros_like(state.airborne_seen)
    ones6 = torch.ones_like(state.airborne_seen)
    zeros = torch.zeros_like(state.first_step)
    ones = torch.ones_like(state.first_step)
    return FirmwareOutput(
        state.previous_joint * signs,
        state.previous_joint,
        base,
        torch.zeros_like(state.previous_twist),
        torch.zeros_like(state.phase_time[:, None].expand(-1, 6)),
        torch.zeros_like(state.previous_leg_state),
        torch.full_like(state.phase_time[:, None].expand(-1, 6), SWING_HEIGHT),
        ones6,
        ones6,
        zeros6,
        zeros,
        ones,
        ones,
    )


def _update_gait(state: FirmwareState, enable: torch.Tensor, contacts: torch.Tensor) -> tuple[dict, dict]:
    new_run = enable & ~state.gait_running
    running = state.gait_running | enable
    stop = torch.where(enable, torch.zeros_like(state.stop_pending), state.stop_pending | running)
    initialized = torch.where(new_run, torch.zeros_like(state.gait_initialized), state.gait_initialized)
    phase_index = torch.where(new_run, torch.zeros_like(state.phase_index), state.phase_index)
    phase_time = torch.where(new_run, torch.zeros_like(state.phase_time), state.phase_time)
    airborne = torch.where(new_run[:, None], torch.zeros_like(state.airborne_seen), state.airborne_seen)
    landed = torch.where(new_run[:, None], torch.zeros_like(state.landed), state.landed)
    phase_time = torch.where(running & initialized, phase_time + DT, phase_time)
    initialized = initialized | running
    progress_scalar = (phase_time / GAIT_PHASE_TIME).clamp(0.0, 1.0)
    group135 = (torch.arange(6, device=contacts.device) % 2) == 0
    swing_group = group135[None] == ((phase_index % 2) == 0)[:, None]
    airborne = airborne | (running[:, None] & swing_group & ~contacts)
    landed = landed | (running[:, None] & swing_group & (airborne | stop[:, None]) & contacts & (progress_scalar >= EARLY_LANDING_PROGRESS)[:, None])
    completed = running & (progress_scalar >= 1.0) & ((~swing_group) | landed).all(-1)
    stop_completed = completed & stop
    running = running & ~stop_completed
    initialized = initialized & ~stop_completed
    stop = stop & ~stop_completed
    phase_index = torch.where(completed & ~stop_completed, phase_index + 1, phase_index)
    phase_time = torch.where(completed, torch.zeros_like(phase_time), phase_time)
    airborne = torch.where(completed[:, None], torch.zeros_like(airborne), airborne)
    landed = torch.where(completed[:, None], torch.zeros_like(landed), landed)
    progress_scalar = torch.where(completed, torch.zeros_like(progress_scalar), progress_scalar)
    swing_group = group135[None] == ((phase_index % 2) == 0)[:, None]
    gait_state = torch.where(running[:, None] & swing_group & ~landed,
                             torch.where((progress_scalar >= 1.0)[:, None], LEG_LATE_LANDING, LEG_SWING), LEG_STANCE).long()
    progress = torch.where(running[:, None], progress_scalar[:, None].expand(-1, 6), torch.zeros_like(contacts, dtype=phase_time.dtype))
    updates = dict(phase_index=phase_index, phase_time=phase_time, airborne_seen=airborne, landed=landed,
                   gait_initialized=initialized, gait_running=running, stop_pending=stop)
    return updates, dict(state=gait_state, progress=progress, startup=running & (phase_index == 0), enabled=running)


def _foot_trajectory(state: FirmwareState, gait: dict, twist: torch.Tensor, enable: torch.Tensor) -> tuple[dict, torch.Tensor]:
    angles, _, base_single, _ = _geometry(twist)
    base = base_single.expand(twist.shape[0], -1, -1)
    displacement = torch.stack((GAIT_PHASE_TIME * (-twist[:, None, 0] + twist[:, None, 3] * base[..., 1]),
                                GAIT_PHASE_TIME * (-twist[:, None, 1] - twist[:, None, 3] * base[..., 0]),
                                GAIT_PHASE_TIME * -twist[:, None, 2].expand(-1, 6)), -1)
    front, rear = base - 0.5 * displacement, base + 0.5 * displacement
    default_start = torch.where(gait["startup"][:, None, None], base, rear)
    current, previous, progress = gait["state"], state.previous_leg_state, gait["progress"]
    entering = (current == LEG_SWING) & (previous != LEG_SWING)
    swing_start = torch.where(entering[..., None], torch.where(state.adapted_stance[..., None], state.foot_memory, default_start), state.swing_start)
    custom = torch.where(entering, state.adapted_stance, state.custom_swing)
    adapted = torch.where(entering, torch.zeros_like(state.adapted_stance), state.adapted_stance)
    actual_start = torch.where(custom[..., None], swing_start, default_start)
    swing_target = _swing(progress, actual_start, front)
    late_repeat = (current == LEG_LATE_LANDING) & (previous == LEG_LATE_LANDING)
    inward = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), -1)
    late_target = state.foot_memory - late_repeat[..., None] * DT * (LATE_INWARD_SPEED * inward + twist.new_tensor((0.0, 0.0, LATE_LANDING_SPEED)))
    adapted = adapted | (current == LEG_LATE_LANDING)
    previous_late = (current == LEG_STANCE) & (previous == LEG_LATE_LANDING)
    early = (current == LEG_STANCE) & (previous == LEG_SWING) & (progress >= EARLY_LANDING_PROGRESS)
    adapted = adapted | previous_late | early
    early_target = _swing(progress, actual_start, front)
    integrated = state.foot_memory + DT * torch.stack((-twist[:, None, 0] + twist[:, None, 3] * state.foot_memory[..., 1],
                                                       -twist[:, None, 1] - twist[:, None, 3] * state.foot_memory[..., 0],
                                                       -twist[:, None, 2].expand(-1, 6)), -1)
    normal = torch.where(gait["startup"][:, None, None], base + progress[..., None] * (rear - base), front + progress[..., None] * (rear - front))
    stance = torch.where((~enable)[:, None, None], integrated,
                         torch.where(previous_late[..., None], state.foot_memory,
                                     torch.where(early[..., None], early_target,
                                                 torch.where(adapted[..., None], integrated, normal))))
    target = torch.where((current == LEG_SWING)[..., None], swing_target,
                         torch.where((current == LEG_LATE_LANDING)[..., None], late_target, stance))
    return dict(foot_memory=target, swing_start=swing_start, previous_leg_state=current,
                adapted_stance=adapted, custom_swing=custom), target


def step(state: FirmwareState, *, target_velocity: torch.Tensor, body_position_world: torch.Tensor,
         attitude_rpy: torch.Tensor, contacts: torch.Tensor, policy_action: torch.Tensor,
         pitch_ff: torch.Tensor | None = None, roll_cmd: torch.Tensor | None = None,
         pitch_cmd: torch.Tensor | None = None, height_offset: torch.Tensor | None = None,
         swing_boost: torch.Tensor | None = None) -> tuple[FirmwareState, FirmwareOutput]:
    """Advance every environment by one 5 ms firmware tick."""
    n = target_velocity.shape[0]
    zero = target_velocity.new_zeros(n)
    pitch_ff = zero if pitch_ff is None else pitch_ff
    roll_cmd = zero if roll_cmd is None else roll_cmd
    pitch_cmd = zero if pitch_cmd is None else pitch_cmd
    height_offset = zero if height_offset is None else height_offset
    swing_boost = zero if swing_boost is None else swing_boost
    target_velocity = torch.maximum(torch.minimum(target_velocity, target_velocity.new_tensor((MAX_LINEAR_SPEED, MAX_YAW_RATE))),
                                    target_velocity.new_tensor((-MAX_LINEAR_SPEED, -MAX_YAW_RATE)))
    alpha = torch.exp(target_velocity.new_tensor(-2.0 * torch.pi * 5.0 * DT))
    throttle_target, yaw_target = _bridge(target_velocity[:, 0], MAX_LINEAR_SPEED, 20.0), _bridge(target_velocity[:, 1], MAX_YAW_RATE, 50.0)
    throttle_filter = alpha * state.throttle_filter + (1 - alpha) * throttle_target
    yaw_filter = alpha * state.yaw_filter + (1 - alpha) * yaw_target
    user_vx, user_wz = MAX_LINEAR_SPEED * throttle_filter, MAX_YAW_RATE * yaw_filter
    yaw_reference = torch.where(state.first_step, attitude_rpy[:, 2],
                                torch.remainder(state.yaw_reference + user_wz * DT + torch.pi, 2 * torch.pi) - torch.pi)
    position_reference = torch.where(state.first_step[:, None], body_position_world[:, :2], state.position_reference)
    position_reference = position_reference + torch.stack((torch.cos(yaw_reference) * user_vx, torch.sin(yaw_reference) * user_vx), -1) * DT
    position_error = (position_reference - body_position_world[:, :2]).clamp(-0.05, 0.05)
    cosine, sine = torch.cos(attitude_rpy[:, 2]), torch.sin(attitude_rpy[:, 2])
    feedback = torch.stack((cosine * position_error[:, 0] + sine * position_error[:, 1],
                            -sine * position_error[:, 0] + cosine * position_error[:, 1]), -1)
    yaw_error = torch.remainder(yaw_reference - attitude_rpy[:, 2] + torch.pi, 2 * torch.pi) - torch.pi
    candidate = torch.stack(((user_vx + feedback[:, 0]).clamp(-MAX_LINEAR_SPEED, MAX_LINEAR_SPEED),
                             feedback[:, 1].clamp(-MAX_LINEAR_SPEED, MAX_LINEAR_SPEED), zero,
                             (user_wz + 2 * yaw_error).clamp(-MAX_YAW_RATE, MAX_YAW_RATE)), -1)
    twist_step = candidate.new_tensor((0.5, 0.5, 0.5, torch.deg2rad(torch.tensor(90.0)).item())) * DT
    candidate = state.previous_twist + (candidate - state.previous_twist).clamp(-twist_step, twist_step)
    preview_valid = _preview_gait(candidate, state.posture_command)
    gait_accepted = state.first_step | preview_valid
    gait_applied = torch.where(state.first_step[:, None], torch.zeros_like(candidate),
                               torch.where(preview_valid[:, None], candidate, state.gait_applied))
    enable = (user_vx.abs() >= 0.005) | (user_wz.abs() >= torch.deg2rad(target_velocity.new_tensor(1.0)))
    gait_updates, gait = _update_gait(state, enable, contacts)
    foot_updates, nominal = _foot_trajectory(state, gait, gait_applied, enable)

    effective_pitch = (pitch_cmd + pitch_ff).clamp(-EFFECTIVE_PITCH_MAX_RAD, EFFECTIVE_PITCH_MAX_RAD)
    posture_error = torch.stack((roll_cmd - attitude_rpy[:, 0], effective_pitch - attitude_rpy[:, 1]), -1)
    rate = (2 * posture_error).clamp(-torch.deg2rad(target_velocity.new_tensor(15.0)), torch.deg2rad(target_velocity.new_tensor(15.0)))
    candidate_xy = (state.posture_command[:, :2] + rate * DT).clamp(-torch.deg2rad(target_velocity.new_tensor(45.0)), torch.deg2rad(target_velocity.new_tensor(45.0)))
    candidate_z = state.posture_command[:, 2] + (-state.posture_command[:, 2]).clamp(-torch.deg2rad(target_velocity.new_tensor(15.0)), torch.deg2rad(target_velocity.new_tensor(15.0))) * DT
    posture_candidate = torch.cat((candidate_xy, candidate_z[:, None]), -1)
    shifted = nominal.clone()
    shifted[..., 2] = shifted[..., 2] - height_offset.clamp(-HEIGHT_OFFSET_MAX_M, HEIGHT_OFFSET_MAX_M)[:, None]
    posture_accepted = _all_feet_valid(shifted, posture_candidate)
    posture = torch.where(state.first_step[:, None], torch.zeros_like(posture_candidate),
                          torch.where(posture_accepted[:, None], posture_candidate, state.posture_command))
    accepted = torch.where(posture_accepted[:, None, None], shifted, nominal)
    nominal_posture = _rotate_inverse(accepted, posture)
    residual_alpha = torch.exp(target_velocity.new_tensor(-DT / 0.10))
    residual_filter = residual_alpha * state.residual_filter + (1 - residual_alpha) * policy_action.clamp(-1.0, 1.0)
    residual_local, swing_height = _phase_gated_residual(residual_filter, gait["state"], gait["progress"], swing_boost)
    residual_feet = _rotate_inverse(_leg_to_body(_body_to_leg(accepted) + residual_local), posture)
    policy_valid = _solve_ik(residual_feet)[1]
    safe = torch.where(policy_valid[..., None], residual_feet, nominal_posture)
    limited, foot_limited = _limit_foot_reach(safe)
    ik_candidate, ik_valid = _solve_ik(limited)
    last_ik = torch.where(ik_valid[..., None], ik_candidate, state.last_ik)
    joint_step = JOINT_RATE * DT
    previous_joint = (state.previous_joint + (last_ik - state.previous_joint).clamp(-joint_step, joint_step)).clamp(-JOINT_LIMIT, JOINT_LIMIT)
    _, _, _, signs = _geometry(previous_joint)
    next_state = state._replace(first_step=torch.zeros_like(state.first_step), throttle_filter=throttle_filter,
                                yaw_filter=yaw_filter, yaw_reference=yaw_reference, position_reference=position_reference,
                                previous_twist=candidate, gait_applied=gait_applied, posture_command=posture,
                                last_ik=last_ik, previous_joint=previous_joint, residual_filter=residual_filter,
                                **gait_updates, **foot_updates)
    output = FirmwareOutput(previous_joint * signs, previous_joint, limited, gait_applied, gait["progress"], gait["state"],
                            swing_height, ik_valid, policy_valid, foot_limited, gait["enabled"], gait_accepted, posture_accepted)
    return next_state, output
