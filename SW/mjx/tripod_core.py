"""Pure-JAX tripod gait, residual authority, safety, and analytical IK core.

The module deliberately contains no MuJoCo state.  It defines the controller
contract shared by the flat-command and terrain environments:

``nominal tripod -> phase/contact-masked residual -> contact adapter ->
workspace projection -> 3DOF IK``.

Keeping these functions side-effect free makes zero-residual parity and the
physical limits testable without constructing an MJX environment.
"""

from __future__ import annotations

import jax.numpy as jp

from urdf_kinematics import (
    FEMUR_LENGTH,
    NOMINAL_FOOT_RADIAL,
    NOMINAL_FOOT_VERTICAL,
    SHOULDER_RADIAL_OFFSET,
    SHOULDER_VERTICAL_OFFSET,
    TIBIA_LENGTH,
)

LINK1 = SHOULDER_RADIAL_OFFSET
LINK2 = FEMUR_LENGTH
LINK3 = TIBIA_LENGTH
MODEL_FORWARD = jp.array((0.0, -1.0, 0.0))
MODEL_LATERAL = jp.array((1.0, 0.0, 0.0))
MODEL_UP = jp.array((0.0, 0.0, 1.0))
CONTROLLER_TO_MODEL = jp.stack((MODEL_FORWARD, MODEL_LATERAL, MODEL_UP), axis=-1)


def _command_components(command: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """Return forward, lateral, and yaw components from a gait command.

    The three-component contract is canonical.  Two-component commands remain
    accepted by the pure core so old NumPy/JAX parity tools can represent
    ``(forward, yaw)`` while they are being migrated.
    """
    if command.shape[-1] == 2:
        return command[0], jp.zeros_like(command[0]), command[1]
    if command.shape[-1] == 3:
        return command[0], command[1], command[2]
    raise ValueError("gait command must be (forward, yaw) or (forward, lateral, yaw)")


def controller_velocity(command: jp.ndarray) -> jp.ndarray:
    """Map controller-frame forward/lateral velocity to the URDF body frame."""
    forward, lateral, _ = _command_components(command)
    return MODEL_FORWARD * forward + MODEL_LATERAL * lateral


def _nominal_body_points(
    origins: jp.ndarray,
    outward: jp.ndarray,
    shoulder_lateral: jp.ndarray,
) -> jp.ndarray:
    tangent = jp.stack(
        (-outward[:, 1], outward[:, 0], jp.zeros(outward.shape[0])), axis=-1
    )
    nominal = (
        origins
        + outward * NOMINAL_FOOT_RADIAL
        + tangent * shoulder_lateral[:, None]
    )
    return nominal.at[:, 2].set(origins[:, 2] + NOMINAL_FOOT_VERTICAL)


def quintic(tau: jp.ndarray) -> jp.ndarray:
    """Quintic timing profile with zero endpoint velocity/acceleration."""
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def scale_asymmetric(action: jp.ndarray, low: float, high: float) -> jp.ndarray:
    """Map a bounded action to ``[low, high]`` while preserving zero.

    This is intentionally piecewise rather than an affine map: an affine map
    would make a zero policy apply a non-zero correction whenever the range is
    asymmetric.  ``low`` must be non-positive and ``high`` non-negative.
    """
    if low > 0.0 or high < 0.0:
        raise ValueError("asymmetric residual range must contain zero")
    bounded = jp.clip(action, -1.0, 1.0)
    return jp.where(bounded >= 0.0, bounded * high, bounded * (-low))


def smooth_gait_action(
    previous: jp.ndarray,
    requested: jp.ndarray,
    *,
    control_dt: float,
    time_constant: float,
) -> jp.ndarray:
    """Low-pass global gait outputs without delaying foot residual safety."""
    if control_dt <= 0.0 or time_constant < 0.0:
        raise ValueError("control_dt must be positive and time_constant non-negative")
    if time_constant == 0.0:
        return requested
    alpha = 1.0 - jp.exp(-control_dt / time_constant)
    return previous + alpha * (requested - previous)


def wrap_angle(angle: jp.ndarray) -> jp.ndarray:
    """Wrap radians to ``[-pi, pi)`` without Python control flow."""
    return (angle + jp.pi) % (2.0 * jp.pi) - jp.pi


def slew_limit(
    current: jp.ndarray,
    requested: jp.ndarray,
    rate_limit: jp.ndarray,
    dt: float,
) -> jp.ndarray:
    """Rate-limit a vector command using physical units per second."""
    return current + jp.clip(requested - current, -rate_limit * dt, rate_limit * dt)


def classical_body_twist(
    *,
    command_target: jp.ndarray,
    filtered_command: jp.ndarray,
    desired_position_xy: jp.ndarray,
    desired_heading: jp.ndarray,
    position_integral: jp.ndarray,
    heading_integral: jp.ndarray,
    applied_twist: jp.ndarray,
    body_position_xy: jp.ndarray,
    body_heading: jp.ndarray,
    dt: float,
    command_deadzone: float,
    command_rate_limit: jp.ndarray,
    position_kp: jp.ndarray,
    position_ki: jp.ndarray,
    position_integral_limit: jp.ndarray,
    position_feedback_limit: jp.ndarray,
    heading_kp: float,
    heading_ki: float,
    heading_integral_limit: float,
    heading_feedback_limit: float,
    twist_limit: jp.ndarray,
    twist_rate_limit: jp.ndarray,
    position_valid: jp.ndarray | None = None,
) -> tuple[jp.ndarray, ...]:
    """Source-controller command filter plus x/y position and heading PI.

    Commands and the returned twist use controller coordinates
    ``(forward, lateral, yaw_rate)``.  Position references live in world XY,
    exactly as in ``SW/Controller/Controller_detail.md``.
    """
    bounded = jp.clip(command_target, -twist_limit, twist_limit)
    bounded = jp.where(jp.abs(bounded) < command_deadzone, 0.0, bounded)
    next_filtered = slew_limit(
        filtered_command, bounded, command_rate_limit, dt
    )

    desired_forward = jp.array((jp.cos(desired_heading), jp.sin(desired_heading)))
    desired_lateral = jp.array((-jp.sin(desired_heading), jp.cos(desired_heading)))
    world_reference_velocity = (
        desired_forward * next_filtered[0]
        + desired_lateral * next_filtered[1]
    )
    if position_valid is None:
        position_valid = jp.ones((), dtype=jp.bool_)
    integrated_desired_position = (
        desired_position_xy + world_reference_velocity * dt
    )
    next_desired_position = jp.where(
        position_valid, integrated_desired_position, body_position_xy
    )
    next_desired_heading = wrap_angle(
        desired_heading + next_filtered[2] * dt
    )

    world_error = next_desired_position - body_position_xy
    body_forward = jp.array((jp.cos(body_heading), jp.sin(body_heading)))
    body_lateral = jp.array((-jp.sin(body_heading), jp.cos(body_heading)))
    candidate_position_integral = jp.clip(
        position_integral + world_error * dt,
        -position_integral_limit,
        position_integral_limit,
    )
    next_position_integral = jp.where(
        position_valid, candidate_position_integral, jp.zeros_like(position_integral)
    )
    position_feedback_world = jp.clip(
        position_kp * world_error + position_ki * next_position_integral,
        -position_feedback_limit,
        position_feedback_limit,
    )
    position_feedback = jp.array(
        (
            jp.dot(position_feedback_world, body_forward),
            jp.dot(position_feedback_world, body_lateral),
        )
    )

    heading_error = wrap_angle(next_desired_heading - body_heading)
    next_heading_integral = jp.clip(
        heading_integral + heading_error * dt,
        -heading_integral_limit,
        heading_integral_limit,
    )
    heading_feedback = jp.clip(
        heading_kp * heading_error + heading_ki * next_heading_integral,
        -heading_feedback_limit,
        heading_feedback_limit,
    )
    requested_twist = jp.clip(
        jp.array(
            (
                next_filtered[0] + position_feedback[0],
                next_filtered[1] + position_feedback[1],
                next_filtered[2] + heading_feedback,
            )
        ),
        -twist_limit,
        twist_limit,
    )
    next_applied_twist = slew_limit(
        applied_twist, requested_twist, twist_rate_limit, dt
    )
    return (
        next_applied_twist,
        next_filtered,
        next_desired_position,
        next_desired_heading,
        next_position_integral,
        next_heading_integral,
        world_error,
        heading_error,
    )


def phase_masks(tripod_a: jp.ndarray, phase: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
    """Return the swing mask and local half-cycle timing for six tripod legs."""
    first_half = phase < 0.5
    tau = jp.where(first_half, phase * 2.0, (phase - 0.5) * 2.0)
    return tripod_a == first_half, tau


def advance_contact_gated_phase(
    *,
    phase: jp.ndarray,
    gait_enabled: jp.ndarray,
    swing: jp.ndarray,
    contacts: jp.ndarray,
    airborne: jp.ndarray,
    dt: float,
    phase_time: float,
    boundary_epsilon: float = 1e-6,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """Advance one tripod phase only after all three swing feet have landed.

    ``plant.slx`` changes tripod groups after both the 0.5 s clock has expired
    and every active swing leg has made contact after becoming airborne.  A
    phase that reaches its time boundary without contact is held just before
    the boundary so the same tripod remains active in late-landing search.
    """
    if phase_time <= 0.0:
        raise ValueError("phase_time must be positive")
    first_half = phase < 0.5
    half_end = jp.where(first_half, 0.5, 1.0)
    candidate = phase + dt / (2.0 * phase_time)
    boundary_reached = gait_enabled & (candidate >= half_end)
    landed = swing & contacts & airborne
    all_swing_landed = jp.all((~swing) | landed)
    cross_boundary = boundary_reached & all_swing_landed
    late_landing = boundary_reached & (~all_swing_landed)
    held_phase = half_end - boundary_epsilon
    next_half_start = jp.where(first_half, 0.5, 0.0)
    next_phase = jp.where(cross_boundary, next_half_start, candidate)
    next_phase = jp.where(late_landing, held_phase, next_phase)
    next_phase = jp.where(gait_enabled, next_phase, phase)
    return jp.mod(next_phase, 1.0), late_landing, cross_boundary


def heading_aligned_points(
    root_xy: jp.ndarray, forward_xy: jp.ndarray, local_offsets: jp.ndarray
) -> jp.ndarray:
    """Convert local forward/lateral terrain-grid offsets to world XY points.

    Only heading enters this mapping.  Callers must project their forward axis
    into XY before calling it, thereby keeping the sampling plane horizontal
    even when the body is pitched or rolled.
    """
    heading = forward_xy / jp.maximum(jp.linalg.norm(forward_xy), 1e-8)
    lateral = jp.array((-heading[1], heading[0]))
    return (
        root_xy[None, :]
        + local_offsets[:, 0, None] * heading[None, :]
        + local_offsets[:, 1, None] * lateral[None, :]
    )


def nominal_foot_targets(
    *,
    origins: jp.ndarray,
    outward: jp.ndarray,
    shoulder_lateral: jp.ndarray,
    tripod_a: jp.ndarray,
    phase: jp.ndarray,
    command: jp.ndarray,
    phase_time: float,
    step_scale: jp.ndarray,
    swing_height: jp.ndarray,
    radial_offset: jp.ndarray,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Generate documented PULL stance and quintic-cubic-Bezier swing targets."""
    swing, tau = phase_masks(tripod_a, phase)
    smooth = quintic(tau)
    # The source controller applies quintic time scaling to a cubic Bezier.
    # With P1.xy=P0.xy and P2.xy=P3.xy its horizontal interpolation is this
    # cubic smoothstep of the quintic clock.  Stance remains the documented
    # rigid-body PULL trajectory from +half stroke to -half stroke.
    bezier_progress = smooth * smooth * (3.0 - 2.0 * smooth)
    phase_position = jp.where(swing, bezier_progress - 0.5, 0.5 - tau)
    lift = jp.where(swing, 4.0 * swing_height * smooth * (1.0 - smooth), 0.0)
    radial = jp.where(swing, 4.0 * radial_offset * smooth * (1.0 - smooth), 0.0)

    nominal_body = _nominal_body_points(origins, outward, shoulder_lateral)
    _, _, yaw_rate = _command_components(command)
    yaw_velocity = jp.cross(
        jp.tile(jp.array((0.0, 0.0, yaw_rate)), (6, 1)), nominal_body
    )
    foot_velocity = controller_velocity(command) + yaw_velocity
    target_body = (
        nominal_body
        + foot_velocity * (phase_time * step_scale * phase_position)[:, None]
        + outward * radial[:, None]
        + jp.stack((jp.zeros(6), jp.zeros(6), lift), axis=-1)
    )
    tangent = jp.stack(
        (-outward[:, 1], outward[:, 0], jp.zeros(6)), axis=-1
    )
    relative = target_body - origins
    feet_local = jp.stack(
        (
            jp.sum(relative * outward, axis=-1),
            jp.sum(relative * tangent, axis=-1),
            relative[:, 2],
        ),
        axis=-1,
    )
    return feet_local, swing


def nominal_touchdown_body_targets(
    *,
    origins: jp.ndarray,
    outward: jp.ndarray,
    shoulder_lateral: jp.ndarray,
    command: jp.ndarray,
    phase_time: float,
) -> jp.ndarray:
    """Classical nominal touchdown locations expressed in the body frame.

    These targets contain no learned action and therefore expose useful
    classical knowledge to the terrain observation without changing the
    residual controller contract.
    """
    nominal_body = _nominal_body_points(origins, outward, shoulder_lateral)
    _, _, yaw_rate = _command_components(command)
    yaw_velocity = jp.cross(
        jp.tile(jp.array((0.0, 0.0, yaw_rate)), (6, 1)), nominal_body
    )
    foot_velocity = controller_velocity(command) + yaw_velocity
    return nominal_body + foot_velocity * (0.5 * phase_time)


def limit_effective_stride(
    *,
    requested_scale: jp.ndarray,
    command: jp.ndarray,
    origins: jp.ndarray,
    outward: jp.ndarray,
    shoulder_lateral: jp.ndarray,
    phase_time: float,
    max_stride: float,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Limit every leg's forward+yaw horizontal stroke to ``max_stride``.

    A yaw command produces a different tangential velocity at each hip.  The
    cap therefore uses the fastest nominal foot rather than forward velocity
    alone.  It is a safety layer after action scaling: low-speed action
    semantics remain unchanged, while high-speed/turn combinations cannot
    request an excessive Cartesian stroke.
    """
    if max_stride <= 0.0:
        raise ValueError("max_stride must be positive")
    nominal_body = _nominal_body_points(origins, outward, shoulder_lateral)
    _, _, yaw_rate = _command_components(command)
    yaw_velocity = jp.cross(
        jp.tile(jp.array((0.0, 0.0, yaw_rate)), (origins.shape[0], 1)),
        nominal_body,
    )
    foot_velocity = controller_velocity(command) + yaw_velocity
    peak_horizontal_speed = jp.max(jp.linalg.norm(foot_velocity[:, :2], axis=-1))
    requested_stride = peak_horizontal_speed * phase_time * requested_scale
    applied_scale = jp.where(
        requested_stride > max_stride,
        requested_scale * max_stride / jp.maximum(requested_stride, 1e-8),
        requested_scale,
    )
    effective_stride = peak_horizontal_speed * phase_time * applied_scale
    return applied_scale, effective_stride


def feasible_yaw_limit(
    *,
    speed: jp.ndarray,
    requested_yaw_limit: jp.ndarray,
    origins: jp.ndarray,
    outward: jp.ndarray,
    shoulder_lateral: jp.ndarray,
    phase_time: float,
    max_stride: float,
    max_frequency_scale: float,
    iterations: int = 12,
) -> jp.ndarray:
    """Return the largest symmetric yaw rate feasible at a given speed.

    At high forward speed, independently sampling maximum yaw can demand more
    stance stroke than the 120 mm Cartesian safety envelope can produce.  The
    available command-space speed is ``max_stride * frequency / phase_time``.
    A fixed-iteration bisection keeps both yaw signs feasible while preserving
    the requested forward speed.
    """
    if phase_time <= 0.0 or max_stride <= 0.0 or max_frequency_scale <= 0.0:
        raise ValueError("gait feasibility limits must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    nominal_body = _nominal_body_points(origins, outward, shoulder_lateral)
    capacity = max_stride * max_frequency_scale / phase_time
    requested = jp.maximum(requested_yaw_limit, 0.0)
    low = jp.zeros_like(requested)
    high = requested
    for _ in range(iterations):
        middle = 0.5 * (low + high)

        def peak(yaw_rate: jp.ndarray) -> jp.ndarray:
            yaw_velocity = jp.cross(
                jp.tile(jp.array((0.0, 0.0, yaw_rate)), (origins.shape[0], 1)),
                nominal_body,
            )
            foot_velocity = MODEL_FORWARD * speed + yaw_velocity
            return jp.max(jp.linalg.norm(foot_velocity[:, :2], axis=-1))

        feasible = jp.maximum(peak(middle), peak(-middle)) <= capacity
        low = jp.where(feasible, middle, low)
        high = jp.where(feasible, high, middle)
    straight_feasible = speed <= capacity
    return jp.where(straight_feasible, low, 0.0)


def self_collision_detected(
    contact_geom: jp.ndarray,
    contact_distance: jp.ndarray,
    geom_body_ids: jp.ndarray,
) -> jp.ndarray:
    """Return whether an active contact joins two robot-owned bodies."""
    bodies = geom_body_ids[contact_geom]
    active = contact_distance < 0.0
    return jp.any(active & (bodies[:, 0] > 0) & (bodies[:, 1] > 0))


def torque_saturation_cost(
    actuator_force: jp.ndarray,
    *,
    force_limit: float,
    threshold_fraction: float = 0.85,
) -> jp.ndarray:
    """Continuous 0..1 cost above a fraction of the hard torque limit."""
    if force_limit <= 0.0:
        raise ValueError("force_limit must be positive")
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must be in (0, 1)")
    threshold = force_limit * threshold_fraction
    span = force_limit - threshold
    normalized = jp.clip((jp.abs(actuator_force) - threshold) / span, 0.0, 1.0)
    return jp.mean(normalized)


def phase_masked_residual(
    raw_xyz: jp.ndarray,
    swing: jp.ndarray,
    *,
    swing_x: float,
    swing_y: float,
    swing_z_low: float,
    swing_z_high: float,
    stance_z: float,
) -> jp.ndarray:
    """Apply XYZ residual in swing and Z-only residual in stance.

    The raw 18 policy dimensions remain unchanged.  Stance X/Y are exactly
    zero by construction, so the nominal stance trajectory keeps ownership of
    horizontal foothold motion.
    """
    raw_xyz = jp.clip(raw_xyz, -1.0, 1.0)
    swing_residual = jp.stack(
        (
            raw_xyz[:, 0] * swing_x,
            raw_xyz[:, 1] * swing_y,
            scale_asymmetric(raw_xyz[:, 2], swing_z_low, swing_z_high),
        ),
        axis=-1,
    )
    stance_residual = jp.stack(
        (
            jp.zeros_like(raw_xyz[:, 0]),
            jp.zeros_like(raw_xyz[:, 1]),
            raw_xyz[:, 2] * stance_z,
        ),
        axis=-1,
    )
    return jp.where(swing[:, None], swing_residual, stance_residual)


def contact_adapt_targets(
    requested: jp.ndarray,
    current: jp.ndarray,
    swing: jp.ndarray,
    contacts: jp.ndarray,
    airborne: jp.ndarray,
    *,
    lost_contact_search: float,
    lost_contact_inward: float = 0.0,
    early_landing_allowed: jp.ndarray | None = None,
    late_landing: jp.ndarray | None = None,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """Prioritize deterministic contact handling over the learned residual.

    A swing foot is considered to have landed early only after it has first
    been observed airborne.  Contact that persists during normal liftoff is
    therefore not mistaken for landing.  A stance foot with no terrain contact
    receives only a small downward search; it cannot receive learned
    horizontal stance motion.
    """
    if early_landing_allowed is None:
        early_landing_allowed = jp.ones_like(swing, dtype=jp.bool_)
    if late_landing is None:
        late_landing = jp.zeros_like(swing, dtype=jp.bool_)
    early_landing = swing & airborne & contacts & early_landing_allowed
    lost_contact = (~swing) & (~contacts)
    search = lost_contact | late_landing
    adapted = jp.where(early_landing[:, None], current, requested)
    adapted = jp.where(late_landing[:, None], current, adapted)
    adapted = adapted.at[:, 0].add(
        jp.where(search, -lost_contact_inward, 0.0)
    )
    adapted = adapted.at[:, 2].add(
        jp.where(search, -lost_contact_search, 0.0)
    )
    return adapted, early_landing, search


def update_airborne_state(
    airborne: jp.ndarray, swing: jp.ndarray, contacts: jp.ndarray
) -> jp.ndarray:
    """Track whether each currently swinging leg has completed liftoff.

    Stance clears the state.  During swing, the state latches as soon as
    contact is absent and remains latched until stance starts again.  This
    pure transition is deliberately separate from contact adaptation so its
    contract can be tested without MuJoCo.
    """
    return swing & (airborne | (~contacts))


def median_support_height(
    terrain_heights: jp.ndarray,
    support_mask: jp.ndarray,
    fallback_height: jp.ndarray,
) -> jp.ndarray:
    """Robust support-surface height from stance feet currently in contact.

    Invalid feet sort to the end.  Odd and even contact counts use the usual
    median definition; no-contact states fall back to the local terrain height
    supplied by the caller.
    """
    count = jp.sum(support_mask.astype(jp.int32))
    ordered = jp.sort(jp.where(support_mask, terrain_heights, jp.inf))
    lower_index = jp.maximum((count - 1) // 2, 0)
    upper_index = jp.maximum(count // 2, 0)
    median = 0.5 * (ordered[lower_index] + ordered[upper_index])
    return jp.where(count > 0, median, fallback_height)


def leg_local_to_body(
    feet: jp.ndarray, origins: jp.ndarray, outward: jp.ndarray
) -> jp.ndarray:
    """Convert six yaw-aligned leg-local foot points to the URDF body frame."""
    tangent = jp.stack(
        (-outward[:, 1], outward[:, 0], jp.zeros(outward.shape[0])), axis=-1
    )
    return (
        origins
        + outward * feet[:, 0, None]
        + tangent * feet[:, 1, None]
        + MODEL_UP * feet[:, 2, None]
    )


def body_to_leg_local(
    feet_body: jp.ndarray, origins: jp.ndarray, outward: jp.ndarray
) -> jp.ndarray:
    """Convert URDF-body foot points to each leg's yaw-aligned frame."""
    tangent = jp.stack(
        (-outward[:, 1], outward[:, 0], jp.zeros(outward.shape[0])), axis=-1
    )
    relative = feet_body - origins
    return jp.stack(
        (
            jp.sum(relative * outward, axis=-1),
            jp.sum(relative * tangent, axis=-1),
            relative[:, 2],
        ),
        axis=-1,
    )


def rpy_matrix(rpy: jp.ndarray) -> jp.ndarray:
    """Controller-frame ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` matrix."""
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


def apply_body_pose_overlay(
    feet_body: jp.ndarray,
    translation_controller: jp.ndarray,
    posture_rpy: jp.ndarray,
) -> jp.ndarray:
    """Apply inverse 6-DOF body pose immediately before leg-frame IK.

    This is the source controller's posture-overlay rule extended with the
    correction-mode body translation.  It changes neither gait phase nor the
    nominal touchdown calculation.
    """
    translation_model = CONTROLLER_TO_MODEL @ translation_controller
    rotation_model = (
        CONTROLLER_TO_MODEL @ rpy_matrix(posture_rpy) @ CONTROLLER_TO_MODEL.T
    )
    return (feet_body - translation_model) @ rotation_model


def posture_pi_candidate(
    *,
    target_rpy: jp.ndarray,
    measured_rpy: jp.ndarray,
    desired_heading: jp.ndarray,
    measured_heading: jp.ndarray,
    posture_integral: jp.ndarray,
    posture_command: jp.ndarray,
    dt: float,
    kp: jp.ndarray,
    ki: jp.ndarray,
    integral_limit: jp.ndarray,
    angular_rate_limit: jp.ndarray,
    posture_limit: jp.ndarray,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """Single attitude PI and integrated posture command from the new design."""
    error = jp.array(
        (
            wrap_angle(target_rpy[0] - measured_rpy[0]),
            wrap_angle(target_rpy[1] - measured_rpy[1]),
            wrap_angle(desired_heading + target_rpy[2] - measured_heading),
        )
    )
    candidate_integral = jp.clip(
        posture_integral + error * dt, -integral_limit, integral_limit
    )
    angular_velocity = jp.clip(
        kp * error + ki * candidate_integral,
        -angular_rate_limit,
        angular_rate_limit,
    )
    candidate_command = jp.clip(
        posture_command + angular_velocity * dt,
        -posture_limit,
        posture_limit,
    )
    return candidate_command, candidate_integral, error


def workspace_valid(
    feet: jp.ndarray,
    shoulder_lateral: jp.ndarray,
    *,
    min_distance: float,
    max_distance: float,
    joint_limit: float,
) -> jp.ndarray:
    """Check an unprojected six-foot candidate before accepting body pose."""
    x, y, z = feet[:, 0], feet[:, 1], feet[:, 2]
    radius = jp.sqrt(x * x + y * y)
    lateral_valid = radius > jp.abs(shoulder_lateral) + 1e-6
    planar_radius = jp.sqrt(
        jp.maximum(radius * radius - shoulder_lateral * shoulder_lateral, 0.0)
    )
    rho = planar_radius - LINK1
    planar_z = z - SHOULDER_VERTICAL_OFFSET
    distance = jp.sqrt(rho * rho + planar_z * planar_z)
    distance_valid = (distance >= min_distance) & (distance <= max_distance)
    angles = analytical_ik(feet, shoulder_lateral)
    joint_valid = jp.all(jp.abs(angles) <= joint_limit, axis=-1)
    return jp.all(lateral_valid & distance_valid & joint_valid)


def project_workspace(
    feet: jp.ndarray,
    shoulder_lateral: jp.ndarray,
    *,
    min_distance: float = 0.112,
    max_distance: float = 0.345,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Project leg-local targets into the safe 2-link reachable annulus.

    The coxa yaw direction is preserved.  Returned cost is the mean squared
    Cartesian displacement caused by projection, i.e. a direct measure of an
    impossible target request rather than a post-hoc IK cosine clipping error.
    """
    if min_distance <= 0.0 or max_distance <= min_distance:
        raise ValueError("workspace distances must satisfy 0 < min < max")
    x, y, z = feet[:, 0], feet[:, 1], feet[:, 2]
    r_xy = jp.sqrt(x * x + y * y)
    planar_radius = jp.sqrt(
        jp.maximum(r_xy * r_xy - shoulder_lateral * shoulder_lateral, 0.0)
    )
    rho = planar_radius - LINK1
    planar_z = z - SHOULDER_VERTICAL_OFFSET
    distance = jp.sqrt(rho * rho + planar_z * planar_z)
    safe_distance = jp.clip(distance, min_distance, max_distance)
    scale = safe_distance / jp.maximum(distance, 1e-8)
    rho_safe = scale * rho
    z_safe = scale * planar_z
    unit_x = jp.where(r_xy > 1e-8, x / r_xy, jp.ones_like(r_xy))
    unit_y = jp.where(r_xy > 1e-8, y / r_xy, jp.zeros_like(r_xy))
    # The ideal radial projection can mathematically request a negative XY
    # radius for targets almost at the yaw axis.  XY radius cannot be
    # negative, so clamp it and restore the safe two-link distance with Z.
    radial_clamped = rho_safe < -LINK1 + 1e-5
    rho_safe = jp.maximum(rho_safe, -LINK1 + 1e-5)
    z_sign = jp.where(z_safe < 0.0, -1.0, 1.0)
    z_safe = jp.where(
        radial_clamped,
        z_sign * jp.sqrt(jp.maximum(safe_distance**2 - rho_safe**2, 0.0)),
        z_safe,
    )
    planar_radius_safe = rho_safe + LINK1
    r_xy_safe = jp.sqrt(
        planar_radius_safe * planar_radius_safe
        + shoulder_lateral * shoulder_lateral
    )
    projected = jp.stack(
        (
            unit_x * r_xy_safe,
            unit_y * r_xy_safe,
            z_safe + SHOULDER_VERTICAL_OFFSET,
        ),
        axis=-1,
    )
    projection_cost = jp.mean(jp.sum(jp.square(feet - projected), axis=-1))
    return projected, projection_cost


def analytical_ik(
    feet: jp.ndarray, shoulder_lateral: jp.ndarray
) -> jp.ndarray:
    """Invert the source URDF chain into common positive servo angles."""
    x, y, z = feet[:, 0], feet[:, 1], feet[:, 2]
    radius = jp.sqrt(x * x + y * y)
    planar_radius = jp.sqrt(
        jp.maximum(radius * radius - shoulder_lateral * shoulder_lateral, 0.0)
    )
    theta1 = jp.arctan2(y, x) - jp.arctan2(
        shoulder_lateral, planar_radius
    )
    theta1 = jp.arctan2(jp.sin(theta1), jp.cos(theta1))
    rho = planar_radius - LINK1
    planar_z = z - SHOULDER_VERTICAL_OFFSET
    cosine3 = (
        rho * rho + planar_z * planar_z - LINK2**2 - LINK3**2
    ) / (2.0 * LINK2 * LINK3)
    # Projection keeps this within range; clipping only absorbs float error.
    cosine3 = jp.clip(cosine3, -1.0, 1.0)
    theta3 = jp.arccos(cosine3)
    theta2 = jp.arctan2(-planar_z, rho) - jp.arctan2(
        LINK3 * jp.sin(theta3), LINK2 + LINK3 * jp.cos(theta3)
    )
    return jp.stack((theta1, theta2, theta3), axis=-1)
