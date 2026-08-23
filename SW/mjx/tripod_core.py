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


LINK1 = 0.074
LINK2 = 0.121
LINK3 = 0.230
MODEL_FORWARD = jp.array((0.0, -1.0, 0.0))


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


def phase_masks(tripod_a: jp.ndarray, phase: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
    """Return the swing mask and local half-cycle timing for six tripod legs."""
    first_half = phase < 0.5
    tau = jp.where(first_half, phase * 2.0, (phase - 0.5) * 2.0)
    return tripod_a == first_half, tau


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
    tripod_a: jp.ndarray,
    phase: jp.ndarray,
    command: jp.ndarray,
    phase_time: float,
    step_scale: jp.ndarray,
    swing_height: jp.ndarray,
    radial_offset: jp.ndarray,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Generate the documented nominal tripod target in each leg frame."""
    swing, tau = phase_masks(tripod_a, phase)
    smooth = quintic(tau)
    phase_position = jp.where(swing, smooth - 0.5, 0.5 - tau)
    lift = jp.where(swing, 4.0 * swing_height * smooth * (1.0 - smooth), 0.0)
    radial = jp.where(swing, 4.0 * radial_offset * smooth * (1.0 - smooth), 0.0)

    nominal_body = origins + outward * 0.218728
    nominal_body = nominal_body.at[:, 2].set(origins[:, 2] - 0.287006)
    yaw_velocity = jp.cross(
        jp.tile(jp.array((0.0, 0.0, command[1])), (6, 1)), nominal_body
    )
    foot_velocity = MODEL_FORWARD * command[0] + yaw_velocity
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
    command: jp.ndarray,
    phase_time: float,
) -> jp.ndarray:
    """Classical nominal touchdown locations expressed in the body frame.

    These targets contain no learned action and therefore expose useful
    classical knowledge to the terrain observation without changing the
    residual controller contract.
    """
    nominal_body = origins + outward * 0.218728
    nominal_body = nominal_body.at[:, 2].set(origins[:, 2] - 0.287006)
    yaw_velocity = jp.cross(
        jp.tile(jp.array((0.0, 0.0, command[1])), (6, 1)), nominal_body
    )
    foot_velocity = MODEL_FORWARD * command[0] + yaw_velocity
    return nominal_body + foot_velocity * (0.5 * phase_time)


def limit_effective_stride(
    *,
    requested_scale: jp.ndarray,
    command: jp.ndarray,
    origins: jp.ndarray,
    outward: jp.ndarray,
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
    nominal_body = origins + outward * 0.218728
    nominal_body = nominal_body.at[:, 2].set(origins[:, 2] - 0.287006)
    yaw_velocity = jp.cross(
        jp.tile(jp.array((0.0, 0.0, command[1])), (origins.shape[0], 1)),
        nominal_body,
    )
    foot_velocity = MODEL_FORWARD * command[0] + yaw_velocity
    peak_horizontal_speed = jp.max(jp.linalg.norm(foot_velocity[:, :2], axis=-1))
    requested_stride = peak_horizontal_speed * phase_time * requested_scale
    applied_scale = jp.where(
        requested_stride > max_stride,
        requested_scale * max_stride / jp.maximum(requested_stride, 1e-8),
        requested_scale,
    )
    effective_stride = peak_horizontal_speed * phase_time * applied_scale
    return applied_scale, effective_stride


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
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """Prioritize deterministic contact handling over the learned residual.

    A swing foot is considered to have landed early only after it has first
    been observed airborne.  Contact that persists during normal liftoff is
    therefore not mistaken for landing.  A stance foot with no terrain contact
    receives only a small downward search; it cannot receive learned
    horizontal stance motion.
    """
    early_landing = swing & airborne & contacts
    lost_contact = (~swing) & (~contacts)
    adapted = jp.where(early_landing[:, None], current, requested)
    adapted = adapted.at[:, 2].add(
        jp.where(lost_contact, -lost_contact_search, 0.0)
    )
    return adapted, early_landing, lost_contact


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


def hysteretic_clearance_contact(
    clearance: jp.ndarray,
    previous: jp.ndarray,
    *,
    enter_clearance: float,
    release_clearance: float,
) -> jp.ndarray:
    """Latch geometric contact until a larger release clearance is reached."""
    if release_clearance <= enter_clearance:
        raise ValueError("contact release clearance must exceed enter clearance")
    threshold = jp.where(previous, release_clearance, enter_clearance)
    return clearance < threshold


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


def project_workspace(
    feet: jp.ndarray,
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
    rho = r_xy - LINK1
    distance = jp.sqrt(rho * rho + z * z)
    safe_distance = jp.clip(distance, min_distance, max_distance)
    scale = safe_distance / jp.maximum(distance, 1e-8)
    rho_safe = scale * rho
    z_safe = scale * z
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
    r_xy_safe = rho_safe + LINK1
    projected = jp.stack((unit_x * r_xy_safe, unit_y * r_xy_safe, z_safe), axis=-1)
    projection_cost = jp.mean(jp.sum(jp.square(feet - projected), axis=-1))
    return projected, projection_cost


def analytical_ik(feet: jp.ndarray) -> jp.ndarray:
    """Convert safe leg-local Cartesian targets to documented servo angles."""
    x, y, z = feet[:, 0], feet[:, 1], feet[:, 2]
    theta1 = jp.arctan2(y, x)
    rho = jp.sqrt(x * x + y * y) - LINK1
    cosine3 = (
        rho * rho + z * z - LINK2**2 - LINK3**2
    ) / (2.0 * LINK2 * LINK3)
    # Projection keeps this within range; clipping only absorbs float error.
    cosine3 = jp.clip(cosine3, -1.0, 1.0)
    theta3 = jp.arctan2(-jp.sqrt(jp.maximum(0.0, 1.0 - cosine3**2)), cosine3)
    theta2 = jp.arctan2(z, rho) - jp.arctan2(
        LINK3 * jp.sin(theta3), LINK2 + LINK3 * jp.cos(theta3)
    )
    return jp.stack((theta1, -theta2, -theta3), axis=-1)
