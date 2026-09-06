"""Fixed top-K foothold combinations and quasi-static support margins."""
import jax
import jax.numpy as jp
import firmware_mjx_controller as fw
from wave_gait_scheduler import WAVE, swing_mask

TOP_K = 3
COMBINATIONS = jp.array([(a, b, c) for a in range(TOP_K) for b in range(TOP_K)
                        for c in range(TOP_K)], dtype=jp.int32)
MIN_SUPPORT_MARGIN = .012
PAIRS = jp.array([(a, b) for a in range(6) for b in range(a+1, 6)])


def support_margin(feet_xy, mask, com_xy):
    """Signed distance to convex hull edges without sorting or dynamic hulls."""
    a, b = PAIRS[:, 0], PAIRS[:, 1]
    start, end = feet_xy[..., a, :], feet_xy[..., b, :]
    edge = end-start
    delta = feet_xy[..., None, :, :]-start[..., :, None, :]
    cross = edge[..., :, 0, None]*delta[..., :, :, 1]-edge[..., :, 1, None]*delta[..., :, :, 0]
    length = jp.linalg.norm(edge, axis=-1)
    left = jp.all(jp.where(mask[..., None, :], cross >= -1e-7, True), axis=-1)
    right = jp.all(jp.where(mask[..., None, :], cross <= 1e-7, True), axis=-1)
    valid = mask[..., a] & mask[..., b] & (length > 1e-6) & (left | right)
    c = com_xy[..., None, :]-start
    distance = (edge[..., 0]*c[..., 1]-edge[..., 1]*c[..., 0])/jp.maximum(length, 1e-6)
    distance *= jp.where(left, 1., -1.)
    margin = jp.min(jp.where(valid, distance, jp.inf), axis=-1)
    return jp.where((jp.sum(mask, axis=-1) >= 3) & jp.isfinite(margin), margin, -1.)


def phase_feasibility(plan, feet, contacts, com, mode, phase):
    mask = swing_mask(mode, phase)
    terrain_observed = jp.any(plan['coverage_ok'], axis=1)
    # Path unknowns cannot establish known-infeasibility or permission to lift.
    leg_safe = plan['safe'] & (plan['path_coverage'] >= .6)
    cost = jp.linalg.norm(plan['xy']-plan['nominal'][:, None, :2], axis=-1)
    _, top_indices = jax.lax.top_k(jp.where(leg_safe, -cost, -1e6), TOP_K)
    stance_margin = support_margin(feet[:, :2], ~mask & contacts, com[:2])
    # Dynamic leg indices still produce a fixed 3-leg array. Wave uses first only.
    legs = jp.nonzero(mask, size=3, fill_value=0)[0]
    choices = top_indices[legs[:, None], COMBINATIONS.T].T
    ids = jp.broadcast_to(jp.maximum(plan['reference_index'], 0), (27, 6))
    for slot in range(3):
        apply = (mode != WAVE) | (slot == 0)
        ids = ids.at[:, legs[slot]].set(jp.where(apply, choices[:, slot], ids[:, legs[slot]]))
    targets = plan['world'][jp.arange(6)[None, :], ids]
    future = jp.where(mask[None, :, None], targets, feet[None, :, :])
    margin_all = support_margin(future[..., :2], jp.ones((27, 6), dtype=jp.bool_), com[:2])
    opposite_margin = support_margin(future[..., :2], jp.broadcast_to(mask, (27, 6)), com[:2])
    landing_margin = jp.where(mode == WAVE, margin_all, jp.minimum(margin_all, opposite_margin))
    chosen_safe = leg_safe[jp.arange(6)[None, :], ids]
    valid = jp.all(~mask[None, :] | chosen_safe, axis=1) & (landing_margin >= MIN_SUPPORT_MARGIN)
    valid &= stance_margin >= MIN_SUPPORT_MARGIN
    costs = jp.sum(jp.where(mask[None, :], cost[jp.arange(6)[None, :], ids], 0.), axis=1)
    best = jp.argmin(jp.where(valid, costs, jp.inf))
    feasible = jp.any(valid)
    observed = jp.all(~mask | terrain_observed)
    # Only classify absence of paths as infeasible when those paths are observed.
    path_observed = jp.all(~mask | jp.any(plan['path_coverage'] >= .6, axis=1))
    known_infeasible = observed & path_observed & ~feasible
    return dict(feasible=feasible, known_infeasible=known_infeasible,
                observed=observed & path_observed, indices=ids[best],
                support_margin=jp.minimum(stance_margin, landing_margin[best]), swing_mask=mask)
