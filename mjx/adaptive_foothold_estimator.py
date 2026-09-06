"""Fixed-size JAX foothold reference search and per-stage rejection diagnostics.

Plane fits use observed cells only. No inferred height is written to the map.
Path checks are sampled spherical-foot checks, not whole-robot collision proofs.
"""
import jax
import jax.numpy as jp

import adaptive_gait_controller as adaptive
import firmware_mjx_controller as fw
from adaptive_gait_perception import MAX_AGE, RESOLUTION

SEARCH_RADIUS = .08
RESIDUAL_RADIUS = .04
CANDIDATE_COUNT = 25
CANDIDATE_OFFSETS = jp.array([(x, y) for x in (-.08, -.04, 0., .04, .08)
                            for y in (-.08, -.04, 0., .04, .08)])
# One grid-cell spacing avoids counting the same cell twice in coverage.
PATCH_OFFSETS = RESOLUTION*jp.array(((0., 0.), (1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
EDGE_OFFSETS = RESOLUTION*jp.array([(x, y) for x in range(-2, 3) for y in range(-2, 3)])
EDGE_PAIRS = jp.array([(x*5+y, (x+1)*5+y) for x in range(4) for y in range(5)] +
                      [(x*5+y, x*5+y+1) for x in range(5) for y in range(4)])
MIN_COVERAGE = .6
MAX_PLANE_RESIDUAL = .008
MAX_SLOPE_RAD = .4363323129985824  # 25 degrees
EDGE_JUMP = .025  # between observed neighbouring 5 cm cells
FOOT_RADIUS = .032
EDGE_CLEARANCE = FOOT_RADIUS + .005
PATH_SAMPLES = 21
CANDIDATE_FEATURES = (
    'dx', 'dy', 'relative_height', 'center_known', 'coverage', 'terrain_ok', 'safe',
    'cell_spread', 'age', 'path_rise', 'path_coverage', 'plane_residual', 'slope',
    'edge_margin', 'edge_observed', 'ik_ok', 'path_ok', 'reference',
    'normal_x', 'normal_y', 'normal_z', 'slope_x', 'slope_y', 'confidence',
    'patch_length', 'patch_width', 'ik_margin', 'joint_margin',
)
UNKNOWN, LOW_COVERAGE, EDGE_ROUGH, IK_REJECTED, PATH_REJECTED, SAFE = range(6)
STATUS_NAMES = ('UNKNOWN', 'LOW_COVERAGE', 'EDGE_ROUGH', 'IK_REJECTED', 'PATH_REJECTED', 'SAFE')


def support_quality(height, known, spread):
    """Centre + at least two non-collinear neighbours; masked local plane fit."""
    weights = known.astype(jp.float32)
    coverage = jp.mean(weights, axis=-1)
    design = jp.concatenate((PATCH_OFFSETS/RESOLUTION, jp.ones((5, 1))), axis=-1)
    gram = jp.einsum('pi,...p,pj->...ij', design, weights, design)
    rhs = jp.einsum('pi,...p->...i', design, jp.where(known, height-height[..., :1], 0.))
    rank_ok = jp.linalg.det(gram) > .1
    # Regularization keeps unknown/collinear fits finite; rank_ok still rejects them.
    plane = jp.linalg.solve(gram + 1e-5*jp.eye(3), rhs[..., None])[..., 0]
    fitted = jp.einsum('pi,...i->...p', design, plane)
    error = jp.where(known, height-height[..., :1]-fitted, 0.)
    residual = jp.sqrt(jp.sum(error**2, axis=-1)/jp.maximum(jp.sum(weights, axis=-1), 1.))
    slope = jp.arctan(jp.linalg.norm(plane[..., :2]/RESOLUTION, axis=-1))
    gradient = plane[..., :2]/RESOLUTION
    normal = jp.concatenate((-gradient, jp.ones_like(gradient[..., :1])), axis=-1)
    normal /= jp.linalg.norm(normal, axis=-1, keepdims=True)
    cell_spread = jp.max(jp.where(known, spread, 0.), axis=-1)
    neighbor_jump = jp.any(known[..., 1:] & known[..., :1] &
                           (jp.abs(height[..., 1:]-height[..., :1]) > EDGE_JUMP), axis=-1)
    coverage_ok = known[..., 0] & (coverage >= MIN_COVERAGE) & rank_ok
    rough = (cell_spread > EDGE_JUMP) | neighbor_jump | (
        rank_ok & ((residual > MAX_PLANE_RESIDUAL) | (slope > MAX_SLOPE_RAD)))
    return dict(center_known=known[..., 0], any_known=jp.any(known, axis=-1),
                complete=jp.all(known, axis=-1), coverage=coverage, coverage_ok=coverage_ok,
                rank_ok=rank_ok, plane_residual=residual, slope=slope, gradient=gradient,
                normal=normal, spread=cell_spread, rough=rough)


def edge_distance(height, known, spread):
    """Conservative distance to observed discontinuities; unknown edges stay unknown."""
    left, right = EDGE_PAIRS[:, 0], EDGE_PAIRS[:, 1]
    discontinuity = known[..., left] & known[..., right] & (jp.abs(height[..., left]-height[..., right]) > EDGE_JUMP)
    midpoints = (EDGE_OFFSETS[left]+EDGE_OFFSETS[right])*.5
    # A grid-edge segment extends half a cell from its midpoint. Subtracting that
    # extent underestimates distance, so a foot is not accepted too close to it.
    distances = jp.maximum(jp.linalg.norm(midpoints, axis=-1)-RESOLUTION*.5, 0.)
    edge = jp.min(jp.where(discontinuity, distances, jp.inf), axis=-1)
    # A vertical spread inside a cell can also reveal a riser.
    cell_edge = known & (spread > EDGE_JUMP)
    cell_dist = jp.maximum(jp.linalg.norm(EDGE_OFFSETS, axis=-1)-RESOLUTION/jp.sqrt(2.), 0.)
    edge = jp.minimum(edge, jp.min(jp.where(cell_edge, cell_dist, jp.inf), axis=-1))
    return jp.minimum(edge, .15), jp.isfinite(edge)


def evaluate_candidates(env, data, info, xy, nominal, basis, lift, *, apex_delta=0., transfer_delta=0., privileged=False):
    """Terrain, endpoint/path IK and observed swing collisions for all 6x25 points."""
    query = lambda points: env._query(info['lidar_map'], points, data.time, privileged=privileged)
    height, known, age, spread = query(xy[..., None, :] + PATCH_OFFSETS)
    quality = support_quality(height, known, spread)
    eh, ek, _, es = query(xy[..., None, :] + EDGE_OFFSETS)
    margin, edge_observed = edge_distance(eh, ek, es)
    edge_rejected = edge_observed & (margin < EDGE_CLEARANCE)
    unsafe = quality['rough'] | edge_rejected
    terrain_ok = quality['coverage_ok'] & ~unsafe
    center_h = jp.where(quality['center_known'], height[..., 0], nominal[:, None, 2]-FOOT_RADIUS)
    world = jp.concatenate((xy, center_h[..., None]), axis=-1)
    cs = info['controller_state']
    rotation = data.xmat[env._root_id]
    model = (world + jp.array((0., 0., FOOT_RADIUS))-data.qpos[:3]) @ rotation
    body = jp.stack((-model[..., 1], model[..., 0], model[..., 2]), axis=-1)
    pre = (body @ fw._rotation_matrix(cs.posture_command).T).at[..., 2].add(cs.height_applied)
    starts = jp.broadcast_to(cs.foot_memory[:, None, :], pre.shape)
    feet = data.site_xpos[env._foot_site_ids]
    fractions = jp.linspace(0., 1., PATH_SAMPLES)
    footprint = PATCH_OFFSETS*(FOOT_RADIUS/RESOLUTION)
    line_xy = feet[:, None, None, :2] + fractions[None, None, :, None]*(xy[:, :, None, :]-feet[:, None, None, :2])
    ph, pk, _, _ = query(line_xy[..., None, :] + footprint)
    path_high = jp.max(jp.where(pk, ph, -jp.inf), axis=(-2, -1))
    path_high = jp.where(jp.any(pk, axis=(-2, -1)), path_high, center_h)
    required = jp.maximum(path_high-jp.maximum(feet[:, None, 2]-FOOT_RADIUS, center_h), 0.)
    minimum_clearance = jp.maximum(.04, required+.02)
    clearance = jp.clip(jp.maximum(required+fw.SWING_HEIGHT+jp.asarray(lift)[:, None], minimum_clearance), .04, .18)
    rise = center_h-(feet[:, None, 2]-FOOT_RADIUS)
    apex_phase = jp.clip(.5-jp.clip(rise/.15, -1., 1.)*.1+apex_delta, .3, .7)
    transfer = jp.clip(.5+transfer_delta, .35, .65)*jp.ones_like(clearance)
    paths = jax.vmap(adaptive.planned_swing, in_axes=(0, None, None, None, None, None))(
        fractions, starts, pre, clearance, apex_phase, transfer).transpose(1, 2, 0, 3)
    shifted = paths.at[..., 2].add(-cs.height_applied)
    controller_paths = fw._rotate_inverse(shifted, cs.posture_command)
    # Firmware IK expects the leg axis immediately before XYZ.
    ik_vectors = controller_paths.transpose(1, 2, 0, 3)
    angles, path_ik = fw._solve_ik(ik_vectors)
    local = fw._body_to_leg(ik_vectors)
    reach = jp.sqrt((jp.linalg.norm(local[..., :2], axis=-1)-fw.LINK_1)**2 + local[..., 2]**2)
    reach_margin = jp.minimum(fw.LINK_2+fw.LINK_3-reach, reach-abs(fw.LINK_2-fw.LINK_3))
    ik_margin = jp.min(reach_margin, axis=1).T
    joint_margin = jp.min(fw.JOINT_LIMIT-jp.max(jp.abs(angles), axis=-1), axis=1).T
    ik_ok = jp.all(path_ik, axis=1).T & (ik_margin >= fw.WORKSPACE_MARGIN) & (joint_margin >= .01745)
    path_model = jp.stack((controller_paths[..., 1], -controller_paths[..., 0], controller_paths[..., 2]), axis=-1)
    path_world = data.qpos[:3] + path_model @ rotation.T
    ch, ck, _, _ = query(path_world[..., None, :2] + footprint)
    # Conservative swept foot footprint, excluding exact takeoff/touchdown.
    interior = (fractions > .05) & (fractions < .95)
    collision = ck & interior[None, None, :, None] & (
        path_world[..., 2, None]-FOOT_RADIUS < ch+.002)
    path_ok = ~jp.any(collision, axis=(-2, -1)) & (clearance >= required+.02)
    path_coverage = jp.mean(ck.astype(jp.float32), axis=(-2, -1))
    path_observed = path_coverage >= MIN_COVERAGE
    safe = terrain_ok & ik_ok & path_ok & path_observed
    status = jp.where(~quality['any_known'], UNKNOWN,
        jp.where(unsafe, EDGE_ROUGH, jp.where(~quality['coverage_ok'], LOW_COVERAGE,
        jp.where(~ik_ok, IK_REJECTED, jp.where(~path_ok, PATH_REJECTED,
        jp.where(~path_observed, LOW_COVERAGE, SAFE)))))).astype(jp.int32)
    distance = jp.linalg.norm(xy-nominal[:, None, :2], axis=-1)
    reference_index = jp.where(jp.any(safe, axis=1), jp.argmin(jp.where(safe, distance, jp.inf), axis=1), -1)
    reference = (jp.arange(CANDIDATE_COUNT)[None, :] == reference_index[:, None])
    relative = center_h-(data.qpos[2]+fw.BASE_FOOT_Z-FOOT_RADIUS)
    confidence = quality['coverage']*jp.exp(-quality['plane_residual']/.008)*jp.clip(1.-age[..., 0]/MAX_AGE, 0., 1.)
    # Contiguous observed support extents through the centre in world-grid axes.
    same_surface = ek & (jp.abs(eh-center_h[..., None]) <= EDGE_JUMP) & (es <= EDGE_JUMP)
    grid_ok = same_surface.reshape(6, CANDIDATE_COUNT, 5, 5)
    lengths = []
    for axis in (0, 1):
        line = grid_ok[..., :, 2] if axis == 0 else grid_ok[..., 2, :]
        neg = line[..., 1].astype(jp.float32) + (line[..., 1] & line[..., 0]).astype(jp.float32)
        pos = line[..., 3].astype(jp.float32) + (line[..., 3] & line[..., 4]).astype(jp.float32)
        lengths.append(jp.where(line[..., 2], (1.+neg+pos)*RESOLUTION, 0.))
    features = jp.concatenate((jp.broadcast_to(CANDIDATE_OFFSETS, (6, CANDIDATE_COUNT, 2))/SEARCH_RADIUS,
        jp.stack((jp.where(quality['center_known'], relative/.2, 0.), quality['center_known'].astype(jp.float32),
            quality['coverage'], terrain_ok.astype(jp.float32), safe.astype(jp.float32),
            jp.clip(quality['spread']/.05, 0., 5.), jp.clip(age[..., 0]/MAX_AGE, 0., 1.),
            jp.where(quality['center_known'], required/.2, 0.), path_coverage,
            jp.clip(quality['plane_residual']/.01, 0., 5.), quality['slope']/MAX_SLOPE_RAD,
            margin/.15, edge_observed.astype(jp.float32), ik_ok.astype(jp.float32),
            path_ok.astype(jp.float32), reference.astype(jp.float32),
            quality['normal'][..., 0], quality['normal'][..., 1], quality['normal'][..., 2],
            quality['gradient'][..., 0], quality['gradient'][..., 1], confidence,
            lengths[0]/.25, lengths[1]/.25, jp.clip(ik_margin/.1, -2., 2.),
            jp.clip(joint_margin, -2., 2.)), axis=-1)), axis=-1)
    return dict(xy=xy, height=center_h, known=quality['coverage_ok'], unsafe=unsafe,
        safe=safe, terrain_ok=terrain_ok, ik_ok=ik_ok, path_ok=path_ok, status=status,
        path_height=path_high, path_coverage=path_coverage, required=required, clearance=clearance,
        pre=pre, world=world, features=features, nominal=nominal, basis=basis,
        edge_margin=margin, edge_observed=edge_observed, edge_rejected=edge_rejected,
        patch_known=known, patch_height=jp.where(known, height, 0.),
        confidence=confidence, patch_length=lengths[0], patch_width=lengths[1],
        ik_margin=ik_margin, joint_margin=joint_margin, apex_phase=apex_phase, transfer=transfer,
        reference_index=reference_index, **quality)
