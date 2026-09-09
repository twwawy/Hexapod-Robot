"""One-foot collision retry: retract, lift, transfer, land; fixed JAX shapes."""
from typing import NamedTuple
import jax
import jax.numpy as jp
import firmware_mjx_controller as fw
from foothold_feasibility import support_margin

DURATION = 1.6
DETECT_SECONDS = .12
KNOTS = jp.array((0., .2, .4, .75, 1.))
BACKOFF = jp.array((.025, .04, .06))
FOOT_RADIUS = .032


class RetryState(NamedTuple):
    active: object
    requested: object
    leg: object
    waypoints: object
    elapsed: object
    attempted_epoch: object
    blocked_time: object
    last_world: object
    rejected: object


def initial_state():
    return RetryState(jp.asarray(False), jp.asarray(False), jp.asarray(0), jp.zeros((5, 3)),
        jp.asarray(0.), jp.full(6, -1, dtype=jp.int32), jp.zeros(6), jp.zeros((6, 3)), jp.asarray(False))


def trajectory(progress, waypoints):
    """The very same piecewise quintic is sampled by planner and controller."""
    i = jp.clip(jp.searchsorted(KNOTS, progress, side='right')-1, 0, 3)
    t = fw._quintic(jp.clip((progress-KNOTS[i])/(KNOTS[i+1]-KNOTS[i]), 0., 1.))
    return waypoints[i] + t[..., None]*(waypoints[i+1]-waypoints[i])


def contact_kinds(geom, dist, frame, foot_ids, body_ids):
    ids = jp.maximum(geom, 0)
    first_foot = ids[:, 0, None] == foot_ids[None, :]
    second_foot = ids[:, 1, None] == foot_ids[None, :]
    pair = (first_foot & (body_ids[ids[:, 1]] == 0)[:, None]) | (second_foot & (body_ids[ids[:, 0]] == 0)[:, None])
    active = jp.all(geom >= 0, axis=-1) & (dist <= 0.)
    vertical = jp.abs(frame.reshape((-1, 3, 3))[:, 0, 2])
    normal_z = frame.reshape((-1, 3, 3))[:, 0, 2]
    upward = normal_z[:, None]*jp.where(first_foot, -1., 1.)
    support = jp.any(pair & active[:, None] & (upward >= .5), axis=0)
    wall = jp.any(pair & (active & (vertical < .5))[:, None], axis=0)
    return support, wall


def prepare(env, data, info, cs):
    r = cs.foot_retry
    feet = data.site_xpos[env._foot_site_ids]
    contacts = info['confirmed_contacts']
    c = data._impl.contact
    _, wall = contact_kinds(c.geom, c.dist, c.frame, env._foot_geom_ids, env._geom_body_ids)
    import wave_gait_scheduler as sched
    mask = sched.swing_mask(cs.scheduler.mode, cs.scheduler.phase)
    speed = jp.linalg.norm(feet-r.last_world, axis=-1)/env.dt
    commanded_body = fw._rotate_inverse(cs.foot_memory.at[:, 2].add(-cs.height_applied), cs.posture_command)
    commanded_model = jp.stack((commanded_body[:, 1], -commanded_body[:, 0], commanded_body[:, 2]), axis=-1)
    commanded_world = data.qpos[:3] + commanded_model @ data.xmat[env._root_id].T
    tracking_error = jp.linalg.norm(commanded_world-feet, axis=-1)
    # Shin/servo blockage may occur without a spherical-foot wall contact.
    # Require sustained tracking error too, so a normal apex pause is not a retry.
    stalled_tracking = (tracking_error > .025) & ~contacts
    blocked = (mask & ~cs.scheduler.landed & (wall | stalled_tracking) & (speed < .015) &
               cs.scheduler.running & (cs.scheduler.elapsed/cs.phase_duration > .15))
    timers = jp.where(blocked, r.blocked_time+env.dt, 0.)
    eligible = (timers >= jp.where(wall, DETECT_SECONDS, .25)) & (r.attempted_epoch != cs.scheduler.epoch)
    # Only one unconfirmed foot; never lift a second support leg to recover.
    eligible &= (jp.sum(contacts) - contacts.astype(jp.int32)) == 5
    leg = jp.argmax(eligible)
    launch = jp.any(eligible) & ~r.active & ~cs.scheduler.fault & ~cs.recenter_active

    def plan(_):
        start = feet[leg]
        # Preserve the active landing endpoint. Map refresh cannot move this goal.
        endpoint_body = fw._rotate_inverse(cs.swing_end.at[:, 2].add(-cs.height_applied), cs.posture_command)[leg]
        endpoint_model = jp.array((endpoint_body[1], -endpoint_body[0], endpoint_body[2]))
        rotation = data.xmat[env._root_id]
        end = data.qpos[:3]+endpoint_model @ rotation.T
        forward = rotation @ jp.array((0., -1., 0.))
        forward = forward.at[2].set(0.)
        forward /= jp.maximum(jp.linalg.norm(forward), 1e-6)
        retreat = start[None, :] - BACKOFF[:, None]*forward
        retreat = retreat.at[:, 2].add(.015)
        top = jp.maximum(start[2], end[2]) + jp.minimum(cs.swing_clearance[leg]+.02, .18)
        lifted = retreat.at[:, 2].set(top)
        transfer = jp.broadcast_to(end, (3, 3)).at[:, 2].set(top)
        waypoints = jp.stack((jp.broadcast_to(start, (3, 3)), retreat, lifted, transfer,
                              jp.broadcast_to(end, (3, 3))), axis=1)
        fractions = jp.linspace(0., 1., 41)
        samples = jax.vmap(lambda w: trajectory(fractions, w))(waypoints)
        footprint = FOOT_RADIUS*jp.array(((0., 0.), (1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
        h, known, _, _ = env._query(info['lidar_map'], samples[..., None, :2]+footprint, data.time)
        margin = samples[..., 2, None]-FOOT_RADIUS-h
        # Initial wall contact may overlap the conservative swept-foot height
        # envelope. Escape must never worsen it and must clear it by retraction end.
        escape_floor = jp.minimum(margin[:, :1, :], -.002)
        allowed = jp.where((fractions <= .2)[None, :, None], margin >= escape_floor-.001, margin >= -.002)
        terrain_safe = jp.all(known & allowed, axis=(1, 2))
        model = (samples-data.qpos[:3]) @ rotation
        body = jp.stack((-model[..., 1], model[..., 0], model[..., 2]), axis=-1)
        _, valid = fw._solve_ik(jp.broadcast_to(body[..., None, :], (3, 41, 6, 3)))
        _, limited = fw._limit_foot_reach(jp.broadcast_to(body[..., None, :], (3, 41, 6, 3)))
        path_ok = jp.all(valid[..., leg] & ~limited[..., leg], axis=1)
        stance_model = (feet-data.qpos[:3]) @ rotation
        stance_body = jp.stack((-stance_model[:, 1], stance_model[:, 0], stance_model[:, 2]), axis=-1)
        _, stance_valid = fw._solve_ik(stance_body)
        _, stance_limited = fw._limit_foot_reach(stance_body)
        stance_ok = jp.all((jp.arange(6) == leg) | (stance_valid & ~stance_limited))
        stance_ok &= support_margin(feet[:, :2], jp.arange(6) != leg, data.subtree_com[env._root_id, :2]) >= .02
        safe = terrain_safe & path_ok & stance_ok
        index = jp.argmax(safe)  # shortest safe backoff
        points_model = (waypoints[index]-data.qpos[:3]) @ rotation
        points_body = jp.stack((-points_model[:, 1], points_model[:, 0], points_model[:, 2]), axis=-1)
        pre = (points_body @ fw._rotation_matrix(cs.posture_command).T).at[:, 2].add(cs.height_applied)
        return pre, jp.any(safe)

    points, valid = jax.lax.cond(launch, plan, lambda _: (r.waypoints, jp.asarray(False)), operand=None)
    requested = launch & valid
    # Failed preflight is also a bounded attempt; do not rebuild every policy tick.
    attempted = r.attempted_epoch.at[leg].set(jp.where(launch, cs.scheduler.epoch, r.attempted_epoch[leg]))
    return r._replace(requested=requested, leg=jp.where(requested, leg, r.leg),
        waypoints=jp.where(requested, points, r.waypoints), blocked_time=timers,
        last_world=feet, attempted_epoch=attempted, rejected=launch & ~valid)
