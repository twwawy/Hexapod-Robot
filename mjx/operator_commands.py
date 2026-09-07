"""Fixed-shape RC command sampler and signed motion diagnostics (SI units)."""
import jax
import jax.numpy as jp

COMMAND_CONTRACT = 'rc_vx_wz_hold_slew_v1'


def next_command(command, target, remaining, key, dt):
    key, mode_key, speed_key, yaw_key, duration_key = jax.random.split(key, 5)
    mode = jax.random.randint(mode_key, (), 0, 9)
    speed = jax.random.uniform(speed_key, (), minval=.03, maxval=.08)
    yaw = jax.random.uniform(yaw_key, (), minval=.12, maxval=.25)
    # stop, forward, reverse, left/right turn, forward/reverse arcs
    vx = jp.array((0., 1., -1., 0., 0., 1., 1., -1., -1.))[mode]*speed
    wz = jp.array((0., 0., 0., 1., -1., 1., -1., 1., -1.))[mode]*yaw
    target = jp.where(remaining <= 0., jp.stack((vx, wz)), target)
    remaining = jp.where(remaining <= 0., jax.random.uniform(duration_key, (), minval=2., maxval=5.), remaining)-dt
    applied = command[:2]+jp.clip(target-command[:2], -jp.array((.08, .4))*dt, jp.array((.08, .4))*dt)
    applied = jp.where(jp.abs(applied) < jp.array((.005, .02)), 0., applied)
    # Retain an un-deadbanded slew state separately at the caller: otherwise
    # a zero command would never leave its deadband in small policy ticks.
    slew = command[:2]+jp.clip(target-command[:2], -jp.array((.08, .4))*dt, jp.array((.08, .4))*dt)
    return applied, slew, target, remaining, key


def motion_score(vx, wz, command):
    desired = jp.array((command[0], .3*command[1]))
    actual = jp.array((vx, .3*wz))
    magnitude = jp.linalg.norm(desired)
    moving = magnitude > .005
    signed_speed = jp.dot(actual, desired)/jp.maximum(magnitude, .005)
    good = (jp.abs(vx-command[0]) <= .02) & (jp.abs(wz-command[1]) <= .07)
    progress = jp.where(moving, jp.clip(signed_speed/jp.maximum(magnitude, .02), -1., 1.5), 0.)
    return signed_speed, moving, good, progress
