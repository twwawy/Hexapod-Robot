#!/usr/bin/env python3
"""Keyboard MJX replay of the 23-D controller, with or without a new checkpoint."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import queue
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terrain', choices=('flat', 'steps', 'ramp'))
    parser.add_argument('--terrain-level', type=int)
    parser.add_argument('--perception', choices=('lidar', 'teacher', 'blind'))
    parser.add_argument('--checkpoint', type=Path, help='new adaptive run or exact checkpoint; stage31 is incompatible')
    parser.add_argument('--residual-scale', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--speed', type=float, default=0.)
    args = parser.parse_args()
    if not 0. <= args.residual_scale <= 1.:
        parser.error('--residual-scale must be in [0,1]')
    if args.terrain is not None and args.terrain_level is not None:
        parser.error('choose --terrain or --terrain-level')
    import jax
    import jax.numpy as jp
    import mujoco
    import mujoco.viewer
    import numpy as np
    from ml_collections import config_dict
    from adaptive_gait_env import AdaptiveGaitEnv, default_config, FOOT_RADIUS
    from adaptive_gait_perception import initial_map, GRID_N, RESOLUTION, MAX_AGE
    from adaptive_gait_policy import contract, load_policy

    policy, metadata = load_policy(args.checkpoint) if args.checkpoint else (None, None)
    perception = args.perception or (metadata['actor_source'] if metadata else 'lidar')
    if metadata and perception != metadata['actor_source']:
        parser.error('checkpoint perception must match training; use --init-teacher to train a LiDAR policy')
    level = args.terrain_level
    if level is None:
        level = {'flat': 0, 'steps': 5, 'ramp': 3}.get(args.terrain, metadata['terrain_level'] if metadata else 0)
    cfg = config_dict.ConfigDict(metadata['config']) if metadata else default_config()
    sensor = metadata['lidar'] if metadata else dict(azimuths=90, elevations=8, dropout=.05, noise_m=.005)
    env = AdaptiveGaitEnv(terrain_level=level, perception=perception, config=cfg,
        azimuths=sensor['azimuths'], elevations=sensor['elevations'], dropout=sensor['dropout'], noise=sensor['noise_m'])
    print('Compiling MJX reset/step; the first frame can take time.', flush=True)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    observe = jax.jit(env._get_obs)
    candidates = jax.jit(env._candidates)
    inference = jax.jit(policy) if policy else None
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = reset(reset_key)
    jax.block_until_ready(state.data.qpos)
    model, display = env.mj_model, mujoco.MjData(env.mj_model)
    keys = queue.SimpleQueue()
    vx, wz, paused, show_map = float(np.clip(args.speed, -.12, .12)), 0., False, True
    history = deque(maxlen=500)
    output = Path(__file__).resolve().parent/'generated/adaptive_gait'
    print(f'23-D adaptive | {perception} | {env.terrain_description} | '+
          ('NEW CHECKPOINT' if policy else 'ZERO ACTION, NO TRAINED POLICY'), flush=True)
    print('Arrows: speed/yaw; Space: stop; Enter: pause; H: reset; C: clear map; M: map; P: save trace', flush=True)

    def sphere(scene, position, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, radius), np.asarray(position), np.eye(3).reshape(9), np.asarray(color, dtype=np.float32))
        scene.ngeom += 1

    with mujoco.viewer.launch_passive(model, display, key_callback=keys.put) as viewer:
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 135.
        viewer.cam.elevation = -30.
        last_log = -1.
        while viewer.is_running():
            tick_start = time.monotonic()
            save = False
            while not keys.empty():
                code = keys.get()
                if code == 265:
                    vx = min(.12, vx+.02)
                elif code == 264:
                    vx = max(-.12, vx-.02)
                elif code == 263:
                    wz = min(.3, wz+.1)
                elif code == 262:
                    wz = max(-.3, wz-.1)
                elif code == 32:
                    vx = wz = 0.
                elif code == 257:
                    paused = not paused
                elif code == ord('H'):
                    key, reset_key = jax.random.split(key)
                    state = reset(reset_key)
                    vx = wz = 0.
                    paused = False
                    history.clear()
                    last_log = -1.
                elif code == ord('C'):
                    state.info['lidar_map'] = initial_map(state.data.qpos[:2])
                elif code == ord('M'):
                    show_map = not show_map
                elif code == ord('P'):
                    save = True
            state.info['command'] = jp.array((vx, wz, 0., 0., 0.))
            state = state.replace(obs=observe(state.data, state.info))
            if not paused:
                key, action_key = jax.random.split(key)
                action = inference(state.obs, action_key)[0] if inference else jp.zeros(23)
                action = action*args.residual_scale
                before = state
                state = step(state, action)
                jax.block_until_ready(state.data.qpos)
                cs = state.info['controller_state']
                history.append(dict(time=float(before.data.time), qpos=np.asarray(before.data.qpos),
                    qvel=np.asarray(before.data.qvel), command=np.asarray(before.info['command']),
                    observation=np.asarray(before.obs['state']), action=np.asarray(action),
                    accepted_action=np.asarray(cs.accepted_action), targets=np.asarray(state.data.ctrl),
                    applied_twist=np.asarray(state.info['controller_output'].applied_twist),
                    posture=np.asarray(cs.adapt_posture), height_applied=float(cs.height_applied),
                    stride_scale=float(cs.stride_scale), phase_duration=float(cs.phase_duration),
                    goal_world=np.asarray(cs.goal_world), contacts=np.asarray(state.info['contact_state']),
                    phase=np.asarray(state.info['controller_output'].gait_progress),
                    done=float(state.done), next_qpos=np.asarray(state.data.qpos)))
                # Idle inspection is allowed indefinitely. A terminal state during
                # commanded walking pauses for inspection rather than auto-reset.
                if float(state.done):
                    paused = True
                    reasons = {k: float(v) for k, v in state.metrics.items() if k.startswith('termination/') and float(v)}
                    print(f'Terminated: {reasons}; success={float(state.metrics["terrain_success"])}. H resets.', flush=True)
            display.qpos[:] = np.asarray(state.data.qpos)
            display.qvel[:] = np.asarray(state.data.qvel)
            display.ctrl[:] = np.asarray(state.data.ctrl)
            display.time = float(state.data.time)
            mujoco.mj_forward(model, display)
            plan = jax.device_get(candidates(state.data, state.info))
            cs = jax.device_get(state.info['controller_state'])
            with viewer.lock():
                viewer.cam.lookat[:] = display.qpos[:3]
                scene = viewer.user_scn
                scene.ngeom = 0
                if show_map:
                    grid = jax.device_get(state.info['lidar_map'])
                    valid = (display.time-grid.timestamp <= MAX_AGE) & (display.time >= grid.timestamp)
                    indices = np.argwhere(valid)
                    indices = indices[::max(1, int(np.ceil(len(indices)/700)))]
                    for i, j in indices:
                        xy = grid.center + (np.array((i, j))+.5-GRID_N/2)*RESOLUTION
                        sphere(scene, (*xy, grid.height[i, j]+.002), .009, (.1, .7, .85, .14))
                for leg in range(6):
                    for candidate in range(9):
                        if plan['safe'][leg, candidate]:
                            sphere(scene, (*plan['xy'][leg, candidate], plan['height'][leg, candidate]+.01),
                                   .011, (.1, .9, .25, .9))
                    if cs.active_known[leg]:
                        sphere(scene, cs.goal_world[leg]+np.array((0., 0., FOOT_RADIUS)), .019, (1., .15, .05, 1.))
            viewer.sync()
            if display.time-last_log >= 1.:
                last_log = display.time
                m = state.metrics
                compared = float(m['map_compared_count'])
                error = f'{float(m["map_mae_m"])*1000:.1f}mm' if compared else 'n/a'
                print(f't={display.time:.1f} v={vx:+.2f} yaw={wz:+.2f} '+
                    f'known={float(m["map_known_fraction"]):.0%} map error={error} '+
                    f'IK={np.asarray(state.info["controller_output"].ik_valid).astype(int)} '+
                    f'reject={bool(cs.plan_rejected)} stride={float(cs.stride_scale):.2f} phase={float(cs.phase_duration):.2f}s', flush=True)
            if save:
                output.mkdir(parents=True, exist_ok=True)
                (output/'.gitignore').write_text('*\n')
                if history:
                    np.savez_compressed(output/'trace.npz', **{name: np.stack([row[name] for row in history]) for name in history[0]})
                grid = jax.device_get(state.info['lidar_map'])
                np.savez_compressed(output/'map.npz', height=grid.height, timestamp=grid.timestamp,
                                    center=grid.center, spread=grid.spread, time=display.time)
                (output/'adaptive_contract.json').write_text(json.dumps(contract(env), indent=2)+'\n')
                mujoco.mj_saveLastXML(str(output/'model.xml'), model)
                print(f'Saved trace/map/model/contract: {output}', flush=True)
            time.sleep(max(0., env.dt-(time.monotonic()-tick_start)))


if __name__ == '__main__':
    main()
