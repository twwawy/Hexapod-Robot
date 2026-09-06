#!/usr/bin/env python3
"""Keyboard MJX replay of the 24-D hybrid controller, with or without a checkpoint."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terrain', choices=('flat', 'steps', 'ramp'))
    parser.add_argument('--terrain-level', type=int)
    parser.add_argument('--perception', choices=('lidar', 'teacher', 'blind', 'oracle'),
                        help='oracle is debug-only GT with 100%% known coverage')
    parser.add_argument('--checkpoint', type=Path, help='new adaptive run or exact checkpoint; stage31 is incompatible')
    parser.add_argument('--residual-scale', type=float, default=1.)
    parser.add_argument('--gait-mode', choices=('hybrid', 'tripod', 'wave'))
    parser.add_argument('--stage0', action='store_true', help='extra same-pose GT oracle comparison; no policy')
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--speed', type=float, default=0.)
    parser.add_argument('--fov-display-radius', type=float, default=1.2,
                        help='MID-360 FOV wire radius [m], independent of sensing range; G toggles')
    args = parser.parse_args()
    if not 0. <= args.residual_scale <= 1.:
        parser.error('--residual-scale must be in [0,1]')
    if args.terrain is not None and args.terrain_level is not None:
        parser.error('choose --terrain or --terrain-level')
    if not math.isfinite(args.fov_display_radius) or args.fov_display_radius <= 0.:
        parser.error('--fov-display-radius must be finite and positive')
    import jax
    import jax.numpy as jp
    import mujoco
    import mujoco.viewer
    import numpy as np
    from ml_collections import config_dict
    from adaptive_gait_env import AdaptiveGaitEnv, default_config, FOOT_RADIUS
    from adaptive_gait_perception import initial_map, GRID_N, RESOLUTION, MAX_AGE
    from adaptive_gait_policy import contract, load_policy
    from foothold_preview_runtime import add_sphere, draw_mid360_fov
    from adaptive_foothold_estimator import CANDIDATE_COUNT, STATUS_NAMES
    from adaptive_gait_controller import LEG_ORDER, ACTION_SIZE
    from firmware_mjx_controller import _rotate_inverse
    from hybrid_gait_supervisor import MODE_NAMES

    policy, metadata = load_policy(args.checkpoint) if args.checkpoint else (None, None)
    if args.stage0 and policy is not None:
        parser.error('--stage0 is planner-only; omit --checkpoint')
    perception = args.perception or (metadata['actor_source'] if metadata else 'lidar')
    if metadata and perception != metadata['actor_source']:
        parser.error('checkpoint perception must match training; use --init-teacher to train a LiDAR policy')
    if perception == 'oracle' and policy is not None:
        parser.error('oracle is a zero-action planner diagnostic; do not use a deployment checkpoint')
    level = args.terrain_level
    if level is None:
        level = {'flat': 0, 'steps': 5, 'ramp': 3}.get(args.terrain, metadata['terrain_level'] if metadata else 0)
    cfg = config_dict.ConfigDict(metadata['config']) if metadata else default_config()
    sensor = metadata['lidar'] if metadata else dict(azimuths=90, elevations=8, dropout=.05, noise_m=.005)
    env = AdaptiveGaitEnv(terrain_level=level, perception=perception, config=cfg,
        azimuths=sensor['azimuths'], elevations=sensor['elevations'], dropout=sensor['dropout'], noise=sensor['noise_m'],
        gait_mode=args.gait_mode or (metadata['gait_mode'] if metadata else 'hybrid'), diagnostics=args.stage0)
    # Use the exact emitter transform held by the JAX raycaster, including the
    # measured 45-degree mount. No separate CAD site or display-only TF is used.
    lidar_tf = np.asarray(env.sensor.tf)
    lidar_root = env.sensor.root_id
    print('Compiling MJX reset/step; the first frame can take time.', flush=True)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    observe = jax.jit(env._get_obs)
    inference = jax.jit(policy) if policy else None
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = reset(reset_key)
    jax.block_until_ready(state.data.qpos)
    model, display = env.mj_model, mujoco.MjData(env.mj_model)
    keys = queue.SimpleQueue()
    vx, wz, paused, show_map = float(np.clip(args.speed, -.12, .12)), 0., False, True
    show_fov = True
    show_rejected = True
    history = deque(maxlen=500)
    output = Path(__file__).resolve().parent/'generated/adaptive_gait'
    print(f'24-D adaptive | {perception} | {env.gait_mode} | {env.terrain_description} | '+
          ('NEW CHECKPOINT' if policy else 'ZERO ACTION, NO TRAINED POLICY'), flush=True)
    print('Arrows: speed/yaw; Space: stop; Enter: pause; H: reset; C: clear map; M: map; G: LiDAR FOV; B: rejected candidates; P: save trace', flush=True)
    print('MID-360 FOV: H360 V[-7,+52]deg; orange=lower, blue=upper; wires show angular limits, not returns.', flush=True)
    print('Candidates: gray=unknown yellow=coverage orange=edge/rough blue=IK purple=path green=safe; white=wide reference cyan=request orange=projected red=latched; local XY grid 3/2 cm shares 5 cm map cells.', flush=True)
    if perception == 'oracle':
        print('DEBUG GT ORACLE: terrain and path queries are 100% known; no LiDAR scans, no RL policy.', flush=True)

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
                elif code == ord('G'):
                    show_fov = not show_fov
                elif code == ord('B'):
                    show_rejected = not show_rejected
                elif code == ord('P'):
                    save = True
            state.info['command'] = jp.array((vx, wz, 0., 0., 0.))
            state = state.replace(obs=observe(state.data, state.info))
            if not paused:
                key, action_key = jax.random.split(key)
                action = inference(state.obs, action_key)[0] if inference else jp.zeros(ACTION_SIZE)
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
                    reference_world=np.asarray(state.info['foothold_plan']['reference_world']),
                    selected_index=np.asarray(state.info['foothold_plan']['selected_index']),
                    active_index=np.asarray(cs.active_index),
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
            # Display the plan actually offered to the controller, not a separate
            # zero-action re-plan that could hide why this action was rejected.
            plan = jax.device_get(state.info['foothold_plan'])
            cs = jax.device_get(state.info['controller_state'])
            with viewer.lock():
                viewer.cam.lookat[:] = display.qpos[:3]
                scene = viewer.user_scn
                scene.ngeom = 0
                if show_fov:
                    root_rotation = display.xmat[lidar_root].reshape(3, 3)
                    lidar_origin = display.xpos[lidar_root] + root_rotation @ lidar_tf[:3, 3]
                    lidar_rotation = root_rotation @ lidar_tf[:3, :3]
                    add_sphere(scene, lidar_origin, .014, (0., .9, 1., 1.), 'MID-360')
                    draw_mid360_fov(scene, lidar_origin, lidar_rotation, args.fov_display_radius)
                if show_map:
                    grid = jax.device_get(state.info['lidar_map'])
                    valid = (display.time-grid.timestamp <= MAX_AGE) & (display.time >= grid.timestamp)
                    indices = np.argwhere(valid)
                    indices = indices[::max(1, int(np.ceil(len(indices)/700)))]
                    for i, j in indices:
                        xy = grid.center + (np.array((i, j))+.5-GRID_N/2)*RESOLUTION
                        sphere(scene, (*xy, grid.height[i, j]+.002), .009, (.1, .7, .85, .14))
                for leg in range(6):
                    for candidate in range(CANDIDATE_COUNT):
                        status = int(plan['status'][leg, candidate])
                        if show_rejected or plan['safe'][leg, candidate]:
                            colors = ((.5, .5, .5, .35), (1., .9, .05, .8),
                                      (1., .4, .05, .9), (.15, .35, 1., .95), (.7, .15, 1., .95), (.1, .9, .25, .9))
                            sphere(scene, (*plan['xy'][leg, candidate], plan['height'][leg, candidate]+.01),
                                   .009, colors[status])
                    add_sphere(scene, plan['wide_nominal'][leg]+np.array((0., 0., .01)),
                               .007, (.5, .5, 1., .8), f'{LEG_ORDER[leg]} nominal')
                    if plan['reference_index'][leg] >= 0:
                        add_sphere(scene, np.array((*plan['requested_xy'][leg], plan['reference_world'][leg, 2]+.065)),
                                   .009, (0., 1., 1., 1.), f'{LEG_ORDER[leg]} RL request')
                        add_sphere(scene, plan['reference_world'][leg]+np.array((0., 0., .025)),
                                   .011, (1., 1., 1., .9), f'{LEG_ORDER[leg]} ref')
                    if plan['selected_index'][leg] >= 0:
                        add_sphere(scene, plan['selected_world'][leg]+np.array((0., 0., .05)),
                                   .012, (1., .5, .05, 1.), f'{LEG_ORDER[leg]} selected')
                    if cs.active_known[leg]:
                        pre = jp.asarray(cs.swing_end[leg]).at[2].add(-cs.height_applied)
                        body = np.asarray(_rotate_inverse(pre, jp.asarray(cs.posture_command)))
                        execution_world = np.asarray(cs.root_position) + np.asarray(cs.root_rotation) @ np.array((body[1], -body[0], body[2]))
                        add_sphere(scene, execution_world,
                                   .019, (.8, .03, .02, 1.), f'{LEG_ORDER[leg]} latched')
            viewer.sync()
            if display.time-last_log >= 1.:
                last_log = display.time
                m = state.metrics
                compared = float(m['map_compared_count'])
                error = f'{float(m["map_mae_m"])*1000:.1f}mm' if compared else 'n/a'
                print(f't={display.time:.1f} plan_t={float(plan["time"]):.2f} source={perception} v={vx:+.2f} yaw={wz:+.2f} '+
                    f'known={float(m["map_known_fraction"]):.0%} map error={error} '+
                    f'IK={np.asarray(state.info["controller_output"].ik_valid).astype(int)} '+
                    f'reject={bool(cs.plan_rejected)} stride={float(cs.stride_scale):.2f} phase={float(cs.phase_duration):.2f}s '+
                    f'mode={MODE_NAMES[int(plan["decision"])]} active_gait={int(cs.scheduler.mode)} '+
                    f'Lmax_scale={float(plan["max_feasible_stride"]):.3f} support={float(plan["support_margin"]):.3f}m '+
                    f'fault={bool(cs.scheduler.fault)}', flush=True)
                print(f'  stride_bank=[1.3,1,.75,.5,.25,.125] feasible={plan["tripod_feasible"].astype(int)} '+
                      f'known_bad={plan["tripod_known_bad"].astype(int)} wave={bool(plan["wave_feasible"])} '+
                      f'two_phase={bool(plan["two_tripod_phases"])} return_confirm={float(plan["return_time"]):.2f}s', flush=True)
                if args.stage0:
                    print('  Stage0: '+ ' '.join(f'{k}={float(v):.4f}' for k, v in m.items()
                                                if k.startswith('oracle_')), flush=True)
                for leg, name in enumerate(LEG_ORDER):
                    terrain_count = int(np.sum(plan['terrain_ok'][leg]))
                    ik_count = int(np.sum(plan['terrain_ok'][leg] & plan['ik_ok'][leg]))
                    safe_count = int(np.sum(plan['safe'][leg]))
                    if safe_count:
                        reason = 'READY'
                    elif terrain_count:
                        path_valid = plan['terrain_ok'][leg] & plan['ik_ok'][leg] & plan['path_ok'][leg]
                        reason = 'IK' if not ik_count else ('LOW_PATH_COVERAGE' if np.any(path_valid) else 'PATH')
                    elif np.any(plan['unsafe'][leg]):
                        reason = 'EDGE_ROUGH'
                    elif np.any(plan['any_known'][leg]):
                        reason = 'LOW_COVERAGE'
                    else:
                        reason = 'UNKNOWN'
                    print(f'  {name}: candidate_total={CANDIDATE_COUNT} known={np.sum(plan["center_known"][leg]):2d} '+
                        f'complete={np.sum(plan["complete"][leg]):2d} coverage_pass={np.sum(plan["coverage_ok"][leg]):2d} '+
                        f'surface_safe={terrain_count:2d} ik_safe={ik_count:2d} path_safe={safe_count:2d} '+
                        f'ref={plan["reference_index"][leg]:2d} selected={plan["selected_index"][leg]:2d} '+
                        f'active={cs.active_index[leg]:2d} phase={int(state.info["controller_output"].gait_state[leg])} '+
                        f'residual_rejected={bool(plan["residual_rejected"][leg])} reason={reason}', flush=True)
            if save:
                output.mkdir(parents=True, exist_ok=True)
                (output/'.gitignore').write_text('*\n')
                if history:
                    np.savez_compressed(output/'trace.npz', **{name: np.stack([row[name] for row in history]) for name in history[0]})
                grid = jax.device_get(state.info['lidar_map'])
                np.savez_compressed(output/'map.npz', height=grid.height, timestamp=grid.timestamp,
                                    center=grid.center, spread=grid.spread, time=display.time)
                (output/'adaptive_contract.json').write_text(json.dumps(contract(env), indent=2)+'\n')
                np.savez_compressed(output/'foothold_plan.npz', **plan)
                diagnostic = dict(perception=perception, plan_time=float(plan['time']),
                    display_time=display.time, status_names=STATUS_NAMES,
                    plan_rejected=bool(cs.plan_rejected), active_index=cs.active_index.tolist(),
                    active_known=cs.active_known.tolist(), goal_world=cs.goal_world.tolist(),
                    controller_ik=np.asarray(state.info['controller_output'].ik_valid).tolist(),
                    candidates={name: value.tolist() for name, value in plan.items()})
                (output/'foothold_diagnostics.json').write_text(json.dumps(diagnostic, indent=2)+'\n')
                mujoco.mj_saveLastXML(str(output/'model.xml'), model)
                print(f'Saved trace/map/model/contract: {output}', flush=True)
            time.sleep(max(0., env.dt-(time.monotonic()-tick_start)))


if __name__ == '__main__':
    main()
