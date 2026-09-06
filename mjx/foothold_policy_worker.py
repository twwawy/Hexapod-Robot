#!/usr/bin/env python3
"""Isolated MJX dynamics/PPO process for the interactive foothold explorer."""
import argparse
from multiprocessing.connection import Connection
from pathlib import Path
import traceback

from view_trained_policy import read_package, prepare_source, load_environment, load_policy


def run(connection):
    request = connection.recv()
    output = Path(request['output'])/'policy'
    package = Path(request['policy_package']).resolve()
    manifest = read_package(package)
    source = prepare_source(manifest, output)

    def factory(module, config, level):
        # These imports happen only AFTER the archived controller/scene imports.
        from foothold_policy_scene import make_explorer_environment
        return make_explorer_environment(module, config, level, request, source)

    print('Foothold explorer: loading stage31 policy and course dynamics...', flush=True)
    env, sensor = load_environment(source, package, manifest,
                                  'flat' if request['terrain'] == 'flat' else 'recorded', factory)
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx
    import numpy as np
    import firmware_mjx_controller as firmware
    jax.config.update('jax_compilation_cache_dir', str(output/'jax_cache'))
    model_path = output/'explorer_policy.mjb'
    mujoco.mj_saveModel(env.mj_model, str(model_path), None)
    connection.send(dict(kind='ready', model=str(model_path), home=np.asarray(env._home_qpos),
                         manifest=dict(commit=manifest['git_commit'], policy=manifest['run_id'],
                                       checkpoint_step=manifest['checkpoint_step'],
                                       robot_model='skeleton', mode='trained PPO dynamics + LiDAR preview',
                                       T_base_lidar=sensor.tolist(), terrain=request['terrain'],
                                       terrain_raster_resolution_m=0.02,
                                       source_fidelity=manifest['source_fidelity'],
                                       foothold_feedback=False)))
    print('Foothold explorer: compiling JAX; the viewer remains responsive.', flush=True)
    policy = load_policy(package, manifest)

    @jax.jit
    def advance(state, key, command):
        info = dict(state.info)
        info['command'] = command
        state = state.replace(info=info, obs=env._get_obs(state.data, info))
        key, sample_key = jax.random.split(key)
        action, _ = policy(state.obs, sample_key)
        return env.step(state, action), key, action

    reset = jax.jit(env.reset)
    seed = int(request['policy_seed'])
    state = reset(jax.random.PRNGKey(seed))
    state.obs.block_until_ready()
    key = jax.random.PRNGKey(seed+1)
    terminal, steps = False, 0
    last_swing = np.zeros(6, dtype=bool)
    touchdowns = 0
    action = np.zeros(18)
    host = mujoco.MjData(env.mj_model)

    def snapshot():
        nonlocal last_swing, touchdowns
        mjx.get_data_into(host, env.mj_model, state.data)
        control = jax.device_get(state.info['controller_output'])
        swing = np.asarray(control.gait_state) == firmware.LEG_SWING
        touchdowns += int(np.count_nonzero(last_swing & ~swing))
        last_swing = swing
        model_feet = np.asarray(control.foot_targets_body)[:, (1, 0, 2)].copy()
        model_feet[:, 1] *= -1
        rotation = host.xmat[env.mj_model.body('hexapod').id].reshape(3, 3)
        targets = model_feet @ rotation.T + host.qpos[:3]
        targets[:, 2] -= 0.032
        reasons = [name.removeprefix('termination/') for name, value in state.metrics.items()
                   if name.startswith('termination/') and float(np.asarray(value)) > 0]
        twist = np.asarray(control.applied_twist)
        return dict(kind='state', qpos=host.qpos.copy(), qvel=host.qvel.copy(), ctrl=host.ctrl.copy(),
                    time=float(host.time), action=np.asarray(action), targets=targets,
                    swing=swing, progress=np.asarray(control.gait_progress),
                    command=np.asarray(state.info['command']), applied=np.array((twist[0], twist[1], twist[3])),
                    terminal=terminal, status=('Stopped: '+(', '.join(reasons) or 'episode ended')+'; H resets'
                                               if terminal else 'Stage31 PPO + firmware dynamics'),
                    completed_swings=touchdowns//3, dt=float(env.dt))

    connection.send(snapshot())
    while True:
        message = connection.recv()
        if message['kind'] == 'close':
            return
        if message['kind'] == 'reset':
            state = reset(jax.random.PRNGKey(seed))
            key = jax.random.PRNGKey(seed+1)
            terminal, steps, touchdowns = False, 0, 0
            last_swing[:] = False
            action = np.zeros(18)
        elif message['kind'] == 'step' and not terminal:
            # One 20 ms policy tick per request; no blocking GPU calls in the UI.
            command = jp.asarray(message['command'], dtype=jp.float32)
            state, key, action = advance(state, key, command)
            state.obs.block_until_ready()
            steps += 1
            terminal = bool(np.asarray(state.done))
            if np.max(np.abs(np.asarray(state.data.qpos[:2]))) > 5.7:
                terminal = True
        connection.send(snapshot())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    args = parser.parse_args()
    connection = Connection(args.fd)
    try:
        run(connection)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            connection.send(dict(kind='error', error=error))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
