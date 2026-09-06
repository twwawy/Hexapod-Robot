#!/usr/bin/env python3
"""Replay the packaged stage31 PPO policy using its recorded v3 source revision."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import tarfile
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = REPO / 'mjx/policies/progress-v2-stage31-level6'


def read_package(package):
    manifest = json.loads((package/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = (package/name).resolve()
        if not path.is_relative_to(package) or not path.is_file():
            raise ValueError(f'Missing package file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Package file changed: {name}')
    return manifest


def prepare_source(manifest, output):
    """Extract committed source/assets into an isolated, ignored directory."""
    revision = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', '--verify', f'{manifest["git_commit"]}^{{commit}}'],
        text=True).strip()
    destination = output/f'source-{revision[:12]}'
    marker = destination/'replay_source.json'
    if marker.is_file() and json.loads(marker.read_text()).get('commit') == revision:
        return destination
    output.mkdir(parents=True, exist_ok=True)
    (output/'.gitignore').write_text('*\n')
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='policy-source-', dir=output) as temporary:
        archive = Path(temporary)/'source.tar'
        subprocess.run(['git', '-C', str(REPO), 'archive', f'--output={archive}',
                        revision, 'mjx', 'HW/urdf'], check=True)
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                path = (destination/member.name).resolve()
                if (not path.is_relative_to(destination.resolve())
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unexpected source archive member: {member.name}')
            bundle.extractall(destination)
    marker.write_text(json.dumps({'commit': revision}, indent=2)+'\n')
    return destination


def load_environment(source, package, manifest, terrain, factory=None):
    # Insert before importing ANY robot modules so local v4/controller edits do
    # not leak into this v3 policy, and generated models stay in the snapshot.
    sys.path.insert(0, str(source/'mjx'))
    curriculum = importlib.import_module('terrain_curriculum')
    level = manifest['terrain_level']
    recorded = replace(curriculum.TERRAIN_LEVELS[level],
                       name=manifest['terrain_name'], kind=manifest['terrain_kind'],
                       stair_count=manifest['terrain_stair_count'],
                       stair_riser=manifest['terrain_step_height_m'])
    levels = list(curriculum.TERRAIN_LEVELS)
    levels[level] = recorded
    curriculum.TERRAIN_LEVELS = tuple(levels)
    module = importlib.import_module('rough_terrain_env')
    for key, name in (('action_contract_version', 'ACTION_CONTRACT_VERSION'),
                      ('observation_contract_version', 'OBSERVATION_CONTRACT_VERSION'),
                      ('action_size', 'ACTION_SIZE'), ('observation_size', 'OBSERVATION_SIZE')):
        if getattr(module, name) != manifest[key]:
            raise ValueError(f'Policy/environment contract mismatch: {key}')
    from ml_collections import config_dict
    config = module.default_config()
    config.update(config_dict.ConfigDict(json.loads((package/manifest['environment']).read_text())))
    selected_level = 0 if terrain == 'flat' else level
    if factory is not None:
        return factory(module, config, selected_level)
    return module.HexapodRoughTerrainEnv(config=config, terrain_level=selected_level)


def load_policy(package, manifest):
    """Restore actor AND observation normalization without legacy JSON shape bugs."""
    from brax.training import networks as common_networks
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks
    path = package/manifest['weights']
    config = json.loads((path/'ppo_network_config.json').read_text())
    if (config['observation_size']['shape'] != [manifest['observation_size']]
            or config['action_size'] != manifest['action_size']):
        raise ValueError('Checkpoint tensor sizes do not match the package manifest')
    kwargs = dict(config['network_factory_kwargs'])
    kwargs['activation'] = common_networks.ACTIVATION[kwargs['activation']]
    for key in ('policy_network_kernel_init_fn', 'value_network_kernel_init_fn',
                'mean_kernel_init_fn'):
        if kwargs.get(key) is not None:
            kwargs[key] = common_networks.KERNEL_INITIALIZER[kwargs[key]]
    normalize = (running_statistics.normalize if config['normalize_observations']
                 else types.identity_observation_preprocessor)
    network = networks.make_ppo_networks(manifest['observation_size'], manifest['action_size'],
                                         preprocess_observations_fn=normalize, **kwargs)
    params = checkpoint.load(path)
    return networks.make_inference_fn(network)(params, deterministic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument('--output', type=Path, default=REPO/'mjx/generated/trained_policy')
    parser.add_argument('--terrain', choices=('recorded', 'flat'), default='recorded')
    parser.add_argument('--seed', type=int, help='defaults to saved best-video seed (20040)')
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0, help='simulation playback speed')
    parser.add_argument('--prepare-only', action='store_true', help='unpack source without loading or running the policy')
    args = parser.parse_args()
    if not 0 < args.speed <= 10:
        parser.error('speed must be between 0 and 10')
    package, output = args.package.expanduser().resolve(), args.output.expanduser().resolve()
    manifest = read_package(package)
    source = prepare_source(manifest, output)
    print(f'Policy: {manifest["run_id"]} / step {manifest["checkpoint_step"]}', flush=True)
    print(f'Isolated source: {source}', flush=True)
    if args.prepare_only:
        return

    print('Loading v3 environment and PPO weights; first JAX compilation may take time.', flush=True)
    env = load_environment(source, package, manifest, args.terrain)
    import glfw
    import jax
    import jax.numpy as jp
    import mujoco
    import mujoco.viewer
    from mujoco import mjx
    import numpy as np
    jax.config.update('jax_compilation_cache_dir', str(output/'jax_cache'))
    policy = load_policy(package, manifest)

    @jax.jit
    def advance(state, key):
        key, action_key = jax.random.split(key)
        action, _ = policy(state.obs, action_key)
        return env.step(state, action), key, action

    @jax.jit
    def change_command(state, command):
        info = dict(state.info)
        info['command'] = command
        return state.replace(info=info, obs=env._get_obs(state.data, info))

    reset = jax.jit(env.reset)
    seed = manifest['replay_seed'] if args.seed is None else args.seed
    state = reset(jax.random.PRNGKey(seed))
    state.obs.block_until_ready()
    key = jax.random.PRNGKey(seed+1)
    host = mjx.get_data(env.mj_model, state.data)
    keys = queue.SimpleQueue()
    paused, terminal, follow = args.paused, False, True
    steps, action_norm = 0, 0.0
    status = 'Paused' if paused else 'Running learned PPO + firmware controller'
    output.mkdir(parents=True, exist_ok=True)
    (output/'replay_manifest.json').write_text(json.dumps(dict(
        policy=manifest['run_id'], checkpoint_step=manifest['checkpoint_step'],
        source=str(source), source_fidelity=manifest['source_fidelity'], seed=seed,
        terrain=env.terrain_description, backend=jax.default_backend(),
        action_contract=manifest['action_contract_version'],
        lidar_foothold_correction=False), indent=2)+'\n')
    with mujoco.viewer.launch_passive(env.mj_model, host, key_callback=keys.put) as viewer:
        with viewer.lock():
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 135, -28
        while viewer.is_running():
            started = time.monotonic()
            command = np.asarray(state.info['command']).copy()
            command_changed = False
            while not keys.empty():
                pressed = keys.get()
                if pressed == glfw.KEY_SPACE and not terminal:
                    paused = not paused
                    status = 'Paused' if paused else 'Running learned PPO + firmware controller'
                elif pressed == ord('R'):
                    state = reset(jax.random.PRNGKey(seed))
                    key = jax.random.PRNGKey(seed+1)
                    steps, terminal, paused, action_norm = 0, False, False, 0.0
                    command = np.asarray(state.info['command']).copy()
                    command_changed = False
                    status = 'Reset: saved seed and sampled command'
                elif pressed in (glfw.KEY_UP, glfw.KEY_DOWN):
                    command[0] = np.clip(command[0]+(0.02 if pressed == glfw.KEY_UP else -0.02), 0, 0.12)
                    command_changed = True
                elif pressed in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                    command[1] = np.clip(command[1]+(0.1 if pressed == glfw.KEY_LEFT else -0.1), -0.3, 0.3)
                    command_changed = True
                elif pressed == ord('F'):
                    follow = not follow
            if command_changed:
                state = change_command(state, jp.asarray(command))
            if not paused and not terminal:
                state, key, action = advance(state, key)
                state.obs.block_until_ready()
                action_norm = float(np.linalg.norm(np.asarray(action)))
                steps += 1
                terminal = bool(np.asarray(state.done)) or steps >= env.episode_length
                if terminal:
                    reasons = [name.removeprefix('termination/') for name, value in state.metrics.items()
                               if name.startswith('termination/') and float(np.asarray(value)) > 0]
                    if float(np.asarray(state.metrics.get('terrain_success', 0))) > 0:
                        reasons.append('terrain_success')
                    status = 'Episode ended: '+(', '.join(reasons) or 'time limit')+' | R resets'
            command = np.asarray(state.info['command'])
            with viewer.lock():
                mjx.get_data_into(host, env.mj_model, state.data)
                if follow:
                    viewer.cam.lookat[:] = host.qpos[:3] + (0.4, 0, -0.1)
            text = (f'TRAINED PPO | stage31 level6 | step {manifest["checkpoint_step"]} | seed {seed}\n'
                    f'{status}\n{env.terrain_description} | simulation {steps*env.dt:.2f}s\n'
                    f'Policy: 146 observations -> 18 actions | action norm {action_norm:.3f}\n'
                    f'Command vx={command[0]:.2f} yaw={command[1]:.2f} height={command[2]:.3f}\n'
                    'SPACE pause/resume | R reset | Up/Down speed | Left/Right yaw | F follow\n'
                    'Recorded v3 source + saved settings | simulator terrain observations | no LiDAR correction')
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, text, '')])
            viewer.sync()
            time.sleep(max(0.001, env.dt/args.speed-(time.monotonic()-started)))


if __name__ == '__main__':
    main()
