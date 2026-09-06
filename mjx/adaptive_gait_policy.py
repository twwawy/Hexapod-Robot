"""Explicit 23-D checkpoint contract shared by training and replay."""
import functools
import hashlib
import json
from pathlib import Path
import subprocess

from adaptive_gait_controller import ACTION_CONTRACT, ACTION_SIZE, LEG_ORDER
from adaptive_gait_env import ACTOR_SIZE, CRITIC_SIZE, OBSERVATION_CONTRACT, REWARD_CONTRACT, CANDIDATE_FEATURES
from adaptive_gait_perception import GRID_N, RESOLUTION, MAX_AGE, SENSOR_PERIOD
from lidar_extrinsics import measurement_metadata


def network_factory():
    from brax.training.agents.ppo import networks
    return functools.partial(networks.make_ppo_networks, policy_hidden_layer_sizes=(256, 256, 128),
                             value_hidden_layer_sizes=(256, 256, 128),
                             policy_obs_key='state', value_obs_key='privileged_state')


def contract(env):
    root = Path(__file__).resolve().parent
    sources = ('adaptive_gait_controller.py', 'adaptive_gait_env.py', 'adaptive_gait_perception.py',
               'adaptive_gait_policy.py', 'firmware_mjx_controller.py', 'rough_terrain_env.py',
               'prepare_rl_scene.py', 'servo_model.py', 'terrain_curriculum.py', 'lidar_extrinsics.py')
    revision = subprocess.check_output(['git', '-C', str(root.parent), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(git_commit=revision, action_contract=ACTION_CONTRACT, observation_contract=OBSERVATION_CONTRACT,
        reward_contract=REWARD_CONTRACT, action_size=ACTION_SIZE,
        observation_size={'state': ACTOR_SIZE, 'privileged_state': CRITIC_SIZE},
        actor_source=env.perception, leg_order=LEG_ORDER,
        action_slices={'xy': [0, 12], 'clearance': [12, 18], 'pitch_roll_height': [18, 21], 'stride': 21, 'phase': 22},
        limits=dict(xy_m=.04, clearance_residual_m=.04, clearance_m=[.04, .18],
                    pitch_deg=10., roll_deg=5., height_m=.03, stride=[.5, 1.3], phase_s=[.3, .7]),
        frames='controller forward/left/up; model forward -Y, left +X; map world XY; metres/radians/seconds',
        geometry='230 mm distal training link; 32 mm sphere centre is IK endpoint',
        joint_order=[f'{leg}_{joint}' for leg in LEG_ORDER for joint in (1, 2, 3)],
        joint_signs=[[1, -1, 1]]*3 + [[1, 1, -1]]*3,
        model_body_mass=env.mj_model.body_mass.tolist(),
        model_actuator_gain=env.mj_model.actuator_gainprm.tolist(),
        model_actuator_bias=env.mj_model.actuator_biasprm.tolist(),
        controller_dt=.005, policy_dt=env.dt, sensor_dt=SENSOR_PERIOD*env.dt,
        map=dict(cells=GRID_N, resolution_m=RESOLUTION, max_age_s=MAX_AGE),
        lidar=dict(tf=measurement_metadata(), azimuths=env.sensor.azimuths, elevations=env.sensor.elevations,
                   horizontal_fov_deg=360, vertical_fov_deg=[-7, 52], range_m=[.1, 8.],
                   dropout=env.sensor.dropout, noise_m=env.sensor.noise),
        candidates=dict(per_leg=9, features=CANDIDATE_FEATURES, patch_radius_m=.035, max_spread_m=.025),
        terrain_level=env.curriculum_level, terrain_description=env.terrain_description,
        episode_length=env.episode_length, config=env._config.to_dict(),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        runtime_validation='left to user; no training or locomotion validation performed by agent')


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if (path/'ppo_network_config.json').is_file():
        return path
    candidates = [p for directory in (path, path/'checkpoints') if directory.is_dir()
                  for p in directory.iterdir() if p.is_dir() and p.name.isdigit()
                  and (p/'ppo_network_config.json').is_file()]
    if not candidates:
        raise ValueError(f'No completed PPO checkpoint in {path}')
    return max(candidates, key=lambda p: int(p.name))


def read_contract(path):
    path = resolve_checkpoint(path)
    manifest = path/'adaptive_contract.json'
    if not manifest.is_file():
        raise ValueError(f'Missing 23-D adaptive contract: {manifest}; stage31 18-D cannot be loaded')
    metadata = json.loads(manifest.read_text())
    for field, expected in (('action_contract', ACTION_CONTRACT), ('observation_contract', OBSERVATION_CONTRACT),
                            ('action_size', ACTION_SIZE),
                            ('observation_size', {'state': ACTOR_SIZE, 'privileged_state': CRITIC_SIZE})):
        if metadata.get(field) != expected:
            raise ValueError(f'Incompatible checkpoint {field}: {metadata.get(field)} != {expected}')
    root = Path(__file__).resolve().parent
    for name, expected in metadata['source_sha256'].items():
        source = (root/name).resolve()
        if source.parent != root or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Controller/source contract changed: {name}; replay with the recorded source revision')
    network = json.loads((path/'ppo_network_config.json').read_text())
    if network['action_size'] != ACTION_SIZE or network['observation_size'] != metadata['observation_size']:
        raise ValueError('Saved PPO network dimensions differ from adaptive contract')
    return path, metadata


def load_policy(path):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks
    path, metadata = read_contract(path)
    params = checkpoint.load(str(path))
    net = network_factory()(observation_size=metadata['observation_size'], action_size=ACTION_SIZE,
                            preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(net)(params, deterministic=True), metadata
