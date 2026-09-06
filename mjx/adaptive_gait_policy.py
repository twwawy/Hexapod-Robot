"""Explicit 24-D checkpoint contract shared by training and replay."""
import functools
import hashlib
import json
from pathlib import Path
import subprocess

from adaptive_gait_controller import ACTION_CONTRACT, ACTION_SIZE, LEG_ORDER
from adaptive_gait_env import ACTOR_SIZE, CRITIC_SIZE, OBSERVATION_CONTRACT, REWARD_CONTRACT, CANDIDATE_FEATURES
from adaptive_gait_perception import GRID_N, RESOLUTION, MAX_AGE, SENSOR_PERIOD
from lidar_extrinsics import measurement_metadata
from adaptive_foothold_estimator import (
    CANDIDATE_COUNT, SEARCH_RADIUS, RESIDUAL_X, RESIDUAL_Y, RESIDUAL_OFFSETS, MIN_COVERAGE,
    MAX_PLANE_RESIDUAL, MAX_SLOPE_RAD, EDGE_JUMP, EDGE_CLEARANCE, PATH_SAMPLES,
)


def network_factory():
    from brax.training.agents.ppo import networks
    return functools.partial(networks.make_ppo_networks, policy_hidden_layer_sizes=(256, 256, 128),
                             value_hidden_layer_sizes=(256, 256, 128),
                             policy_obs_key='state', value_obs_key='privileged_state')


def contract(env):
    root = Path(__file__).resolve().parent
    sources = ('adaptive_contract.py', 'adaptive_gait_controller.py', 'adaptive_gait_env.py', 'adaptive_gait_perception.py',
               'adaptive_foothold_estimator.py', 'foothold_feasibility.py',
               'hybrid_gait_supervisor.py', 'wave_gait_scheduler.py',
               'adaptive_gait_policy.py', 'firmware_mjx_controller.py', 'rough_terrain_env.py',
               'prepare_rl_scene.py', 'servo_model.py', 'terrain_curriculum.py', 'lidar_extrinsics.py')
    revision = subprocess.check_output(['git', '-C', str(root.parent), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(git_commit=revision, action_contract=ACTION_CONTRACT, observation_contract=OBSERVATION_CONTRACT,
        reward_contract=REWARD_CONTRACT, action_size=ACTION_SIZE,
        observation_size={'state': ACTOR_SIZE, 'privileged_state': CRITIC_SIZE},
        actor_source=env.perception, leg_order=LEG_ORDER,
        gait_mode=env.gait_mode, action_profile=env.action_profile,
        action_slices={'xy': [0, 12], 'clearance': [12, 18], 'roll_pitch_height': [18, 21],
                       'stride': 21, 'apex_phase': 22, 'transfer_timing': 23},
        limits=dict(xy_x_m=.06, xy_y_m=.04, clearance_residual_m=.04, clearance_m=[.04, .18],
                    pitch_deg=10., roll_deg=5., height_m=.03, stride=[.5, 1.3],
                    tripod_phase_s=[.5, 1.4], wave_phase_s=[.6, 1.4], timing_residual=.15),
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
        candidates=dict(per_leg=CANDIDATE_COUNT, features=CANDIDATE_FEATURES,
                        search_radius_m=SEARCH_RADIUS, residual_about_reference_m=[RESIDUAL_X, RESIDUAL_Y],
                        residual_offsets_m=RESIDUAL_OFFSETS.tolist(),
                        shared_map_cells=True,
                        patch_radius_m=RESOLUTION, min_coverage=MIN_COVERAGE,
                        plane_residual_m=MAX_PLANE_RESIDUAL, max_slope_rad=MAX_SLOPE_RAD,
                        edge_jump_m=EDGE_JUMP, edge_clearance_m=EDGE_CLEARANCE,
                        path_samples=PATH_SAMPLES, path_check='observed swept foot samples; not whole-body collision'),
        oracle_debug_only=env.perception == 'oracle',
        terrain_level=env.curriculum_level, terrain_description=env.terrain_description,
        rough_representation='adaptive_boxes_8x4_v1',
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


def read_contract(path, *, migrate_flat_boxes=False):
    path = resolve_checkpoint(path)
    manifest = path/'adaptive_contract.json'
    if not manifest.is_file():
        raise ValueError(f'Missing 24-D adaptive contract: {manifest}; stage31 18-D cannot be loaded')
    metadata = json.loads(manifest.read_text())
    for field, expected in (('action_contract', ACTION_CONTRACT), ('observation_contract', OBSERVATION_CONTRACT),
                            ('action_size', ACTION_SIZE),
                            ('observation_size', {'state': ACTOR_SIZE, 'privileged_state': CRITIC_SIZE})):
        if metadata.get(field) != expected:
            raise ValueError(f'Incompatible checkpoint {field}: {metadata.get(field)} != {expected}')
    root = Path(__file__).resolve().parent
    if migrate_flat_boxes and metadata.get('terrain_level') != 0:
        raise ValueError('Flat-to-box migration accepts only a flat terrain checkpoint')
    migrated_sources = []
    migration_files = {'adaptive_gait_env.py', 'rough_terrain_env.py', 'prepare_rl_scene.py', 'adaptive_gait_policy.py'}
    for name, expected in metadata['source_sha256'].items():
        source = (root/name).resolve()
        if source.parent != root:
            raise ValueError(f'Invalid contract source: {name}')
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            if not migrate_flat_boxes or name not in migration_files:
                raise ValueError(f'Controller/source contract changed: {name}; replay with the recorded source revision')
            # Only permit the reviewed flat checkpoint source, not arbitrary
            # source-hash bypasses. Action/obs/network checks still apply.
            previous = subprocess.check_output(['git', '-C', str(root.parent), 'show',
                f'786ff09f284b290b4c499abd79159754642375bf:mjx/{name}'])
            if hashlib.sha256(previous).hexdigest() != expected:
                raise ValueError(f'Unreviewed migration source: {name}')
            migrated_sources.append(name)
    network = json.loads((path/'ppo_network_config.json').read_text())
    if network['action_size'] != ACTION_SIZE or network['observation_size'] != metadata['observation_size']:
        raise ValueError('Saved PPO network dimensions differ from adaptive contract')
    if migrate_flat_boxes:
        metadata['explicit_migration'] = dict(kind='flat_786ff09_to_boxes',
                                             changed_sources=migrated_sources)
    return path, metadata


def load_policy(path):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint, networks
    path, metadata = read_contract(path)
    params = checkpoint.load(str(path))
    net = network_factory()(observation_size=metadata['observation_size'], action_size=ACTION_SIZE,
                            preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(net)(params, deterministic=True), metadata
