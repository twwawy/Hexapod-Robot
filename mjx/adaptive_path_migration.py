"""Explicit d0370b3 terrain v5 -> path-v6 warm start; never edits source checkpoint."""
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_REVISION = 'd0370b3f30b0d3c83541c61a1df44bc87ebcf7d7'


def validate_source(path):
    from adaptive_gait_policy import resolve_checkpoint
    path = resolve_checkpoint(path)
    metadata = json.loads((path/'adaptive_contract.json').read_text())
    already_v6 = metadata.get('observation_contract') == 'adaptive_hybrid_elevation_grid24x24x6_path_v6'
    # Explicitly reviewed source revisions only. Never accept an arbitrary v6
    # source just because tensor dimensions happen to match.
    recorded_revision = metadata.get('git_commit', '')
    revision = '585bee2' if already_v6 else SOURCE_REVISION
    if already_v6 and recorded_revision.startswith('53bab78'):
        revision = '53bab78ab54e2bbf0c97582319348d73f2ea5cec'
    if already_v6 and recorded_revision.startswith('bd979c9'):
        revision = 'bd979c9f2864bf88599bc9b1f642de1af727025a'
    expected = dict(command_mode='terrain', observation_contract='adaptive_hybrid_elevation_grid24x24x6_v5',
        action_contract='adaptive_hybrid_geometry_residual_24_v4', action_size=24,
        network_contract='elevation_cnn_16_32_dense64_v1', reward_contract='adaptive_completion_outcome_v5',
        observation_size={'state': 7890, 'privileged_state': 8205})
    if already_v6:
        expected['observation_contract'] = 'adaptive_hybrid_elevation_grid24x24x6_path_v6'
        expected['observation_size'] = {'state': 8790, 'privileged_state': 9105}
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f'Unreviewed path-v6 migration {field}: {metadata.get(field)}')
    repo = Path(__file__).resolve().parent.parent
    recorded = metadata['source_sha256']
    original_policy = subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:mjx/adaptive_gait_policy.py'], text=True)
    # Require the complete recorded v5 source set, not a partial hash bypass.
    import ast
    tree = ast.parse(original_policy)
    source_names = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sources' for t in n.targets))
    if set(recorded) != set(source_names):
        raise ValueError('Migration requires the full reviewed v5 source manifest')
    for name, digest in recorded.items():
        source = subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:mjx/{name}'])
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError(f'Unreviewed path-v6 migration source: {name}')
    config = json.loads((path/'ppo_network_config.json').read_text())
    if config['action_size'] != 24 or config['observation_size'] != expected['observation_size']:
        raise ValueError('Saved network dimensions do not match reviewed v5 metadata')
    metadata['explicit_migration'] = dict(kind='path_v6_contact_foot_retry_update' if already_v6 else 'terrain_v5_to_path_v6', source=str(path),
        source_revision=revision, new_input_weights='zero', optimizer='fresh',
        note='Warm start only; changed planner geometry can change behavior immediately')
    return path, metadata


def input_indices(tail):
    """Old -> new indices: 234 shared + 150*(28 -> 34) + unchanged tail."""
    import numpy as np
    return np.concatenate((np.arange(234),
        (234+np.arange(150)[:, None]*34+np.arange(28)).reshape(-1),
        5334+np.arange(tail)))


def migrate_params(params):
    import copy
    import jax.numpy as jp
    import numpy as np
    from flax.core import FrozenDict, freeze, unfreeze
    from adaptive_gait_env import VECTOR_SIZE, ACTOR_SIZE, CRITIC_SIZE
    if (VECTOR_SIZE, ACTOR_SIZE, CRITIC_SIZE) != (5334, 8790, 9105):
        raise ValueError('Unreviewed target observation layout')
    normalizer, actor, critic = params
    if (normalizer.mean['state'].shape == (8790,) and
        normalizer.mean['privileged_state'].shape == (9105,)):
        if actor['params']['Dense_1']['kernel'].shape != (5398, 256) or critic['params']['Dense_1']['kernel'].shape != (5713, 256):
            raise ValueError('Unexpected v6 kernel shape')
        return params  # Explicit source-only migration; no tensor expansion.
    def expand(values, tail, fill):
        old_size = 4434+tail
        if values.shape[0] != old_size:
            raise ValueError(f'Unexpected migration input shape {values.shape}, expected {old_size}')
        out = np.full((5334+tail,)+values.shape[1:], fill, dtype=values.dtype)
        out[input_indices(tail)] = np.asarray(values)
        return jp.asarray(out)
    stats = {}
    for field, fill in (('mean', 0.), ('std', 1.), ('summed_variance', 0.)):
        old = getattr(normalizer, field)
        if set(old) != {'state', 'privileged_state'}:
            raise ValueError('Unexpected normalization observation keys')
        stats[field] = {k: expand(v, 3456+(315 if k == 'privileged_state' else 0), fill) for k, v in old.items()}
    # Retain all old normalization values; new features start at mean 0/std 1.
    normalizer = normalizer.replace(**stats)
    def network(old, tail):
        frozen = isinstance(old, FrozenDict)
        new = unfreeze(old) if frozen else copy.deepcopy(old)
        kernel = new['params']['Dense_1']['kernel']
        if kernel.shape != (4434+tail, 256):
            raise ValueError(f'Unexpected first vector Dense kernel: {kernel.shape}')
        new['params']['Dense_1']['kernel'] = expand(kernel, tail, 0.)
        return freeze(new) if frozen else new
    return (normalizer, network(actor, 64), network(critic, 315+64))
