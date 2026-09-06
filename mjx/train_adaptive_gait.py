#!/usr/bin/env python3
"""Train a 24-D residual policy after user validation of Stage 0 geometry."""
import argparse
from datetime import datetime
import functools
import json
import os
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
if not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perception', choices=('lidar', 'teacher', 'blind'), default='lidar')
    parser.add_argument('--stage', type=int, choices=(1, 2, 3), default=1,
                        help='1: Tripod, 2: Wave, 3: deterministic hybrid; validate Stage 0 first')
    parser.add_argument('--terrain-level', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=10_000_000)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-minibatches', type=int, default=4)
    parser.add_argument('--episode-length', type=int, default=2000)
    parser.add_argument('--num-evals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--azimuths', type=int, default=90)
    parser.add_argument('--elevations', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=.05)
    parser.add_argument('--range-noise', type=float, default=.005)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--restore', type=Path, help='24-D weights and normalizer; PPO optimizer restarts; stage can change')
    parser.add_argument('--init-teacher', type=Path, help='initialize lidar PPO from a new 24-D GT teacher; no legacy conversion')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', default='hexapod-adaptive-gait')
    args = parser.parse_args()
    if min(args.timesteps, args.num_envs, args.batch_size, args.num_minibatches, args.episode_length, args.num_evals) < 1:
        parser.error('training counts must be positive')
    if (args.batch_size*args.num_minibatches) % args.num_envs:
        parser.error('batch-size * num-minibatches must be divisible by num-envs')
    if args.restore and args.init_teacher:
        parser.error('choose --restore or --init-teacher')
    if args.init_teacher and args.perception != 'lidar':
        parser.error('--init-teacher is for --perception lidar')

    from brax.training.agents.ppo import train as ppo, checkpoint
    from mujoco_playground._src import wrapper
    from adaptive_gait_env import AdaptiveGaitEnv, default_config
    from adaptive_gait_policy import contract, network_factory, read_contract

    cfg = default_config()
    cfg.episode_length = args.episode_length
    env = AdaptiveGaitEnv(terrain_level=args.terrain_level, perception=args.perception, config=cfg,
                          azimuths=args.azimuths, elevations=args.elevations,
                          dropout=args.dropout, noise=args.range_noise,
                          gait_mode={1: 'tripod', 2: 'wave', 3: 'hybrid'}[args.stage])
    restore = None
    if args.restore or args.init_teacher:
        restore, old = read_contract(args.restore or args.init_teacher)
        expected_source = 'teacher' if args.init_teacher else args.perception
        if old['actor_source'] != expected_source:
            parser.error(f'checkpoint actor source must be {expected_source}, got {old["actor_source"]}')
    output = (args.output or Path(__file__).resolve().parent/'runs'/
              f'adaptive-{args.perception}-level{args.terrain_level}-{datetime.now():%Y%m%d-%H%M%S}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'.gitignore').write_text('*\n')
    metadata = contract(env)
    metadata['initial_checkpoint'] = str(restore) if restore else None
    metadata['initialization'] = 'teacher_weight_transfer_then_asymmetric_ppo' if args.init_teacher else 'ppo'
    (output/'adaptive_contract.json').write_text(json.dumps(metadata, indent=2)+'\n')
    factory = network_factory()
    network_config = checkpoint.network_config(env.observation_size, env.action_size, True, factory)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=output.name, dir=str(output), config=metadata)

    def progress(step, metrics):
        numeric = {name: float(value) for name, value in metrics.items()}
        payload = dict(step=int(step), **numeric)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(payload)+'\n')
        print(json.dumps(payload), flush=True)
        if wandb_run:
            wandb_run.log(numeric, step=int(step))

    def save_policy(step, make_policy, params):
        # Persist the contract with each exported checkpoint, not only the run.
        if int(step) == 0:
            return
        checkpoint.save(str(output/'checkpoints'), int(step), params, network_config)
        completed = [p for p in (output/'checkpoints').iterdir() if p.name.isdigit() and int(p.name) == int(step)]
        for path in completed:
            (path/'adaptive_contract.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (output/'latest_checkpoint.json').write_text(json.dumps({'step': int(step), 'path': str(completed[0])})+'\n')

    print(f'Output: {output}\nActor: {args.perception}, gait={env.gait_mode}, action={env.action_size}, obs={env.observation_size}', flush=True)
    try:
        ppo.train(environment=env, num_timesteps=args.timesteps, num_envs=args.num_envs,
            episode_length=args.episode_length, action_repeat=1, learning_rate=3e-4,
            entropy_cost=.005, discounting=.99, unroll_length=20, batch_size=args.batch_size,
            num_minibatches=args.num_minibatches, num_updates_per_batch=4,
            normalize_observations=True, clipping_epsilon=.2, gae_lambda=.95, max_grad_norm=1.,
            num_evals=args.num_evals, num_eval_envs=min(args.num_envs, 16), deterministic_eval=True,
            network_factory=factory, wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
            restore_checkpoint_path=str(restore) if restore else None,
            seed=args.seed, progress_fn=progress, policy_params_fn=save_policy, use_pmap_on_reset=False)
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    main()
