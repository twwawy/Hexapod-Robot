#!/usr/bin/env python3
"""Bounded zero-action baseline check, without a viewer or PPO training."""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--speed', type=float, default=.08)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--terrain-level', type=int, default=0)
    parser.add_argument('--perception', choices=('oracle', 'teacher', 'lidar'), default='oracle')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('mjx/generated/adaptive_baseline.json'))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 0 < args.speed <= .10:
        parser.error('seconds must be positive and finite; speed must be in (0, .10]')
    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    import jax.numpy as jp
    from adaptive_gait_env import AdaptiveGaitEnv

    env = AdaptiveGaitEnv(terrain_level=args.terrain_level, perception=args.perception,
                          gait_mode='tripod')
    state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    step = jax.jit(env.step)
    initial_x = float(state.data.qpos[0])
    for tick in range(math.ceil(args.seconds / env.dt)):
        state.info['command'] = jp.array((args.speed, 0., 0., 0., 0.))
        state = step(state, jp.zeros(24))
        cs = state.info['controller_state']
        if tick % 100 == 0 or bool(state.done):
            print(f't={(tick+1)*env.dt:.2f}s x={float(state.data.qpos[0]):.3f}m '
                  f'completed_phases={int(cs.scheduler.epoch)} '
                  f'supervisor={int(state.metrics["supervisor_mode"])}', flush=True)
        if bool(state.done):
            break
    reasons = {k: float(v) for k, v in state.metrics.items()
               if k.startswith('termination/') and float(v) != 0.}
    report = dict(elapsed_s=(tick+1)*env.dt, displacement_m=float(state.data.qpos[0])-initial_x,
                  completed_phases=int(cs.scheduler.epoch), done=bool(state.done),
                  success=bool(state.metrics['terrain_success']), termination=reasons,
                  seed=args.seed, perception=args.perception, speed=args.speed,
                  terrain_level=args.terrain_level, policy='zero action')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return int(bool(reasons) or int(cs.scheduler.epoch) == 0)


if __name__ == '__main__':
    raise SystemExit(main())
