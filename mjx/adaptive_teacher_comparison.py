"""Bounded cycle-end comparison; invoked only by the user's training command."""
import math
import jax
import jax.numpy as jp
from rough_terrain_env import MODEL_FORWARD


def compare_baseline(env, make_policy, params, seed, duration=20., seeds=2):
    """Same reset seeds/horizon for zero residual and deterministic cycle best.

    No reset after termination. Report truncated episodes as truncated, not
    successes. This small diagnostic does not replace the PPO evaluation score.
    """
    if not math.isfinite(duration) or duration <= 0 or seeds < 1:
        raise ValueError('Comparison requires positive finite duration and seed count')
    policy = make_policy(params, deterministic=True)
    horizon = max(1, math.ceil(duration/float(env.dt)))
    metric_names = ('hold_planner_s', 'hold_contact_wait_s', 'foot_slip', 'plan_rejected')

    def rollout(key, use_policy):
        initial = env.reset(key)
        def tick(carry, _):
            state, key, totals, steps = carry
            key, action_key = jax.random.split(key)
            action = jax.lax.cond(use_policy, lambda: policy(state.obs, action_key)[0],
                                  lambda: jp.zeros(env.action_size))
            live = ~state.done.astype(jp.bool_)
            state = jax.lax.cond(live, lambda s: env.step(s, action), lambda s: s, state)
            increments = jp.stack((state.reward, *(state.metrics[name] for name in metric_names)))
            return (state, key, totals+jp.where(live, increments, 0.), steps+live), None
        (last, _, totals, steps), _ = jax.lax.scan(tick,
            (initial, key, jp.zeros(1+len(metric_names)), jp.asarray(0)), None, length=horizon)
        direction = initial.data.xmat[env._root_id] @ MODEL_FORWARD
        forward = jp.dot(last.data.qpos[:3]-initial.data.qpos[:3], direction)
        return jp.concatenate((totals, jp.array((forward, steps*env.dt, last.done,
            last.metrics['terrain_success'], last.metrics['termination/no_progress']))))

    keys = jax.random.split(jax.random.PRNGKey(seed), seeds)
    run = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
    names = ('reward', *metric_names, 'forward_m', 'duration_s', 'terminated', 'success', 'no_progress')
    report = {}
    for label, enabled in (('baseline', False), ('policy', True)):
        values = jax.device_get(run(keys, jp.asarray(enabled)))
        for column, name in enumerate(names):
            report[f'comparison/{label}/{name}'] = float(values[:, column].mean())
        report[f'comparison/{label}/truncated'] = 1.-report[f'comparison/{label}/terminated']
    report['comparison/policy_minus_baseline_forward_m'] = report['comparison/policy/forward_m']-report['comparison/baseline/forward_m']
    report['comparison/horizon_s'] = horizon*env.dt
    report['comparison/seed_count'] = seeds
    return report
