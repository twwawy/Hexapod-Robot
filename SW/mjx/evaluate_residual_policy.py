from __future__ import annotations

"""Evaluate a saved residual-RL policy in the MJX locomotion environment."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from hexapod_mjx.model import load_hexapod_model, repo_root_from
from hexapod_mjx.residual_controller import ACTION_DIM, ResidualControllerConfig, build_residual_controller
from hexapod_mjx.residual_env import ResidualEnvConfig, joint_group_index, reset_env, step_env
from hexapod_mjx.residual_rl import load_checkpoint, policy_mean



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved residual-RL hexapod policy.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default="SW/mjx/artifacts/residual_rl_policy.pkl")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-path", type=str, default="SW/mjx/artifacts/residual_rl_eval.json")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)
    bundle = load_hexapod_model(repo_root)
    controller_config = ResidualControllerConfig()
    controller_bundle = build_residual_controller(bundle, controller_config)
    env_config = ResidualEnvConfig(episode_steps=args.rollout_steps)
    policy_path = (repo_root / args.policy_path).resolve()
    train_state, metadata = load_checkpoint(policy_path)

    group_index = joint_group_index(bundle)
    key = jax.random.key(args.seed)
    state, obs = reset_env(bundle, controller_bundle, controller_config, key, args.num_envs)
    step_fn = jax.jit(
        lambda env_state, action: step_env(
            bundle,
            controller_bundle,
            controller_config,
            env_config,
            group_index,
            env_state,
            action,
        )
    )

    reward_history = []
    metric_history = []
    done_history = []
    for _ in range(args.rollout_steps):
        action = policy_mean(train_state.params, obs)
        state, obs, reward, done, metrics = step_fn(state, action)
        reward_history.append(reward)
        done_history.append(done)
        metric_history.append(metrics)

    rewards = jnp.stack(reward_history, axis=0)
    dones = jnp.stack(done_history, axis=0)
    metrics = jnp.stack(metric_history, axis=0)
    metric_means = jnp.mean(metrics.reshape(-1, metrics.shape[-1]), axis=0)
    report = {
        "policy_path": str(policy_path),
        "metadata": metadata,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "mean_reward": float(jnp.mean(rewards)),
        "mean_done": float(jnp.mean(dones)),
        "velocity_reward": float(metric_means[0]),
        "yaw_reward": float(metric_means[1]),
        "attitude_reward": float(metric_means[2]),
        "height_reward": float(metric_means[3]),
        "slip_cost": float(metric_means[4]),
        "control_cost": float(metric_means[5]),
        "action_cost": float(metric_means[6]),
        "forward_velocity": float(metric_means[7]),
        "lateral_velocity": float(metric_means[8]),
        "yaw_rate": float(metric_means[9]),
    }

    report_path = (repo_root / args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"repo_root: {repo_root}")
    print(f"policy_path: {policy_path}")
    print(f"report_path: {report_path}")
    print(f"mean_reward: {report['mean_reward']:.4f}")
    print(f"mean_done: {report['mean_done']:.4f}")


if __name__ == "__main__":
    main()
