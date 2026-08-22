from __future__ import annotations

"""Small PPO implementation without extra dependencies."""

from dataclasses import dataclass, asdict
from typing import Any, NamedTuple
import json
import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


LOG_2PI = float(math.log(2.0 * math.pi))


class AdamState(NamedTuple):
    step: jnp.ndarray
    m: Any
    v: Any


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 32
    rollout_steps: int = 64
    num_updates: int = 8
    minibatch_size: int = 128
    ppo_epochs: int = 4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.002
    min_policy_log_std: float = -3.0
    max_policy_log_std: float = -1.0
    hidden_size: int = 128
    seed: int = 0
    output_path: str = "SW/mjx/artifacts/residual_rl_policy.pkl"
    metrics_path: str = "SW/mjx/artifacts/residual_rl_metrics.json"


class PolicyBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    old_log_prob: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray


class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    log_prob: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    metrics: jnp.ndarray
    last_obs: jnp.ndarray


class TrainState(NamedTuple):
    params: dict[str, Any]
    optimizer_state: AdamState



def init_mlp(key: jax.Array, layer_sizes: tuple[int, ...], scale: float = 1.0) -> list[dict[str, jnp.ndarray]]:
    params: list[dict[str, jnp.ndarray]] = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for k, in_size, out_size in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        weight_scale = scale * math.sqrt(2.0 / max(1, in_size))
        params.append(
            {
                "w": jax.random.normal(k, (in_size, out_size), dtype=jnp.float32) * weight_scale,
                "b": jnp.zeros((out_size,), dtype=jnp.float32),
            }
        )
    return params



def apply_mlp(params: list[dict[str, jnp.ndarray]], obs: jnp.ndarray, *, final_tanh: bool = False) -> jnp.ndarray:
    x = obs
    for layer_idx, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if layer_idx < len(params) - 1:
            x = jnp.tanh(x)
        elif final_tanh:
            x = jnp.tanh(x)
    return x



def init_train_state(obs_dim: int, action_dim: int, config: PPOConfig) -> TrainState:
    key = jax.random.key(config.seed)
    policy_key, value_key = jax.random.split(key)
    params = {
        "policy_layers": init_mlp(policy_key, (obs_dim, config.hidden_size, config.hidden_size, action_dim), scale=0.8),
        # The residual is only ±3 cm.  Start with a conservative 0.37
        # normalized-action standard deviation and never let PPO turn it into
        # a saturated bang-bang foot-height controller.
        "policy_log_std": jnp.full((action_dim,), -1.0, dtype=jnp.float32),
        "value_layers": init_mlp(value_key, (obs_dim, config.hidden_size, config.hidden_size, 1), scale=0.8),
    }
    optimizer_state = adam_init(params)
    return TrainState(params=params, optimizer_state=optimizer_state)



def policy_mean(params: dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
    return apply_mlp(params["policy_layers"], obs, final_tanh=True)



def value_predict(params: dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
    return apply_mlp(params["value_layers"], obs).squeeze(-1)



def gaussian_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(((action - mean) / std) ** 2 + 2.0 * log_std + LOG_2PI, axis=-1)



def gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(log_std + 0.5 * (1.0 + LOG_2PI), axis=-1)



def _bounded_policy_log_std(params: dict[str, Any], config: PPOConfig) -> jnp.ndarray:
    return jnp.clip(params["policy_log_std"], config.min_policy_log_std, config.max_policy_log_std)


def sample_action(
    params: dict[str, Any],
    obs: jnp.ndarray,
    key: jax.Array,
    config: PPOConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = policy_mean(params, obs)
    log_std = _bounded_policy_log_std(params, config)[None, :]
    noise = jax.random.normal(key, mean.shape, dtype=jnp.float32)
    action = mean + noise * jnp.exp(log_std)
    log_prob = gaussian_log_prob(mean, log_std, action)
    value = value_predict(params, obs)
    return action, log_prob, value



def adam_init(params: Any) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(step=jnp.array(0, dtype=jnp.int32), m=zeros, v=zeros)



def adam_update(params: Any, grads: Any, state: AdamState, lr: float) -> tuple[Any, AdamState]:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = state.step + 1
    m = jax.tree_util.tree_map(lambda m_prev, g: beta1 * m_prev + (1.0 - beta1) * g, state.m, grads)
    v = jax.tree_util.tree_map(lambda v_prev, g: beta2 * v_prev + (1.0 - beta2) * (g * g), state.v, grads)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**step), v)
    new_params = jax.tree_util.tree_map(
        lambda p, m_value, v_value: p - lr * m_value / (jnp.sqrt(v_value) + eps),
        params,
        m_hat,
        v_hat,
    )
    return new_params, AdamState(step=step, m=m, v=v)



def compute_gae(rewards: jnp.ndarray, dones: jnp.ndarray, values: jnp.ndarray, last_values: jnp.ndarray, config: PPOConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    advantages = []
    gae = jnp.zeros_like(last_values)
    next_values = last_values
    for step in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - dones[step]
        delta = rewards[step] + config.gamma * next_values * mask - values[step]
        gae = delta + config.gamma * config.gae_lambda * mask * gae
        advantages.append(gae)
        next_values = values[step]
    advantages = jnp.stack(list(reversed(advantages)), axis=0)
    returns = advantages + values
    return advantages, returns



def _ppo_loss(params: dict[str, Any], batch: PolicyBatch, config: PPOConfig) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    mean = policy_mean(params, batch.obs)
    log_std = _bounded_policy_log_std(params, config)[None, :]
    new_log_prob = gaussian_log_prob(mean, log_std, batch.actions)
    ratio = jnp.exp(new_log_prob - batch.old_log_prob)
    clipped_ratio = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    actor_loss = -jnp.mean(jnp.minimum(ratio * batch.advantages, clipped_ratio * batch.advantages))

    values = value_predict(params, batch.obs)
    value_loss = 0.5 * jnp.mean((values - batch.returns) ** 2)
    entropy = jnp.mean(gaussian_entropy(log_std))
    total = actor_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    return total, (actor_loss, value_loss, entropy)


PPO_GRAD = jax.jit(jax.value_and_grad(_ppo_loss, has_aux=True), static_argnums=(2,))



def ppo_update(train_state: TrainState, batch: PolicyBatch, config: PPOConfig) -> tuple[TrainState, dict[str, float]]:
    (loss, (actor_loss, value_loss, entropy)), grads = PPO_GRAD(train_state.params, batch, config)
    new_params, new_opt_state = adam_update(train_state.params, grads, train_state.optimizer_state, config.learning_rate)
    metrics = {
        "loss": float(loss),
        "actor_loss": float(actor_loss),
        "value_loss": float(value_loss),
        "entropy": float(entropy),
    }
    return TrainState(params=new_params, optimizer_state=new_opt_state), metrics



def flatten_rollout(rollout: RolloutBatch, advantages: jnp.ndarray, returns: jnp.ndarray) -> PolicyBatch:
    obs = rollout.obs.reshape((-1, rollout.obs.shape[-1]))
    actions = rollout.actions.reshape((-1, rollout.actions.shape[-1]))
    old_log_prob = rollout.log_prob.reshape((-1,))
    advantages = advantages.reshape((-1,))
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    returns = returns.reshape((-1,))
    return PolicyBatch(obs=obs, actions=actions, old_log_prob=old_log_prob, advantages=advantages, returns=returns)



def minibatches(batch: PolicyBatch, config: PPOConfig, seed: int) -> list[PolicyBatch]:
    size = batch.obs.shape[0]
    permutation = np.random.default_rng(seed).permutation(size)
    items: list[PolicyBatch] = []
    for start in range(0, size, config.minibatch_size):
        index = permutation[start : start + config.minibatch_size]
        if index.size == 0:
            continue
        items.append(
            PolicyBatch(
                obs=batch.obs[index],
                actions=batch.actions[index],
                old_log_prob=batch.old_log_prob[index],
                advantages=batch.advantages[index],
                returns=batch.returns[index],
            )
        )
    return items



def save_checkpoint(path: Path, train_state: TrainState, metadata: dict[str, Any]) -> None:
    payload = {
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x), train_state.params),
        "optimizer_state": jax.tree_util.tree_map(lambda x: np.asarray(x), train_state.optimizer_state),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)



def load_checkpoint(path: Path) -> tuple[TrainState, dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload["params"])
    optimizer_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload["optimizer_state"])
    return TrainState(params=params, optimizer_state=AdamState(**optimizer_state) if isinstance(optimizer_state, dict) else optimizer_state), payload["metadata"]



def write_metrics(path: Path, config: PPOConfig, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "history": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
