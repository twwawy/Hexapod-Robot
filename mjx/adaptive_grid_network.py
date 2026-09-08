"""PPO actor/critic with spatial elevation encoder and existing vector features."""
from collections.abc import Mapping
import operator
import jax
import jax.numpy as jp
from flax import linen as nn
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks
from adaptive_grid import GRID_SIDE, GRID_CHANNELS, GRID_SIZE
from adaptive_gait_env import VECTOR_SIZE

NETWORK_CONTRACT = 'elevation_cnn_16_32_dense64_v1'


def observation_width(spec):
    """Accept environment integer sizes and PPO/checkpoint 1-D shape specs."""
    if isinstance(spec, (tuple, list)):
        if len(spec) != 1:
            raise ValueError(f'Elevation network requires a 1-D observation, got {spec!r}')
        spec = spec[0]
    try:
        width = operator.index(spec)
    except TypeError as exc:
        raise ValueError(f'Observation width must be a concrete integer, got {spec!r}') from exc
    if width < VECTOR_SIZE+GRID_SIZE:
        raise ValueError(f'Observation width {width} cannot contain vector and elevation grid')
    return width


class GridHead(nn.Module):
    output_size: int
    hidden_sizes: tuple = (256, 256, 128)

    @nn.compact
    def __call__(self, obs):
        batch = obs.shape[:-1]
        grid = obs[..., VECTOR_SIZE:VECTOR_SIZE+GRID_SIZE].reshape((-1, GRID_SIDE, GRID_SIDE, GRID_CHANNELS))
        grid = nn.swish(nn.Conv(16, (3, 3), strides=(2, 2))(grid))
        grid = nn.swish(nn.Conv(32, (3, 3), strides=(2, 2))(grid))
        # Flatten instead of global pooling: location of a step matters.
        grid = nn.swish(nn.Dense(64)(grid.reshape((grid.shape[0], -1)))).reshape(batch+(64,))
        vector = jp.concatenate((obs[..., :VECTOR_SIZE], obs[..., VECTOR_SIZE+GRID_SIZE:]), axis=-1)
        x = jp.concatenate((vector, grid), axis=-1)
        for size in self.hidden_sizes:
            x = nn.swish(nn.Dense(size)(x))
        return nn.Dense(self.output_size, kernel_init=nn.initializers.variance_scaling(.01, 'fan_in', 'uniform'))(x)


def make_grid_networks(observation_size, action_size,
                       preprocess_observations_fn=types.identity_observation_preprocessor,
                       policy_hidden_layer_sizes=(256, 256, 128),
                       value_hidden_layer_sizes=(256, 256, 128),
                       policy_obs_key='state', value_obs_key='privileged_state'):
    dist = distribution.NormalTanhDistribution(event_size=action_size)

    def make(key_name, outputs, hidden, value=False):
        module = GridHead(outputs, tuple(hidden))
        size = observation_width(observation_size[key_name] if isinstance(observation_size, Mapping) else observation_size)
        def apply(processor, params, obs):
            raw = obs[key_name] if isinstance(obs, Mapping) else obs
            selected = networks.normalizer_select(processor, key_name) if isinstance(obs, Mapping) and processor is not None else processor
            normalized = preprocess_observations_fn(raw, selected)
            # Channels already have physical scaling; retain known masks and
            # spatially shared meaning instead of per-cell running statistics.
            normalized = normalized.at[..., VECTOR_SIZE:VECTOR_SIZE+GRID_SIZE].set(raw[..., VECTOR_SIZE:VECTOR_SIZE+GRID_SIZE])
            result = module.apply(params, normalized)
            return jp.squeeze(result, -1) if value else result
        return networks.FeedForwardNetwork(
            init=lambda key: module.init(key, jp.zeros((1, size))), apply=apply)

    return PPONetworks(make(policy_obs_key, dist.param_size, policy_hidden_layer_sizes),
                       make(value_obs_key, 1, value_hidden_layer_sizes, True), dist)
