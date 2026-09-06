"""Asymmetric RSL-RL PPO defaults for the perceptive residual policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HexapodPerceptivePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100
    experiment_name = "hexapod_perceptive_residual"
    logger = "wandb"
    wandb_project = "hexapod-isaac-lidar-depth-curriculum"
    clip_actions = 1.0
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        # A unit action requests up to 100 mm Cartesian residual.  Keep the
        # initial exploration near the stable firmware gait (about 10 mm).
        init_noise_std=0.1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
