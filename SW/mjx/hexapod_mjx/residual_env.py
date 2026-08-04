from __future__ import annotations

"""MJX batch environment for residual-RL hexapod locomotion."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from .model import BASE_COLLISION_HALF_SIZE, BASE_COLLISION_POS, FOOT_PROXY_RADIUS, HexapodModelBundle

from .residual_controller import (
    ACTION_DIM,
    ResidualControllerBundle,
    ResidualControllerConfig,
    ResidualControllerState,
    body_to_world,
    body_velocity_components,
    controller_step,
    policy_dt,
    quat_roll_pitch_yaw,
    reset_controller_state,
)


class ResidualEnvState(NamedTuple):
    data: mjx.Data
    controller_state: ResidualControllerState
    command: jnp.ndarray
    prev_root_pos: jnp.ndarray
    prev_yaw: jnp.ndarray
    prev_foot_world: jnp.ndarray
    body_velocity_world: jnp.ndarray
    yaw_rate: jnp.ndarray
    terminated: jnp.ndarray


class ResidualTransition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    metrics: jnp.ndarray


@dataclass(frozen=True)
class ResidualEnvConfig:
    episode_steps: int = 96
    velocity_reward_sigma: float = 0.18
    yaw_reward_sigma: float = 0.45
    attitude_reward_sigma: float = 0.35
    height_reward_sigma: float = 0.03
    slip_penalty: float = 0.10
    energy_penalty: float = 0.0015
    action_penalty: float = 0.015
    body_contact_penalty: float = 10.0
    yaw_rate_clip: float = 3.0
    foot_contact_height: float = 0.03
    termination_contact_z: float = 0.0



@dataclass(frozen=True)
class CommandSamplingConfig:
    forward_min: float
    forward_max: float
    lateral_limit: float
    yaw_limit: float


@dataclass(frozen=True)
class CommandCurriculumConfig:
    mode: str = "staged"
    forward_only_updates: int = 120
    yaw_stage_updates: int = 120
    forward_only_scale: float = 0.60
    yaw_stage_scale: float = 0.35


_BASE_COLLISION_CORNERS = jnp.asarray(
    [
        [
            BASE_COLLISION_POS[0] + sx * BASE_COLLISION_HALF_SIZE[0],
            BASE_COLLISION_POS[1] + sy * BASE_COLLISION_HALF_SIZE[1],
            BASE_COLLISION_POS[2] + sz * BASE_COLLISION_HALF_SIZE[2],
        ]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ],
    dtype=jnp.float32,
)


def joint_group_index(bundle: HexapodModelBundle) -> jnp.ndarray:
    return jnp.asarray([int(name.split("_")[-1]) - 1 for name in bundle.joint_names], dtype=jnp.int32)



def _foot_support_world(data: mjx.Data, controller_bundle: ResidualControllerBundle) -> jnp.ndarray:
    ids = controller_bundle.foot_support_geom_ids.reshape(-1)
    supports = jnp.take(data.geom_xpos, ids, axis=1)
    batch_size = data.geom_xpos.shape[0]
    leg_count = controller_bundle.foot_support_geom_ids.shape[0]
    point_count = controller_bundle.foot_support_geom_ids.shape[1]
    return supports.reshape(batch_size, leg_count, point_count, 3)


def _foot_world(data: mjx.Data, controller_bundle: ResidualControllerBundle) -> jnp.ndarray:
    return jnp.mean(_foot_support_world(data, controller_bundle), axis=2)


def _foot_support_height(data: mjx.Data, controller_bundle: ResidualControllerBundle) -> jnp.ndarray:
    support_world = _foot_support_world(data, controller_bundle)
    return jnp.min(support_world[:, :, :, 2] - FOOT_PROXY_RADIUS, axis=2)



def _make_batched_data(
    bundle: HexapodModelBundle,
    controller_bundle: ResidualControllerBundle,
    batch_size: int,
) -> mjx.Data:
    qpos0 = np.zeros((batch_size, bundle.model.nq), dtype=np.float32)
    qvel0 = np.zeros((batch_size, bundle.model.nv), dtype=np.float32)
    qpos0[:, 0:3] = np.array([0.0, 0.0, float(controller_bundle.reset_root_height)], dtype=np.float32)
    qpos0[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    qpos0[:, bundle.joint_qpos_adr] = np.asarray(controller_bundle.neutral_joint_pose)
    batch_data = jax.vmap(lambda _: mjx.make_data(bundle.mjx_model))(jnp.arange(batch_size, dtype=jnp.int32))
    batch_data = batch_data.replace(qpos=jnp.asarray(qpos0), qvel=jnp.asarray(qvel0))
    batch_data = jax.vmap(mjx.forward, in_axes=(None, 0))(bundle.mjx_model, batch_data)
    return batch_data


def command_sampling_config(
    controller_config: ResidualControllerConfig,
    curriculum_config: CommandCurriculumConfig | None,
    update_index: int,
) -> CommandSamplingConfig:
    forward_limit, lateral_limit, yaw_limit = map(float, controller_config.command_limits)
    if curriculum_config is None or curriculum_config.mode == "none":
        return CommandSamplingConfig(
            forward_min=-forward_limit,
            forward_max=forward_limit,
            lateral_limit=lateral_limit,
            yaw_limit=yaw_limit,
        )

    if update_index <= curriculum_config.forward_only_updates:
        return CommandSamplingConfig(
            forward_min=0.0,
            forward_max=forward_limit * curriculum_config.forward_only_scale,
            lateral_limit=0.0,
            yaw_limit=0.0,
        )

    if update_index <= curriculum_config.forward_only_updates + curriculum_config.yaw_stage_updates:
        return CommandSamplingConfig(
            forward_min=0.0,
            forward_max=forward_limit,
            lateral_limit=0.0,
            yaw_limit=yaw_limit * curriculum_config.yaw_stage_scale,
        )

    return CommandSamplingConfig(
        forward_min=-forward_limit,
        forward_max=forward_limit,
        lateral_limit=lateral_limit,
        yaw_limit=yaw_limit,
    )



def sample_commands(
    key: jax.Array,
    batch_size: int,
    command_config: CommandSamplingConfig,
) -> jnp.ndarray:
    forward_key, lateral_key, yaw_key = jax.random.split(key, 3)
    forward = jax.random.uniform(
        forward_key,
        (batch_size,),
        minval=command_config.forward_min,
        maxval=command_config.forward_max,
    )
    lateral = (
        jax.random.uniform(
            lateral_key,
            (batch_size,),
            minval=-command_config.lateral_limit,
            maxval=command_config.lateral_limit,
        )
        if command_config.lateral_limit > 0.0
        else jnp.zeros((batch_size,), dtype=jnp.float32)
    )
    yaw = (
        jax.random.uniform(
            yaw_key,
            (batch_size,),
            minval=-command_config.yaw_limit,
            maxval=command_config.yaw_limit,
        )
        if command_config.yaw_limit > 0.0
        else jnp.zeros((batch_size,), dtype=jnp.float32)
    )
    return jnp.stack([forward, lateral, yaw], axis=-1).astype(jnp.float32)



def reset_env(
    bundle: HexapodModelBundle,
    controller_bundle: ResidualControllerBundle,
    controller_config: ResidualControllerConfig,
    key: jax.Array,
    batch_size: int,
    command_config: CommandSamplingConfig | None = None,
) -> tuple[ResidualEnvState, jnp.ndarray]:
    data = _make_batched_data(bundle, controller_bundle, batch_size)
    sampling = command_config or command_sampling_config(controller_config, None, 0)
    command = sample_commands(key, batch_size, sampling)
    controller_state = reset_controller_state(batch_size)
    root_pos = data.qpos[:, 0:3]
    _, _, yaw = quat_roll_pitch_yaw(data.qpos[:, 3:7])
    foot_world = _foot_world(data, controller_bundle)
    state = ResidualEnvState(
        data=data,
        controller_state=controller_state,
        command=command,
        prev_root_pos=root_pos,
        prev_yaw=yaw,
        prev_foot_world=foot_world,

        body_velocity_world=jnp.zeros((batch_size, 3), dtype=jnp.float32),
        yaw_rate=jnp.zeros((batch_size,), dtype=jnp.float32),
        terminated=jnp.zeros((batch_size,), dtype=bool),
    )
    return state, make_observation(state, controller_bundle, controller_config)



def _pd_torque(
    bundle: HexapodModelBundle,
    joint_targets: jnp.ndarray,
    data: mjx.Data,
    group_index: jnp.ndarray,
    controller_config: ResidualControllerConfig,
) -> jnp.ndarray:
    qj = data.qpos[:, jnp.asarray(bundle.joint_qpos_adr)]
    qv = data.qvel[:, jnp.asarray(bundle.joint_dof_adr)]
    kp = jnp.asarray(controller_config.pd_kp, dtype=jnp.float32)[group_index][None, :]
    kd = jnp.asarray(controller_config.pd_kd, dtype=jnp.float32)[group_index][None, :]
    tau_limit = jnp.asarray(controller_config.torque_limit, dtype=jnp.float32)[group_index][None, :]
    tau = kp * (joint_targets - qj) - kd * qv
    return jnp.clip(tau, -tau_limit, tau_limit)


def _masked_tree(mask: jnp.ndarray, old_tree, new_tree):
    def select(old_leaf, new_leaf):
        expand = (mask.shape[0],) + (1,) * (new_leaf.ndim - 1)
        return jnp.where(mask.reshape(expand), old_leaf, new_leaf)

    return jax.tree_util.tree_map(select, old_tree, new_tree)



def _termination_mask(
    env_config: ResidualEnvConfig,
    data: mjx.Data,
) -> jnp.ndarray:
    """Terminate when the simplified body box itself touches the ground.

    The user-facing meaning of "fallen" here is no longer a tilt-angle or root-
    height threshold. We only stop the rollout once the main body/back collision
    box reaches the floor plane.
    """
    batch_size = data.qpos.shape[0]
    body_corners = jnp.broadcast_to(_BASE_COLLISION_CORNERS[None, :, :], (batch_size, _BASE_COLLISION_CORNERS.shape[0], 3))
    world_corners = body_to_world(data.qpos[:, 0:3], data.qpos[:, 3:7], body_corners)
    lowest_body_z = jnp.min(world_corners[:, :, 2], axis=1)
    return lowest_body_z <= env_config.termination_contact_z



def make_observation(
    state: ResidualEnvState,
    controller_bundle: ResidualControllerBundle,
    controller_config: ResidualControllerConfig,
) -> jnp.ndarray:
    quat = state.data.qpos[:, 3:7]
    roll, pitch, _ = quat_roll_pitch_yaw(quat)
    forward_velocity, lateral_velocity = body_velocity_components(quat, state.body_velocity_world)
    body_height_error = state.data.qpos[:, 2] - controller_bundle.reset_root_height
    joint_qpos = state.data.qpos[:, 7:]
    joint_qvel = state.data.qvel[:, 6:]
    foot_contact_height = _foot_support_height(state.data, controller_bundle)
    contacts = (foot_contact_height < controller_config.foot_contact_height).astype(jnp.float32)

    phase_angle = state.controller_state.phase * (2.0 * jnp.pi)
    return jnp.concatenate(
        [
            state.command,
            jnp.stack([forward_velocity, lateral_velocity, state.yaw_rate], axis=-1),
            jnp.stack([roll, pitch, body_height_error], axis=-1),
            joint_qpos,
            joint_qvel,
            contacts,
            jnp.stack([jnp.sin(phase_angle), jnp.cos(phase_angle)], axis=-1),
            state.controller_state.prev_action,
        ],
        axis=-1,
    )



def step_env(
    bundle: HexapodModelBundle,
    controller_bundle: ResidualControllerBundle,
    controller_config: ResidualControllerConfig,
    env_config: ResidualEnvConfig,
    group_index: jnp.ndarray,
    state: ResidualEnvState,
    action: jnp.ndarray,
) -> tuple[ResidualEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance the batched environment by one policy action.

    If the simplified main body touches the floor during any inner MuJoCo step,
    that environment is marked done immediately and frozen for the rest of the
    action rollout.
    """
    action = jnp.clip(action, -1.0, 1.0)
    previous_action = state.controller_state.prev_action

    def control_tick(carry, _):
        data, controller_state, torque_accum, terminated = carry
        next_controller_state, joint_targets, _ = controller_step(
            bundle,
            controller_bundle,
            controller_config,
            controller_state,
            state.command,
            action,
        )
        next_controller_state = _masked_tree(terminated, controller_state, next_controller_state)
        tau = _pd_torque(bundle, joint_targets, data, group_index, controller_config)
        active = (~terminated).astype(jnp.float32)

        def physics_step(inner_carry, _):
            inner_data, inner_terminated = inner_carry
            qfrc = inner_data.qfrc_applied.at[:, :].set(0.0)
            qfrc = qfrc.at[:, jnp.asarray(bundle.joint_dof_adr)].set(tau)
            inner_data = inner_data.replace(qfrc_applied=qfrc)
            stepped_data = jax.vmap(mjx.step, in_axes=(None, 0))(bundle.mjx_model, inner_data)
            updated_terminated = jnp.logical_or(inner_terminated, _termination_mask(env_config, stepped_data))
            kept_data = _masked_tree(inner_terminated, inner_data, stepped_data)
            return (kept_data, updated_terminated), None

        (next_data, terminated), _ = jax.lax.scan(
            physics_step,
            (data, terminated),
            xs=None,
            length=controller_config.physics_steps_per_control,
        )
        torque_accum = torque_accum + active * jnp.mean(tau * tau, axis=1)
        return (next_data, next_controller_state, torque_accum, terminated), None

    batch_size = action.shape[0]
    init_torque = jnp.zeros((batch_size,), dtype=jnp.float32)
    (data_f, controller_state_f, torque_accum, terminated_f), _ = jax.lax.scan(
        control_tick,
        (state.data, state.controller_state, init_torque, state.terminated),
        xs=None,
        length=controller_config.policy_controls_per_action,
    )

    step_dt = policy_dt(bundle, controller_config)
    root_pos = data_f.qpos[:, 0:3]
    world_velocity = (root_pos - state.prev_root_pos) / step_dt
    _, _, yaw = quat_roll_pitch_yaw(data_f.qpos[:, 3:7])
    yaw_delta = jnp.arctan2(jnp.sin(yaw - state.prev_yaw), jnp.cos(yaw - state.prev_yaw))
    yaw_rate = jnp.clip(yaw_delta / step_dt, -env_config.yaw_rate_clip, env_config.yaw_rate_clip)
    foot_world = _foot_world(data_f, controller_bundle)
    foot_xy_velocity = (foot_world[:, :, 0:2] - state.prev_foot_world[:, :, 0:2]) / step_dt


    next_state = ResidualEnvState(
        data=data_f,
        controller_state=controller_state_f,
        command=state.command,
        prev_root_pos=root_pos,
        prev_yaw=yaw,
        prev_foot_world=foot_world,
        body_velocity_world=world_velocity,
        yaw_rate=yaw_rate,
        terminated=terminated_f,
    )

    reward, done, metrics = compute_reward(
        controller_bundle,
        controller_config,
        env_config,
        next_state,
        action,
        torque_accum / controller_config.policy_controls_per_action,
        foot_xy_velocity,
        previous_action,
    )
    reward = jnp.where(state.terminated, 0.0, reward)
    obs = make_observation(next_state, controller_bundle, controller_config)
    return next_state, obs, reward, done, metrics



def compute_reward(
    controller_bundle: ResidualControllerBundle,
    controller_config: ResidualControllerConfig,
    env_config: ResidualEnvConfig,
    state: ResidualEnvState,
    action: jnp.ndarray,
    control_cost: jnp.ndarray,
    foot_xy_velocity: jnp.ndarray,
    previous_action: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    quat = state.data.qpos[:, 3:7]
    roll, pitch, _ = quat_roll_pitch_yaw(quat)
    forward_velocity, lateral_velocity = body_velocity_components(quat, state.body_velocity_world)
    velocity_error = (forward_velocity - state.command[:, 0]) ** 2 + (lateral_velocity - state.command[:, 1]) ** 2
    yaw_error = (state.yaw_rate - state.command[:, 2]) ** 2
    height_error = (state.data.qpos[:, 2] - controller_bundle.reset_root_height) ** 2

    velocity_reward = jnp.exp(-velocity_error / env_config.velocity_reward_sigma)
    yaw_reward = jnp.exp(-yaw_error / env_config.yaw_reward_sigma)
    attitude_reward = jnp.exp(-(roll * roll + pitch * pitch) / env_config.attitude_reward_sigma)
    height_reward = jnp.exp(-height_error / env_config.height_reward_sigma)

    contacts = (_foot_support_height(state.data, controller_bundle) < env_config.foot_contact_height).astype(jnp.float32)
    slip_cost = jnp.sum(contacts * jnp.sum(foot_xy_velocity * foot_xy_velocity, axis=-1), axis=1)
    action_cost = jnp.mean((action - previous_action) ** 2, axis=1)
    body_contact = state.terminated.astype(jnp.float32)

    reward = (
        1.5 * velocity_reward
        + 0.6 * yaw_reward
        + 0.8 * attitude_reward
        + 0.5 * height_reward
        - env_config.slip_penalty * slip_cost
        - env_config.energy_penalty * control_cost
        - env_config.action_penalty * action_cost
        - env_config.body_contact_penalty * body_contact
    )


    done = state.terminated

    metrics = jnp.stack(
        [
            velocity_reward,
            yaw_reward,
            attitude_reward,
            height_reward,
            slip_cost,
            control_cost,
            action_cost,
            forward_velocity,
            lateral_velocity,
            state.yaw_rate,
            body_contact,
        ],
        axis=-1,
    )
    return reward.astype(jnp.float32), done.astype(jnp.float32), metrics.astype(jnp.float32)
