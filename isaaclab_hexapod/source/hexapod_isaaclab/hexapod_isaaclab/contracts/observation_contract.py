"""Versioned legacy and perceptive observation builders.

The actor cutover deliberately removes the 15 simulator terrain heights from
the deployable observation.  The critic may receive them as privileged input.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


LEGACY_SIZE = 146
TERRAIN_HEIGHT_SLICE = slice(76, 91)
ACTOR_PROPRIO_SIZE = 131
TERRAIN_LATENT_SIZE = 64
ACTOR_OBSERVATION_SIZE = ACTOR_PROPRIO_SIZE + TERRAIN_LATENT_SIZE
CRITIC_PRIVILEGED_SIZE = 30
CRITIC_OBSERVATION_SIZE = ACTOR_OBSERVATION_SIZE + CRITIC_PRIVILEGED_SIZE

LEGACY_SLICES: Mapping[str, slice] = {
    "command": slice(0, 5),
    "root_local_linear_velocity": slice(5, 8),
    "root_world_angular_velocity": slice(8, 11),
    "projected_gravity": slice(11, 14),
    "relative_roll_pitch": slice(14, 16),
    "joint_position_error": slice(16, 34),
    "joint_velocity_scaled": slice(34, 52),
    "foot_position_controller": slice(52, 70),
    "foot_contact": slice(70, 76),
    "terrain_height_gt": TERRAIN_HEIGHT_SLICE,
    "gait_progress": slice(91, 97),
    "gait_state_scaled": slice(97, 103),
    "applied_twist": slice(103, 107),
    "ik_valid": slice(107, 113),
    "policy_valid": slice(113, 119),
    "foot_limited": slice(119, 125),
    "controller_acceptance": slice(125, 127),
    "last_action": slice(127, 145),
    "pitch_feedforward": slice(145, 146),
}


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


def build_legacy_observation(
    *,
    command: torch.Tensor,
    root_local_linear_velocity: torch.Tensor,
    root_world_angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    relative_roll_pitch: torch.Tensor,
    joint_position_error: torch.Tensor,
    joint_velocity: torch.Tensor,
    foot_position_controller: torch.Tensor,
    foot_contact: torch.Tensor,
    terrain_height_gt: torch.Tensor,
    gait_progress: torch.Tensor,
    gait_state: torch.Tensor,
    applied_twist: torch.Tensor,
    ik_valid: torch.Tensor,
    policy_valid: torch.Tensor,
    foot_limited: torch.Tensor,
    gait_accepted: torch.Tensor,
    posture_accepted: torch.Tensor,
    last_action: torch.Tensor,
    pitch_feedforward: torch.Tensor,
) -> torch.Tensor:
    """Build the exact 146-D MJX-compatible observation."""
    observation = torch.cat(
        (
            _flat(command),
            _flat(root_local_linear_velocity),
            0.2 * _flat(root_world_angular_velocity),
            _flat(projected_gravity),
            _flat(relative_roll_pitch),
            _flat(joint_position_error),
            0.1 * _flat(joint_velocity),
            _flat(foot_position_controller),
            _flat(foot_contact).to(dtype=command.dtype),
            _flat(terrain_height_gt),
            _flat(gait_progress),
            _flat(gait_state).to(dtype=command.dtype) / 2.0,
            _flat(applied_twist),
            _flat(ik_valid).to(dtype=command.dtype),
            _flat(policy_valid).to(dtype=command.dtype),
            _flat(foot_limited).to(dtype=command.dtype),
            _flat(gait_accepted).to(dtype=command.dtype),
            _flat(posture_accepted).to(dtype=command.dtype),
            _flat(last_action),
            _flat(pitch_feedforward),
        ),
        dim=-1,
    )
    if observation.shape[-1] != LEGACY_SIZE:
        raise ValueError(f"legacy observation must be {LEGACY_SIZE}-D, got {observation.shape}")
    return observation


def actor_proprioception(legacy_observation: torch.Tensor) -> torch.Tensor:
    """Remove simulator ground-truth terrain from the deployable actor input."""
    if legacy_observation.shape[-1] != LEGACY_SIZE:
        raise ValueError(f"expected {LEGACY_SIZE}-D observation")
    result = torch.cat(
        (
            legacy_observation[..., : TERRAIN_HEIGHT_SLICE.start],
            legacy_observation[..., TERRAIN_HEIGHT_SLICE.stop :],
        ),
        dim=-1,
    )
    if result.shape[-1] != ACTOR_PROPRIO_SIZE:
        raise RuntimeError("actor proprioception contract drifted")
    return result


def build_actor_observation(
    legacy_observation: torch.Tensor, terrain_latent: torch.Tensor
) -> torch.Tensor:
    if terrain_latent.shape[-1] != TERRAIN_LATENT_SIZE:
        raise ValueError(f"terrain latent must be {TERRAIN_LATENT_SIZE}-D")
    return torch.cat((actor_proprioception(legacy_observation), terrain_latent), dim=-1)


def build_critic_observation(
    actor_observation: torch.Tensor, privileged: torch.Tensor
) -> torch.Tensor:
    if actor_observation.shape[-1] != ACTOR_OBSERVATION_SIZE:
        raise ValueError(f"actor observation must be {ACTOR_OBSERVATION_SIZE}-D")
    if privileged.shape[-1] != CRITIC_PRIVILEGED_SIZE:
        raise ValueError(f"critic privileged input must be {CRITIC_PRIVILEGED_SIZE}-D")
    return torch.cat((actor_observation, privileged), dim=-1)
