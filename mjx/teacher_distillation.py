"""Frozen legacy-policy distillation helpers for command-aware PPO.

The firmware environment owns a 146-D observation while the proven walking
policies use the preceding 142-D terrain contract.  This module keeps those
teachers frozen, projects current observations back to their exact layout, and
adds action imitation to the stock Brax PPO objective without changing the
environment reward or firmware controller.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jp
import numpy as np


LEGACY_OBSERVATION_SIZE = 142
STUDENT_OBSERVATION_SIZE = 146
LEGACY_OBSERVATION_CONTRACT = "firmware_state_collision_terrain_curriculum_v2"
V2_ACTION_CONTRACT = "stm32_firmware_cartesian_foot_residual_v2"
V3_ACTION_CONTRACT = "stm32_firmware_adaptive_swing_residual_v3"
XY_ACTION_MASK = jp.tile(jp.asarray((1.0, 1.0, 0.0)), 6)
_KERNEL_INITIALIZER_KEYS = (
    "policy_network_kernel_init_fn",
    "value_network_kernel_init_fn",
    "mean_kernel_init_fn",
)


@dataclass(frozen=True)
class FrozenTeacher:
    """A deterministic frozen PPO policy plus its semantic provenance."""

    name: str
    checkpoint: Path
    action_contract: str
    network_layers: tuple[int, ...]
    policy: Callable[[jax.Array, jax.Array], tuple[jax.Array, dict[str, Any]]]
    params: Any


def legacy_teacher_observation(observation: jax.Array) -> jax.Array:
    """Project command5+pitch observations onto the exact legacy 142-D layout."""
    if observation.shape[-1] != STUDENT_OBSERVATION_SIZE:
        raise ValueError(
            "teacher adapter requires a 146-D student observation, got "
            f"{observation.shape[-1]}"
        )
    # Legacy command=[speed, yaw].  Current indices 2:5 are
    # [height, pitch, roll], and index 145 is pitch feedforward.
    return jp.concatenate((observation[..., :2], observation[..., 5:145]), axis=-1)


def resolve_teacher_checkpoint(path: Path) -> Path:
    """Resolve an exact checkpoint, a run directory, or a checkpoints directory."""
    resolved = path.expanduser().resolve()
    if (resolved / "ppo_network_config.json").exists():
        return resolved
    best_pointer = resolved / "monitor" / "best_checkpoint.json"
    if best_pointer.exists():
        payload = json.loads(best_pointer.read_text(encoding="utf-8"))
        best = Path(payload.get("path", "")).expanduser().resolve()
        if (best / "ppo_network_config.json").exists():
            return best
        raise ValueError(f"invalid best teacher checkpoint in {best_pointer}: {best}")
    candidates = sorted(
        (
            child
            for child in resolved.iterdir()
            if child.is_dir()
            and child.name.isdigit()
            and (child / "ppo_network_config.json").exists()
        ),
        key=lambda child: int(child.name),
    ) if resolved.is_dir() else []
    if not candidates:
        raise ValueError(f"no PPO teacher checkpoint found at {resolved}")
    return candidates[-1]


def _run_metadata(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    for parent in checkpoint.parents:
        metadata_path = parent / "run_metadata.json"
        if metadata_path.exists():
            return metadata_path, json.loads(metadata_path.read_text(encoding="utf-8"))
    raise ValueError(f"teacher checkpoint has no run_metadata.json: {checkpoint}")


def _legacy_ppo_policy(params: Any, config: dict[str, Any]) -> Callable[..., Any]:
    """Rebuild a PPO policy while tolerating legacy null initializer fields.

    Brax serializes optional initializers such as ``mean_kernel_init_fn`` as
    JSON null.  Some Brax releases then try to resolve that value through
    ``KERNEL_INITIALIZER[None]`` while loading the policy.  Reconstructing the
    network from the already validated config avoids mutating old checkpoints
    and preserves their exact architecture.
    """
    from brax.training import networks as brax_networks
    from brax.training import types as brax_types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    kwargs = dict(config.get("network_factory_kwargs", {}))
    activation_name = kwargs.get("activation")
    if isinstance(activation_name, str):
        try:
            kwargs["activation"] = brax_networks.ACTIVATION[activation_name]
        except KeyError as exc:
            raise ValueError(
                f"unsupported teacher activation {activation_name!r}"
            ) from exc
    for key in _KERNEL_INITIALIZER_KEYS:
        initializer_name = kwargs.get(key)
        if initializer_name is None:
            continue
        if not isinstance(initializer_name, str):
            raise ValueError(f"invalid teacher initializer {key}={initializer_name!r}")
        try:
            kwargs[key] = brax_networks.KERNEL_INITIALIZER[initializer_name]
        except KeyError as exc:
            raise ValueError(
                f"unsupported teacher initializer {key}={initializer_name!r}"
            ) from exc

    preprocess = brax_types.identity_observation_preprocessor
    if bool(config.get("normalize_observations", False)):
        preprocess = running_statistics.normalize
    networks = ppo_networks.make_ppo_networks(
        observation_size=LEGACY_OBSERVATION_SIZE,
        action_size=18,
        preprocess_observations_fn=preprocess,
        **kwargs,
    )
    return ppo_networks.make_inference_fn(networks)(params, deterministic=True)


def load_frozen_teacher(
    path: Path,
    *,
    name: str,
    expected_action_contract: str,
) -> FrozenTeacher:
    """Load and semantically validate a deterministic legacy PPO teacher."""
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    checkpoint = resolve_teacher_checkpoint(path)
    config = json.loads(
        (checkpoint / "ppo_network_config.json").read_text(encoding="utf-8")
    )
    shape = config.get("observation_size", {}).get("shape")
    if shape != [LEGACY_OBSERVATION_SIZE] or config.get("action_size") != 18:
        raise ValueError(
            f"{name} tensor contract must be obs=142/action=18, got "
            f"obs={shape} action={config.get('action_size')}"
        )
    network_layers = tuple(
        config.get("network_factory_kwargs", {}).get(
            "policy_hidden_layer_sizes", ()
        )
    )
    metadata_path, metadata = _run_metadata(checkpoint)
    observation_contract = metadata.get("observation_contract_version")
    action_contract = metadata.get("action_contract_version")
    if observation_contract != LEGACY_OBSERVATION_CONTRACT:
        raise ValueError(
            f"{name} observation contract {observation_contract!r} is not "
            f"{LEGACY_OBSERVATION_CONTRACT!r} ({metadata_path})"
        )
    if action_contract != expected_action_contract:
        raise ValueError(
            f"{name} action contract {action_contract!r} is not "
            f"{expected_action_contract!r} ({metadata_path})"
        )
    params = ppo_checkpoint.load(checkpoint)
    policy = _legacy_ppo_policy(params, config)
    return FrozenTeacher(
        name, checkpoint, action_contract, network_layers, policy, params
    )


def _expand_legacy_vector(value: jax.Array, fill: float) -> jax.Array:
    if value.shape != (LEGACY_OBSERVATION_SIZE,):
        raise ValueError(f"expected legacy vector shape (142,), got {value.shape}")
    expanded = jp.full((STUDENT_OBSERVATION_SIZE,), fill, dtype=value.dtype)
    expanded = expanded.at[:2].set(value[:2])
    return expanded.at[5:145].set(value[2:])


def expand_v3_teacher_for_student(teacher: FrozenTeacher) -> tuple[Any, Any, Any]:
    """Expand a v3 teacher actor/normalizer from 142 to 146 student inputs.

    The critic is returned only to satisfy Brax's restore tuple and is ignored
    when ``restore_value_fn`` is false.  New command dimensions start with zero
    first-layer weights, so the initial student actor reproduces the teacher on
    the legacy observation projection.
    """
    if teacher.action_contract != V3_ACTION_CONTRACT:
        raise ValueError("only an adaptive-swing v3 teacher can initialize a student")
    normalizer, policy_params, value_params = teacher.params
    expanded_kernels = 0

    def expand_kernel(value: Any) -> Any:
        nonlocal expanded_kernels
        if hasattr(value, "shape") and len(value.shape) == 2 and value.shape[0] == 142:
            expanded_kernels += 1
            expanded = jp.zeros((146, value.shape[1]), dtype=value.dtype)
            expanded = expanded.at[:2].set(value[:2])
            return expanded.at[5:145].set(value[2:])
        return value

    policy_params = jax.tree_util.tree_map(expand_kernel, policy_params)
    if expanded_kernels != 1:
        raise ValueError(
            "expected exactly one 142-row teacher policy input kernel, found "
            f"{expanded_kernels}"
        )

    count = normalizer.count.to_numpy() if hasattr(normalizer.count, "to_numpy") else 1
    variance_fill = max(float(np.asarray(count).item()) - 1.0, 1.0)
    normalizer = normalizer.replace(
        mean=_expand_legacy_vector(normalizer.mean, 0.0),
        std=_expand_legacy_vector(normalizer.std, 1.0),
        summed_variance=_expand_legacy_vector(
            normalizer.summed_variance, variance_fill
        ),
    )
    return normalizer, policy_params, value_params


def _huber_action_loss(
    student_action: jax.Array,
    teacher_action: jax.Array,
    mask: jax.Array,
    delta: float,
) -> jax.Array:
    error = jp.abs(student_action - jp.stop_gradient(teacher_action))
    loss = jp.where(error <= delta, 0.5 * jp.square(error), delta * (error - 0.5 * delta))
    return jp.mean(jp.sum(loss * mask, axis=-1) / jp.maximum(jp.sum(mask), 1.0))


@contextmanager
def install_distillation_loss(
    *,
    v3_teacher: FrozenTeacher | None,
    v2_teacher: FrozenTeacher | None,
    v3_weight: float,
    v2_xy_weight: float,
    huber_delta: float,
) -> Iterator[None]:
    """Temporarily augment Brax PPO with frozen-teacher imitation losses."""
    from brax.training.agents.ppo import losses as ppo_losses

    original = ppo_losses.compute_ppo_loss

    def compute_teacher_ppo_loss(
        params: Any,
        normalizer_params: Any,
        data: Any,
        rng: jax.Array,
        ppo_network: Any,
        **kwargs: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        total, metrics = original(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network,
            **kwargs,
        )
        legacy_observation = legacy_teacher_observation(data.observation)
        student_logits = ppo_network.policy_network.apply(
            normalizer_params, params.policy, data.observation
        )
        student_action = ppo_network.parametric_action_distribution.mode(
            student_logits
        )
        distill_total = jp.zeros(())
        result = dict(metrics)
        if v3_teacher is not None and v3_weight > 0.0:
            teacher_action, _ = v3_teacher.policy(legacy_observation, rng)
            v3_loss = _huber_action_loss(
                student_action, teacher_action, jp.ones(18), huber_delta
            )
            distill_total = distill_total + v3_weight * v3_loss
            result["distill_v3_action_loss"] = v3_loss
            result["distill_v3_action_rmse"] = jp.sqrt(
                jp.mean(jp.square(student_action - teacher_action))
            )
        if v2_teacher is not None and v2_xy_weight > 0.0:
            teacher_action, _ = v2_teacher.policy(legacy_observation, rng)
            v2_loss = _huber_action_loss(
                student_action, teacher_action, XY_ACTION_MASK, huber_delta
            )
            distill_total = distill_total + v2_xy_weight * v2_loss
            result["distill_v2_xy_loss"] = v2_loss
            squared = jp.square(student_action - teacher_action) * XY_ACTION_MASK
            result["distill_v2_xy_rmse"] = jp.sqrt(
                jp.mean(jp.sum(squared, axis=-1) / jp.sum(XY_ACTION_MASK))
            )
        total = total + distill_total
        result["distill_loss"] = distill_total
        result["total_loss"] = total
        return total, result

    ppo_losses.compute_ppo_loss = compute_teacher_ppo_loss
    try:
        yield
    finally:
        ppo_losses.compute_ppo_loss = original
